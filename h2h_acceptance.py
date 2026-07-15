"""H2H 本机与服务器验收共用的正确性、性能和多进程测量工具。"""

from __future__ import annotations

import ctypes
import gc
import math
import multiprocessing as mp
import os
import pickle
import platform
import queue
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx


@dataclass(frozen=True)
class QueryWorkload:
    """保存可复现的有向查询节点对及实际使用的源节点集合。"""

    sources: tuple[int, ...]
    targets: tuple[int, ...]
    unique_sources: tuple[int, ...]
    seed: int

    @property
    def query_count(self) -> int:
        """返回节点对数量。"""
        return len(self.sources)


@dataclass(frozen=True)
class QueryValidationReport:
    """保存 H2H 与分组单源 Dijkstra 的逐项比较结果。"""

    dataset_name: str
    node_count: int
    edge_count: int
    query_count: int
    source_count: int
    failure_count: int
    max_absolute_error: float
    average_absolute_error: float
    h2h_batch_seconds: float
    dijkstra_seconds: float
    directional_pairs_checked: int
    asymmetric_pair_count: int

    def to_dict(self) -> dict[str, Any]:
        """将报告转换为 JSON 可序列化字典。"""
        return asdict(self)


@dataclass(frozen=True)
class QueryPerformanceReport:
    """保存三条公开查询路径的吞吐、校验和与常驻内存变化。"""

    query_count: int
    batch_seconds: float
    batch_queries_per_second: float
    scalar_seconds: float
    scalar_queries_per_second: float
    double_indexed_seconds: float
    double_indexed_queries_per_second: float
    repeat_double_indexed_seconds: float
    repeat_double_indexed_queries_per_second: float
    checksum: float
    rss_before_bytes: int | None
    rss_after_first_pass_bytes: int | None
    rss_after_repeat_bytes: int | None
    stable_rss_growth_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        """将报告转换为 JSON 可序列化字典。"""
        return asdict(self)


def build_query_workload(
    node_count: int,
    query_count: int = 100_000,
    source_count: int = 200,
    seed: int = 20260715,
) -> QueryWorkload:
    """
    构造固定种子的有向节点对，并限制 Dijkstra 只需运行指定数量的源。

    输入：节点数、查询数、不同源节点数和随机种子。
    输出：源/目标等长的不可变工作负载；每个选中源至少出现一次。
    """
    if node_count <= 0:
        raise ValueError('node_count 必须为正整数。')
    if query_count <= 0:
        raise ValueError('query_count 必须为正整数。')
    if source_count <= 0 or source_count > node_count:
        raise ValueError(f'source_count 必须位于 [1, {node_count}]。')
    if query_count < source_count:
        raise ValueError('query_count 不能小于 source_count，否则无法覆盖所有源节点。')

    generator = random.Random(seed)
    unique_sources = tuple(generator.sample(range(node_count), source_count))
    # 轮转源节点保证每个源都有查询；目标保持独立随机，保留有向性。
    sources = tuple(unique_sources[index % source_count] for index in range(query_count))
    targets = tuple(generator.randrange(node_count) for _ in range(query_count))
    return QueryWorkload(sources, targets, unique_sources, seed)


def _within_tolerance(actual: float, expected: float, tolerance: float) -> bool:
    """按实施计划的绝对误差或相对误差规则判断一个结果。"""
    absolute_error = abs(actual - expected)
    scale = max(1.0, abs(actual), abs(expected))
    return absolute_error <= tolerance or absolute_error / scale <= tolerance


def validate_query_workload(
    graph: nx.Graph,
    matrix,
    workload: QueryWorkload,
    dataset_name: str,
    tolerance: float = 1e-10,
    directional_sample_count: int = 1_000,
) -> QueryValidationReport:
    """
    用 H2H batch 查询并按源分组运行 NetworkX Dijkstra 逐项比较。

    输入：标准化强连通图、H2H 矩阵、查询工作负载、数据集名称和误差容限。
    输出：失败数、最大/平均误差、耗时和方向非对称统计。

    实现逻辑：只保留 H2H 的 100,000 个结果；每个 Dijkstra 距离行比较后立即释放，
    避免中图验收退化成全对矩阵或长期保存 200 个完整距离行。
    """
    if matrix.node_count != graph.number_of_nodes():
        raise ValueError('H2H 矩阵节点数与待验收图不一致。')
    if workload.query_count == 0 or len(workload.targets) != workload.query_count:
        raise ValueError('查询工作负载为空或 sources/targets 长度不一致。')

    h2h_started_at = time.perf_counter()
    actual_distances = matrix.query_batch(workload.sources, workload.targets)
    h2h_seconds = time.perf_counter() - h2h_started_at

    # `indices_by_source` 只保存 100,000 个整数位置，不复制图或距离行。
    indices_by_source: dict[int, list[int]] = {
        source: [] for source in workload.unique_sources
    }
    for index, source in enumerate(workload.sources):
        indices_by_source[source].append(index)

    failure_count = 0
    max_absolute_error = 0.0
    total_absolute_error = 0.0
    dijkstra_started_at = time.perf_counter()
    for source in workload.unique_sources:
        expected_row = nx.single_source_dijkstra_path_length(graph, source, weight='weight')
        for index in indices_by_source[source]:
            target = workload.targets[index]
            expected = float(expected_row[target])
            actual = float(actual_distances[index])
            absolute_error = abs(actual - expected)
            total_absolute_error += absolute_error
            max_absolute_error = max(max_absolute_error, absolute_error)
            if not _within_tolerance(actual, expected, tolerance):
                failure_count += 1
    dijkstra_seconds = time.perf_counter() - dijkstra_started_at

    checked = min(directional_sample_count, workload.query_count)
    reverse_distances = matrix.query_batch(
        workload.targets[:checked], workload.sources[:checked]
    ) if checked else []
    asymmetric_pair_count = sum(
        not _within_tolerance(actual_distances[index], reverse_distances[index], tolerance)
        for index in range(checked)
        if workload.sources[index] != workload.targets[index]
    )
    return QueryValidationReport(
        dataset_name=dataset_name,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        query_count=workload.query_count,
        source_count=len(workload.unique_sources),
        failure_count=failure_count,
        max_absolute_error=max_absolute_error,
        average_absolute_error=total_absolute_error / workload.query_count,
        h2h_batch_seconds=h2h_seconds,
        dijkstra_seconds=dijkstra_seconds,
        directional_pairs_checked=checked,
        asymmetric_pair_count=asymmetric_pair_count,
    )


def current_process_rss_bytes() -> int | None:
    """
    返回当前进程常驻内存，供查询前后稳定性比较。

    输出：Windows 工作集或 Linux `/proc/self/statm` 的 RSS 字节数；平台不支持时为 None。
    """
    if platform.system() == 'Windows':
        class ProcessMemoryCounters(ctypes.Structure):
            """对应 Win32 PROCESS_MEMORY_COUNTERS，只读取 WorkingSetSize。"""

            _fields_ = [
                ('cb', ctypes.c_ulong),
                ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        handle = get_current_process()
        if get_process_memory_info(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return None
    if platform.system() == 'Linux':
        try:
            with open('/proc/self/statm', encoding='ascii') as statm_file:
                resident_pages = int(statm_file.read().split()[1])
            return resident_pages * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            return None
    return None


def peak_process_rss_bytes(include_children: bool = False) -> int | None:
    """
    返回当前进程或已结束子进程的峰值 RSS。

    输入：是否读取 `RUSAGE_CHILDREN`；Linux ru_maxrss 单位为 KiB。
    输出：字节数；不支持 resource 的平台返回 None。
    """
    try:
        import resource
    except ImportError:
        return None
    target = resource.RUSAGE_CHILDREN if include_children else resource.RUSAGE_SELF
    value = int(resource.getrusage(target).ru_maxrss)
    if platform.system() == 'Darwin':
        return value
    return value * 1024


def _timed_scalar_queries(matrix, workload: QueryWorkload, double_indexed: bool) -> tuple[float, float]:
    """执行一遍标量查询，返回耗时和防止结果被优化/忽略的距离校验和。"""
    checksum = 0.0
    started_at = time.perf_counter()
    if double_indexed:
        for source, target in zip(workload.sources, workload.targets):
            checksum += matrix[source][target]
    else:
        for source, target in zip(workload.sources, workload.targets):
            checksum += matrix.query(source, target)
    return time.perf_counter() - started_at, checksum


def benchmark_query_paths(matrix, workload: QueryWorkload) -> QueryPerformanceReport:
    """
    测量 batch、ctypes 标量和完整双下标三条查询路径，并复跑检查 RSS 稳定性。

    输入：已构建的 H2H 矩阵与 100,000 对工作负载。
    输出：三条吞吐、共同校验和及第二遍双下标查询后的 RSS 增量。
    """
    warm_count = min(4_096, workload.query_count)
    matrix.query_batch(workload.sources[:warm_count], workload.targets[:warm_count])
    gc.collect()
    rss_before = current_process_rss_bytes()

    batch_started_at = time.perf_counter()
    batch_values = matrix.query_batch(workload.sources, workload.targets)
    batch_seconds = time.perf_counter() - batch_started_at
    batch_checksum = math.fsum(batch_values)
    del batch_values

    scalar_seconds, scalar_checksum = _timed_scalar_queries(matrix, workload, False)
    double_seconds, double_checksum = _timed_scalar_queries(matrix, workload, True)
    if not (
        math.isclose(batch_checksum, scalar_checksum, rel_tol=1e-12, abs_tol=1e-9)
        and math.isclose(batch_checksum, double_checksum, rel_tol=1e-12, abs_tol=1e-9)
    ):
        raise AssertionError('三条 H2H 查询路径的校验和不一致。')

    gc.collect()
    rss_after_first_pass = current_process_rss_bytes()
    repeat_seconds, repeat_checksum = _timed_scalar_queries(matrix, workload, True)
    if not math.isclose(batch_checksum, repeat_checksum, rel_tol=1e-12, abs_tol=1e-9):
        raise AssertionError('重复双下标查询的校验和发生变化。')
    gc.collect()
    rss_after_repeat = current_process_rss_bytes()
    stable_growth = None
    if rss_after_first_pass is not None and rss_after_repeat is not None:
        stable_growth = rss_after_repeat - rss_after_first_pass

    count = workload.query_count
    return QueryPerformanceReport(
        query_count=count,
        batch_seconds=batch_seconds,
        batch_queries_per_second=count / batch_seconds,
        scalar_seconds=scalar_seconds,
        scalar_queries_per_second=count / scalar_seconds,
        double_indexed_seconds=double_seconds,
        double_indexed_queries_per_second=count / double_seconds,
        repeat_double_indexed_seconds=repeat_seconds,
        repeat_double_indexed_queries_per_second=count / repeat_seconds,
        checksum=batch_checksum,
        rss_before_bytes=rss_before,
        rss_after_first_pass_bytes=rss_after_first_pass,
        rss_after_repeat_bytes=rss_after_repeat,
        stable_rss_growth_bytes=stable_growth,
    )


def _spawn_batch_worker(
    payload: bytes,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
    ready_queue,
    start_event,
    result_queue,
) -> None:
    """
    在 spawn worker 中恢复 mmap 距离矩阵并完成一批查询。

    输入：小型 pickle、当前 worker 的节点对、就绪队列、起跑事件和结果队列。
    输出：队列中的成功统计或异常信息；退出前主动关闭本进程 mmap。
    """
    matrix = None
    ready_reported = False
    try:
        matrix = pickle.loads(payload)
        # 先在计时外完成 DLL/mmap 延迟打开，所有 worker 就绪后统一起跑。
        if sources:
            matrix.query(sources[0], targets[0])
        ready_queue.put({'ok': True})
        ready_reported = True
        if not start_event.wait(timeout=600.0):
            raise TimeoutError('等待 H2H worker 统一起跑信号超时。')
        started_at = time.perf_counter()
        values = matrix.query_batch(sources, targets)
        elapsed = time.perf_counter() - started_at
        result_queue.put({
            'ok': True,
            'query_count': len(values),
            'query_seconds': elapsed,
            'checksum': math.fsum(values),
            'rss_bytes': current_process_rss_bytes(),
        })
    except Exception as exc:
        if not ready_reported:
            ready_queue.put({
                'ok': False,
                'error_type': type(exc).__name__,
                'error': str(exc),
            })
        result_queue.put({
            'ok': False,
            'error_type': type(exc).__name__,
            'error': str(exc),
        })
    finally:
        if matrix is not None:
            matrix.close()


def benchmark_spawn_workers(
    matrix,
    workload: QueryWorkload,
    worker_count: int,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """
    用 spawn 启动指定数量 worker，验证 pickle 重开并测量共享 mmap 总吞吐。

    输入：H2H 矩阵、总工作负载、worker 数和超时。
    输出：总查询数、吞吐、每个 worker 统计及 pickle 大小。
    """
    if worker_count <= 0:
        raise ValueError('worker_count 必须为正整数。')
    payload = pickle.dumps(matrix, protocol=pickle.HIGHEST_PROTOCOL)
    context = mp.get_context('spawn')
    ready_queue = context.Queue()
    start_event = context.Event()
    result_queue = context.Queue()
    processes = []
    startup_started_at = time.perf_counter()
    for worker_index in range(worker_count):
        # 步进切片让各 worker 的源节点分布一致，同时保证每个查询只执行一次。
        sources = workload.sources[worker_index::worker_count]
        targets = workload.targets[worker_index::worker_count]
        process = context.Process(
            target=_spawn_batch_worker,
            args=(payload, sources, targets, ready_queue, start_event, result_queue),
        )
        process.start()
        processes.append(process)

    results = []
    deadline = time.monotonic() + timeout_seconds
    try:
        ready_results = []
        for _ in processes:
            remaining = max(0.1, deadline - time.monotonic())
            ready_results.append(ready_queue.get(timeout=remaining))
        readiness_failures = [result for result in ready_results if not result.get('ok')]
        if readiness_failures:
            raise RuntimeError(f'H2H spawn worker 初始化失败：{readiness_failures}')
        startup_seconds = time.perf_counter() - startup_started_at
        measured_started_at = time.perf_counter()
        start_event.set()
        for _ in processes:
            remaining = max(0.1, deadline - time.monotonic())
            results.append(result_queue.get(timeout=remaining))
    except queue.Empty as exc:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise TimeoutError(f'{worker_count} 个 H2H worker 验收超时。') from exc
    except Exception:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in processes:
            process.join(timeout=min(5.0, max(0.1, deadline - time.monotonic())))

    elapsed = time.perf_counter() - measured_started_at
    bad_exit_codes = [process.exitcode for process in processes if process.exitcode != 0]
    failures = [result for result in results if not result.get('ok')]
    if bad_exit_codes or failures:
        raise RuntimeError(
            f'H2H spawn worker 失败：exit_codes={bad_exit_codes}, failures={failures}'
        )
    total_queries = sum(int(result['query_count']) for result in results)
    if total_queries != workload.query_count:
        raise AssertionError(
            f'worker 总查询数 {total_queries} 与工作负载 {workload.query_count} 不一致。'
        )
    return {
        'worker_count': worker_count,
        'query_count': total_queries,
        'startup_seconds': startup_seconds,
        'wall_seconds': elapsed,
        'queries_per_second': total_queries / elapsed,
        'pickle_bytes': len(payload),
        'checksum': math.fsum(float(result['checksum']) for result in results),
        'max_worker_rss_bytes': max(
            (int(result['rss_bytes']) for result in results if result['rss_bytes'] is not None),
            default=None,
        ),
        'workers': results,
    }
