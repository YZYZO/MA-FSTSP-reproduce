"""在 Linux 大内存服务器上执行 55k H2H 构建、校验、性能和端到端验收。"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    H2H_INDEX_DIR,
    H2H_NATIVE_BUILD_DIR,
    MANHATTAN_GRAPH_PATH,
)
from h2h_acceptance import (  # noqa: E402
    benchmark_query_paths,
    benchmark_spawn_workers,
    build_query_workload,
    current_process_rss_bytes,
    peak_process_rss_bytes,
    validate_query_workload,
)


def _parse_positive_csv(text: str, option: str) -> tuple[int, ...]:
    """
    解析逗号分隔的正整数并保持用户顺序去重。

    输入：命令行文本和选项名。
    输出：非空正整数元组；非法值抛出 argparse.ArgumentTypeError。
    """
    try:
        values = tuple(dict.fromkeys(int(item.strip()) for item in text.split(',')))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'{option} 必须是逗号分隔的正整数。') from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f'{option} 必须是逗号分隔的正整数。')
    return values


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    解析服务器验收参数。

    输入：可选参数列表；省略时读取当前命令行。
    输出：包含显式 55k 确认、资源阈值、查询和实例规模的命名空间。
    """
    parser = argparse.ArgumentParser(
        description='在 Linux 大内存服务器执行 MA-FSTSP 55k H2H 验收。'
    )
    parser.add_argument(
        '--confirm-server-55k',
        action='store_true',
        help='明确确认当前是允许构建 55k 索引的服务器；缺失时立即退出。',
    )
    parser.add_argument('--graph', type=Path, default=MANHATTAN_GRAPH_PATH)
    parser.add_argument('--index-dir', type=Path, default=H2H_INDEX_DIR)
    parser.add_argument('--build-dir', type=Path, default=H2H_NATIVE_BUILD_DIR)
    parser.add_argument('--compiler', default='g++')
    parser.add_argument(
        '--skip-native-build',
        action='store_true',
        help='仅在同一服务器已有本次源码编译产物时跳过 Linux 重编译。',
    )
    parser.add_argument('--minimum-memory-gb', type=float, default=150.0)
    parser.add_argument('--query-count', type=int, default=100_000)
    parser.add_argument('--source-count', type=int, default=200)
    parser.add_argument('--performance-query-count', type=int, default=100_000)
    parser.add_argument('--parallel-query-count', type=int, default=1_000_000)
    parser.add_argument('--worker-counts', default='1,4,8,16')
    parser.add_argument('--customer-counts', default='20')
    parser.add_argument('--depots', type=int, default=5)
    parser.add_argument('--drones', type=int, default=3)
    parser.add_argument('--gurobi-threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260715)
    parser.add_argument(
        '--report',
        type=Path,
        default=PROJECT_ROOT / 'results' / 'h2h-server-55k-acceptance.json',
    )
    arguments = parser.parse_args(argv)
    arguments.worker_counts = _parse_positive_csv(arguments.worker_counts, '--worker-counts')
    arguments.customer_counts = _parse_positive_csv(
        arguments.customer_counts, '--customer-counts'
    )
    return arguments


def linux_total_memory_bytes() -> int | None:
    """
    从 `/proc/meminfo` 读取 Linux 物理内存总量。

    输出：字节数；文件不可用或字段无效时返回 None。
    """
    try:
        for line in Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
            if line.startswith('MemTotal:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def validate_server_preconditions(
    arguments: argparse.Namespace,
    system_name: str | None = None,
    total_memory_bytes: int | None = None,
) -> dict[str, Any]:
    """
    在读取 GraphML、创建缓存或编译前验证显式授权、Linux 与内存规模。

    输入：参数及测试可覆盖的平台/内存值。
    输出：可写入报告的服务器环境摘要。
    """
    if not arguments.confirm_server_55k:
        raise PermissionError(
            '未提供 --confirm-server-55k；没有读取 nyc.graphml、编译或创建索引。'
        )
    selected_system = system_name or platform.system()
    if selected_system != 'Linux':
        raise RuntimeError('55k 验收脚本只允许在 Linux 服务器运行。')
    memory_bytes = total_memory_bytes
    if memory_bytes is None:
        memory_bytes = linux_total_memory_bytes()
    minimum_bytes = int(arguments.minimum_memory_gb * 1024 ** 3)
    if memory_bytes is None:
        raise RuntimeError('无法读取服务器物理内存，拒绝启动 55k 构建。')
    if memory_bytes < minimum_bytes:
        raise RuntimeError(
            f'服务器物理内存 {memory_bytes / 1024 ** 3:.1f} GiB 低于要求的 '
            f'{arguments.minimum_memory_gb:.1f} GiB，拒绝启动。'
        )
    graph_path = arguments.graph.expanduser().resolve()
    if not graph_path.is_file():
        raise FileNotFoundError(f'55k GraphML 不存在：{graph_path}')
    return {
        'platform': selected_system,
        'python': sys.version,
        'cpu_count': os.cpu_count(),
        'total_memory_bytes': memory_bytes,
        'minimum_memory_gb': arguments.minimum_memory_gb,
        'graph_path': str(graph_path),
    }


def _compile_linux_native(arguments: argparse.Namespace) -> dict[str, Any]:
    """
    通过项目统一 Python 脚本在服务器重编译 builder 和 `.so`。

    输入：编译器和构建目录参数。
    输出：命令与耗时；失败时 subprocess 直接抛出异常并保留原输出。
    """
    command = [
        sys.executable,
        str(PROJECT_ROOT / 'scripts' / 'build_h2h_native.py'),
        '--compiler', arguments.compiler,
        '--build-dir', str(arguments.build_dir),
        '--release',
    ]
    started_at = time.perf_counter()
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return {
        'command': command,
        'seconds': time.perf_counter() - started_at,
        'skipped': False,
    }


def _validate_end_to_end_solution(model, solution, depots, cities, distance) -> dict[str, Any]:
    """
    验证主算法的客户分组、仓库端点、无人机航程和路线结构。

    输入：已求解模型、公开路线、仓库/客户数组和距离提供器。
    输出：服务客户数、sortie 数和最大 sortie 航程摘要。
    """
    assigned = [
        int(city)
        for depot in depots
        for city in model.groups[depot]
    ]
    if Counter(assigned) != Counter(map(int, cities)):
        raise AssertionError('55k 端到端实例没有把每个客户恰好分配一次。')
    if len(solution) != len(depots):
        raise AssertionError('端到端路线数量与仓库数量不一致。')

    sortie_count = 0
    max_sortie_distance = 0.0
    for depot, route in zip(depots, solution):
        truck_route = route['truck']
        if len(truck_route) < 2 or int(truck_route[0]) != int(depot) \
                or int(truck_route[-1]) != int(depot):
            raise AssertionError(f'仓库 {int(depot)} 的卡车路线端点不闭合。')
        if len(route['drone']) != model.drone:
            raise AssertionError('公开路线的无人机槽位数与配置不一致。')
        for drone_routes in route['drone']:
            for sortie in drone_routes:
                if len(sortie) != 3:
                    raise AssertionError(f'无人机 sortie 不是三元组：{sortie!r}')
                launch, city, recovery = map(int, sortie)
                sortie_distance = (
                    distance['drone'][launch][city]
                    + distance['drone'][city][recovery]
                )
                if sortie_distance > model.limit + 1e-10:
                    raise AssertionError(
                        f'无人机 sortie {sortie!r} 超过航程 {model.limit}。'
                    )
                sortie_count += 1
                max_sortie_distance = max(max_sortie_distance, sortie_distance)
    return {
        'served_customer_count': len(assigned),
        'sortie_count': sortie_count,
        'max_sortie_distance': max_sortie_distance,
    }


def _run_end_to_end_instances(
    graph,
    distance,
    customer_counts: tuple[int, ...],
    depot_count: int,
    drone_count: int,
    seed: int,
    gurobi_threads: int,
) -> list[dict[str, Any]]:
    """
    按 20、50、100、150 等配置顺序运行 55k 主算法实例。

    输入：图、H2H 距离、客户规模、仓库/无人机数、种子和 Gurobi 线程数。
    输出：每个规模的成本、时间、内存和可行性统计。
    """
    import gurobipy as gp
    from src.fstsp import MultiAgentFlyingSidekickTSP

    if gurobi_threads < 0:
        raise ValueError('gurobi_threads 不能为负数。')
    gp.setParam('Threads', gurobi_threads)
    results = []
    for offset, customer_count in enumerate(customer_counts):
        generator = np.random.default_rng(seed + offset)
        locations = generator.choice(
            graph.number_of_nodes(),
            size=depot_count + customer_count,
            replace=False,
        )
        depots = np.asarray(locations[:depot_count], dtype=int)
        cities = np.asarray(locations[depot_count:], dtype=int)
        rss_before = current_process_rss_bytes()
        model = MultiAgentFlyingSidekickTSP(
            graph, depots, cities, distance, drone_count
        )
        started_at = time.perf_counter()
        solution, cost = model.solve()
        solve_seconds = time.perf_counter() - started_at
        if cost is None or not math.isfinite(float(cost)):
            raise AssertionError(f'{customer_count} 客户实例没有返回有限成本。')
        feasibility = _validate_end_to_end_solution(
            model, solution, depots, cities, distance
        )
        results.append({
            'customer_count': customer_count,
            'depot_count': depot_count,
            'drone_count': drone_count,
            'gurobi_threads': gurobi_threads,
            'cost': float(cost),
            'solve_seconds': solve_seconds,
            'rss_before_bytes': rss_before,
            'rss_after_bytes': current_process_rss_bytes(),
            'peak_process_rss_bytes': peak_process_rss_bytes(),
            **feasibility,
        })
    return results


def _write_json_report(path: Path, report: dict[str, Any]) -> None:
    """
    先 fsync 临时文件再原子替换服务器 JSON 报告。

    输入：报告路径和 JSON 字典。
    输出：无；不会覆盖 GraphML 或 H2H 缓存内容。
    """
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.tmp-{os.getpid()}')
    with temporary.open('w', encoding='utf-8', newline='\n') as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)


def run_server_acceptance(arguments: argparse.Namespace) -> dict[str, Any]:
    """
    执行 Linux 编译、55k 索引、100k 正确性、吞吐、多 worker 和端到端验收。

    输入：已解析且带显式确认的服务器参数。
    输出：完整报告字典；任一退出条件失败则抛出异常。
    """
    environment = validate_server_preconditions(arguments)
    report: dict[str, Any] = {
        'schema_version': 1,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'environment': environment,
        'configuration': {
            'query_count': arguments.query_count,
            'source_count': arguments.source_count,
            'performance_query_count': arguments.performance_query_count,
            'parallel_query_count': arguments.parallel_query_count,
            'worker_counts': arguments.worker_counts,
            'customer_counts': arguments.customer_counts,
            'depots': arguments.depots,
            'drones': arguments.drones,
            'gurobi_threads': arguments.gurobi_threads,
            'seed': arguments.seed,
        },
    }
    try:
        if arguments.skip_native_build:
            report['native_build'] = {'skipped': True, 'seconds': 0.0}
        else:
            report['native_build'] = _compile_linux_native(arguments)

        # 只有全部服务器前置条件通过后，才在本进程显式放开 55k 保护。
        import h2h_backend
        from distance_oracle import build_distance_provider
        from h2h_backend import (
            ensure_h2h_index,
            native_artifact_paths,
            read_h2h_index_statistics,
        )
        from problem import manhattan

        h2h_backend.H2H_ENABLE_55K = True
        graph_path = arguments.graph.expanduser().resolve()
        graph = manhattan(graph_path)
        builder_path, library_path = native_artifact_paths(arguments.build_dir)
        cache = ensure_h2h_index(
            graph,
            dataset_name='nyc-55k',
            index_dir=arguments.index_dir,
            builder_path=builder_path,
        )
        report['cache'] = {
            'cache_dir': str(cache.cache_dir),
            'built_this_run': cache.built,
            'graph_hash': cache.graph_hash,
            'node_count': cache.node_count,
            'statistics': read_h2h_index_statistics(cache.cache_dir),
        }
        distance = build_distance_provider(
            graph,
            backend='h2h',
            dataset_name='nyc-55k',
            graph_path=str(graph_path),
            index_dir=str(arguments.index_dir),
            builder_path=str(builder_path),
            library_path=str(library_path),
        )
        try:
            correctness_workload = build_query_workload(
                graph.number_of_nodes(),
                arguments.query_count,
                arguments.source_count,
                arguments.seed,
            )
            correctness = validate_query_workload(
                graph,
                distance['truck'],
                correctness_workload,
                dataset_name='nyc-55k',
            )
            report['correctness'] = correctness.to_dict()
            if correctness.failure_count != 0:
                raise AssertionError(
                    f'55k 查询存在 {correctness.failure_count} 个错误结果。'
                )

            performance_workload = build_query_workload(
                graph.number_of_nodes(),
                arguments.performance_query_count,
                min(arguments.source_count, arguments.performance_query_count),
                arguments.seed + 1,
            )
            performance = benchmark_query_paths(distance['truck'], performance_workload)
            report['query_performance'] = performance.to_dict()
            if performance.double_indexed_queries_per_second < 100_000:
                raise AssertionError(
                    '完整双下标查询低于 100,000 queries/s，需按计划定位热点。'
                )

            parallel_workload = build_query_workload(
                graph.number_of_nodes(),
                arguments.parallel_query_count,
                min(arguments.source_count, arguments.parallel_query_count),
                arguments.seed + 2,
            )
            worker_reports = [
                benchmark_spawn_workers(
                    distance['truck'], parallel_workload, worker_count
                )
                for worker_count in arguments.worker_counts
            ]
            report['worker_scaling'] = worker_reports
            single_worker = next(
                (item for item in worker_reports if item['worker_count'] == 1), None
            )
            multi_worker = [item for item in worker_reports if item['worker_count'] > 1]
            if single_worker is not None and multi_worker:
                measured_gain = max(
                    item['queries_per_second'] for item in multi_worker
                ) > single_worker['queries_per_second']
                report['worker_scaling_has_measured_gain'] = measured_gain
                if not measured_gain:
                    raise AssertionError(
                        '4/8/16 worker 查询吞吐均未超过单 worker，服务器并行退出条件未满足。'
                    )

            report['end_to_end'] = _run_end_to_end_instances(
                graph,
                distance,
                arguments.customer_counts,
                arguments.depots,
                arguments.drones,
                arguments.seed + 3,
                arguments.gurobi_threads,
            )
        finally:
            distance['truck'].close()

        report['status'] = 'passed'
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        report['peak_process_rss_bytes'] = peak_process_rss_bytes()
        report['peak_children_rss_bytes'] = peak_process_rss_bytes(include_children=True)
        return report
    except Exception:
        report['status'] = 'failed'
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        report['error'] = traceback.format_exc()
        raise
    finally:
        _write_json_report(arguments.report, report)


def main(argv: list[str] | None = None) -> int:
    """
    运行服务器验收并打印最终报告路径。

    输入：可选命令行参数。
    输出：全部退出条件通过为 0，否则打印明确异常并返回 1。
    """
    arguments = parse_arguments(argv)
    try:
        report = run_server_acceptance(arguments)
    except Exception as exc:
        print(f'H2H_SERVER_ACCEPTANCE_FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(f'H2H_SERVER_ACCEPTANCE_OK report={arguments.report.resolve()}')
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
