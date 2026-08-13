"""
本文件是论文实验的主入口脚本。

主要内容：
1. 组织小规模示例、Manhattan 路网实验、Cambridge 路网实验。
2. 组织论文中的消融实验与可扩展性实验。
3. 在缺少真实地图数据时，自动退化为轻量级离线演示。
4. 调用 `src/` 目录中的多种算法，并将成本与耗时保存为 `.npy` 文件。
"""

# 导入 `sys`，保留给潜在的命令行扩展使用。
import sys
# 导入 CSV 与 JSON，用于保存剪枝消融的逐仓库扁平记录。
import csv
import json
# 导入数学工具，用于阻止非有限目标值进入正确性比较。
import math
# 导入日期时间工具，用运行开始时间生成不会覆盖旧结果的文件名。
from datetime import datetime
# 导入路径工具，用传入的 GraphML 文件名生成稳定的地图标识。
from pathlib import Path
# 导入论文主算法。
from src.fstsp import MultiAgentFlyingSidekickTSP
# 导入仅替换 Phase 2 的剪枝算法入口。
from src.fstsp_pruned import PrunedMultiAgentFlyingSidekickTSP
# 导入剪枝分组与统一 Gurobi 求解配置。
from src.pruning import PruningOptions, SetTSPSolverOptions
# 导入实例构造函数与 Manhattan 路网读取函数。   
from problem import small_instance, multiagent_instance_on_manhattan, multiagent_instance_on_cambridge, manhattan
# 导入 `numpy`，用于均值、数组保存与随机采样。
import numpy as np
# 导入 `time`，用于统计实验耗时。
import time
# 导入进度条工具 `tqdm`。
from tqdm import tqdm
# 导入统一距离工厂，规模实验禁止再构造全对嵌套字典。
from src.distance.distance_oracle import build_distance_provider
from config import RESULTS_DIR
from utils import ensure_dir, result_path
from experiment_results import (
    _save_npz,
    _save_stsp_batch_result,
    _solve_model_with_process_data,
)

# 导入 `os`，用于读取/设置 CPU 亲和性。
import os
# 导入进程池工具，用于并行运行独立实验实例。
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import MANHATTAN_GRAPH_PATH,MANHATTAN_BASELINE_GRAPH_PATH,MANHATTAN1k_GRAPH_PATH,MANHATTAN11k_GRAPH_PATH



SMALL_DATA_DIR = result_path('small', 'data')
MANHATTAN_DATA_DIR = result_path('manhattan', 'data')
BOSTON_DATA_DIR = result_path('boston', 'data')


def _save_array(path, array):
    ensure_dir(path.parent)
    np.save(path, np.array(array))


# 固定使用 10 个进程并行运行实例。
PROCESS_WORKERS = 1
# 默认把这 10 个进程限制在当前允许 CPU 集合中的前 10 个逻辑 CPU 上。
# 如果只想限制进程数、不想绑定 CPU，可改为 False。
PIN_TO_PROCESS_CORES = True

# 子进程中的共享只读上下文。Linux fork 模式下，大对象通常通过 copy-on-write 共享，
# 避免每个任务反复把 graph / distance 序列化传给子进程。
_PAR_GRAPH = None
_PAR_DEPOTS = None
_PAR_CITIES = None
_PAR_DISTANCE = None


def _init_process_worker(graph, depots, cities, distance, cpu_affinity=None):
    """
    初始化每个子进程的只读实验上下文，并按需设置 CPU 亲和性。
    """
    if cpu_affinity is not None and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(cpu_affinity))

    global _PAR_GRAPH, _PAR_DEPOTS, _PAR_CITIES, _PAR_DISTANCE
    _PAR_GRAPH = graph
    _PAR_DEPOTS = depots
    _PAR_CITIES = cities
    _PAR_DISTANCE = distance


def _solve_instance_job(task):
    """
    子进程执行单个实例。

    task: (algorithm, index, drones, rounds, theta)
    返回: `(index, cost, elapsed, solution, process_data)`。
    """
    algorithm, i, drones, rounds, theta = task
    depots = _PAR_DEPOTS[i]
    cities = _PAR_CITIES[i]


    if theta is None:
        model = MultiAgentFlyingSidekickTSP(_PAR_GRAPH, depots, cities, _PAR_DISTANCE, drones)
    else:
        model = MultiAgentFlyingSidekickTSP(
            _PAR_GRAPH, depots, cities, _PAR_DISTANCE, drones, theta=theta
        )

    if algorithm != 'stsp':
        raise ValueError(f'当前并行实验记录器只支持 stsp，收到 algorithm={algorithm!r}')

    solution, cost, process_data = _solve_model_with_process_data(model)
    elapsed = process_data['solve_seconds']

    return i, cost, elapsed, solution, process_data


def _clear_process_worker_context():
    """
    清理当前进程内的只读实验上下文。

    输入：无。
    输出：无；将图、仓库、客户和距离对象的模块级引用重置为空。
    """
    global _PAR_GRAPH, _PAR_DEPOTS, _PAR_CITIES, _PAR_DISTANCE
    _PAR_GRAPH = None
    _PAR_DEPOTS = None
    _PAR_CITIES = None
    _PAR_DISTANCE = None


def _run_single_worker_instances(tasks, num, graph, depots, cities, distance, desc=None):
    """
    在主进程中顺序执行单 worker 实验，避免 Windows 调试器创建 spawn 子进程。

    输入：任务列表、实例数，以及求解所需的图、仓库、客户、距离对象和进度条名称。
    输出：按实例编号排序的 ``(cost, elapsed, solution, process_data)`` 结果列表。

    主进程路径不设置 CPU 亲和性，避免把 VS Code 调试会话及其后续代码固定到单核。
    无论求解成功还是抛出异常，都会清理临时上下文，避免长期持有大型路网和索引对象。
    """
    # 结果槽位与并行路径保持相同顺序，任务完成后按实例编号写回。
    results = [None] * num
    _init_process_worker(graph, depots, cities, distance, cpu_affinity=None)
    try:
        for task in tqdm(tasks, total=num, desc=desc):
            i, cost, elapsed, solution, process_data = _solve_instance_job(task)
            results[i] = (cost, elapsed, solution, process_data)
        return results
    finally:
        _clear_process_worker_context()


def _run_parallel_instances(
    num,
    graph,
    depots,
    cities,
    distance,
    algorithm,
    drones,
    rounds=None,
    theta=None,
    max_workers=PROCESS_WORKERS,
    desc=None,
):
    """
    用 10 个进程并行执行 num 个相互独立的实验实例。

    注意：
    - 这里使用 ProcessPoolExecutor，不再使用 Python 线程。
    - 对纯 Python CPU 密集型代码，多进程可以绕开 GIL。
    - 每个子进程通常会占用一个 CPU 核心；如果模型内部又启动 Gurobi/BLAS 多线程，
      实际 CPU 占用可能超过 10 个核心，需要在对应求解器内部限制线程数。
    """
    if num <= 0:
        return []

    worker_count = min(max_workers, num)
    results = [None] * num
    tasks = [
        (algorithm, i, drones, rounds, theta)
        for i in range(num)
    ]

    if worker_count == 1:
        print('Using one worker in the main process.')
        return _run_single_worker_instances(
            tasks,
            num,
            graph,
            depots,
            cities,
            distance,
            desc=desc,
        )

    try:
        ctx = mp.get_context('fork')
    except ValueError:
        # 非 Linux 环境没有 fork 时退回默认启动方式。
        ctx = mp.get_context()

    cpu_affinity = None
    if PIN_TO_PROCESS_CORES and hasattr(os, 'sched_getaffinity'):
        allowed_cpus = sorted(os.sched_getaffinity(0))
        cpu_affinity = tuple(allowed_cpus[:worker_count])
        print(f'Using process pool with {worker_count} processes; CPU affinity={cpu_affinity}')
    else:
        print(f'Using process pool with {worker_count} processes.')

    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=ctx,
        initializer=_init_process_worker,
        initargs=(graph, depots, cities, distance, cpu_affinity),
    ) as executor:
        futures = {
            executor.submit(_solve_instance_job, task): task[1]
            for task in tasks
        }

        for future in tqdm(as_completed(futures), total=num, desc=desc):
            i, cost, elapsed, solution, process_data = future.result()
            results[i] = (cost, elapsed, solution, process_data)

    return results


def _store_cost_time(costs, times, key, results):
    """
    将并行任务返回记录中的 cost/elapsed 写回成本表和耗时表。

    输入：成本表、耗时表、算法键和实例结果列表。
    输出：无；只更新传入的两个结果表，忽略路线与过程数据字段。
    """
    costs[key] = [item[0] for item in results]
    times[key] = [item[1] for item in results]


def test_small_instance(num, size):
    """
    在小规模子图实例上运行四种算法，并统计平均成本与平均耗时。

    本版本将每一种算法在 num 个实例上的求解改为 10 进程并行。
    """
    # 生成共享的小规模图、多个 depot/city 采样以及距离矩阵。
    graph, _depots, _cities, distance = small_instance(num, 1100, 2, size)
    # 为四种算法准备成本记录容器。
    costs = {'lrmp': [], 'hc': [], 'stsp': [], 'lp': []}
    # 为四种算法准备耗时记录容器。
    times = {'lrmp': [], 'hc': [], 'stsp': [], 'lp': []}

    print(f'Running small instance experiment with {PROCESS_WORKERS} processes.')

    # lrmp_results = _run_parallel_instances(
    #     num, graph, _depots, _cities, distance,
    #     algorithm='lrmp', drones=2, desc='LRMP'
    # )
    # _store_cost_time(costs, times, 'lrmp', lrmp_results)
    # print(f'LRMP gives solution with cost {sum(costs["lrmp"]) / num} in {sum(times["lrmp"]) / num}s')

    # hc_results = _run_parallel_instances(
    #     num, graph, _depots, _cities, distance,
    #     algorithm='hc', drones=2, rounds=1000, desc='HC'
    # )
    # _store_cost_time(costs, times, 'hc', hc_results)
    # print(f'Hill Climbing gives solution with cost {sum(costs["hc"]) / num} in {sum(times["hc"]) / num}s')

    stsp_results = _run_parallel_instances(
        num, graph, _depots, _cities, distance,
        algorithm='stsp', drones=2, desc='STSP'
    )
    _store_cost_time(costs, times, 'stsp', stsp_results)
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')

    # lp_results = _run_parallel_instances(
    #     num, graph, _depots, _cities, distance,
    #     algorithm='lp', drones=2, desc='LP'
    # )
    # _store_cost_time(costs, times, 'lp', lp_results)
    # print(f'LP gives solution with cost {sum(costs["lp"]) / num} in {sum(times["lp"]) / num}s')

    _save_npz(
        SMALL_DATA_DIR / f'small-size-{size}.npz',
        # lrmp_cost=np.array(costs['lrmp']),
        # lrmp_time=np.array(times['lrmp']),
        # hc_cost=np.array(costs['hc']),
        # hc_time=np.array(times['hc']),
        stsp_cost=np.array(costs['stsp']),
        stsp_time=np.array(times['stsp']),
        # lp_cost=np.array(costs['lp']),
        # lp_time=np.array(times['lp']),
    )


def test_manhattan(num, size,map):
    """
    在 Manhattan 路网实例上运行 HC 和论文主算法。

    输入：
    - num: 本次需要求解的随机实例数量。
    - size: 每个实例的客户数量。
    - map: 本次实验使用的 Manhattan/NYC GraphML 路径。

    输出：
    - 无显式返回值；将本次全部实例汇总到一个带运行时间和地图名的 `.npz` 文件。

    实现逻辑：
    1. 在函数开始时记录秒级时间戳，并从 GraphML 文件名提取地图名。
    2. 生成指定地图上的随机实例并运行论文主算法。
    3. 将成本和耗时写入 Manhattan 结果目录。

    本版本将论文主算法在 num 个实例上的求解改为 10 进程并行。
    原文件中的 HC 部分本来就是注释状态，这里保持不运行 HC。
    """
    # 每次函数调用只写一个文件，因此使用秒级运行开始时间即可区分实验批次。
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # 从传入路径提取不含扩展名的地图名，例如 `manhatten` 或 `nyc`。
    map_name = Path(map).stem
    # 生成 Manhattan 路网实例集合。
    graph, depots, cities, distance = multiagent_instance_on_manhattan(num, 5, size,map)
    # 为两种算法创建成本记录表。
    costs = {'hc': [], 'stsp': [], 'lp': []}
    # 为两种算法创建耗时记录表。
    times = {'hc': [], 'stsp': [], 'lp': []}

    print(f'Running Manhattan experiment with {PROCESS_WORKERS} processes.')

    stsp_results = _run_parallel_instances(
        num, graph, depots, cities, distance,
        algorithm='stsp', drones=3, theta=(0.5, 0.5), desc='Manhattan-STSP'
    )
    _store_cost_time(costs, times, 'stsp', stsp_results)

    # 输出论文主算法平均表现。
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')
    _save_stsp_batch_result(
        MANHATTAN_DATA_DIR / f'{run_timestamp}-{map_name}-{size}.npz',
        stsp_results,
        depots,
        cities,
        distance,
        3,
        costs,
        times,
    )


def test_cambridge(num, size):
    """
    在 Cambridge 路网实例上运行 HC 和论文主算法。

    输入：
    - num: 本次需要求解的随机实例数量。
    - size: 每个实例的客户数量。

    输出：
    - 无显式返回值；将本次全部实例汇总到一个带运行时间和地图名的 `.npz` 文件。

    实现逻辑：
    1. 在函数开始时记录秒级时间戳，并以 `boston` 作为结果地图名。
    2. 生成 Cambridge/Boston 随机实例并运行论文主算法。
    3. 将成本和耗时写入 Boston 结果目录。

    本版本将论文主算法在 num 个实例上的求解改为 10 进程并行。
    原文件中的 HC 部分本来就是注释状态，这里保持不运行 HC。
    """
    # 每次函数调用只写一个文件，因此使用秒级运行开始时间即可区分实验批次。
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # 生成 Cambridge 路网实例集合。
    graph, depots, cities, distance = multiagent_instance_on_cambridge(num, 10, size)
    # 为两种算法创建成本记录表。
    costs = {'hc': [], 'stsp': [], 'lp': []}
    # 为两种算法创建耗时记录表。
    times = {'hc': [], 'stsp': [], 'lp': []}

    print(f'Running Cambridge experiment with {PROCESS_WORKERS} processes.')

    stsp_results = _run_parallel_instances(
        num, graph, depots, cities, distance,
        algorithm='stsp', drones=3, theta=(0.5, 0.5), desc='Cambridge-STSP'
    )
    _store_cost_time(costs, times, 'stsp', stsp_results)

    # 输出主算法平均表现。
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')
    _save_stsp_batch_result(
        BOSTON_DATA_DIR / f'{run_timestamp}-boston-{size}.npz',
        stsp_results,
        depots,
        cities,
        distance,
        3,
        costs,
        times,
    )


def test_manhattan_1k(num, size):
    """
    在 1,024 节点 Manhattan 路网上运行论文主算法并保存汇总结果。

    输入：
    - num: 本次需要求解的随机实例数量；论文完整实验使用 100。
    - size: 每个实例的客户数量；论文使用 50、100、150。

    输出：
    - 无显式返回值；将本次所有实例的成本和耗时保存为一个 `.npz` 文件。

    实现逻辑：
    1. 记录函数运行开始时间，作为结果文件名的一部分。
    2. 从 `nyc_1024.graphml` 生成 5 仓库随机实例。
    3. 按论文 Manhattan 口径使用每车 3 架无人机和 `(0.5, 0.5)` 阈值求解。
    4. 将全部实例的汇总结果写入 Manhattan 结果目录。
    """
    # 秒级运行时间用于区分不同批次；一次函数调用只生成一个汇总文件。
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # 论文 Manhattan 场景固定使用 5 个仓库，并显式选择 1,024 节点地图。
    graph, depots, cities, distance = multiagent_instance_on_manhattan(
        num, 5, size, MANHATTAN1k_GRAPH_PATH
    )
    # 保留与现有路网实验一致的结果结构，HC/LP 当前不执行。
    costs = {'hc': [], 'stsp': [], 'lp': []}
    times = {'hc': [], 'stsp': [], 'lp': []}

    print(f'Running Manhattan 1K experiment with {PROCESS_WORKERS} processes.')

    # `limit=1.5` 与 `speed=1.6` 使用主算法构造函数的论文默认值。
    stsp_results = _run_parallel_instances(
        num, graph, depots, cities, distance,
        algorithm='stsp', drones=3, theta=(0.5, 0.5), desc='Manhattan-1K-STSP'
    )
    _store_cost_time(costs, times, 'stsp', stsp_results)

    # 输出本批次论文主算法的平均成本和平均耗时。
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')
    # 文件名口径为“运行时间-地图名-客户数量”，本批次只写一个汇总文件。
    _save_stsp_batch_result(
        MANHATTAN_DATA_DIR / f'{run_timestamp}-manhattan_1k-{size}.npz',
        stsp_results,
        depots,
        cities,
        distance,
        3,
        costs,
        times,
    )


def test_manhattan_11k(num, size):
    """
    在 11,000 节点 NYC 路网上按论文 Boston 场景口径运行主算法。

    输入：
    - num: 本次需要求解的随机实例数量；论文完整实验使用 100。
    - size: 每个实例的客户数量；论文使用 50、100、150。

    输出：
    - 无显式返回值；将本次所有实例的成本和耗时保存为一个 `.npz` 文件。

    实现逻辑：
    1. 记录函数运行开始时间，作为结果文件名的一部分。
    2. 从 `nyc_11000.graphml` 生成 10 仓库随机实例。
    3. 将该 11k 地图视作 Boston 复现，按论文口径使用每车 4 架无人机。
    4. 将全部实例的汇总结果写入 Boston 结果目录。
    """
    # 秒级运行时间用于区分不同批次；一次函数调用只生成一个汇总文件。
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # 按用户确认的 Boston 复现口径使用 10 个仓库，但读取指定的 NYC 11k 地图。
    graph, depots, cities, distance = multiagent_instance_on_manhattan(
        num, 10, size, MANHATTAN11k_GRAPH_PATH
    )
    # 保留与现有路网实验一致的结果结构，HC/LP 当前不执行。
    costs = {'hc': [], 'stsp': [], 'lp': []}
    times = {'hc': [], 'stsp': [], 'lp': []}

    print(f'Running Boston 11K reproduction with {PROCESS_WORKERS} processes.')

    # `limit=1.5` 与 `speed=1.6` 使用主算法构造函数的论文默认值。
    stsp_results = _run_parallel_instances(
        num, graph, depots, cities, distance,
        algorithm='stsp', drones=4, theta=(0.5, 0.5), desc='Boston-11K-STSP'
    )
    _store_cost_time(costs, times, 'stsp', stsp_results)

    # 输出本批次论文主算法的平均成本和平均耗时。
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')
    # 文件名口径为“运行时间-地图名-客户数量”，本批次只写一个汇总文件。
    _save_stsp_batch_result(
        BOSTON_DATA_DIR / f'{run_timestamp}-boston_11k-{size}.npz',
        stsp_results,
        depots,
        cities,
        distance,
        4,
        costs,
        times,
    )


# def ablation_r():
#     """
#     研究飞行距离上限 `limit` 对算法性能的影响。

#     输入：
#     - 无显式输入，函数内部固定使用 20 组实例、5 个仓库、100 个客户。

#     输出：
#     - 将每个 `limit` 下的耗时保存到 `r-time.npy`。
#     - 将每个 `limit` 下的成本保存到 `r-cost.npy`。

#     实现逻辑：
#     1. 生成固定的一批 Manhattan 实例。
#     2. 枚举不同的飞行半径上限。
#     3. 对每个上限重复求解，并保存全部原始结果。
#     """
#     # 打印实验主题。
#     print('Studying the effect of radius limit')
#     # 生成固定的一批 Manhattan 实例。
#     graph, depots, cities, distance = multiagent_instance_on_manhattan(20, 5, 100)
#     # 初始化成本和耗时容器。
#     costs, times = [], []
#     # 枚举不同半径参数。
#     for r in range(5, 16, 2):
#         # 为当前半径开辟成本列表。
#         costs.append([])
#         # 为当前半径开辟耗时列表。
#         times.append([])
#         # 遍历 100 个随机实例。
#         for i in tqdm(range(100)):
#             # 构造主算法模型，其中 `limit=r/10`。
#             model = MultiAgentFlyingSidekickTSP(graph, depots[i], cities[i], distance, 3, limit=r / 10, theta=(0.5, 0.5))
#             # 开始计时。
#             start = time.time()
#             # 求解当前实例。
#             _, cost = model.solve()
#             # 保存当前实例耗时。
#             times[-1].append(time.time() - start)
#             # 保存当前实例成本。
#             costs[-1].append(cost)
#     # 将耗时数组保存到磁盘。
#     _save_array(MANHATTAN_DATA_DIR / 'r-time.npy', times)
#     # 将成本数组保存到磁盘。
#     _save_array(MANHATTAN_DATA_DIR / 'r-cost.npy', costs)


# def ablation_speed():
#     """
#     研究无人机速度比对算法性能的影响。

#     输入：
#     - 无显式输入，函数内部固定使用一批 Manhattan 实例。

#     输出：
#     - 将速度实验耗时保存到 `speed-time.npy`。
#     - 将速度实验成本保存到 `speed-cost.npy`。

#     实现逻辑：
#     1. 生成固定实例。
#     2. 枚举不同的无人机速度参数。
#     3. 对每个速度统计总耗时与所有成本样本。
#     """
#     # 打印实验主题。
#     print('Studying the effect of speed')
#     # 生成固定的一批 Manhattan 实例。
#     graph, depots, cities, distance = multiagent_instance_on_manhattan(20, 5, 100)
#     # 初始化成本和耗时容器。
#     costs, times = [], []
#     # 枚举不同速度比。
#     for speed in [i / 30 for i in range(10, 120, 20)]:
#         # 为当前速度开辟成本列表。
#         costs.append([])
#         # 为当前速度累计总耗时。
#         times.append(0)
#         # 遍历 100 个随机实例。
#         for i in tqdm(range(100)):
#             # 构造主算法模型，其中速度设为当前 `speed`。
#             model = MultiAgentFlyingSidekickTSP(graph, depots[i], cities[i], distance, 3, speed=speed)
#             # 开始计时。
#             start = time.time()
#             # 求解当前实例。
#             _, cost = model.solve()
#             # 累加总耗时。
#             times[-1] += time.time() - start
#             # 记录当前实例成本。
#             costs[-1].append(cost)
#     # 保存耗时数组。
#     _save_array(MANHATTAN_DATA_DIR / 'speed-time.npy', times)
#     # 保存成本数组。
#     _save_array(MANHATTAN_DATA_DIR / 'speed-cost.npy', costs)


# def ablation_k():
#     """
#     研究无人机数量对主算法性能的影响。

#     输入：
#     - 无显式输入，函数内部枚举客户规模并测试 0 到 5 架无人机。

#     输出：
#     - 将所有规模下的无人机数量成本矩阵保存为 `k-cost.npy`。

#     实现逻辑：
#     1. 枚举不同客户规模。
#     2. 对每个规模运行 `solve_multiple_drones`。
#     3. 保存每个规模下不同无人机数量的成本结果。
#     """
#     # 打印实验主题。
#     print('studying the effect of drone number')
#     # 用于汇总所有规模的成本矩阵。
#     all_costs = []
#     # 枚举客户规模。
#     for size in range(50, 160, 20):
#         # 生成当前规模的 Manhattan 实例。
#         graph, depots, cities, distance = multiagent_instance_on_manhattan(10, 5, size)
#         # 存放当前规模下所有样本的成本。
#         costs, times = [], []
#         # 遍历 100 个随机实例。
#         for i in tqdm(range(100)):
#             # 构造主算法模型，初始无人机数给 0，随后由 `solve_multiple_drones` 枚举。
#             model = MultiAgentFlyingSidekickTSP(graph, depots[i], cities[i], distance, 0)
#             # 求出 0~5 架无人机对应的成本序列。
#             cost = model.solve_multiple_drones()
#             # 保存当前实例结果。
#             costs.append(cost.copy())
#         # 将当前规模的结果转为数组并加入总表。
#         all_costs.append(np.array(costs))
#         # 打印当前规模下的平均成本。
#         print(f'size {size} gives {np.mean(costs, axis=0)}')
#     # 将总结果保存到磁盘。
#     _save_array(MANHATTAN_DATA_DIR / 'k-cost.npy', np.array(all_costs, dtype=float))


# def scale_cities():
#     """
#     固定仓库数量，研究客户数量增长时主算法的可扩展性。

#     输入：
#     - 无显式输入，内部固定使用 Manhattan 路网。

#     输出：
#     - 保存耗时到 `city-time.npy`。
#     - 保存成本到 `city-cost.npy`。

#     实现逻辑：
#     1. 先构造 Manhattan 路网和距离矩阵。
#     2. 枚举不同客户数量。
#     3. 每个数量下重复采样并统计性能。
#     """
#     # 打印实验主题。
#     print('studying the scalability of fix depot case')
#     # 读取 Manhattan 路网。
#     graph = manhattan()
#     # 三类规模实验共享同一图哈希 H2H 缓存，无人机距离保持按需计算。
#     distance = build_distance_provider(
#         graph,
#         dataset_name=graph.graph.get('dataset_name', 'nyc'),
#         graph_path=graph.graph.get('source_path'),
#     )
#     # 初始化成本和耗时容器。
#     costs, times = [], []
#     # 枚举客户数量。
#     for num in range(120, 350, 40):
#         # 为当前规模开辟成本列表。
#         costs.append([])
#         # 为当前规模开辟耗时列表。
#         times.append([])
#         # 重复 100 次随机采样。
#         for _ in tqdm(range(100)):
#             # 从图节点中随机采样仓库和客户。
#             locations = np.random.choice(graph.nodes, num + 10, replace=False)
#             # 构造主算法模型。
#             model = MultiAgentFlyingSidekickTSP(graph, locations[:10], locations[10:], distance, 3)
#             # 开始计时。
#             start = time.time()
#             # 求解当前实例。
#             _, cost = model.solve()
#             # 保存耗时。
#             times[-1].append(time.time() - start)
#             # 保存成本。
#             costs[-1].append(cost)
#     # 保存耗时结果。
#     _save_array(MANHATTAN_DATA_DIR / 'city-time.npy', times)
#     # 保存成本结果。
#     _save_array(MANHATTAN_DATA_DIR / 'city-cost.npy', costs)


# def scale_rates():
#     """
#     固定客户与仓库比例，研究规模增长时主算法的可扩展性。

#     输入：
#     - 无显式输入，内部固定客户仓库比例为 20:1。

#     输出：
#     - 保存耗时到 `rates-time.npy`。
#     - 保存成本到 `rates-cost.npy`。

#     实现逻辑：
#     1. 读取 Manhattan 路网。
#     2. 枚举仓库数量。
#     3. 按照固定比例同时增加客户数量。
#     """
#     # 打印实验主题。
#     print('studying the scalability of fix rates case')
#     # 读取 Manhattan 路网。
#     graph = manhattan()
#     # 使用与其他规模实验相同的数据集名，确保只构建一份版本化索引。
#     distance = build_distance_provider(
#         graph,
#         dataset_name=graph.graph.get('dataset_name', 'nyc'),
#         graph_path=graph.graph.get('source_path'),
#     )
#     # 初始化成本和耗时容器。
#     costs, times = [], []
#     # 枚举仓库数。
#     for num in range(3, 21, 3):
#         # 为当前规模开辟成本列表。
#         costs.append([])
#         # 为当前规模开辟耗时列表。
#         times.append([])
#         # 重复 100 次随机采样。
#         for _ in tqdm(range(100)):
#             # 按固定比例采样节点。
#             locations = np.random.choice(graph.nodes, num * (1 + 20), replace=False)
#             # 构造主算法模型。
#             model = MultiAgentFlyingSidekickTSP(graph, locations[:num], locations[num:], distance, 3)
#             # 开始计时。
#             start = time.time()
#             # 求解当前实例。
#             _, cost = model.solve()
#             # 保存耗时。
#             times[-1].append(time.time() - start)
#             # 保存成本。
#             costs[-1].append(cost)
#     # 保存耗时结果。
#     _save_array(MANHATTAN_DATA_DIR / 'rates-time.npy', times)
#     # 保存成本结果。
#     _save_array(MANHATTAN_DATA_DIR / 'rates-cost.npy', costs)


# def scale_depots():
#     """
#     固定客户数量，研究仓库数量增长时主算法的可扩展性。

#     输入：
#     - 无显式输入，内部固定客户数为 150。

#     输出：
#     - 保存耗时到 `depots-time.npy`。
#     - 保存成本到 `depots-cost.npy`。

#     实现逻辑：
#     1. 读取 Manhattan 路网。
#     2. 枚举仓库数。
#     3. 固定客户数 150，重复采样并统计性能。
#     """
#     # 打印实验主题。
#     print('studying the scalability of fix cities case')
#     # 读取 Manhattan 路网。
#     graph = manhattan()
#     # 距离工厂命中前两类实验的 H2H 缓存，不再按实验重复物化矩阵。
#     distance = build_distance_provider(
#         graph,
#         dataset_name=graph.graph.get('dataset_name', 'nyc'),
#         graph_path=graph.graph.get('source_path'),
#     )
#     # 初始化成本和耗时容器。
#     costs, times = [], []
#     # 枚举仓库数量。
#     for num in range(5, 16, 2):
#         # 为当前规模开辟成本列表。
#         costs.append([])
#         # 为当前规模开辟耗时列表。
#         times.append([])
#         # 重复 100 次随机采样。
#         for _ in tqdm(range(100)):
#             # 随机采样仓库与客户。
#             locations = np.random.choice(graph.nodes, num + 150, replace=False)
#             # 构造主算法模型。
#             model = MultiAgentFlyingSidekickTSP(graph, locations[:num], locations[num:], distance, 3)
#             # 开始计时。
#             start = time.time()
#             # 求解当前实例。
#             _, cost = model.solve()
#             # 保存耗时。
#             times[-1].append(time.time() - start)
#             # 保存成本。
#             costs[-1].append(cost)
#     # 保存耗时结果。
#     _save_array(MANHATTAN_DATA_DIR / 'depots-time.npy', times)
#     # 保存成本结果。
#     _save_array(MANHATTAN_DATA_DIR / 'depots-cost.npy', costs)





def run_full_experiments():
    """
    运行 README 中对应的论文全量实验。

    输入：
    - 无显式输入。

    输出：
    - 无显式返回值，实验结果以终端输出和 `.npy` 文件形式保存。

    实现逻辑：
    1. 运行小规模对比实验。
    2. 运行 Manhattan 和 Cambridge 大规模对比实验。
    3. 运行消融与可扩展性实验。
    """
    # 枚举小规模实验中的客户数量。
    # for size in [5, 10, 15]:
    #size = 10
    #     # 运行当前规模的小规模实验。
    #test_small_instance(1, size)


    # 枚举大规模路网实验中的客户数量。
    #for size in [50, 100, 150]:
    size = 20
        # 运行 Manhattan 4k 实验。
    test_manhattan(1, size, MANHATTAN_BASELINE_GRAPH_PATH)
        # 运行 Cambridge 实验。
    test_cambridge(1, size)
        # # 运行 Manhattan 55k 实验。
        # test_manhattan(40, size, MANHATTAN_GRAPH_PATH)

    test_manhattan_1k(1, size)

    test_manhattan_11k(1, size)





    # # 运行距离上限消融实验。
    # ablation_r()
    # # 运行速度消融实验。
    # ablation_speed()
    # # 运行无人机数量消融实验。
    # ablation_k()
    # # 运行固定仓库规模的可扩展性实验。
    # scale_cities()
    # # 运行固定比例的可扩展性实验。
    # scale_rates()
    # # 运行固定客户规模的可扩展性实验。
    # scale_depots()


def run_pruning_experiments(
    num_instances=1,
    customer_counts=(20,),
    solver_seed=0,
    objective_tolerance=1e-7,
    output_dir=None,
):
    """在 Manhattan 1K 与 Boston 11K 路网上运行 Phase 2 剪枝消融。

    输入：每个场景的实例数、客户规模序列、Gurobi 随机种子、目标一致性
    容差和可选输出目录。
    输出：最终 JSON 与 CSV 路径；文件包含逐场景、逐实例、逐仓库、逐实验
    组的 Phase 2 剪枝和求解指标。

    逻辑：每个“地图×客户规模”只生成一次实例，随后让 C0、P1、P3、P7
    和 P1→P3→P7 复用完全相同的仓库、客户与距离对象。C0 先运行并作为
    Phase 2 目标与总时间基线；任一剪枝组目标超过容差时立即保存已有记录
    并终止，避免服务器继续产生不可用的性能结果。
    """

    if isinstance(num_instances, bool) or not isinstance(num_instances, int) or num_instances <= 0:
        raise ValueError('剪枝实验的 num_instances 必须为正整数。')
    normalized_customer_counts = tuple(int(value) for value in customer_counts)
    if not normalized_customer_counts or any(value <= 0 for value in normalized_customer_counts):
        raise ValueError('剪枝实验的 customer_counts 必须包含至少一个正整数。')
    if not math.isfinite(float(objective_tolerance)) or objective_tolerance < 0:
        raise ValueError('剪枝实验的 objective_tolerance 必须为有限非负数。')

    # 两个服务器实验场景沿用论文复现中的仓库数、无人机数和路网文件。
    scenarios = (
        {
            'name': 'manhattan_1k',
            'map_path': MANHATTAN1k_GRAPH_PATH,
            'depot_count': 5,
            'drone_count': 3,
        },
        {
            'name': 'boston_11k',
            'map_path': MANHATTAN11k_GRAPH_PATH,
            'depot_count': 10,
            'drone_count': 4,
        },
    )
    # 单项组用于测量每条规则的独立贡献，最终组按 P1→P3→P7 顺序组合。
    pruning_groups = (
        ('C0', PruningOptions()),
        ('P1', PruningOptions(structural_stsp=True)),
        ('P3', PruningOptions(assignment_bound=True)),
        ('P7', PruningOptions(endpoint_pair_dominance=True)),
        ('P1_P3_P7', PruningOptions.p1_p3_p7()),
    )
    solver_options = SetTSPSolverOptions(seed=int(solver_seed))
    selected_output_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR / 'pruning'
    selected_output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    json_path = selected_output_dir / f'road_pruning_{run_timestamp}.json'
    csv_path = selected_output_dir / f'road_pruning_{run_timestamp}.csv'
    records = []

    def write_current_records():
        """原子写入当前全部扁平记录，防止中断留下半个结果文件。"""

        json_temporary = json_path.with_suffix('.json.tmp')
        json_temporary.write_text(
            json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False),
            encoding='utf-8',
        )
        json_temporary.replace(json_path)

        csv_temporary = csv_path.with_suffix('.csv.tmp')
        fieldnames = sorted({key for record in records for key in record})
        with csv_temporary.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        csv_temporary.replace(csv_path)

    # 每个基线键对应同一地图、规模、实例和仓库，防止跨实例错误比较。
    phase2_baselines = {}
    full_cost_baselines = {}
    for scenario in scenarios:
        for customer_count in normalized_customer_counts:
            print(
                f'Preparing pruning scenario={scenario["name"]}, '
                f'customers={customer_count}, instances={num_instances}.'
            )
            graph, depots, cities, distance = multiagent_instance_on_manhattan(
                num_instances,
                scenario['depot_count'],
                customer_count,
                scenario['map_path'],
            )

            for instance_index in range(num_instances):
                for group_name, pruning_options in pruning_groups:
                    print(
                        f'Running pruning group={group_name}, scenario={scenario["name"]}, '
                        f'customers={customer_count}, instance={instance_index}.'
                    )
                    model = PrunedMultiAgentFlyingSidekickTSP(
                        graph,
                        depots[instance_index],
                        cities[instance_index],
                        distance,
                        scenario['drone_count'],
                        theta=(0.5, 0.5),
                        pruning_options=pruning_options,
                        solver_options=solver_options,
                    )
                    _, full_cost, process_data = _solve_model_with_process_data(model)
                    phase2_metrics = model.phase2_metrics()
                    phase2_reports = model.phase2_pruning_reports
                    active_depot_records = [
                        item for item in process_data['depot_records'] if item['customers']
                    ]
                    if not (
                        len(active_depot_records)
                        == len(phase2_metrics)
                        == len(phase2_reports)
                    ):
                        raise RuntimeError(
                            f'{scenario["name"]}/实例 {instance_index}/{group_name} 的 '
                            'Phase 2 仓库记录、指标和剪枝报告数量不一致。'
                        )

                    full_key = (scenario['name'], customer_count, instance_index)
                    if group_name == 'C0':
                        full_cost_baselines[full_key] = float(full_cost)
                    baseline_full_cost = full_cost_baselines[full_key]

                    phase2_index = 0
                    for depot_record in process_data['depot_records']:
                        if not depot_record['customers']:
                            continue
                        metrics = phase2_metrics[phase2_index]
                        report = phase2_reports[phase2_index]
                        phase2_index += 1
                        objective = float(metrics['phase2_objective'])
                        if not math.isfinite(objective):
                            raise RuntimeError(
                                f'{scenario["name"]}/实例 {instance_index}/{group_name}/'
                                f'仓库 {depot_record["depot_index"]} 的 Phase 2 目标非有限。'
                            )

                        baseline_key = (
                            scenario['name'],
                            customer_count,
                            instance_index,
                            int(depot_record['depot_index']),
                        )
                        if group_name == 'C0':
                            phase2_baselines[baseline_key] = {
                                'objective': objective,
                                'total_seconds': float(metrics['phase2_total_seconds']),
                            }
                        baseline = phase2_baselines[baseline_key]
                        objective_delta = abs(objective - baseline['objective'])
                        objective_consistent = objective_delta <= objective_tolerance
                        phase2_total_seconds = float(metrics['phase2_total_seconds'])

                        record = {
                            'scenario': scenario['name'],
                            'map_path': str(scenario['map_path']),
                            'customer_count': customer_count,
                            'instance_index': instance_index,
                            'group': group_name,
                            'depot_index': int(depot_record['depot_index']),
                            'depot_node': str(depot_record['depot_node']),
                            'assigned_customer_count': len(depot_record['customers']),
                            'convex_set_sizes': json.dumps(depot_record['convex_set_sizes']),
                            'set_tsp_sequence': json.dumps(depot_record['set_tsp_sequence']),
                            'full_objective': float(full_cost),
                            'full_objective_delta_from_c0': abs(
                                float(full_cost) - baseline_full_cost
                            ),
                            'full_solve_seconds': float(process_data['solve_seconds']),
                            'phase2_objective_delta_from_c0': objective_delta,
                            'phase2_objective_consistent': objective_consistent,
                            'phase2_speedup_vs_c0': (
                                baseline['total_seconds'] / phase2_total_seconds
                                if phase2_total_seconds > 0.0
                                else None
                            ),
                        }
                        record.update(metrics)
                        for name, count in report.initial_counts.items():
                            record[f'phase2_initial_{name}'] = int(count)
                        for name, count in report.final_counts.items():
                            record[f'phase2_final_{name}'] = int(count)
                        records.append(record)

                        if not objective_consistent:
                            write_current_records()
                            raise RuntimeError(
                                f'{scenario["name"]}/实例 {instance_index}/{group_name}/'
                                f'仓库 {depot_record["depot_index"]} 的 Phase 2 目标与 C0 '
                                f'不一致：delta={objective_delta}，'
                                f'tolerance={objective_tolerance}。'
                            )

                    # 每个实验组结束后立即刷新结果，服务器中断时最多损失当前组。
                    write_current_records()

    print(f'Pruning experiment JSON: {json_path}')
    print(f'Pruning experiment CSV: {csv_path}')
    return json_path, csv_path


if __name__ == '__main__':
    # 当前入口按用户确认直接运行 Manhattan 1K 与 Boston 11K 剪枝消融。
    run_pruning_experiments()


    # # 执行论文全量实验。
    # run_full_experiments()