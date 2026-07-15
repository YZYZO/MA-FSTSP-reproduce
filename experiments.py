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
# 导入 LRMP 基线算法。
from src.lrmp import LinearRelaxedMasterProblem
# 导入爬山 + 变邻域搜索基线算法。
from src.hc_vns import HillClimbingVariableNeighborhoodSearch
# 导入论文主算法。
from src.fstsp import MultiAgentFlyingSidekickTSP
# 导入线性规划基线算法。
from src.lp import LinearProgramming
# 导入实例构造函数与 Manhattan 路网读取函数。   
from problem import small_instance, multiagent_instance_on_manhattan, multiagent_instance_on_cambridge, manhattan
# 导入 `numpy`，用于均值、数组保存与随机采样。
import numpy as np
# 导入 `time`，用于统计实验耗时。
import time
# 导入进度条工具 `tqdm`。
from tqdm import tqdm
# 导入统一距离工厂，规模实验禁止再构造全对嵌套字典。
from distance_oracle import build_distance_provider
from config import RUN_FULL_EXPERIMENTS, ensure_dir, result_path

# 导入 `os`，用于读取/设置 CPU 亲和性。
import os
# 导入进程池工具，用于并行运行独立实验实例。
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


SMALL_DATA_DIR = result_path('small', 'data')
MANHATTAN_DATA_DIR = result_path('manhattan', 'data')
BOSTON_DATA_DIR = result_path('boston', 'data')


def _save_array(path, array):
    ensure_dir(path.parent)
    np.save(path, np.array(array))


def _save_npz(path, **arrays):
    ensure_dir(path.parent)
    np.savez(path, **arrays)


    
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
    返回: (index, cost, elapsed)
    """
    algorithm, i, drones, rounds, theta = task
    depots = _PAR_DEPOTS[i]
    cities = _PAR_CITIES[i]

    if algorithm == 'lrmp':
        model = LinearRelaxedMasterProblem(_PAR_GRAPH, depots, cities, _PAR_DISTANCE, drones)
    elif algorithm == 'hc':
        model = HillClimbingVariableNeighborhoodSearch(
            _PAR_GRAPH, depots, cities, _PAR_DISTANCE, drones, rounds=rounds
        )
    elif algorithm == 'stsp':
        if theta is None:
            model = MultiAgentFlyingSidekickTSP(_PAR_GRAPH, depots, cities, _PAR_DISTANCE, drones)
        else:
            model = MultiAgentFlyingSidekickTSP(
                _PAR_GRAPH, depots, cities, _PAR_DISTANCE, drones, theta=theta
            )
    elif algorithm == 'lp':
        model = LinearProgramming(_PAR_GRAPH, depots, cities, _PAR_DISTANCE, drones)
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')

    start = time.time()
    _, cost = model.solve()
    elapsed = time.time() - start

    if algorithm == 'lp' and not cost:
        cost = 0x3f3f3f3f3f3f

    return i, cost, elapsed


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
            i, cost, elapsed = future.result()
            results[i] = (cost, elapsed)

    return results


def _store_cost_time(costs, times, key, results):
    """
    将并行任务返回的 (cost, elapsed) 写回成本表和耗时表。
    """
    costs[key] = [item[0] for item in results]
    times[key] = [item[1] for item in results]


def run_quick_case(name, graph, depots, cities, distance, drones, hc_rounds=100, theta=(0.5, 0.5)):
    """
    运行一个轻量级对比实验，并打印启发式算法与主算法的结果。

    输入：
    - name: 当前实验名称，字符串。
    - graph: 路网图，`networkx` 图对象。
    - depots: 仓库节点集合。
    - cities: 客户节点集合。
    - distance: 预计算好的卡车/无人机距离字典。
    - drones: 每辆卡车可携带的无人机数量。
    - hc_rounds: 爬山算法的迭代轮数。
    - theta: 主算法中使用的两个阈值参数。

    输出：
    - 无显式返回值，直接在终端打印两种算法的成本和耗时。

    实现逻辑：
    1. 先构造爬山算法模型并求解。
    2. 再构造论文主算法模型并求解。
    3. 分别记录并打印两者耗时与成本。
    """
    # 打印当前用例名称，便于观察日志。
    print(f'\n{name}')
    # 构造爬山 + 变邻域搜索模型。
    hc_model = HillClimbingVariableNeighborhoodSearch(graph, depots, cities, distance, drones, rounds=hc_rounds)
    # 记录起始时间。
    start = time.time()
    # 求解模型，并忽略具体路径只保留成本。
    _, hc_cost = hc_model.solve()
    hc_time = time.time() - start
    # 打印爬山算法的结果。
    print(f'Hill Climbing cost {hc_cost} in {hc_time}s')

    # 构造论文主算法模型。
    fstsp_model = MultiAgentFlyingSidekickTSP(graph, depots, cities, distance, drones, theta=theta)
    # 重置计时起点。
    start = time.time()
    # 求解模型，并忽略具体路径只保留成本。
    _, fstsp_cost = fstsp_model.solve()
    fstsp_time = time.time() - start
    # 打印论文主算法的结果。
    print(f'Our algorithm cost {fstsp_cost} in {fstsp_time}s')
    return {
        'hc_cost': hc_cost,
        'hc_time': hc_time,
        'stsp_cost': fstsp_cost,
        'stsp_time': fstsp_time,
    }


# def test_small_instance(num, size):
#     """
#     在小规模子图实例上运行四种算法，并统计平均成本与平均耗时。

#     输入：
#     - num: 随机实例数量。
#     - size: 每个实例中的客户数量。

#     输出：
#     - 无显式返回值，直接打印 LRMP、HC、主算法、LP 的平均表现。

#     实现逻辑：
#     1. 调用 `small_instance` 生成若干个固定子图上的随机仓库/客户配置。
#     2. 依次运行 LRMP、HC、主算法和 LP。
#     3. 用列表累计成本与耗时，最后输出平均值。
#     """
#     # 生成共享的小规模图、多个 depot/city 采样以及距离矩阵。
#     graph, _depots, _cities, distance = small_instance(num, 20, 2, size)
#     # 为四种算法准备成本记录容器。
#     costs = {'lrmp': [], 'hc': [], 'stsp': [], 'lp': []}
#     # 为四种算法准备耗时记录容器。
#     times = {'lrmp': [], 'hc': [], 'stsp': [], 'lp': []}

#     # 遍历所有随机实例，评估 LRMP。
#     for i in tqdm(range(num)):
#         # 取出当前实例的仓库与客户。
#         depots, cities = _depots[i], _cities[i]
#         # 构造 LRMP 模型。
#         model = LinearRelaxedMasterProblem(graph, depots, cities, distance, 2)
#         # 记录起始时间。
#         start = time.time()
#         # 求解模型。
#         solution, cost = model.solve()
#         # 保存当前实例的成本。
#         costs['lrmp'].append(cost)
#         # 保存当前实例的耗时。
#         times['lrmp'].append(time.time() - start)
#     # 打印 LRMP 的平均结果。
#     print(f'LRMP gives solution with cost {sum(costs["lrmp"]) / num} in {sum(times["lrmp"]) / num}s')

#     # 遍历所有随机实例，评估 HC。
#     for i in tqdm(range(num)):
#         # 取出当前实例的仓库与客户。
#         depots, cities = _depots[i], _cities[i]
#         # 构造 HC 模型，并设置 1000 轮局部搜索。
#         model = HillClimbingVariableNeighborhoodSearch(graph, depots, cities, distance, 2, rounds=1000)
#         # 记录起始时间。
#         start = time.time()
#         # 求解模型。
#         solution, cost = model.solve()
#         # 保存成本。
#         costs['hc'].append(cost)
#         # 保存耗时。
#         times['hc'].append(time.time() - start)
#     # 打印 HC 的平均结果。
#     print(f'Hill Climbing gives solution with cost {sum(costs["hc"]) / num} in {sum(times["hc"]) / num}s')

#     # 遍历所有随机实例，评估论文主算法。
#     for i in tqdm(range(num)):
#         # 取出当前实例的仓库与客户。
#         depots, cities = _depots[i], _cities[i]
#         # 构造主算法模型。
#         model = MultiAgentFlyingSidekickTSP(graph, depots, cities, distance, 2)
#         # 记录起始时间。
#         start = time.time()
#         # 求解模型。
#         solution, cost = model.solve()
#         # 保存成本。
#         costs['stsp'].append(cost)
#         # 保存耗时。
#         times['stsp'].append(time.time() - start)
#     # 打印主算法的平均结果。
#     print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')

#     # 遍历所有随机实例，评估 LP。
#     for i in tqdm(range(num)):
#         # 取出当前实例的仓库与客户。
#         depots, cities = _depots[i], _cities[i]
#         # 构造 LP 模型。
#         model = LinearProgramming(graph, depots, cities, distance, 2)
#         # 记录起始时间。
#         start = time.time()
#         # 求解模型。
#         solution, cost = model.solve()
#         # 保存成本。
#         if cost:
#             costs['lp'].append(cost)
#         else:
#             costs['lp'].append(0x3f3f3f3f3f3f)
#         # 保存耗时。
#         times['lp'].append(time.time() - start)
#     # 打印 LP 的平均结果。
#     print(f'LP gives solution with cost {sum(costs["lp"]) / num} in {sum(times["lp"]) / num}s')
#     _save_npz(
#         SMALL_DATA_DIR / f'small-size-{size}.npz',
#         lrmp_cost=np.array(costs['lrmp']),
#         lrmp_time=np.array(times['lrmp']),
#         hc_cost=np.array(costs['hc']),
#         hc_time=np.array(times['hc']),
#         stsp_cost=np.array(costs['stsp']),
#         stsp_time=np.array(times['stsp']),
#         lp_cost=np.array(costs['lp']),
#         lp_time=np.array(times['lp']),
#     )


# def test_manhattan(num, size):
#     """
#     在 Manhattan 路网实例上运行 HC 和论文主算法。

#     输入：
#     - num: 随机实例数量。
#     - size: 每个实例的客户数量。

#     输出：
#     - 无显式返回值，直接打印两种算法的平均成本与平均耗时。

#     实现逻辑：
#     1. 生成 `num` 组 Manhattan 随机实例。
#     2. 先评估 HC，再评估论文主算法。
#     3. 汇总平均成本与平均耗时。
#     """
#     # 生成 Manhattan 路网实例集合。
#     graph, depots, cities, distance = multiagent_instance_on_manhattan(num, 5, size)
#     # 为两种算法创建成本记录表。
#     costs = {'hc': [], 'stsp': [], 'lp': []}
#     # 为两种算法创建耗时记录表。
#     times = {'hc': [], 'stsp': [], 'lp': []}

#     # 依次运行 HC。
#     # for i in tqdm(range(num)):
#     #     # 取出当前实例的仓库和客户。
#     #     depot, city = depots[i], cities[i]
#     #     # 构造 HC 模型。
#     #     model = HillClimbingVariableNeighborhoodSearch(graph, depot, city, distance, 3, rounds=5000)
#     #     # 开始计时。
#     #     start = time.time()
#     #     # 求解当前实例。
#     #     solution, cost = model.solve()
#     #     # 记录耗时。
#     #     times['hc'].append(time.time() - start)
#     #     # 记录成本。
#     #     costs['hc'].append(cost)
#     # # 输出 HC 平均表现。
#     # print(f'Hill Climbing gives solution with cost {sum(costs["hc"]) / num} in {sum(times["hc"])/ num}')

#     # 依次运行论文主算法。
#     for i in tqdm(range(num)):
#         # 取出当前实例的仓库和客户。
#         depot, city = depots[i], cities[i]
#         # 构造论文主算法模型。
#         model = MultiAgentFlyingSidekickTSP(graph, depot, city, distance, 3, theta=(0.5, 0.5))
#         # 开始计时。
#         start = time.time()
#         # 求解当前实例。
#         solution, cost = model.solve()
#         # 记录耗时。
#         times['stsp'].append(time.time() - start)
#         # 记录成本。
#         costs['stsp'].append(cost)
#     # 输出论文主算法平均表现。
#     print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')
#     _save_npz(
#         MANHATTAN_DATA_DIR / f'road-size-{size}.npz',
#         hc_cost=np.array(costs['hc']),
#         hc_time=np.array(times['hc']),
#         stsp_cost=np.array(costs['stsp']),
#         stsp_time=np.array(times['stsp']),
#     )


# def test_cambridge(num, size):
#     """
#     在 Cambridge 路网实例上运行 HC 和论文主算法。

#     输入：
#     - num: 随机实例数量。
#     - size: 每个实例的客户数量。

#     输出：
#     - 无显式返回值，直接打印两种算法的平均成本与平均耗时。

#     实现逻辑：
#     1. 生成 `num` 组 Cambridge 随机实例。
#     2. 分别运行 HC 和论文主算法。
#     3. 输出汇总结果。
#     """
#     # 生成 Cambridge 路网实例集合。
#     graph, depots, cities, distance = multiagent_instance_on_cambridge(num, 10, size)
#     # 为两种算法创建成本记录表。
#     costs = {'hc': [], 'stsp': [], 'lp': []}
#     # 为两种算法创建耗时记录表。
#     times = {'hc': [], 'stsp': [], 'lp': []}

#     # 运行 HC。
#     # for i in tqdm(range(num)):
#     #     # 取出当前实例。
#     #     depot, city = depots[i], cities[i]
#     #     # 构造 HC 模型。
#     #     model = HillClimbingVariableNeighborhoodSearch(graph, depot, city, distance, 3, rounds=5000)
#     #     # 开始计时。
#     #     start = time.time()
#     #     # 求解当前实例。
#     #     solution, cost = model.solve()
#     #     # 记录耗时。
#     #     times['hc'].append(time.time() - start)
#     #     # 记录成本。
#     #     costs['hc'].append(cost)
#     # # 输出 HC 平均表现。
#     # print(f'Hill Climbing gives solution with cost {sum(costs["hc"]) / num} in {sum(times["hc"]) / num}')

#     # 运行论文主算法。
#     for i in tqdm(range(num)):
#         # 取出当前实例。
#         depot, city = depots[i], cities[i]
#         # 构造主算法模型。
#         model = MultiAgentFlyingSidekickTSP(graph, depot, city, distance, 3, theta=(0.5, 0.5))
#         # 开始计时。
#         start = time.time()
#         # 求解当前实例。
#         solution, cost = model.solve()
#         # 记录耗时。
#         times['stsp'].append(time.time() - start)
#         # 记录成本。
#         costs['stsp'].append(cost)
#     # 输出主算法平均表现。
#     print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')
#     _save_npz(
#         BOSTON_DATA_DIR / f'road-size-{size}.npz',
#         hc_cost=np.array(costs['hc']),
#         hc_time=np.array(times['hc']),
#         stsp_cost=np.array(costs['stsp']),
#         stsp_time=np.array(times['stsp']),
#     )


def test_small_instance(num, size):
    """
    在小规模子图实例上运行四种算法，并统计平均成本与平均耗时。

    本版本将每一种算法在 num 个实例上的求解改为 10 进程并行。
    """
    # 生成共享的小规模图、多个 depot/city 采样以及距离矩阵。
    graph, _depots, _cities, distance = small_instance(num, 20, 2, size)
    # 为四种算法准备成本记录容器。
    costs = {'lrmp': [], 'hc': [], 'stsp': [], 'lp': []}
    # 为四种算法准备耗时记录容器。
    times = {'lrmp': [], 'hc': [], 'stsp': [], 'lp': []}

    print(f'Running small instance experiment with {PROCESS_WORKERS} processes.')

    lrmp_results = _run_parallel_instances(
        num, graph, _depots, _cities, distance,
        algorithm='lrmp', drones=2, desc='LRMP'
    )
    _store_cost_time(costs, times, 'lrmp', lrmp_results)
    print(f'LRMP gives solution with cost {sum(costs["lrmp"]) / num} in {sum(times["lrmp"]) / num}s')

    hc_results = _run_parallel_instances(
        num, graph, _depots, _cities, distance,
        algorithm='hc', drones=2, rounds=1000, desc='HC'
    )
    _store_cost_time(costs, times, 'hc', hc_results)
    print(f'Hill Climbing gives solution with cost {sum(costs["hc"]) / num} in {sum(times["hc"]) / num}s')

    stsp_results = _run_parallel_instances(
        num, graph, _depots, _cities, distance,
        algorithm='stsp', drones=2, desc='STSP'
    )
    _store_cost_time(costs, times, 'stsp', stsp_results)
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')

    lp_results = _run_parallel_instances(
        num, graph, _depots, _cities, distance,
        algorithm='lp', drones=2, desc='LP'
    )
    _store_cost_time(costs, times, 'lp', lp_results)
    print(f'LP gives solution with cost {sum(costs["lp"]) / num} in {sum(times["lp"]) / num}s')

    _save_npz(
        SMALL_DATA_DIR / f'small-size-{size}.npz',
        lrmp_cost=np.array(costs['lrmp']),
        lrmp_time=np.array(times['lrmp']),
        hc_cost=np.array(costs['hc']),
        hc_time=np.array(times['hc']),
        stsp_cost=np.array(costs['stsp']),
        stsp_time=np.array(times['stsp']),
        lp_cost=np.array(costs['lp']),
        lp_time=np.array(times['lp']),
    )


def test_manhattan(num, size):
    """
    在 Manhattan 路网实例上运行 HC 和论文主算法。

    本版本将论文主算法在 num 个实例上的求解改为 10 进程并行。
    原文件中的 HC 部分本来就是注释状态，这里保持不运行 HC。
    """
    # 生成 Manhattan 路网实例集合。
    graph, depots, cities, distance = multiagent_instance_on_manhattan(num, 5, size)
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
    _save_npz(
        MANHATTAN_DATA_DIR / f'road-size-{size}.npz',
        hc_cost=np.array(costs['hc']),
        hc_time=np.array(times['hc']),
        stsp_cost=np.array(costs['stsp']),
        stsp_time=np.array(times['stsp']),
    )


def test_cambridge(num, size):
    """
    在 Cambridge 路网实例上运行 HC 和论文主算法。

    本版本将论文主算法在 num 个实例上的求解改为 10 进程并行。
    原文件中的 HC 部分本来就是注释状态，这里保持不运行 HC。
    """
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
    _save_npz(
        BOSTON_DATA_DIR / f'road-size-{size}.npz',
        hc_cost=np.array(costs['hc']),
        hc_time=np.array(times['hc']),
        stsp_cost=np.array(costs['stsp']),
        stsp_time=np.array(times['stsp']),
    )


def ablation_r():
    """
    研究飞行距离上限 `limit` 对算法性能的影响。

    输入：
    - 无显式输入，函数内部固定使用 20 组实例、5 个仓库、100 个客户。

    输出：
    - 将每个 `limit` 下的耗时保存到 `r-time.npy`。
    - 将每个 `limit` 下的成本保存到 `r-cost.npy`。

    实现逻辑：
    1. 生成固定的一批 Manhattan 实例。
    2. 枚举不同的飞行半径上限。
    3. 对每个上限重复求解，并保存全部原始结果。
    """
    # 打印实验主题。
    print('Studying the effect of radius limit')
    # 生成固定的一批 Manhattan 实例。
    graph, depots, cities, distance = multiagent_instance_on_manhattan(20, 5, 100)
    # 初始化成本和耗时容器。
    costs, times = [], []
    # 枚举不同半径参数。
    for r in range(5, 16, 2):
        # 为当前半径开辟成本列表。
        costs.append([])
        # 为当前半径开辟耗时列表。
        times.append([])
        # 遍历 100 个随机实例。
        for i in tqdm(range(100)):
            # 构造主算法模型，其中 `limit=r/10`。
            model = MultiAgentFlyingSidekickTSP(graph, depots[i], cities[i], distance, 3, limit=r / 10, theta=(0.5, 0.5))
            # 开始计时。
            start = time.time()
            # 求解当前实例。
            _, cost = model.solve()
            # 保存当前实例耗时。
            times[-1].append(time.time() - start)
            # 保存当前实例成本。
            costs[-1].append(cost)
    # 将耗时数组保存到磁盘。
    _save_array(MANHATTAN_DATA_DIR / 'r-time.npy', times)
    # 将成本数组保存到磁盘。
    _save_array(MANHATTAN_DATA_DIR / 'r-cost.npy', costs)


def ablation_speed():
    """
    研究无人机速度比对算法性能的影响。

    输入：
    - 无显式输入，函数内部固定使用一批 Manhattan 实例。

    输出：
    - 将速度实验耗时保存到 `speed-time.npy`。
    - 将速度实验成本保存到 `speed-cost.npy`。

    实现逻辑：
    1. 生成固定实例。
    2. 枚举不同的无人机速度参数。
    3. 对每个速度统计总耗时与所有成本样本。
    """
    # 打印实验主题。
    print('Studying the effect of speed')
    # 生成固定的一批 Manhattan 实例。
    graph, depots, cities, distance = multiagent_instance_on_manhattan(20, 5, 100)
    # 初始化成本和耗时容器。
    costs, times = [], []
    # 枚举不同速度比。
    for speed in [i / 30 for i in range(10, 120, 20)]:
        # 为当前速度开辟成本列表。
        costs.append([])
        # 为当前速度累计总耗时。
        times.append(0)
        # 遍历 100 个随机实例。
        for i in tqdm(range(100)):
            # 构造主算法模型，其中速度设为当前 `speed`。
            model = MultiAgentFlyingSidekickTSP(graph, depots[i], cities[i], distance, 3, speed=speed)
            # 开始计时。
            start = time.time()
            # 求解当前实例。
            _, cost = model.solve()
            # 累加总耗时。
            times[-1] += time.time() - start
            # 记录当前实例成本。
            costs[-1].append(cost)
    # 保存耗时数组。
    _save_array(MANHATTAN_DATA_DIR / 'speed-time.npy', times)
    # 保存成本数组。
    _save_array(MANHATTAN_DATA_DIR / 'speed-cost.npy', costs)


def ablation_k():
    """
    研究无人机数量对主算法性能的影响。

    输入：
    - 无显式输入，函数内部枚举客户规模并测试 0 到 5 架无人机。

    输出：
    - 将所有规模下的无人机数量成本矩阵保存为 `k-cost.npy`。

    实现逻辑：
    1. 枚举不同客户规模。
    2. 对每个规模运行 `solve_multiple_drones`。
    3. 保存每个规模下不同无人机数量的成本结果。
    """
    # 打印实验主题。
    print('studying the effect of drone number')
    # 用于汇总所有规模的成本矩阵。
    all_costs = []
    # 枚举客户规模。
    for size in range(50, 160, 20):
        # 生成当前规模的 Manhattan 实例。
        graph, depots, cities, distance = multiagent_instance_on_manhattan(10, 5, size)
        # 存放当前规模下所有样本的成本。
        costs, times = [], []
        # 遍历 100 个随机实例。
        for i in tqdm(range(100)):
            # 构造主算法模型，初始无人机数给 0，随后由 `solve_multiple_drones` 枚举。
            model = MultiAgentFlyingSidekickTSP(graph, depots[i], cities[i], distance, 0)
            # 求出 0~5 架无人机对应的成本序列。
            cost = model.solve_multiple_drones()
            # 保存当前实例结果。
            costs.append(cost.copy())
        # 将当前规模的结果转为数组并加入总表。
        all_costs.append(np.array(costs))
        # 打印当前规模下的平均成本。
        print(f'size {size} gives {np.mean(costs, axis=0)}')
    # 将总结果保存到磁盘。
    _save_array(MANHATTAN_DATA_DIR / 'k-cost.npy', np.array(all_costs, dtype=float))


def scale_cities():
    """
    固定仓库数量，研究客户数量增长时主算法的可扩展性。

    输入：
    - 无显式输入，内部固定使用 Manhattan 路网。

    输出：
    - 保存耗时到 `city-time.npy`。
    - 保存成本到 `city-cost.npy`。

    实现逻辑：
    1. 先构造 Manhattan 路网和距离矩阵。
    2. 枚举不同客户数量。
    3. 每个数量下重复采样并统计性能。
    """
    # 打印实验主题。
    print('studying the scalability of fix depot case')
    # 读取 Manhattan 路网。
    graph = manhattan()
    # 三类规模实验共享同一图哈希 H2H 缓存，无人机距离保持按需计算。
    distance = build_distance_provider(
        graph,
        dataset_name=graph.graph.get('dataset_name', 'nyc'),
        graph_path=graph.graph.get('source_path'),
    )
    # 初始化成本和耗时容器。
    costs, times = [], []
    # 枚举客户数量。
    for num in range(120, 350, 40):
        # 为当前规模开辟成本列表。
        costs.append([])
        # 为当前规模开辟耗时列表。
        times.append([])
        # 重复 100 次随机采样。
        for _ in tqdm(range(100)):
            # 从图节点中随机采样仓库和客户。
            locations = np.random.choice(graph.nodes, num + 10, replace=False)
            # 构造主算法模型。
            model = MultiAgentFlyingSidekickTSP(graph, locations[:10], locations[10:], distance, 3)
            # 开始计时。
            start = time.time()
            # 求解当前实例。
            _, cost = model.solve()
            # 保存耗时。
            times[-1].append(time.time() - start)
            # 保存成本。
            costs[-1].append(cost)
    # 保存耗时结果。
    _save_array(MANHATTAN_DATA_DIR / 'city-time.npy', times)
    # 保存成本结果。
    _save_array(MANHATTAN_DATA_DIR / 'city-cost.npy', costs)


def scale_rates():
    """
    固定客户与仓库比例，研究规模增长时主算法的可扩展性。

    输入：
    - 无显式输入，内部固定客户仓库比例为 20:1。

    输出：
    - 保存耗时到 `rates-time.npy`。
    - 保存成本到 `rates-cost.npy`。

    实现逻辑：
    1. 读取 Manhattan 路网。
    2. 枚举仓库数量。
    3. 按照固定比例同时增加客户数量。
    """
    # 打印实验主题。
    print('studying the scalability of fix rates case')
    # 读取 Manhattan 路网。
    graph = manhattan()
    # 使用与其他规模实验相同的数据集名，确保只构建一份版本化索引。
    distance = build_distance_provider(
        graph,
        dataset_name=graph.graph.get('dataset_name', 'nyc'),
        graph_path=graph.graph.get('source_path'),
    )
    # 初始化成本和耗时容器。
    costs, times = [], []
    # 枚举仓库数。
    for num in range(3, 21, 3):
        # 为当前规模开辟成本列表。
        costs.append([])
        # 为当前规模开辟耗时列表。
        times.append([])
        # 重复 100 次随机采样。
        for _ in tqdm(range(100)):
            # 按固定比例采样节点。
            locations = np.random.choice(graph.nodes, num * (1 + 20), replace=False)
            # 构造主算法模型。
            model = MultiAgentFlyingSidekickTSP(graph, locations[:num], locations[num:], distance, 3)
            # 开始计时。
            start = time.time()
            # 求解当前实例。
            _, cost = model.solve()
            # 保存耗时。
            times[-1].append(time.time() - start)
            # 保存成本。
            costs[-1].append(cost)
    # 保存耗时结果。
    _save_array(MANHATTAN_DATA_DIR / 'rates-time.npy', times)
    # 保存成本结果。
    _save_array(MANHATTAN_DATA_DIR / 'rates-cost.npy', costs)


def scale_depots():
    """
    固定客户数量，研究仓库数量增长时主算法的可扩展性。

    输入：
    - 无显式输入，内部固定客户数为 150。

    输出：
    - 保存耗时到 `depots-time.npy`。
    - 保存成本到 `depots-cost.npy`。

    实现逻辑：
    1. 读取 Manhattan 路网。
    2. 枚举仓库数。
    3. 固定客户数 150，重复采样并统计性能。
    """
    # 打印实验主题。
    print('studying the scalability of fix cities case')
    # 读取 Manhattan 路网。
    graph = manhattan()
    # 距离工厂命中前两类实验的 H2H 缓存，不再按实验重复物化矩阵。
    distance = build_distance_provider(
        graph,
        dataset_name=graph.graph.get('dataset_name', 'nyc'),
        graph_path=graph.graph.get('source_path'),
    )
    # 初始化成本和耗时容器。
    costs, times = [], []
    # 枚举仓库数量。
    for num in range(5, 16, 2):
        # 为当前规模开辟成本列表。
        costs.append([])
        # 为当前规模开辟耗时列表。
        times.append([])
        # 重复 100 次随机采样。
        for _ in tqdm(range(100)):
            # 随机采样仓库与客户。
            locations = np.random.choice(graph.nodes, num + 150, replace=False)
            # 构造主算法模型。
            model = MultiAgentFlyingSidekickTSP(graph, locations[:num], locations[num:], distance, 3)
            # 开始计时。
            start = time.time()
            # 求解当前实例。
            _, cost = model.solve()
            # 保存耗时。
            times[-1].append(time.time() - start)
            # 保存成本。
            costs[-1].append(cost)
    # 保存耗时结果。
    _save_array(MANHATTAN_DATA_DIR / 'depots-time.npy', times)
    # 保存成本结果。
    _save_array(MANHATTAN_DATA_DIR / 'depots-cost.npy', costs)


def run_demo_experiments():
    """
    运行离线轻量级演示实验。

    输入：
    - 无显式输入。

    输出：
    - 无返回值，直接打印三个小规模案例的实验结果。

    实现逻辑：
    1. 构造一个小型子图案例。
    2. 构造一个合成 Manhattan 案例。
    3. 构造一个合成 Cambridge 案例。
    """
    # 打印演示模式提示。
    print('Running a lightweight demo suite.')
    # 提示如何切换到论文全量实验。
    print('Set RUN_FULL_EXPERIMENTS=True in config.py if you want the paper-scale experiments.')
    # 构造小型子图案例。
    graph, depots, cities, distance = small_instance(1, 10, 2, 3)
    # 运行小型子图案例。
    result = run_quick_case('Small synthetic subset', graph, depots[0], cities[0], distance, 2, hc_rounds=80)
    _save_npz(SMALL_DATA_DIR / 'quick-small-subset.npz', **result)

    # 构造 Manhattan 局部路网案例。
    graph, depots, cities, distance = small_instance(1, 80, 3, 6)
    # 运行 Manhattan 案例。
    result = run_quick_case('Local Manhattan road subset', graph, depots[0], cities[0], distance, 3, hc_rounds=80)
    _save_npz(MANHATTAN_DATA_DIR / 'quick-road-subset.npz', **result)

    # 构造 Boston/Cambridge 路网案例。
    graph, depots, cities, distance = multiagent_instance_on_cambridge(1, 2, 3)
    # 运行 Boston/Cambridge 案例。
    result = run_quick_case('Boston road instance', graph, depots[0], cities[0], distance, 2, hc_rounds=40)
    _save_npz(BOSTON_DATA_DIR / 'quick-road-instance.npz', **result)


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
    # #size = 5
    #     # 运行当前规模的小规模实验。
    #     test_small_instance(100, size)
    # 枚举大规模路网实验中的客户数量。
    #for size in [50, 100, 150]:
    size = 20
        # 运行 Manhattan 实验。
    test_manhattan(1, size)
        # 运行 Cambridge 实验。
    test_cambridge(1, size)
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


if __name__ == '__main__':
    # 当配置显式要求全量实验时，运行完整实验。
    if RUN_FULL_EXPERIMENTS:
        # 执行论文全量实验。
        run_full_experiments()
    else:
        # 否则执行离线演示实验。
        run_demo_experiments()
