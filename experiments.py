"""
本文件是论文实验的主入口脚本。

主要内容：
1. 组织小规模示例、Manhattan 路网实验、Cambridge 路网实验。
2. 组织论文中的消融实验与可扩展性实验。
3. 在缺少真实地图数据时，自动退化为轻量级离线演示。
4. 调用 `src/` 目录中的多种算法，并将成本与耗时保存为 `.npy` 文件。
"""

# 导入 `networkx`，用于图最短路和全对最短路计算。
import networkx as nx
# 导入 `sys`，保留给潜在的命令行扩展使用。
import sys
# 导入日期时间工具，用实验启动时间生成不会覆盖旧结果的文件名。
from datetime import datetime

# 导入论文主算法。
from src.fstsp import MultiAgentFlyingSidekickTSP

# 导入实例构造函数与 Manhattan 路网读取函数。   
from problem import small_instance, multiagent_instance_on_manhattan, multiagent_instance_on_cambridge, manhattan
# 导入 `numpy`，用于均值、数组保存与随机采样。
import numpy as np
# 导入 `time`，用于统计实验耗时。
import time
# 导入进度条工具 `tqdm`。
from tqdm import tqdm
# 导入球面距离与统一结果目录工具。
from utils import ensure_dir, haversine, result_path
from config import (
    MANHATTAN1k_GRAPH_PATH,
    MANHATTAN11k_GRAPH_PATH,
)
from experiment_results import (
    _save_stsp_batch_result,
    _solve_model_with_process_data,
)


SMALL_DATA_DIR = result_path('small', 'data')
MANHATTAN_DATA_DIR = result_path('manhattan', 'data')
BOSTON_DATA_DIR = result_path('boston', 'data')


def _save_array(path, array):
    ensure_dir(path.parent)
    np.save(path, np.array(array))


def _print_distance_initialization_stats(label, stats):
    '''
    输出一批路网实验的全点对距离初始化耗时。

    输入：数据集显示名称和批次级耗时字典。
    输出：无；打印卡车、无人机和总初始化耗时。
    '''
    print(
        '{} distance initialization: truck APSP={:.3f}s, '
        'drone pairwise={:.3f}s, total={:.3f}s.'.format(
            label,
            stats['truck_apsp_seconds'],
            stats['drone_pairwise_seconds'],
            stats['distance_initialization_seconds'],
        )
    )





def test_manhattan(num, size):
    """
    在 Manhattan 路网实例上顺序运行论文主算法。

    输入：
    - num: 随机实例数量。
    - size: 每个实例的客户数量。

    输出：
    - 无显式返回值；打印平均成本与耗时，并保存包含路线和三阶段记录的批次结果。

    实现逻辑：
    1. 一次性生成共享路网、距离数据和全部随机实例。
    2. 按实例索引在主进程中依次构造并求解模型。
    3. 汇总结果，同时保留现有 NPZ 结果结构。
    """
    # 生成 Manhattan 路网实例集合。
    graph, depots, cities, distance = multiagent_instance_on_manhattan(num, 5, size)
    # 为两种算法创建成本记录表。
    costs = {'hc': [], 'stsp': [], 'lp': []}
    # 为两种算法创建耗时记录表。
    times = {'hc': [], 'stsp': [], 'lp': []}
    # 按实例顺序保存成本、耗时、路线和三阶段过程数据。
    stsp_results = []

    print('Running Manhattan experiment serially.')

    # 恢复官方仓库的顺序循环风格，避免为每批实例创建额外子进程。
    for i in tqdm(range(num), desc='Manhattan-STSP'):
        # 当前实例的仓库和客户必须使用同一个索引配对。
        depot, city = depots[i], cities[i]
        model = MultiAgentFlyingSidekickTSP(
            graph, depot, city, distance, 3, theta=(0.5, 0.5)
        )
        # 保留扩展结果模块需要的最终路线与三阶段紧凑记录。
        solution, cost, process_data = _solve_model_with_process_data(model)
        elapsed = process_data['solve_seconds']
        stsp_results.append((cost, elapsed, solution, process_data))
        costs['stsp'].append(cost)
        times['stsp'].append(elapsed)

    # 输出论文主算法平均表现。
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')
    _save_stsp_batch_result(
        MANHATTAN_DATA_DIR / f'road-size-{size}.npz',
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
    在 Cambridge 路网实例上顺序运行论文主算法。

    输入：
    - num: 随机实例数量。
    - size: 每个实例的客户数量。

    输出：
    - 无显式返回值；打印平均成本与耗时，并保存包含路线和三阶段记录的批次结果。

    实现逻辑：
    1. 一次性生成共享路网、距离数据和全部随机实例。
    2. 按实例索引在主进程中依次构造并求解模型。
    3. 汇总结果，同时保留现有 NPZ 结果结构。
    """
    # 生成 Cambridge 路网实例集合。
    graph, depots, cities, distance = multiagent_instance_on_cambridge(num, 10, size)
    # 为两种算法创建成本记录表。
    costs = {'hc': [], 'stsp': [], 'lp': []}
    # 为两种算法创建耗时记录表。
    times = {'hc': [], 'stsp': [], 'lp': []}
    # 按实例顺序保存成本、耗时、路线和三阶段过程数据。
    stsp_results = []

    print('Running Cambridge experiment serially.')

    # 恢复官方仓库的顺序循环风格，避免为每批实例创建额外子进程。
    for i in tqdm(range(num), desc='Cambridge-STSP'):
        # 当前实例的仓库和客户必须使用同一个索引配对。
        depot, city = depots[i], cities[i]
        model = MultiAgentFlyingSidekickTSP(
            graph, depot, city, distance, 3, theta=(0.5, 0.5)
        )
        # 保留扩展结果模块需要的最终路线与三阶段紧凑记录。
        solution, cost, process_data = _solve_model_with_process_data(model)
        elapsed = process_data['solve_seconds']
        stsp_results.append((cost, elapsed, solution, process_data))
        costs['stsp'].append(cost)
        times['stsp'].append(elapsed)

    # 输出主算法平均表现。
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')
    _save_stsp_batch_result(
        BOSTON_DATA_DIR / f'road-size-{size}.npz',
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
    在 1,024 节点 NYC 路网上按论文 Manhattan 场景运行主算法。

    输入：
    - num: 本次需要求解的随机实例数量；论文完整实验使用 100。
    - size: 每个实例的客户数量；论文使用 50、100、150。

    输出：
    - 无显式返回值；将全部实例的成本和耗时保存到一个 `.npz` 文件。

    实现逻辑：
    1. 从 `nyc_1024.graphml` 生成 5 仓库随机实例。
    2. 使用每车 3 架无人机和 `(0.5, 0.5)` 阈值运行论文主算法。
    3. 使用实验启动时间区分批次，并将结果写入 Manhattan 目录。
    """
    # 秒级时间戳用于避免相同客户规模的不同实验批次互相覆盖。
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # 中型 Manhattan 论文场景固定为 5 个仓库，并显式选择 1,024 节点地图。
    graph, depots, cities, distance, distance_stats = multiagent_instance_on_manhattan(
        num,
        5,
        size,
        MANHATTAN1k_GRAPH_PATH,
        return_distance_stats=True,
    )
    # 距离矩阵由本批次全部实例共享，因此初始化耗时只报告和保存一次。
    _print_distance_initialization_stats('Manhattan 1K', distance_stats)
    # 保持与现有路网实验相同的结果结构；HC 和 LP 在当前分支不执行。
    costs = {'hc': [], 'stsp': [], 'lp': []}
    times = {'hc': [], 'stsp': [], 'lp': []}

    # 按实例顺序保存成本、耗时、路线和三阶段过程数据。
    stsp_results = []

    print('Running Manhattan 1K experiment serially.')

    # 主算法默认使用 1.5 km 航程和 1.6 倍无人机速度，仅显式传入论文阈值。
    for i in tqdm(range(num), desc='Manhattan-1K-STSP'):
        # 当前实例的仓库和客户必须使用同一个索引配对。
        depot, city = depots[i], cities[i]
        model = MultiAgentFlyingSidekickTSP(
            graph, depot, city, distance, 3, theta=(0.5, 0.5)
        )
        # 顺序求解并保留绘图、校验所需的紧凑过程数据。
        solution, cost, process_data = _solve_model_with_process_data(model)
        elapsed = process_data['solve_seconds']
        stsp_results.append((cost, elapsed, solution, process_data))
        costs['stsp'].append(cost)
        times['stsp'].append(elapsed)

    # 输出本批次主算法的平均成本和平均求解时间。
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')
    # 结果文件命名与目标分支保持一致。
    _save_stsp_batch_result(
        MANHATTAN_DATA_DIR / f'{run_timestamp}-manhattan_1k-{size}.npz',
        stsp_results,
        depots,
        cities,
        distance,
        3,
        costs,
        times,
        distance_initialization_stats=distance_stats,
    )


def test_manhattan_11k(num, size):
    """
    在 11,000 节点 NYC 路网上按论文 Boston 场景口径运行主算法。

    输入：
    - num: 本次需要求解的随机实例数量；论文完整实验使用 100。
    - size: 每个实例的客户数量；论文使用 50、100、150。

    输出：
    - 无显式返回值；将全部实例的成本和耗时保存到一个 `.npz` 文件。

    实现逻辑：
    1. 从 `nyc_11000.graphml` 生成 10 仓库随机实例。
    2. 按 Boston 实验口径使用每车 4 架无人机和 `(0.5, 0.5)` 阈值。
    3. 使用实验启动时间区分批次，并将结果写入 Boston 结果目录。
    """
    # 秒级时间戳用于避免相同客户规模的不同实验批次互相覆盖。
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # 该 NYC 地图仅替代 Boston 的路网规模，仓库数量仍采用论文大型场景参数。
    graph, depots, cities, distance, distance_stats = multiagent_instance_on_manhattan(
        num,
        10,
        size,
        MANHATTAN11k_GRAPH_PATH,
        return_distance_stats=True,
    )
    # 11K NYC 替代图继续按 Boston 实验口径命名，初始化耗时按批次记录一次。
    _print_distance_initialization_stats('Boston 11K', distance_stats)
    # 保持与现有路网实验相同的结果结构；HC 和 LP 在当前分支不执行。
    costs = {'hc': [], 'stsp': [], 'lp': []}
    times = {'hc': [], 'stsp': [], 'lp': []}

    # 按实例顺序保存成本、耗时、路线和三阶段过程数据。
    stsp_results = []

    print('Running Boston 11K reproduction serially.')

    # 主算法默认使用 1.5 km 航程和 1.6 倍无人机速度，仅显式传入论文阈值。
    for i in tqdm(range(num), desc='Boston-11K-STSP'):
        # 当前实例的仓库和客户必须使用同一个索引配对。
        depot, city = depots[i], cities[i]
        model = MultiAgentFlyingSidekickTSP(
            graph, depot, city, distance, 4, theta=(0.5, 0.5)
        )
        # 顺序求解并保留绘图、校验所需的紧凑过程数据。
        solution, cost, process_data = _solve_model_with_process_data(model)
        elapsed = process_data['solve_seconds']
        stsp_results.append((cost, elapsed, solution, process_data))
        costs['stsp'].append(cost)
        times['stsp'].append(elapsed)

    # 输出本批次主算法的平均成本和平均求解时间。
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')
    # 按用户要求沿用目标分支的 Boston 目录和 `boston_11k` 命名。
    _save_stsp_batch_result(
        BOSTON_DATA_DIR / f'{run_timestamp}-boston_11k-{size}.npz',
        stsp_results,
        depots,
        cities,
        distance,
        4,
        costs,
        times,
        distance_initialization_stats=distance_stats,
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
#     # 构造卡车和无人机距离矩阵。
#     distance = {'truck': dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight')),
#                 'drone': {i: {j: haversine(graph.nodes[i]['pos'], graph.nodes[j]['pos']) for j in graph.nodes}
#                           for i in graph.nodes}}
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
#     # 构造卡车和无人机距离矩阵。
#     distance = {'truck': dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight')),
#                 'drone': {i: {j: haversine(graph.nodes[i]['pos'], graph.nodes[j]['pos']) for j in graph.nodes}
#                           for i in graph.nodes}}
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
#     # 构造卡车和无人机距离矩阵。
#     distance = {'truck': dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight')),
#                 'drone': {i: {j: haversine(graph.nodes[i]['pos'], graph.nodes[j]['pos']) for j in graph.nodes}
#                           for i in graph.nodes}}
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
    # #size = 5
    #     # 运行当前规模的小规模实验。
    #     test_small_instance(100, size)
    # 枚举大规模路网实验中的客户数量。
    #for size in [50, 100, 150]:
    size = 20
    #     # 运行 Manhattan 实验。
    # test_manhattan(1, size)
    #     # 运行 Cambridge 实验。
    # test_cambridge(1, size)
    # 运行论文 1,024 节点 Manhattan 场景实验。
    test_manhattan_1k(1, size)
    # 使用 11,000 节点 NYC 地图运行论文 Boston 规模替代实验。
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


if __name__ == '__main__':

    # 执行论文全量实验。
    run_full_experiments()

