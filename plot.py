"""
本文件负责读取实验结果并生成论文中的可视化图表。

主要内容：
1. 绘制不同消融实验与扩展性实验的统计图。
2. 绘制算法示意图、区域重叠图与地图 HTML。
3. 在缺少某些 `.npy` 实验结果时自动跳过对应图表。
4. 适配当前仓库的离线小规模示例与合成路网。
"""

import folium
import networkx as nx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging
import json
import time
from pathlib import Path
from functools import lru_cache
from html import escape
from matplotlib.patches import Circle
from shapely.geometry import MultiPoint
from src.fstsp import MultiAgentFlyingSidekickTSP
from problem import (
    cambridge,
    manhattan,
    _pairwise_distance,
    multiagent_instance_on_cambridge,
    multiagent_instance_on_manhattan,
)
from config import MANHATTAN1k_GRAPH_PATH, MANHATTAN11k_GRAPH_PATH
from experiment_results import _load_large_road_saved_result
from utils import ensure_dir, haversine, result_path


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
fontsize = 18
parameters = {
    'font.family': 'cmr10',
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'legend.fontsize': fontsize,
    'axes.axisbelow': True
}
plt.rcParams.update(parameters)
colors = sns.color_palette()
# 绘图示例使用的无人机航程上限，只影响本文件生成的 demo 图。
DEMO_DRONE_LIMIT = 0.5
SMALL_FIGURE_DIR = result_path('small', 'figures')
MANHATTAN_DATA_DIR = result_path('manhattan', 'data')
MANHATTAN_FIGURE_DIR = result_path('manhattan', 'figures')
MANHATTAN_MAP_DIR = result_path('manhattan', 'maps')
BOSTON_FIGURE_DIR = result_path('boston', 'figures')
BOSTON_MAP_DIR = result_path('boston', 'maps')
ARTIFACTS = {
    'plot_r': (MANHATTAN_DATA_DIR / 'r-time.npy', MANHATTAN_DATA_DIR / 'r-cost.npy'),
    'plot_speed': (MANHATTAN_DATA_DIR / 'speed-time.npy', MANHATTAN_DATA_DIR / 'speed-cost.npy'),
    'plot_k': (MANHATTAN_DATA_DIR / 'k-cost.npy',),
    'plot_cities': (MANHATTAN_DATA_DIR / 'city-time.npy',),
    'plot_rates': (MANHATTAN_DATA_DIR / 'rates-time.npy',),
    'plot_depots': (MANHATTAN_DATA_DIR / 'depots-time.npy',),
}

# Large road-network solution-map settings. Edit these values directly when you
# want to inspect another result directory, customer size, instance, or fleet
# configuration.
LARGE_ROAD_RESULT_ROOT = result_path()
LARGE_ROAD_OUTPUT_ROOT = result_path()
LARGE_ROAD_CITIES = ('manhattan', 'boston')
LARGE_ROAD_CUSTOMER_COUNT = 100
# 新格式结果默认展示目标函数最接近中位数的代表实例；传入整数仍可指定任意实例。
LARGE_ROAD_INSTANCE_INDEX = None
LARGE_ROAD_NUM_INSTANCES = 100
# 旧版 11k NPZ 没有保存路线；默认禁止绘图阶段隐式重建全对距离并重新求解。
LARGE_ROAD_ALLOW_LEGACY_11K_RESOLVE = False
LARGE_ROAD_LIMIT = 1.5
LARGE_ROAD_SPEED = 1.6
LARGE_ROAD_THETA = (0.5, 0.5)
LARGE_ROAD_DRAW_ROAD_EDGES = False
LARGE_ROAD_MAX_DRAW_EDGES = 10000
LARGE_ROAD_DEPOT_RADIUS = 7
LARGE_ROAD_CUSTOMER_RADIUS = 4
LARGE_ROAD_ROUTE_WEIGHT = 4
LARGE_ROAD_DRONE_WEIGHT = 3
LARGE_ROAD_DRONE_COLOR = '#111827'
LARGE_ROAD_TRUCK_OPACITY = 0.95
LARGE_ROAD_DRONE_OPACITY = 0.85
LARGE_ROAD_COLORS = (
    '#1f77b4', '#d62728', '#9467bd', '#ff7f0e', '#17becf',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#0f4c81',
)
LARGE_ROAD_CITY_CONFIGS = {
    'manhattan': {
        'label': 'Manhattan 1K',
        'result_subdir': 'manhattan',
        'result_map_name': 'manhattan_1k',
        'instance_builder': 'manhattan',
        'graph_loader': 'manhattan',
        'graph_path': MANHATTAN1k_GRAPH_PATH,
        'num_depots': 5,
        'drones_per_truck': 3,
    },
    'boston': {
        'label': 'Boston 11K (NYC substitute)',
        'result_subdir': 'boston',
        'result_map_name': 'boston_11k',
        'instance_builder': 'manhattan',
        'graph_loader': 'manhattan',
        'graph_path': MANHATTAN11k_GRAPH_PATH,
        'num_depots': 10,
        'drones_per_truck': 4,
    },
}
LARGE_ROAD_INSTANCE_BUILDERS = {
    'manhattan': multiagent_instance_on_manhattan,
    'boston': multiagent_instance_on_cambridge,
}


def _savefig(path, fig=None, **kwargs):
    ensure_dir(path.parent)
    if fig is None:
        plt.savefig(path, **kwargs)
    else:
        fig.savefig(path, **kwargs)


def _save_map(map_obj, path):
    ensure_dir(path.parent)
    map_obj.save(str(path))


def _require_files(*paths):
    """
    检查一组文件是否全部存在。

    输入：
    - paths: 任意数量的文件路径字符串。

    输出：
    - 若全部存在则返回 `True`，否则打印缺失项并返回 `False`。

    实现逻辑：
    - 逐个检测路径是否存在，缺失时统一提示。
    """
    missing = [Path(path) for path in paths if not Path(path).is_file()]
    if missing:
        print(f"Skipping because required files are missing: {', '.join(str(path) for path in missing)}")
        return False
    return True


def _graph_bounds(graph, margin_ratio=0.08):
    """
    根据图中节点坐标计算可视化边界框。

    输入：
    - graph: 路网图。
    - margin_ratio: 四周额外留白比例。

    输出：
    - `(x_min, x_max, y_min, y_max)`。

    实现逻辑：
    1. 收集所有节点坐标。
    2. 计算最小/最大值。
    3. 按比例增加边距，避免图形贴边显示。
    """
    xs = [graph.nodes[node]['pos'][0] for node in graph.nodes]
    ys = [graph.nodes[node]['pos'][1] for node in graph.nodes]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_margin = max((x_max - x_min) * margin_ratio, 1e-4)
    y_margin = max((y_max - y_min) * margin_ratio, 1e-4)
    return x_min - x_margin, x_max + x_margin, y_min - y_margin, y_max + y_margin


def _draw_path(map_obj, graph, route, color, weight=5, opacity=1.0):
    """
    在 `folium` 地图上绘制一条按照路网最短路展开的路线。

    输入：
    - map_obj: `folium.Map` 对象。
    - graph: 路网图。
    - route: 关键节点访问序列。
    - color: 线条颜色。
    - weight: 线宽。
    - opacity: 透明度。

    输出：
    - 无显式返回值，直接往地图对象中添加折线。

    实现逻辑：
    1. 遍历相邻关键节点对。
    2. 在原图上展开为最短路。
    3. 将完整节点序列转成经纬度折线绘制到地图上。
    """
    locations = [[graph.nodes[route[0]]['pos'][1], graph.nodes[route[0]]['pos'][0]]]
    for start, end in zip(route[:-1], route[1:]):
        path = nx.dijkstra_path(graph, start, end, weight='weight')
        for node in path[1:]:
            locations.append([graph.nodes[node]['pos'][1], graph.nodes[node]['pos'][0]])
    folium.PolyLine(locations=locations, color=color, weight=weight, opacity=opacity).add_to(map_obj)


def _draw_path_on_axis(ax, graph, route, color, linewidth=2.6, alpha=1.0):
    if len(route) < 2:
        return
    for start, end in zip(route[:-1], route[1:]):
        path = nx.dijkstra_path(graph, start, end, weight='weight')
        xs = [graph.nodes[node]['pos'][0] for node in path]
        ys = [graph.nodes[node]['pos'][1] for node in path]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, solid_capstyle='round')


def _draw_demo_base(ax, graph):
    ax.set_facecolor('#eef2f2')
    for edge in graph.edges:
        start, end = edge[:2]
        ax.plot([graph.nodes[start]['pos'][0], graph.nodes[end]['pos'][0]],
                [graph.nodes[start]['pos'][1], graph.nodes[end]['pos'][1]],
                color='black', linewidth=0.35, alpha=0.68)
    x_min, x_max, y_min, y_max = _graph_bounds(graph, margin_ratio=0.015)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _nearest_unique_nodes(graph, locations):
    selected = []
    used = set()
    for location in locations:
        candidates = sorted(graph.nodes, key=lambda node: haversine(location, graph.nodes[node]['pos']))
        node = next(candidate for candidate in candidates if candidate not in used)
        selected.append(node)
        used.add(node)
    return np.array(selected)


@lru_cache(maxsize=1)
def _manhattan_demo_instance():
    graph = manhattan()
    bbox = (-73.990, -73.955, 40.760, 40.830)
    nodes = [node for node in graph.nodes
             if bbox[0] <= graph.nodes[node]['pos'][0] <= bbox[1]
             and bbox[2] <= graph.nodes[node]['pos'][1] <= bbox[3]]
    graph = nx.MultiDiGraph(graph.subgraph(nodes).copy())
    component = max(nx.strongly_connected_components(graph), key=len)
    graph = nx.MultiDiGraph(graph.subgraph(component).copy())

    depot_locations = [(-73.984, 40.772), (-73.962, 40.772)]
    customer_locations = [
        (-73.987, 40.764), (-73.982, 40.778), (-73.975, 40.794),
        (-73.967, 40.812), (-73.958, 40.826),
        (-73.960, 40.764), (-73.960, 40.779), (-73.958, 40.795),
        (-73.957, 40.812), (-73.956, 40.826),
    ]
    locations = depot_locations + customer_locations
    selected = _nearest_unique_nodes(graph, locations)
    depots = selected[:len(depot_locations)]
    cities = selected[len(depot_locations):]
    distance = _pairwise_distance(graph)
    return graph, depots, cities, distance


@lru_cache(maxsize=1)
def _boston_demo_instance():
    graph = cambridge()
    depot_locations = [(-71.064, 42.355), (-71.052, 42.365)]
    customer_locations = [
        (-71.066, 42.351), (-71.063, 42.358), (-71.060, 42.364),
        (-71.055, 42.369), (-71.049, 42.372),
        (-71.049, 42.352), (-71.047, 42.358), (-71.051, 42.361),
        (-71.057, 42.366), (-71.063, 42.370),
    ]
    locations = depot_locations + customer_locations
    selected = _nearest_unique_nodes(graph, locations)
    depots = selected[:len(depot_locations)]
    cities = selected[len(depot_locations):]
    distance = _pairwise_distance(graph)
    return graph, depots, cities, distance


def _demo_groups(depots, cities):
    half = len(cities) // 2
    return {depots[0]: list(cities[:half]), depots[1]: list(cities[half:])}


def _safe_polygon(points, alpha=0.1):
    """
    稳健地从点集构造一个多边形区域。

    输入：
    - points: 二维点集。
    - alpha: `alphashape` 参数。

    输出：
    - 一个拥有 `exterior` 的多边形对象；若退化则回退到凸包或微小缓冲区。

    实现逻辑：
    1. 若点集太少或共线，则直接用凸包。
    2. 否则调用 `alphashape`。
    3. 若返回结果不是普通多边形，则继续回退到凸包。
    """
    import alphashape

    points = np.asarray(points, dtype=float)
    if len(points) < 4 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        hull = MultiPoint(points).convex_hull
        if hasattr(hull, 'exterior'):
            return hull
        return hull.buffer(1e-4)

    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.ERROR)
    try:
        shape = alphashape.alphashape(points, alpha)
    finally:
        root_logger.setLevel(previous_level)
    if hasattr(shape, 'exterior'):
        return shape
    hull = MultiPoint(points).convex_hull
    if hasattr(hull, 'exterior'):
        return hull
    return hull.buffer(1e-4)


def plot_graph(graph):
    """
    使用 `matplotlib` 绘制路网的节点与边。

    输入：
    - graph: 路网图。

    输出：
    - 无显式返回值，直接在当前画布上作图。

    实现逻辑：
    - 先画所有节点，再逐条画边。
    """
    plt.scatter([graph.nodes[node]['pos'][0] for node in graph.nodes],
                [graph.nodes[node]['pos'][1] for node in graph.nodes], s=2)
    for edge in graph.edges:
        plt.plot([graph.nodes[edge[0]]['pos'][0], graph.nodes[edge[1]]['pos'][0]],
                 [graph.nodes[edge[0]]['pos'][1], graph.nodes[edge[1]]['pos'][1]], color='black', linewidth=1)


def plot_truck_solution(graph, solution):
    """
    在当前画布上绘制卡车路线。

    输入：
    - graph: 路网图。
    - solution: 卡车的关键节点访问序列。

    输出：
    - 无显式返回值，直接在当前画布上作图。

    实现逻辑：
    - 将关键节点间的连线展开为原图中的最短路并逐段绘制。
    """
    for start, end in zip(solution[:-1], solution[1:]):
        path = nx.shortest_path(G=graph, source=start, target=end, weight='weight')
        for edge_start, edge_end in zip(path[:-1], path[1:]):
            plt.plot([graph.nodes[edge_start]['pos'][0], graph.nodes[edge_end]['pos'][0]],
                     [graph.nodes[edge_start]['pos'][1], graph.nodes[edge_end]['pos'][1]], color='red')


def plot_multiagent_solution(graph, solution, depots, cities):
    """
    绘制一个完整的多仓库卡车-无人机联合解。

    输入：
    - graph: 路网图。
    - solution: 统一格式的多条路线。
    - depots: 仓库集合。
    - cities: 客户集合。

    输出：
    - 无显式返回值，直接在当前画布上绘制并关闭图像。

    实现逻辑：
    1. 先画底图。
    2. 再画仓库和客户节点。
    3. 分别叠加卡车和无人机路线。
    """
    plot_graph(graph)
    plt.scatter([graph.nodes[node]['pos'][0] for node in depots],
                [graph.nodes[node]['pos'][1] for node in depots], s=20, c='red', marker='o')
    plt.scatter([graph.nodes[node]['pos'][0] for node in cities],
                [graph.nodes[node]['pos'][1] for node in cities], s=20, c='blue', marker='o')
    for node in np.concatenate((depots, cities)):
        plt.text(graph.nodes[node]['pos'][0], graph.nodes[node]['pos'][1], node)
    for route in solution:
        plot_truck_solution(graph, route['truck'])
        for drone_route in route['drone']:
            for route_d in drone_route:
                for start, end in zip(route_d[:-1], route_d[1:]):
                    plt.plot([graph.nodes[start]['pos'][0], graph.nodes[end]['pos'][0]],
                             [graph.nodes[start]['pos'][1], graph.nodes[end]['pos'][1]], linestyle='-', color='green')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.close()


def plot_r():
    """
    读取 `r-time.npy` 与 `r-cost.npy`，绘制飞行距离上限消融图。

    输入：
    - 无，直接读取当前目录中的实验结果文件。

    输出：
    - 保存 `r.pdf`。

    实现逻辑：
    - 用箱线图显示成本分布，用折线图显示平均耗时。
    """
    if not _require_files(*ARTIFACTS['plot_r']):
        return
    times = np.load(ARTIFACTS['plot_r'][0])
    costs = np.load(ARTIFACTS['plot_r'][1])
    fig, ax1 = plt.subplots()
    box = ax1.boxplot(costs.T, patch_artist=True, boxprops=dict(facecolor='C0'), showfliers=False)
    for patch in box['boxes']:
        patch.set_facecolor(colors[0])
    ax1.set_xlabel('Distance Limit')
    ax1.set_ylabel('Cost (Boxes)')

    times = times
    ax2 = ax1.twinx()
    ax2.plot(range(1, 7), np.mean(times, axis=1), marker='d', markersize=10)
    ax2.set_ylabel('Time(s) (Line)')
    ax2.set_yscale('log')
    plt.xticks(ticks=range(1, 7), labels=[f'{i / 10:.1f}' for i in range(5, 16, 2)])
    plt.tight_layout()
    _savefig(MANHATTAN_FIGURE_DIR / 'r.pdf')
    plt.close(fig)


def plot_speed():
    """
    读取速度消融结果，并生成速度-成本/时间图。

    输入：
    - 无。

    输出：
    - 保存 `speed.pdf`。

    实现逻辑：
    - 箱线图展示成本分布，折线图展示不同速度下的运行时间。
    """
    if not _require_files(*ARTIFACTS['plot_speed']):
        return
    times = np.load(ARTIFACTS['plot_speed'][0])
    costs = np.load(ARTIFACTS['plot_speed'][1])
    fig, ax1 = plt.subplots()
    ax1.boxplot(costs.T, patch_artist=True, boxprops=dict(facecolor='C0'), showfliers=True)
    ax1.set_xlabel('Ratio of Speed')
    ax1.set_ylabel('Cost (Boxes)')

    times = times
    ax2 = ax1.twinx()
    ax2.plot(range(1, 7), times / 25, marker='d', color=colors[0], markersize=10)
    ax2.set_ylabel('Time(s) (Line)')
    plt.xticks(ticks=range(1, 7), labels=[f'{i / 30:.2f}' for i in range(10, 120, 20)])
    plt.tight_layout()
    _savefig(MANHATTAN_FIGURE_DIR / 'speed.pdf')
    plt.close(fig)


def plot_k():
    """
    读取不同无人机数量下的成本结果，并打印平均值。

    输入：
    - 无。

    输出：
    - 无图像文件，直接打印不同规模下的平均成本。

    实现逻辑：
    - 逐个规模读取成本矩阵并在终端输出均值。
    """
    if not _require_files(*ARTIFACTS['plot_k']):
        return
    costs = np.load(ARTIFACTS['plot_k'][0])
    for i in range(5):
        size = 50 + 20 * i
        cost = costs[i]
        print(f'at size {size}, the average cost is {np.mean(cost, axis=0)}')


def plot_cities():
    """
    绘制固定仓库数时，客户数量增长对应的耗时箱线图。

    输入：
    - 无。

    输出：
    - 保存 `city.pdf`。
    """
    if not _require_files(*ARTIFACTS['plot_cities']):
        return
    times = np.load(ARTIFACTS['plot_cities'][0])[2:]
    plt.grid()
    plt.boxplot(times.T, patch_artist=True, boxprops=dict(facecolor='C0'), showfliers=True)
    plt.xticks(ticks=range(1, 7), labels=[120 + 40 * i for i in range(6)])
    plt.text(0.05, 0.9, "$|\mathcal{P}|=10$", fontsize=22, transform=plt.gca().transAxes, verticalalignment='top')
    plt.ylabel('Time(s)')
    plt.xlabel('Customers')
    plt.tight_layout()
    _savefig(MANHATTAN_FIGURE_DIR / 'city.pdf')
    plt.close()


def plot_rates():
    """
    绘制固定客户仓库比例下的可扩展性耗时箱线图。

    输入：
    - 无。

    输出：
    - 保存 `rates.pdf`。
    """
    if not _require_files(*ARTIFACTS['plot_rates']):
        return
    times = np.load(ARTIFACTS['plot_rates'][0])
    # times[times > 300] = 300
    # times = times[4:]
    plt.grid()
    plt.boxplot(times.T, patch_artist=True, boxprops=dict(facecolor='C0'), showfliers=True)
    plt.xticks(ticks=range(1, 7), labels=[20 * i for i in range(3, 21, 3)])
    plt.text(0.05, 0.9, "$|\mathcal{C}|/|\mathcal{P}|=20$", fontsize=22, transform=plt.gca().transAxes, verticalalignment='top')
    plt.ylabel('Time(s)')
    plt.xlabel('Customers')
    plt.tight_layout()
    _savefig(MANHATTAN_FIGURE_DIR / 'rates.pdf')
    plt.close()


def plot_depots():
    """
    绘制固定客户数时，仓库数量增长对应的耗时箱线图。

    输入：
    - 无。

    输出：
    - 保存 `depots.pdf`。
    """
    if not _require_files(*ARTIFACTS['plot_depots']):
        return
    times = np.load(ARTIFACTS['plot_depots'][0])
    plt.grid()
    plt.boxplot(times.T, patch_artist=True, boxprops=dict(facecolor='C0'), showfliers=True)
    plt.xticks(ticks=range(1, 7), labels=[5 + 2 * i for i in range(6)])
    plt.text(0.75, 0.9, "$|\mathcal{C}|=150$", fontsize=22, transform=plt.gca().transAxes, verticalalignment='top')
    plt.ylabel('Time(s)')
    plt.xlabel('Depots')
    plt.tight_layout()
    _savefig(MANHATTAN_FIGURE_DIR / 'depots.pdf')
    plt.close()


def plot_accelerate():
    """
    绘制论文中用于说明区域重叠与边界构造的三张示意图。

    输入：
    - 无，函数内部自动生成一个小规模子图并选取两个客户点。

    输出：
    - 保存 `overlap.pdf`、`overlap-2.pdf`、`overlap-3.pdf`。

    实现逻辑：
    1. 画两个客户邻域的重叠区域。
    2. 用 `alphashape` 或凸包近似客户归属区域。
    3. 额外标出边界点和边界包络。
    """
    from problem import small_instance
    from matplotlib.patches import Circle
    from utils import euclidean
    graph, depots, sampled_cities, distance = small_instance(1, 50, 1, 2)
    cities = sampled_cities[0].tolist()
    plot_graph(graph)
    plt.scatter([graph.nodes[cities[0]]['pos'][0]], [graph.nodes[cities[0]]['pos'][1]],
                s=200, color=colors[0], marker='o')
    plt.scatter([graph.nodes[cities[1]]['pos'][0]], [graph.nodes[cities[1]]['pos'][1]],
                s=200, color=colors[1], marker='o')

    blue_region = Circle((graph.nodes[cities[0]]['pos'][0], graph.nodes[cities[0]]['pos'][1]),
                         0.007, color=colors[0], alpha=0.4)
    red_region = Circle((graph.nodes[cities[1]]['pos'][0], graph.nodes[cities[1]]['pos'][1]),
                        0.007, color=colors[1], alpha=0.4)
    ax = plt.gca()
    ax.add_patch(blue_region)
    ax.add_patch(red_region)
    for node in graph.nodes:
        if euclidean(graph.nodes[node]['pos'], graph.nodes[cities[0]]['pos']) < 0.007:
            if euclidean(graph.nodes[node]['pos'], graph.nodes[cities[1]]['pos']) < 0.007:
                plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]], s=50, color=colors[4], marker='o')
            else:
                plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]], s=50, color=colors[0], marker='o')
        else:
            plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]], s=50, color=colors[1], marker='o')
    # x_min = min([graph.nodes[node]['pos'][0] for node in graph.nodes])
    # x_max = max([graph.nodes[node]['pos'][0] for node in graph.nodes])
    # y_min = min([graph.nodes[node]['pos'][1] for node in graph.nodes])
    # y_max = max([graph.nodes[node]['pos'][1] for node in graph.nodes])
    x_min, x_max, y_min, y_max = _graph_bounds(graph)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    _savefig(SMALL_FIGURE_DIR / 'overlap.pdf')
    plt.close()

    plot_graph(graph)
    plt.scatter([graph.nodes[cities[0]]['pos'][0]], [graph.nodes[cities[0]]['pos'][1]],
                s=200, color=colors[0], marker='o')
    plt.scatter([graph.nodes[cities[1]]['pos'][0]], [graph.nodes[cities[1]]['pos'][1]],
                s=200, color=colors[1], marker='o')
    blue_vertices = []
    red_vertices = []
    for node in graph.nodes:
        d_1 = euclidean(graph.nodes[node]['pos'], graph.nodes[cities[0]]['pos'])
        d_2 = euclidean(graph.nodes[node]['pos'], graph.nodes[cities[1]]['pos'])
        if d_1 < d_2:
            blue_vertices.append(graph.nodes[node]['pos'])
        else:
            red_vertices.append(graph.nodes[node]['pos'])
        if d_1 < 0.007 and d_2 < 0.007:
            plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]],
                        s=50, color=colors[0] if d_1 < d_2 else colors[1], marker='o')
        elif d_1 < 0.007:
            plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]],
                        s=50, color=colors[0], marker='o')
        elif d_2 < 0.007:
            plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]],
                        s=50, color=colors[1], marker='o')
    blue_vertices = _safe_polygon(blue_vertices, 0.1)
    red_vertices = _safe_polygon(red_vertices, 0.1)
    ax = plt.gca()
    x, y = blue_vertices.exterior.xy
    ax.fill(x, y, facecolor=colors[0], edgecolor=colors[0], alpha=0.4)
    x, y = red_vertices.exterior.xy
    ax.fill(x, y, facecolor=colors[1], edgecolor=colors[1], alpha=0.4)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    _savefig(SMALL_FIGURE_DIR / 'overlap-2.pdf')
    plt.close()

    plot_graph(graph)
    plt.scatter([graph.nodes[cities[0]]['pos'][0]], [graph.nodes[cities[0]]['pos'][1]],
                s=200, color=colors[0], marker='o')
    plt.scatter([graph.nodes[cities[1]]['pos'][0]], [graph.nodes[cities[1]]['pos'][1]],
                s=200, color=colors[1], marker='o')
    blue_vertices, red_vertices = [], []
    for node in graph.nodes:
        d_1 = euclidean(graph.nodes[node]['pos'], graph.nodes[cities[0]]['pos'])
        d_2 = euclidean(graph.nodes[node]['pos'], graph.nodes[cities[1]]['pos'])
        if d_1 < d_2:
            blue_vertices.append(node)
        else:
            red_vertices.append(node)
    for node in blue_vertices:
        for neighbor in graph.neighbors(node):
            if neighbor in red_vertices:
                plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]], s=50, color=colors[0], marker='o')
                break
    for node in red_vertices:
        for neighbor in graph.neighbors(node):
            if neighbor in blue_vertices:
                plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]], s=50, color=colors[1], marker='o')
                break
    blue_vertices = [graph.nodes[node]['pos'] for node in blue_vertices]
    red_vertices = [graph.nodes[node]['pos'] for node in red_vertices]
    blue_boundary = _safe_polygon(blue_vertices, 0.1)
    red_boundary = _safe_polygon(red_vertices, 0.1)
    plt.scatter(np.array(blue_boundary.exterior.coords)[:, 0], np.array(blue_boundary.exterior.coords)[:, 1],
                s=50, color=colors[0])
    plt.scatter(np.array(red_boundary.exterior.coords)[:, 0], np.array(red_boundary.exterior.coords)[:, 1],
                s=50, color=colors[1])
    ax = plt.gca()
    x, y = blue_boundary.exterior.xy
    ax.fill(x, y, facecolor=colors[0], edgecolor=colors[0], alpha=0.4)
    x, y = red_boundary.exterior.xy
    ax.fill(x, y, facecolor=colors[1], edgecolor=colors[1], alpha=0.4)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    _savefig(SMALL_FIGURE_DIR / 'overlap-3.pdf')
    plt.close()


def _plot_phase_demo(instance, output_dir, basename):
    graph, depots, cities, distance = instance
    model = MultiAgentFlyingSidekickTSP(graph, depots, cities, distance, 2, limit=DEMO_DRONE_LIMIT, theta=(0.5, 0))
    groups = _demo_groups(depots, cities)
    model.groups = groups
    color = {depots[0]: 'blue', depots[1]: 'red'}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.4))
    fig.subplots_adjust(left=0.02, right=0.995, top=0.98, bottom=0.18, wspace=0.02)
    for ax in axes:
        _draw_demo_base(ax, graph)

    # Phase 1: customer assignment.
    for depot in depots:
        axes[0].scatter([graph.nodes[depot]['pos'][0]], [graph.nodes[depot]['pos'][1]],
                        s=40, color=color[depot], zorder=4)
        for city in groups[depot]:
            axes[0].scatter([graph.nodes[city]['pos'][0]], [graph.nodes[city]['pos'][1]],
                            s=28, facecolors='none', edgecolors=color[depot], linewidths=1.3, zorder=4)

    # Phase 2: visiting order with candidate drone-service neighborhoods.
    phase2_routes = {}
    for depot in depots:
        axes[1].scatter([graph.nodes[depot]['pos'][0]], [graph.nodes[depot]['pos'][1]],
                        s=40, color=color[depot], zorder=5)
        for city in groups[depot]:
            center = graph.nodes[city]['pos']
            axes[1].add_patch(Circle(center, 0.0055, facecolor=color[depot],
                                     edgecolor=color[depot], linewidth=0.8, alpha=0.18, zorder=2))
        seq = model.get_seq(depot, [[depot]] + [[city] for city in groups[depot]])
        route = [depot] + [groups[depot][i - 1] for i in seq[1:-1]] + [depot]
        phase2_routes[depot] = route
        _draw_path_on_axis(axes[1], graph, route, color[depot], linewidth=2.8)

    # Phase 3: truck routes plus drone sorties.
    for depot in depots:
        route = phase2_routes[depot]
        solution, _ = model.local_search_multi_drone_appr(route, depot)
        _draw_path_on_axis(axes[2], graph, solution['truck'], color[depot], linewidth=2.4)
        axes[2].scatter([graph.nodes[depot]['pos'][0]], [graph.nodes[depot]['pos'][1]],
                        s=40, color=color[depot], zorder=5)
        for drone_route in solution['drone']:
            xs = [graph.nodes[node]['pos'][0] for node in drone_route]
            ys = [graph.nodes[node]['pos'][1] for node in drone_route]
            axes[2].plot(xs, ys, color='green', linewidth=2.0, alpha=0.95, zorder=5)
        for city in groups[depot]:
            axes[2].scatter([graph.nodes[city]['pos'][0]], [graph.nodes[city]['pos'][1]],
                            s=18, facecolors='white', edgecolors=color[depot], linewidths=1.0, zorder=6)

    captions = [
        '(a) Phase 1: assign customers to truck groups',
        '(b) Phase 2: solve the Set-TSP to get visiting\norders of customers',
        '(c) Phase 3: optimize routes for trucks and\ndrones simultaneously based on visiting orders',
    ]
    for ax, caption in zip(axes, captions):
        ax.text(0.5, -0.035, caption, transform=ax.transAxes, ha='center', va='top', fontsize=12)

    _savefig(output_dir / f'{basename}.pdf', fig=fig)
    _savefig(output_dir / f'{basename}.png', fig=fig, dpi=220)
    plt.close(fig)


def _plot_solution_maps(instance, output_dir):
    graph, depots, cities, distance = instance
    model = MultiAgentFlyingSidekickTSP(graph, depots, cities, distance, 2, limit=DEMO_DRONE_LIMIT)
    center = [np.mean([graph.nodes[node]['pos'][1] for node in graph.nodes]),
              np.mean([graph.nodes[node]['pos'][0] for node in graph.nodes])]

    # plot the map
    m = folium.Map(location=center, zoom_start=14, tiles='Cartodb Positron')
    for edge in graph.edges:
        lon0, lat0 = graph.nodes[edge[0]]['pos'][0], graph.nodes[edge[0]]['pos'][1]
        lon1, lat1 = graph.nodes[edge[1]]['pos'][0], graph.nodes[edge[1]]['pos'][1]
        folium.PolyLine(locations=[[lat0, lon0], [lat1, lon1]], color='black', weight=1).add_to(m)

    # draw the partition
    model.groups = _demo_groups(depots, cities)
    groups = model.groups
    color = {depots[0]: 'blue', depots[1]: 'red'}
    for depot in groups.keys():
        folium.Circle(location=[graph.nodes[depot]['pos'][1], graph.nodes[depot]['pos'][0]], color=color[depot],
                      fill_color=color[depot], fill_opacity=1, radius=40).add_to(m)
        for city in groups[depot]:
            folium.Circle(location=[graph.nodes[city]['pos'][1], graph.nodes[city]['pos'][0]],
                          color=color[depot], weight=2, radius=40, fill=False).add_to(m)
    _save_map(m, output_dir / 'map.html')

    n = folium.Map(location=center, zoom_start=14, tiles='Cartodb Positron')
    o = folium.Map(location=center, zoom_start=14, tiles='Cartodb Positron')
    for edge in graph.edges:
        lon0, lat0 = graph.nodes[edge[0]]['pos'][0], graph.nodes[edge[0]]['pos'][1]
        lon1, lat1 = graph.nodes[edge[1]]['pos'][0], graph.nodes[edge[1]]['pos'][1]
        folium.PolyLine(locations=[[lat0, lon0], [lat1, lon1]], color='black', weight=1, opacity=0.5).add_to(n)
        folium.PolyLine(locations=[[lat0, lon0], [lat1, lon1]], color='black', weight=1, opacity=0.5).add_to(o)

    for depot in groups.keys():
        folium.Circle(location=[graph.nodes[depot]['pos'][1], graph.nodes[depot]['pos'][0]], color=color[depot],
                      fill_color=color[depot], fill_opacity=1, radius=40).add_to(n)
        folium.Circle(location=[graph.nodes[depot]['pos'][1], graph.nodes[depot]['pos'][0]], color=color[depot],
                      fill_color=color[depot], fill_opacity=1, radius=40).add_to(o)
        for city in groups[depot]:
            folium.Circle(location=[graph.nodes[city]['pos'][1], graph.nodes[city]['pos'][0]],
                          color=color[depot], weight=2, radius=40, fill=False).add_to(n)

    # draw the set TSP
    convex_sets = model.get_boundary_convex_sets(DEMO_DRONE_LIMIT)
    for depot in model.depots:
        for city in groups[depot]:
            folium.Circle(location=[graph.nodes[city]['pos'][1], graph.nodes[city]['pos'][0]],
                          radius=400, color=color[depot], weight=0.5, fill_color=color[depot], fill_opacity=0.2).add_to(
                o)
        local_convex_sets = [[depot]] + [convex_sets[city] for city in groups[depot]]
        solution = model.single_solution(depot, local_convex_sets)
        seq = model.get_seq(depot, local_convex_sets)
        route = [depot] + [groups[depot][i - 1] for i in seq[1:-1]] + [depot]
        _draw_path(o, graph, route, color[depot])
        _draw_path(n, graph, solution['truck'], color[depot])
        for drone_route in solution['drone']:
            folium.PolyLine(
                locations=[[graph.nodes[node]['pos'][1], graph.nodes[node]['pos'][0]] for node in drone_route],
                color='green', weight=5).add_to(n)
    _save_map(o, output_dir / 'tsp.html')
    _save_map(n, output_dir / 'solution.html')


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _save_json(path, payload):
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(_jsonable(payload), f, indent=2)


def _node_text(nodes, limit=24):
    nodes = list(nodes)
    shown = ', '.join(str(_jsonable(node)) for node in nodes[:limit])
    if len(nodes) > limit:
        shown += f', ... (+{len(nodes) - limit})'
    return shown


def _html(value):
    return escape(str(value), quote=True)


def _edge_weight(graph, start, end):
    data = graph.get_edge_data(start, end)
    if data is None:
        return 0.0
    if 'weight' in data:
        return float(data.get('weight', 0.0))
    weights = []
    for item in data.values():
        if isinstance(item, dict) and 'weight' in item:
            weights.append(float(item['weight']))
    return min(weights) if weights else 0.0


def _truck_route_metrics(graph, route):
    route = list(route)
    if len(route) == 0:
        return {
            'key_route': [],
            'expanded_nodes': [],
            'locations': [],
            'segments': [],
            'distance': 0.0,
        }

    locations = [_folium_node_location(graph, route[0])]
    expanded_nodes = [route[0]]
    segments = []
    total_distance = 0.0
    for segment_index, (start, end) in enumerate(zip(route[:-1], route[1:])):
        if start == end:
            path = [start]
            segment_distance = 0.0
        else:
            path = nx.dijkstra_path(graph, start, end, weight='weight')
            segment_distance = sum(_edge_weight(graph, u, v) for u, v in zip(path[:-1], path[1:]))
        total_distance += segment_distance
        for node in path[1:]:
            locations.append(_folium_node_location(graph, node))
            expanded_nodes.append(node)
        segments.append({
            'index': segment_index,
            'from': start,
            'to': end,
            'distance': segment_distance,
            'path_node_count': len(path),
            'path_nodes': path,
        })

    return {
        'key_route': route,
        'expanded_nodes': expanded_nodes,
        'locations': locations,
        'segments': segments,
        'distance': total_distance,
    }


def _drone_sortie_metrics(graph, sortie, limit, speed):
    sortie = list(sortie)
    distance = sum(
        haversine(graph.nodes[start]['pos'], graph.nodes[end]['pos'])
        for start, end in zip(sortie[:-1], sortie[1:])
    )
    customer = sortie[1] if len(sortie) >= 3 else None
    return {
        'nodes': sortie,
        'launch': sortie[0] if sortie else None,
        'customer': customer,
        'recovery': sortie[-1] if sortie else None,
        'distance': distance,
        'flight_time': distance / speed if speed else None,
        'within_limit': distance <= limit,
    }


def _large_road_route_stats(graph, solution, depot_records, limit, speed):
    stats = []
    for depot_index, route in enumerate(solution):
        record = depot_records[depot_index] if depot_index < len(depot_records) else {}
        truck_metrics = _truck_route_metrics(graph, route.get('truck', []))
        drone_metrics = [
            _drone_sortie_metrics(graph, sortie, limit, speed)
            for sortie in _iter_drone_sorties(route)
        ]
        stats.append({
            'depot_index': depot_index,
            'depot_node': record.get('depot_node'),
            'customer_count': record.get('customer_count', 0),
            'truck': {
                'key_route': truck_metrics['key_route'],
                'distance': truck_metrics['distance'],
                'stop_count': len(truck_metrics['key_route']),
                'expanded_node_count': len(truck_metrics['expanded_nodes']),
                'segments': truck_metrics['segments'],
            },
            'drone': {
                'sortie_count': len(drone_metrics),
                'total_distance': sum(item['distance'] for item in drone_metrics),
                'sorties': drone_metrics,
            },
        })
    return stats


def _solve_large_road_with_telemetry(model):
    total_start = time.perf_counter()
    telemetry = {
        'timings': {},
        'depot_records': [],
    }
    model.solution = []
    model.cost = 0

    start = time.perf_counter()
    convex_sets = model.get_boundary_convex_sets(model.theta[0])
    telemetry['timings']['boundary_convex_sets_seconds'] = time.perf_counter() - start

    start = time.perf_counter()
    model.set_mst(convex_sets)
    telemetry['timings']['mst_partition_seconds'] = time.perf_counter() - start

    for depot_index, depot in enumerate(model.depots):
        group = list(model.groups[depot])
        convex_set = [[depot]] + [convex_sets[city] for city in group]
        record = {
            'depot_index': depot_index,
            'depot_node': depot,
            'customer_count': len(group),
            'customers': group,
            'convex_set_sizes': [len(nodes) for nodes in convex_set],
            'set_tsp_solver': 'none',
            'set_tsp_sequence': [],
            'visit_route': [depot, depot],
            'objective_contribution': 0.0,
            'timings': {
                'set_tsp_seconds': 0.0,
                'local_search_seconds': 0.0,
            },
        }

        if len(group) == 0:
            raw_solution = {'truck': [depot, depot], 'drone': []}
            converted_solution = model.convert(raw_solution)
        else:
            record['set_tsp_solver'] = 'LKH' if model.theta[1] == 0 else 'Set-TSP'
            start = time.perf_counter()
            seq = model.get_seq(depot, convex_set)
            record['timings']['set_tsp_seconds'] = time.perf_counter() - start
            visit_route = [depot] + [group[i - 1] for i in seq[1:-1]] + [depot]
            record['set_tsp_sequence'] = seq
            record['visit_route'] = visit_route

            start = time.perf_counter()
            raw_solution, cost = model.local_search_multi_drone_appr(visit_route, depot)
            record['timings']['local_search_seconds'] = time.perf_counter() - start
            record['objective_contribution'] = cost
            model.cost += cost
            converted_solution = model.convert(raw_solution)

        model.solution.append(converted_solution)
        telemetry['depot_records'].append(record)

    telemetry['timings']['solve_seconds'] = time.perf_counter() - total_start
    telemetry['objective_value'] = model.cost
    return model.solution, model.cost, telemetry


def _build_large_road_summary(
    graph,
    depots,
    cities,
    config,
    result_file,
    output_path,
    summary_path,
    city_label,
    customer_count,
    instance_index,
    num_instances,
    saved_result,
    solved_cost,
    telemetry,
    route_stats,
):
    groups = [
        {
            'depot_index': record['depot_index'],
            'depot_node': record['depot_node'],
            'customer_count': record['customer_count'],
            'customers': record['customers'],
        }
        for record in telemetry['depot_records']
    ]
    return {
        'city': city_label,
        'input': {
            'customer_count': customer_count,
            'instance_index': instance_index,
            'num_instances': num_instances,
            'num_depots': config['num_depots'],
            'drones_per_truck': config.get('drones_per_truck'),
            'limit': config.get('limit', LARGE_ROAD_LIMIT),
            'speed': config.get('speed', LARGE_ROAD_SPEED),
            'theta': config.get('theta', LARGE_ROAD_THETA),
            'depots': list(depots),
            'customers': list(cities),
        },
        'files': {
            'result_file': result_file,
            'output_html': output_path,
            'summary_json': summary_path,
        },
        'graph': {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
        },
        'results': {
            'objective_value': solved_cost,
            'saved_cost': saved_result.get('cost') if saved_result else None,
            'saved_time': saved_result.get('time') if saved_result else None,
            'solve_seconds': telemetry['timings']['solve_seconds'],
        },
        'groups': groups,
        'set_tsp': telemetry['depot_records'],
        'route_stats': route_stats,
        'timings': telemetry['timings'],
    }


def _summary_number(value, digits=3):
    if value is None:
        return 'n/a'
    return f'{float(value):.{digits}f}'


def _render_large_road_sidebar(summary):
    input_data = summary['input']
    results = summary['results']
    graph = summary['graph']
    files = summary['files']
    route_stats = summary['route_stats']
    group_rows = []
    for group in summary['groups']:
        group_rows.append(
            '<details class="ma-details">'
            f"<summary>Depot {group['depot_index']} node {group['depot_node']} "
            f"({group['customer_count']} customers)</summary>"
            f"<div class=\"ma-list\">{_html(_node_text(group['customers'], limit=60))}</div>"
            '</details>'
        )

    set_tsp_rows = []
    for record in summary['set_tsp']:
        set_tsp_rows.append(
            '<details class="ma-details">'
            f"<summary>Depot {record['depot_index']} - {record['set_tsp_solver']} "
            f"({_summary_number(record['timings']['set_tsp_seconds'])}s)</summary>"
            f"<div>Customers: {record['customer_count']}</div>"
            f"<div>Objective part: {_summary_number(record['objective_contribution'], 6)}</div>"
            f"<div>Sequence: {_html(_node_text(record['set_tsp_sequence'], limit=80))}</div>"
            f"<div>Visit route: {_html(_node_text(record['visit_route'], limit=80))}</div>"
            '</details>'
        )

    route_rows = []
    for stat in route_stats:
        route_rows.append(
            '<details class="ma-details">'
            f"<summary>Depot {stat['depot_index']} routes</summary>"
            f"<div>Truck distance: {_summary_number(stat['truck']['distance'], 6)}</div>"
            f"<div>Truck stops: {stat['truck']['stop_count']}</div>"
            f"<div>Drone sorties: {stat['drone']['sortie_count']}</div>"
            f"<div>Drone total distance: {_summary_number(stat['drone']['total_distance'], 6)}</div>"
            '</details>'
        )

    legend = (
        '<div class="ma-legend-row"><span class="ma-line ma-truck"></span> Truck route: solid, depot color</div>'
        '<div class="ma-legend-row"><span class="ma-line ma-drone"></span> Drone sortie: dark dashed line</div>'
        '<div class="ma-legend-row"><span class="ma-dot ma-depot"></span> Depot</div>'
        '<div class="ma-legend-row"><span class="ma-dot ma-customer"></span> Customer</div>'
    )

    return f"""
    <style>
      .ma-sidebar {{
        position: fixed;
        top: 12px;
        right: 12px;
        bottom: 12px;
        width: 380px;
        z-index: 9999;
        overflow: auto;
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #cbd5e1;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
        border-radius: 6px;
        padding: 12px 14px;
        color: #111827;
        font-family: Arial, sans-serif;
        font-size: 12px;
        line-height: 1.42;
      }}
      .ma-sidebar h2 {{
        margin: 0 0 8px 0;
        font-size: 16px;
      }}
      .ma-sidebar h3 {{
        margin: 12px 0 6px 0;
        font-size: 13px;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 3px;
      }}
      .ma-kv {{
        display: grid;
        grid-template-columns: 132px 1fr;
        gap: 3px 8px;
      }}
      .ma-key {{
        color: #475569;
      }}
      .ma-value {{
        overflow-wrap: anywhere;
      }}
      .ma-details {{
        margin: 4px 0;
        padding: 4px 0;
        border-bottom: 1px solid #f1f5f9;
      }}
      .ma-details summary {{
        cursor: pointer;
        font-weight: 600;
      }}
      .ma-list {{
        margin-top: 3px;
        color: #334155;
        overflow-wrap: anywhere;
      }}
      .ma-line {{
        display: inline-block;
        width: 30px;
        height: 0;
        margin-right: 6px;
        vertical-align: middle;
      }}
      .ma-truck {{
        border-top: 4px solid #1f77b4;
      }}
      .ma-drone {{
        border-top: 3px dashed {LARGE_ROAD_DRONE_COLOR};
      }}
      .ma-dot {{
        display: inline-block;
        width: 10px;
        height: 10px;
        margin-right: 8px;
        border-radius: 50%;
        vertical-align: middle;
      }}
      .ma-depot {{
        background: #1f77b4;
      }}
      .ma-customer {{
        background: #ffffff;
        border: 2px solid #1f77b4;
      }}
    </style>
    <div class="ma-sidebar">
      <h2>{_html(summary['city'])} Run Details</h2>
      <h3>Legend</h3>
      {legend}
      <h3>Inputs</h3>
      <div class="ma-kv">
        <div class="ma-key">Customers</div><div class="ma-value">{input_data['customer_count']}</div>
        <div class="ma-key">Instance</div><div class="ma-value">{input_data['instance_index']} / {input_data['num_instances'] - 1}</div>
        <div class="ma-key">Depots</div><div class="ma-value">{input_data['num_depots']}</div>
        <div class="ma-key">Drones/truck</div><div class="ma-value">{input_data['drones_per_truck']}</div>
        <div class="ma-key">Limit</div><div class="ma-value">{input_data['limit']}</div>
        <div class="ma-key">Speed</div><div class="ma-value">{input_data['speed']}</div>
        <div class="ma-key">Theta</div><div class="ma-value">{_html(input_data['theta'])}</div>
        <div class="ma-key">Depot nodes</div><div class="ma-value">{_html(_node_text(input_data['depots'], limit=40))}</div>
        <div class="ma-key">Customer nodes</div><div class="ma-value">{_html(_node_text(input_data['customers'], limit=60))}</div>
      </div>
      <h3>Files</h3>
      <div class="ma-kv">
        <div class="ma-key">Result npz</div><div class="ma-value">{_html(files['result_file'])}</div>
        <div class="ma-key">HTML output</div><div class="ma-value">{_html(files['output_html'])}</div>
        <div class="ma-key">Summary JSON</div><div class="ma-value">{_html(files['summary_json'])}</div>
      </div>
      <h3>Results</h3>
      <div class="ma-kv">
        <div class="ma-key">Objective</div><div class="ma-value">{_summary_number(results['objective_value'], 6)}</div>
        <div class="ma-key">Saved cost</div><div class="ma-value">{_summary_number(results['saved_cost'], 6)}</div>
        <div class="ma-key">Saved time</div><div class="ma-value">{_summary_number(results['saved_time'])}s</div>
        <div class="ma-key">Solve time</div><div class="ma-value">{_summary_number(results['solve_seconds'])}s</div>
        <div class="ma-key">Road nodes</div><div class="ma-value">{graph['nodes']}</div>
        <div class="ma-key">Road edges</div><div class="ma-value">{graph['edges']}</div>
      </div>
      <h3>Customer Groups</h3>
      {''.join(group_rows)}
      <h3>Set-TSP / Order</h3>
      {''.join(set_tsp_rows)}
      <h3>Route Stats</h3>
      {''.join(route_rows)}
    </div>
    """


def _line_popup(title, rows):
    body = ''.join(
        f'<tr><th style="text-align:left;padding:2px 8px 2px 0">{_html(key)}</th>'
        f'<td style="padding:2px 0">{_html(value)}</td></tr>'
        for key, value in rows
    )
    return folium.Popup(f'<b>{_html(title)}</b><table>{body}</table>', max_width=520)


def _large_road_result_file(result_root, city_key, config, customer_count):
    """
    查找指定路网与客户规模的最新一批实验结果。

    输入：结果根目录、城市键、城市配置和客户数量。
    输出：显式配置文件、最新时间戳文件，或用于兼容旧格式的固定文件路径。
    """
    if config.get('result_file'):
        return Path(config['result_file'])
    # 时间戳文件以名称排序即可确定同地图、同规模的最新批次。
    result_dir = Path(result_root) / config.get('result_subdir', city_key) / 'data'
    map_name = config.get('result_map_name')
    if map_name:
        candidates = sorted(result_dir.glob(f'*-{map_name}-{customer_count}.npz'))
        if candidates:
            return candidates[-1]
    return result_dir / f'road-size-{customer_count}.npz'


def _large_road_output_dir(output_root, city_key, config):
    if config.get('output_dir'):
        return Path(config['output_dir'])
    return Path(output_root) / config.get('result_subdir', city_key) / 'maps'


def _telemetry_from_saved_result(saved_result):
    """
    将新格式 NPZ 中的阶段数组恢复为地图摘要所需的 telemetry 结构。

    输入：`_load_large_road_saved_result` 返回的新格式实例记录。
    输出：包含总计时和逐仓库记录的字典。
    """
    # 三类变长记录按仓库编号对齐；缺失字段使用空列表兼容早期 schema。
    groups = saved_result.get('phase1_groups') or []
    orders = saved_result.get('phase2_orders') or []
    phase2_times = saved_result.get('phase2_time') or []
    phase3_times = saved_result.get('phase3_time') or []
    phase_costs = saved_result.get('phase_costs') or []
    depot_records = []
    for depot_index, depot in enumerate(saved_result['depots']):
        group = groups[depot_index] if depot_index < len(groups) else {}
        order = orders[depot_index] if depot_index < len(orders) else {}
        customers = list(group.get('customers', order.get('customers', [])))
        depot_records.append({
            'depot_index': depot_index,
            'depot_node': depot,
            'customer_count': len(customers),
            'customers': customers,
            'convex_set_sizes': order.get('convex_set_sizes', []),
            'set_tsp_solver': order.get('set_tsp_solver', 'unknown'),
            'set_tsp_sequence': order.get('set_tsp_sequence', []),
            'visit_route': order.get('visit_route', [depot, depot]),
            'objective_contribution': (
                phase_costs[depot_index] if depot_index < len(phase_costs) else None
            ),
            'timings': {
                'set_tsp_seconds': (
                    phase2_times[depot_index] if depot_index < len(phase2_times) else None
                ),
                'local_search_seconds': (
                    phase3_times[depot_index] if depot_index < len(phase3_times) else None
                ),
            },
        })
    return {
        'timings': {
            'boundary_convex_sets_seconds': saved_result.get('phase1_boundary_time'),
            'mst_partition_seconds': saved_result.get('phase1_partition_time'),
            'solve_seconds': saved_result.get('time'),
        },
        'depot_records': depot_records,
        'objective_value': saved_result['cost'],
    }


def _load_graph_for_saved_result(config):
    """
    根据城市配置只加载结果对应的路网，不重新生成实例或运行求解器。

    输入：包含 `graph_loader` 和可选 `graph_path` 的城市配置。
    输出：标准化后的 NetworkX 路网图。
    """
    graph_loader = config.get('graph_loader', config.get('instance_builder'))
    graph_path = config.get('graph_path')
    if graph_loader == 'manhattan':
        return manhattan(graph_path)
    if graph_loader == 'boston':
        # 当前基础版本的 `cambridge()` 通过内部候选路径选择本地 Boston 图。
        return cambridge()
    raise ValueError(f'unknown graph_loader={graph_loader!r}')


def _is_legacy_11k_result(config):
    """
    判断当前城市配置是否对应需要保护的旧版 11k 结果。

    输入：包含结果地图名和 GraphML 路径的城市配置。
    输出：若旧格式回退会在 11k 地图上重新构造距离并求解，则返回 `True`。
    """
    # 文件命名是首要判据；路径判据兼容调用方只覆盖 `graph_path` 的情况。
    if config.get('result_map_name') == 'boston_11k':
        return True
    graph_path = config.get('graph_path')
    if graph_path is None:
        return False
    return Path(graph_path).resolve() == Path(MANHATTAN11k_GRAPH_PATH).resolve()


def _folium_graph_center_and_bounds(graph):
    lons = [graph.nodes[node]['pos'][0] for node in graph.nodes]
    lats = [graph.nodes[node]['pos'][1] for node in graph.nodes]
    center = [float(np.mean(lats)), float(np.mean(lons))]
    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    return center, bounds


def _folium_node_location(graph, node):
    return [graph.nodes[node]['pos'][1], graph.nodes[node]['pos'][0]]


def _iter_drone_sorties(route):
    for drone_routes in route.get('drone', []):
        if len(drone_routes) == 0:
            continue
        first = drone_routes[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            for sortie in drone_routes:
                if len(sortie) >= 2:
                    yield list(sortie)
        elif len(drone_routes) >= 2:
            yield list(drone_routes)


def _draw_large_road_edges(map_obj, graph):
    road_layer = folium.FeatureGroup(name='Road network', show=False)
    for edge in graph.edges:
        start, end = edge[:2]
        folium.PolyLine(
            locations=[_folium_node_location(graph, start), _folium_node_location(graph, end)],
            color='#222222',
            weight=1,
            opacity=0.28,
        ).add_to(road_layer)
    road_layer.add_to(map_obj)


def _plot_large_road_solution_map(
    graph,
    depots,
    cities,
    groups,
    solution,
    output_path,
    summary,
    draw_road_edges,
):
    city_label = summary['city']
    center, bounds = _folium_graph_center_and_bounds(graph)
    map_obj = folium.Map(location=center, zoom_start=13, tiles='Cartodb Positron')

    if draw_road_edges and graph.number_of_edges() <= LARGE_ROAD_MAX_DRAW_EDGES:
        _draw_large_road_edges(map_obj, graph)
    elif draw_road_edges:
        print(
            f'Skipping road-edge layer for {city_label}: '
            f'{graph.number_of_edges()} edges exceed LARGE_ROAD_MAX_DRAW_EDGES={LARGE_ROAD_MAX_DRAW_EDGES}.'
        )

    assigned_customers = set()
    for depot_index, depot in enumerate(depots):
        color = LARGE_ROAD_COLORS[depot_index % len(LARGE_ROAD_COLORS)]
        layer = folium.FeatureGroup(name=f'{city_label} depot {depot_index}', show=True)
        depot_location = _folium_node_location(graph, depot)
        folium.CircleMarker(
            location=depot_location,
            radius=LARGE_ROAD_DEPOT_RADIUS,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=f'Depot {depot_index}: node {depot}',
            popup=f'Depot {depot_index}<br>Node: {depot}<br>Assigned customers: {len(groups.get(depot, []))}',
        ).add_to(layer)

        for city in groups.get(depot, []):
            assigned_customers.add(city)
            folium.CircleMarker(
                location=_folium_node_location(graph, city),
                radius=LARGE_ROAD_CUSTOMER_RADIUS,
                color=color,
                weight=1,
                fill=True,
                fill_color='white',
                fill_opacity=0.9,
                tooltip=f'Depot {depot_index} customer: node {city}',
                popup=f'Customer node: {city}<br>Assigned depot: {depot_index}',
            ).add_to(layer)

        if depot_index < len(solution):
            route = solution[depot_index]
            truck_route = route.get('truck', [])
            if len(truck_route) >= 2:
                truck_metrics = _truck_route_metrics(graph, truck_route)
                if len(truck_metrics['locations']) >= 2:
                    tooltip = (
                        f"<b>Truck route</b><br>"
                        f"Depot {depot_index} node {depot}<br>"
                        f"Distance: {truck_metrics['distance']:.6f}<br>"
                        f"Stops: {len(truck_route)}<br>"
                        f"Assigned customers: {len(groups.get(depot, []))}"
                    )
                    popup = _line_popup(
                        f'Depot {depot_index} truck route',
                        [
                            ('type', 'truck'),
                            ('depot node', depot),
                            ('key route', _node_text(truck_route, limit=120)),
                            ('expanded nodes', truck_metrics['expanded_node_count']
                             if 'expanded_node_count' in truck_metrics else len(truck_metrics['expanded_nodes'])),
                            ('distance', f"{truck_metrics['distance']:.6f}"),
                            ('segments', len(truck_metrics['segments'])),
                        ],
                    )
                    folium.PolyLine(
                        locations=truck_metrics['locations'],
                        color=color,
                        weight=LARGE_ROAD_ROUTE_WEIGHT,
                        opacity=LARGE_ROAD_TRUCK_OPACITY,
                        tooltip=folium.Tooltip(tooltip, sticky=True),
                        popup=popup,
                    ).add_to(layer)
            for sortie in _iter_drone_sorties(route):
                drone_metrics = _drone_sortie_metrics(
                    graph,
                    sortie,
                    summary['input']['limit'],
                    summary['input']['speed'],
                )
                status = 'within limit' if drone_metrics['within_limit'] else 'exceeds limit'
                tooltip = (
                    f"<b>Drone sortie</b><br>"
                    f"Depot {depot_index} node {depot}<br>"
                    f"Launch: {drone_metrics['launch']}<br>"
                    f"Customer: {drone_metrics['customer']}<br>"
                    f"Recovery: {drone_metrics['recovery']}<br>"
                    f"Distance: {drone_metrics['distance']:.6f}<br>"
                    f"{status}"
                )
                popup = _line_popup(
                    f'Depot {depot_index} drone sortie',
                    [
                        ('type', 'drone'),
                        ('depot node', depot),
                        ('nodes', _node_text(drone_metrics['nodes'], limit=20)),
                        ('launch', drone_metrics['launch']),
                        ('customer', drone_metrics['customer']),
                        ('recovery', drone_metrics['recovery']),
                        ('distance', f"{drone_metrics['distance']:.6f}"),
                        ('flight time', _summary_number(drone_metrics['flight_time'], 6)),
                        ('limit', summary['input']['limit']),
                        ('status', status),
                    ],
                )
                folium.PolyLine(
                    locations=[_folium_node_location(graph, node) for node in sortie],
                    color=LARGE_ROAD_DRONE_COLOR,
                    weight=LARGE_ROAD_DRONE_WEIGHT,
                    opacity=LARGE_ROAD_DRONE_OPACITY,
                    dash_array='8, 6',
                    tooltip=folium.Tooltip(tooltip, sticky=True),
                    popup=popup,
                ).add_to(layer)

        layer.add_to(map_obj)

    unassigned = [city for city in cities if city not in assigned_customers]
    if unassigned:
        layer = folium.FeatureGroup(name=f'{city_label} unassigned customers', show=True)
        for city in unassigned:
            folium.CircleMarker(
                location=_folium_node_location(graph, city),
                radius=LARGE_ROAD_CUSTOMER_RADIUS,
                color='#444444',
                weight=1,
                fill=True,
                fill_color='#dddddd',
                fill_opacity=0.9,
                tooltip=f'Unassigned customer: node {city}',
                popup=f'Unassigned customer node: {city}',
            ).add_to(layer)
        layer.add_to(map_obj)

    map_obj.get_root().html.add_child(folium.Element(_render_large_road_sidebar(summary)))
    folium.LayerControl(position='topleft', collapsed=False).add_to(map_obj)
    map_obj.fit_bounds(bounds)
    _save_map(map_obj, output_path)


def plot_large_road_experiment_results(
    result_root=None,
    output_root=None,
    city_configs=None,
    cities=None,
    customer_count=None,
    instance_index=None,
    num_instances=None,
    draw_road_edges=None,
    allow_legacy_11k_resolve=None,
):
    """
    可视化 Manhattan 1K 与 Boston 11K 路网实验结果。

    输入：结果/输出根目录、城市配置、客户规模、可选实例编号、绘图开关，以及
    是否允许旧版 11k 结果在绘图阶段重新求解。
    输出：每个成功生成城市对应的地图、摘要和源结果路径。

    新格式 NPZ 直接恢复保存的输入、三阶段记录和最终路线；未指定实例编号时选择
    median 代表实例。旧格式只有成本/耗时时，才进入重新生成实例并求解的兼容路径。
    """
    result_root = LARGE_ROAD_RESULT_ROOT if result_root is None else Path(result_root)
    output_root = LARGE_ROAD_OUTPUT_ROOT if output_root is None else Path(output_root)
    city_configs = LARGE_ROAD_CITY_CONFIGS if city_configs is None else city_configs
    cities = LARGE_ROAD_CITIES if cities is None else cities
    customer_count = LARGE_ROAD_CUSTOMER_COUNT if customer_count is None else customer_count
    instance_index = LARGE_ROAD_INSTANCE_INDEX if instance_index is None else instance_index
    num_instances = LARGE_ROAD_NUM_INSTANCES if num_instances is None else num_instances
    draw_road_edges = LARGE_ROAD_DRAW_ROAD_EDGES if draw_road_edges is None else draw_road_edges
    if allow_legacy_11k_resolve is None:
        allow_legacy_11k_resolve = LARGE_ROAD_ALLOW_LEGACY_11K_RESOLVE

    generated = {}
    for city_key in cities:
        if city_key not in city_configs:
            print(f'Skipping unknown large-road city config: {city_key}')
            continue

        config = city_configs[city_key]
        local_customer_count = config.get('customer_count', customer_count)
        requested_instance_index = config.get('instance_index', instance_index)
        local_num_depots = config['num_depots']
        default_drones = LARGE_ROAD_CITY_CONFIGS.get(city_key, {}).get('drones_per_truck', 3)
        local_drones = config.get('drones_per_truck', default_drones)
        local_limit = config.get('limit', LARGE_ROAD_LIMIT)
        local_speed = config.get('speed', LARGE_ROAD_SPEED)
        local_theta = config.get('theta', LARGE_ROAD_THETA)
        city_label = config.get('label', city_key.title())

        result_file = _large_road_result_file(result_root, city_key, config, local_customer_count)
        if not _require_files(result_file):
            continue

        saved_result, saved_count = _load_large_road_saved_result(
            result_file, requested_instance_index
        )
        if saved_result is None and saved_count == 0:
            continue
        if saved_result is None:
            continue
        local_instance_index = saved_result['instance_index']
        local_num_instances = max(
            config.get('num_instances', num_instances), local_instance_index + 1
        )

        if saved_result.get('solution') is not None:
            # 新格式直接使用保存的实例与最终解，绘图过程不再调用优化求解器。
            print(
                f'Loading saved {city_label} instance {local_instance_index} from {result_file}.'
            )
            graph = _load_graph_for_saved_result(config)
            selected_depots = saved_result['depots']
            selected_cities = saved_result['cities']
            solution = saved_result['solution']
            solved_cost = saved_result['cost']
            telemetry = _telemetry_from_saved_result(saved_result)
            groups = {
                record['depot_node']: list(record['customers'])
                for record in telemetry['depot_records']
            }
            local_num_depots = len(selected_depots)
            local_customer_count = len(selected_cities)
            local_drones = saved_result.get('drones_per_truck', local_drones)
            local_limit = saved_result.get('limit', local_limit)
            local_speed = saved_result.get('speed', local_speed)
            local_theta = saved_result.get('theta', local_theta)
        else:
            # 旧格式结果没有路线，只能按旧逻辑重新生成同编号实例并求解。
            legacy_resolve_allowed = config.get(
                'allow_legacy_resolve', allow_legacy_11k_resolve
            )
            if _is_legacy_11k_result(config) and not legacy_resolve_allowed:
                print(
                    f'Skipping {city_label}: {result_file} 是不含路线的旧格式 11k 结果。'
                    '请用新结果格式重跑实验，或显式传入 '
                    'allow_legacy_11k_resolve=True 后再重新求解。'
                )
                continue
            builder_key = config.get('instance_builder', city_key)
            if builder_key not in LARGE_ROAD_INSTANCE_BUILDERS:
                print(f'Skipping {city_label}: unknown instance_builder={builder_key}.')
                continue
            builder = LARGE_ROAD_INSTANCE_BUILDERS[builder_key]
            print(
                f'Building legacy {city_label} instance {local_instance_index}: '
                f'depots={local_num_depots}, customers={local_customer_count}, drones={local_drones}.'
            )
            if builder_key == 'manhattan':
                graph, depots, city_nodes, distance = builder(
                    local_num_instances,
                    local_num_depots,
                    local_customer_count,
                    config.get('graph_path'),
                )
            else:
                graph, depots, city_nodes, distance = builder(
                    local_num_instances, local_num_depots, local_customer_count
                )
            selected_depots = depots[local_instance_index]
            selected_cities = city_nodes[local_instance_index]
            model = MultiAgentFlyingSidekickTSP(
                graph,
                selected_depots,
                selected_cities,
                distance,
                local_drones,
                limit=local_limit,
                speed=local_speed,
                theta=local_theta,
            )
            solution, solved_cost, telemetry = _solve_large_road_with_telemetry(model)
            groups = model.groups

        output_dir = _large_road_output_dir(output_root, city_key, config)
        # 使用源结果文件名作为前缀，使地图携带运行时刻、地图名和客户数量。
        output_basename = f'{result_file.stem}-instance-{local_instance_index:03d}'
        output_path = output_dir / f'{output_basename}-solution.html'
        summary_path = output_dir / f'{output_basename}-summary.json'
        route_stats = _large_road_route_stats(
            graph,
            solution,
            telemetry['depot_records'],
            local_limit,
            local_speed,
        )
        summary_config = {
            **config,
            'num_depots': local_num_depots,
            'limit': local_limit,
            'speed': local_speed,
            'theta': local_theta,
            'drones_per_truck': local_drones,
        }
        summary = _build_large_road_summary(
            graph,
            selected_depots,
            selected_cities,
            summary_config,
            result_file,
            output_path,
            summary_path,
            city_label,
            local_customer_count,
            local_instance_index,
            saved_count or local_num_instances,
            saved_result,
            solved_cost,
            telemetry,
            route_stats,
        )
        _save_json(summary_path, summary)
        _plot_large_road_solution_map(
            graph,
            selected_depots,
            selected_cities,
            groups,
            solution,
            output_path,
            summary,
            config.get('draw_road_edges', draw_road_edges),
        )
        print(f'Saved {city_label} large-road map to {output_path}')
        print(f'Saved {city_label} large-road summary to {summary_path}')
        generated[city_key] = {
            'map': output_path,
            'summary': summary_path,
            'result_file': result_file,
            'solved_cost': solved_cost,
            'saved_cost': saved_result.get('cost') if saved_result else None,
            'saved_time': saved_result.get('time') if saved_result else None,
        }

    return generated


def plot_paper_demo():
    """
    使用 Manhattan 与 Boston 子图生成论文风格三阶段示意图。
    """
    _plot_phase_demo(_manhattan_demo_instance(), MANHATTAN_FIGURE_DIR, 'paper_demo')
    _plot_phase_demo(_boston_demo_instance(), BOSTON_FIGURE_DIR, 'paper_demo')


def plot_example():
    """
    生成 HTML 地图示例，展示 Manhattan 与 Boston 上的分组、集合 TSP 路线与最终联合解。
    """
    _plot_solution_maps(_manhattan_demo_instance(), MANHATTAN_MAP_DIR)
    _plot_solution_maps(_boston_demo_instance(), BOSTON_MAP_DIR)


if __name__ == '__main__':
    # plot_speed()
    # plot_k()
    # plot_cities()
    # plot_rates()
    # plot_depots()
    # plot_r()
    # plot_accelerate()
    # plot_paper_demo()
    # 默认入口读取最新 1K/11K 实验结果；示例地图可按需单独启用。
    # plot_example()
    plot_large_road_experiment_results()
