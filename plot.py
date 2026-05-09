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
from pathlib import Path
from functools import lru_cache
from matplotlib.patches import Circle
from shapely.geometry import MultiPoint
from src.fstsp import MultiAgentFlyingSidekickTSP
from problem import cambridge, manhattan, _pairwise_distance
from utils import haversine
from config import DEMO_DRONE_LIMIT, ensure_dir, result_path


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
    plot_speed()
    plot_k()
    plot_cities()
    plot_rates()
    plot_depots()
    plot_r()
    plot_accelerate()
    plot_paper_demo()
    plot_example()
