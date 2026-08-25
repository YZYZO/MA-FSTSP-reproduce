"""
本文件负责构造实验实例与读取/生成路网数据。

主要内容：
1. 读取 Manhattan 与 Cambridge 路网。
2. 在缺少真实数据时自动生成合成网格路网。
3. 预计算卡车路网距离与无人机直线距离。
4. 生成论文实验所需的小规模、多仓库随机实例。
"""

import json
import time
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")


import osmnx as ox
import pickle
from functools import lru_cache
from pathlib import Path
from utils import haversine
from tqdm import tqdm
from config import (
    DATASETS_DIR,
    PROJECT_ROOT,
)

# 以下参数只服务于 Boston/OSM 路网的下载与裁剪，不属于全局项目配置。
# 只有同时开启下载授权和刷新开关，`cambridge()` 才会访问 Overpass 网络服务。
ALLOW_OSM_DOWNLOAD = False
REFRESH_OSM = False
OSM_CENTER_POINT = (42.3601, -71.0589)
OSM_DIST_METERS = 1600
OSM_MAX_NODES = 11000
OSM_TIMEOUT = 300
OVERPASS_ENDPOINTS = (
    'https://overpass.kumi.systems/api',
    'https://overpass-api.de/api',
)

MANHATTAN_CACHE = DATASETS_DIR / 'manhattan.json'
CAMBRIDGE_CACHE = DATASETS_DIR / 'cambridge_all_pair_road_distance.pkl'
MANHATTAN_GRAPH_CANDIDATES = (
    PROJECT_ROOT / 'manhatten.graphml',
    DATASETS_DIR / 'manhatten.graphml',
    PROJECT_ROOT / 'manhattan.graphml',
    DATASETS_DIR / 'manhattan.graphml',
    PROJECT_ROOT / 'nyc.graphml',
    DATASETS_DIR / 'nyc.graphml',
)
CAMBRIDGE_GRAPH_CANDIDATES = (
    PROJECT_ROOT / 'boston.graphml',
    DATASETS_DIR / 'boston.graphml',
    PROJECT_ROOT / 'cambridge.graphml',
    DATASETS_DIR / 'cambridge.graphml',
)
BOSTON_GRAPH_CACHE = DATASETS_DIR / 'boston.graphml'


def _coordinate_keys(graph):
    first_node = next(iter(graph.nodes))
    attrs = graph.nodes[first_node]
    if 'lon' in attrs and 'lat' in attrs:
        return 'lon', 'lat'
    if 'x' in attrs and 'y' in attrs:
        return 'x', 'y'
    raise KeyError(f"GraphML nodes must contain lon/lat or x/y coordinates, got {sorted(attrs.keys())}")


def _largest_strong_component(graph):
    if nx.is_strongly_connected(graph):
        return sorted(graph.nodes, key=str)
    component = max(nx.strongly_connected_components(graph), key=len)
    skipped = graph.number_of_nodes() - len(component)
    print(f'Using largest strongly connected road component: {len(component)} nodes, skipped {skipped} unreachable nodes.')
    return sorted(component, key=str)


def _boston_center_point():
    return OSM_CENTER_POINT


def _nearest_node_to_point(graph, center_point):
    center_pos = [center_point[1], center_point[0]]
    return min(graph.nodes, key=lambda node: haversine(center_pos, [float(graph.nodes[node]['x']), float(graph.nodes[node]['y'])]))


def _limit_nodes_near_center(graph, max_nodes, center_point):
    if max_nodes <= 0 or graph.number_of_nodes() <= max_nodes:
        return graph

    start = _nearest_node_to_point(graph, center_point)
    undirected = graph.to_undirected(as_view=True)
    selected = []
    seen = {start}
    queue = [start]
    while queue and len(selected) < max_nodes:
        node = queue.pop(0)
        selected.append(node)
        neighbors = sorted(
            undirected.neighbors(node),
            key=lambda item: haversine(
                [float(graph.nodes[item]['x']), float(graph.nodes[item]['y'])],
                [float(graph.nodes[node]['x']), float(graph.nodes[node]['y'])],
            ),
        )
        for neighbor in neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
            if len(selected) + len(queue) >= max_nodes:
                break

    limited = nx.MultiDiGraph(graph.subgraph(selected).copy())
    if limited.number_of_nodes() == 0:
        return graph
    nodes = _largest_strong_component(limited)
    print(f'Limited Boston road graph to {len(nodes)} nodes near {center_point}.')
    return nx.MultiDiGraph(limited.subgraph(nodes).copy())


def _configure_osmnx():
    _ensure_datasets_dir()
    cache_dir = DATASETS_DIR / 'osmnx_cache'
    cache_dir.mkdir(exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.timeout = OSM_TIMEOUT
    ox.settings.cache_folder = str(cache_dir)
    # OSMnx 1.8.1 can fail inside the Overpass /status rate-limit check.
    ox.settings.overpass_rate_limit = False


def _read_osm_graphml(path):
    graph = ox.load_graphml(str(path)) if hasattr(ox, 'load_graphml') else nx.MultiDiGraph(nx.read_graphml(path))
    return nx.MultiDiGraph(graph)


def _download_boston_graph():
    _configure_osmnx()
    center_point = _boston_center_point()
    distance = OSM_DIST_METERS
    errors = []

    for endpoint in OVERPASS_ENDPOINTS:
        ox.settings.overpass_endpoint = endpoint
        print(f'Trying Overpass endpoint: {endpoint}')
        try:
            graph = ox.graph_from_point(
                center_point,
                dist=distance,
                network_type='drive',
                simplify=True,
                retain_all=False,
            )
            _ensure_datasets_dir()
            ox.save_graphml(graph, filepath=BOSTON_GRAPH_CACHE)
            print(
                f'Downloaded Boston road graph with {graph.number_of_nodes()} nodes '
                f'and {graph.number_of_edges()} edges; saved to {BOSTON_GRAPH_CACHE}.'
            )
            return graph
        except Exception as exc:
            message = f'{endpoint}: {type(exc).__name__}: {exc}'
            errors.append(message)
            print(f'Failed with {message}')

    raise RuntimeError('Could not download Boston road graph.\n' + '\n'.join(errors))


def _graph_signature(graph):
    xs = [round(float(graph.nodes[node]['pos'][0]), 7) for node in graph.nodes]
    ys = [round(float(graph.nodes[node]['pos'][1]), 7) for node in graph.nodes]
    return {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'bounds': (min(xs), max(xs), min(ys), max(ys)),
    }


def _load_pairwise_cache(path, graph):
    if not path.is_file():
        return None
    with path.open('rb') as f:
        payload = pickle.load(f)
    if 'truck' in payload and 'drone' in payload:
        return None
    if payload.get('signature') == _graph_signature(graph):
        return payload['distance']
    return None


def _save_pairwise_cache(path, graph, distance):
    with path.open('wb') as f:
        pickle.dump({'signature': _graph_signature(graph), 'distance': distance}, f)


def _ensure_datasets_dir():
    """
    确保 `datasets` 缓存目录存在。

    输入：
    - 无。

    输出：
    - 无显式返回值，只在磁盘上创建目录。

    实现逻辑：
    - 调用 `Path.mkdir`，若目录已存在则不报错。
    """
    # 创建数据目录；若已存在则忽略错误。
    DATASETS_DIR.mkdir(exist_ok=True)


def _normalize_graph(graph, x_key, y_key, nodes=None):
    """
    将原始路网图重新编号，并统一转成带 `pos` 属性的 `MultiDiGraph`。

    输入：
    - graph: 原始图对象。
    - x_key: 原始图中横坐标字段名。
    - y_key: 原始图中纵坐标字段名。
    - nodes: 可选，指定保留的节点子集。

    输出：
    - road_graph: 重新编号后的 `networkx.MultiDiGraph`。

    实现逻辑：
    1. 先选取需要保留的节点集合。
    2. 为节点建立连续整数编号。
    3. 拷贝节点坐标并写入统一的 `pos=[lon, lat]` 结构。
    4. 用球面距离作为边权重新建立边。
    """
    road_graph = nx.MultiDiGraph()
    selected_nodes = list(graph.nodes if nodes is None else nodes)
    index = {node: i for i, node in enumerate(selected_nodes)}
    for node in selected_nodes:
        road_graph.add_node(index[node], pos=[float(graph.nodes[node][x_key]), float(graph.nodes[node][y_key])])
    for start, end, _ in graph.edges(data=True):
        if start in index and end in index:
            start_pos = road_graph.nodes[index[start]]['pos']
            end_pos = road_graph.nodes[index[end]]['pos']
            road_graph.add_edge(index[start], index[end], weight=haversine(start_pos, end_pos))
    return road_graph


def _synthetic_grid_graph(rows, cols, origin, step):
    """
    生成一个规则网格状的合成路网，用作离线演示数据。

    输入：
    - rows: 网格行数。
    - cols: 网格列数。
    - origin: 左上角近似经纬度坐标 `(lon, lat)`。
    - step: 相邻网格点的经纬度步长。

    输出：
    - graph: 构造好的 `networkx.MultiDiGraph`。

    实现逻辑：
    1. 按行列生成节点，并赋予经纬度坐标。
    2. 在上下左右相邻节点之间建立双向道路边。
    3. 用球面距离作为边权。
    """
    graph = nx.MultiDiGraph()
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            lon = origin[0] + col * step
            lat = origin[1] + row * step
            graph.add_node(node, pos=[lon, lat])
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if col + 1 < cols:
                neighbor = node + 1
                weight = haversine(graph.nodes[node]['pos'], graph.nodes[neighbor]['pos'])
                graph.add_edge(node, neighbor, weight=weight)
                graph.add_edge(neighbor, node, weight=weight)
            if row + 1 < rows:
                neighbor = node + cols
                weight = haversine(graph.nodes[node]['pos'], graph.nodes[neighbor]['pos'])
                graph.add_edge(node, neighbor, weight=weight)
                graph.add_edge(neighbor, node, weight=weight)
    return graph


def _pairwise_distance(graph, return_stats=False, show_progress=True):
    """
    为给定路网一次性构造卡车与无人机的全对距离矩阵。

    输入：
    - graph: 路网图。
    - return_stats: 是否在距离字典之外返回初始化耗时统计。
    - show_progress: 是否显示卡车与无人机距离构造进度条。

    输出：
    - 默认返回一个字典：
      - `truck`: 路网最短路距离。
      - `drone`: 节点间球面直线距离。
    - `return_stats=True` 时返回 `(distance, stats)`，附带三个墙钟耗时字段。

    实现逻辑：
    1. 用 Dijkstra 计算卡车在路网上的全对最短路。
    2. 用 `haversine` 计算无人机的节点间直线距离。
    3. 以源节点为进度单位显示两个阶段，并分别记录耗时。
    """
    # 固化节点顺序，使两个进度条的总量与实际源节点数保持一致。
    nodes = list(graph.nodes)
    node_count = len(nodes)
    total_start = time.perf_counter()

    # 卡车阶段每完成一个源节点的单源 Dijkstra，进度前进一步。
    truck_start = time.perf_counter()
    truck_rows = nx.all_pairs_dijkstra_path_length(graph, weight='weight')
    if show_progress:
        truck_rows = tqdm(
            truck_rows,
            total=node_count,
            desc='Truck all-pairs Dijkstra',
            unit='source',
        )
    truck_distance = dict(truck_rows)
    truck_apsp_seconds = time.perf_counter() - truck_start

    # 无人机阶段仍按原 haversine 语义物化矩阵，仅为外层源节点增加进度。
    drone_start = time.perf_counter()
    drone_sources = nodes
    if show_progress:
        drone_sources = tqdm(
            nodes,
            total=node_count,
            desc='Drone all-pairs distance',
            unit='source',
        )
    drone_distance = {
        source: {
            target: haversine(graph.nodes[source]['pos'], graph.nodes[target]['pos'])
            for target in nodes
        }
        for source in drone_sources
    }
    drone_pairwise_seconds = time.perf_counter() - drone_start

    # 总计时不包含地图读取和随机实例采样，只覆盖距离矩阵初始化。
    distance_initialization_seconds = time.perf_counter() - total_start
    stats = {
        'truck_apsp_seconds': truck_apsp_seconds,
        'drone_pairwise_seconds': drone_pairwise_seconds,
        'distance_initialization_seconds': distance_initialization_seconds,
    }
    print(
        'Pairwise distance initialization finished: '
        f'truck={truck_apsp_seconds:.3f}s, '
        f'drone={drone_pairwise_seconds:.3f}s, '
        f'total={distance_initialization_seconds:.3f}s.'
    )

    # 默认返回值保持兼容；实验入口可以显式取得批次级统计。
    distance = {'truck': truck_distance, 'drone': drone_distance}
    if return_stats:
        return distance, stats
    return distance


@lru_cache(maxsize=4)
def manhattan(graph_path=None):
    """
    读取指定的 Manhattan/NYC 路网，并标准化节点编号、坐标和边权。

    输入：
    - graph_path: 可选 GraphML 路径；省略时沿用现有 Manhattan 候选路径逻辑。

    输出：
    - manhattan_graph: `networkx.MultiDiGraph` 路网图。

    实现逻辑：
    1. 调用方传入路径时严格读取该文件，路径不存在则直接报错。
    2. 未传入路径时，继续从历史候选位置查找默认 Manhattan 地图。
    3. 默认地图缺失时构造规则网格，保持原有离线回退行为。
    4. 仅对默认地图保留原有 Manhattan 距离缓存生成逻辑。
    """
    # 显式路径用于 1k/11k 实验；默认调用继续采用原有候选路径优先级。
    if graph_path is None:
        selected_graph_path = next(
            (path for path in MANHATTAN_GRAPH_CANDIDATES if path.is_file()),
            None,
        )
    else:
        selected_graph_path = Path(graph_path)
        if not selected_graph_path.is_file():
            raise FileNotFoundError(
                f'Manhattan/NYC GraphML 文件不存在：{selected_graph_path.resolve()}'
            )
    # 如果没找到真实图，就使用合成网格图。

    g = nx.MultiDiGraph(nx.read_graphml(selected_graph_path))
     # 将真实图标准化为统一编号与坐标结构。
    x_key, y_key = _coordinate_keys(g)
    nodes = _largest_strong_component(g)
    manhattan_graph = _normalize_graph(g, x_key, y_key, nodes=nodes)
    # 确保缓存目录存在。
    _ensure_datasets_dir()

    return manhattan_graph

# @lru_cache(maxsize=4)
# def manhattan(graph_path=None):
#     """
#     读取指定的 Manhattan/NYC 路网，并标准化节点编号、坐标和边权。

#     输入：
#     - graph_path: 可选 GraphML 路径；省略时沿用现有 Manhattan 候选路径逻辑。

#     输出：
#     - manhattan_graph: `networkx.MultiDiGraph` 路网图。

#     实现逻辑：
#     1. 调用方传入路径时严格读取该文件，路径不存在则直接报错。
#     2. 未传入路径时，继续从历史候选位置查找默认 Manhattan 地图。
#     3. 默认地图缺失时构造规则网格，保持原有离线回退行为。
#     4. 仅对默认地图保留原有 Manhattan 距离缓存生成逻辑。
#     """
#     # 显式路径用于 1k/11k 实验；默认调用继续采用原有候选路径优先级。
#     if graph_path is None:
#         selected_graph_path = next(
#             (path for path in MANHATTAN_GRAPH_CANDIDATES if path.is_file()),
#             None,
#         )
#     else:
#         selected_graph_path = Path(graph_path)
#         if not selected_graph_path.is_file():
#             raise FileNotFoundError(
#                 f'Manhattan/NYC GraphML 文件不存在：{selected_graph_path.resolve()}'
#             )
#     # 如果没找到真实图，就使用合成网格图。
#     if selected_graph_path is None:
#         # 打印提示信息。
#         print('Manhattan GraphML not found, using a synthetic Manhattan-style grid instance instead.')
#         # 生成一个合成 Manhattan 风格网格图。
#         manhattan_graph = _synthetic_grid_graph(20, 20, origin=(-73.99, 40.75), step=0.002)
#     else:
#         # 读取真实 GraphML 文件。
#         g = nx.MultiDiGraph(nx.read_graphml(selected_graph_path))
#         # 将真实图标准化为统一编号与坐标结构。
#         x_key, y_key = _coordinate_keys(g)
#         nodes = _largest_strong_component(g)
#         manhattan_graph = _normalize_graph(g, x_key, y_key, nodes=nodes)
#     # 确保缓存目录存在。
#     _ensure_datasets_dir()
#     # 如果 Manhattan 距离缓存不存在，就创建它。
#     if (
#         graph_path is None
#         and not MANHATTAN_CACHE.is_file()
#         and manhattan_graph.number_of_nodes() <= 5000
#     ):
#         print('=============preparing pairwise data=================')
#         lengths = dict(nx.all_pairs_dijkstra_path_length(manhattan_graph, weight='weight'))
#         pairwise_distances = [[lengths[i][j] for j in manhattan_graph.nodes] for i in manhattan_graph.nodes]
#         with MANHATTAN_CACHE.open('w') as f:
#             json.dump(pairwise_distances, f)
#     return manhattan_graph

def cambridge():
    """
    读取 Cambridge/Boston 路网；若缺失真实数据，可下载或回退到合成网格图。

    输入：
    - 无。

    输出：
    - Cambridge/Boston 路网图。

    实现逻辑：
    1. 优先尝试读取本地 `boston.graphml` 或 `cambridge.graphml`。
    2. 若环境变量允许，则下载小范围 Boston 路网并截取最大强连通分量。
    3. 若仍失败，则回退到合成 Cambridge 风格网格图。
    """
    center_point = _boston_center_point()
    max_nodes = OSM_MAX_NODES

    # 在候选位置中查找 Boston/Cambridge 图文件。
    refresh_osm = REFRESH_OSM and ALLOW_OSM_DOWNLOAD
    graph_path = next((path for path in CAMBRIDGE_GRAPH_CANDIDATES if path.is_file()), None)
    # 如果本地已有图文件，则直接读取它。
    if graph_path is not None and not refresh_osm:
        print(f'Loading road graph from {graph_path}.')
        g = _read_osm_graphml(graph_path)
        x_key, y_key = _coordinate_keys(g)
        if x_key == 'x' and y_key == 'y':
            g = _limit_nodes_near_center(g, max_nodes, center_point)
        nodes = _largest_strong_component(g)
        return _normalize_graph(g, x_key, y_key, nodes=nodes)
    # 如果环境变量允许联网下载，则尝试使用 OSM 数据。
    if ALLOW_OSM_DOWNLOAD:
        try:
            graph = _download_boston_graph()
            graph = _limit_nodes_near_center(graph, max_nodes, center_point)
            nodes = _largest_strong_component(graph)
            return _normalize_graph(graph, 'x', 'y', nodes=nodes)
        except Exception as exc:
            print(f'Unable to download the Boston road network: {type(exc).__name__}: {exc}')
            print('Using a synthetic Cambridge-style grid instead.')
    else:
        print('boston.graphml/cambridge.graphml not found, using a synthetic Cambridge-style grid instance instead.')
    return _synthetic_grid_graph(14, 14, origin=(-71.11, 42.37), step=0.003)


def random_multiagent_instance(graph, num_depots, num_destinations):
    """
    在给定图上随机采样多仓库多客户实例。

    输入：
    - graph: 路网图。
    - num_depots: 仓库数量。
    - num_destinations: 客户数量。

    输出：
    - `(graph, depots, destinations)`。

    实现逻辑：
    1. 固定随机种子，保证可复现。
    2. 检查图节点数是否足够。
    3. 从图中随机采样仓库和客户节点。
    """
    np.random.seed(0)
    assert len(graph.nodes) > num_depots + num_destinations, \
        f"impossible to sample {num_depots + num_destinations} locations from {len(graph.nodes)} nodes"
    assert num_depots > 1, f"fewer than 2 depots, try to use random_instance function to generate for single agent"
    locations = np.random.choice(graph.nodes, size=num_depots + num_destinations)
    return graph, locations[:num_depots], locations[num_depots:]


def small_instance(num, nodes, depots, cities):
    """
    从 Manhattan 路网中截取一个小型连通子图，并生成若干随机实例。

    输入：
    - num: 需要生成的实例数量。
    - nodes: 子图节点数。
    - depots: 每个实例中的仓库数。
    - cities: 每个实例中的客户数。

    输出：
    - `(subgraph, depot_list, city_list, distance)`。

    实现逻辑：
    1. 读取 Manhattan 路网。
    2. 随机挑选一个起点并做 BFS 扩张，截取出指定大小子图。
    3. 在子图上构造卡车/无人机距离矩阵。
    4. 重复随机采样得到多个实例。
    """
    np.random.seed(0)
    graph = manhattan()
    node = np.random.choice(graph.nodes, 1).item()
    _nodes = [node]
    subgraph = nx.DiGraph()
    subgraph.add_node(node, pos=graph.nodes[node]['pos'])
    while subgraph.number_of_nodes() < nodes:
        neighbors = graph.neighbors(_nodes.pop(0))
        for n in neighbors:
            if not subgraph.has_node(n):
                _nodes.append(n)
                subgraph.add_node(n, pos=graph.nodes[n]['pos'])
            if subgraph.number_of_nodes() >= nodes:
                break
    assert subgraph.number_of_nodes() == nodes, 'wrong number of nodes, check the code'
    for start in subgraph.nodes:
        for end in subgraph.nodes:
            if graph.has_edge(start, end):
                subgraph.add_edge(start, end, weight=graph.edges[start, end, 0]['weight'])
                subgraph.add_edge(end, start, weight=graph.edges[start, end, 0]['weight'])
    distance = {'truck': dict(nx.all_pairs_dijkstra_path_length(subgraph, weight='weight')),
                'drone': {i: {j: haversine(subgraph.nodes[i]['pos'], subgraph.nodes[j]['pos'])
                              for j in subgraph.nodes} for i in subgraph.nodes}}
    _depots, _cities = [], []
    for _ in range(num):
        locations = np.random.choice(subgraph.nodes, depots + cities, replace=False)
        _depots.append(locations[:depots])
        _cities.append(locations[depots:])
    return subgraph, _depots, _cities, distance


def prepare_manhattan_road_network(
    graph_path=None,
    show_distance_progress=True,
):
    """
    读取指定 Manhattan/NYC 路网，并一次性构造可供多个客户规模复用的距离数据。

    输入：
    - graph_path: 可选 GraphML 路径；省略时使用默认 Manhattan 路网。
    - show_distance_progress: 是否显示全点对距离初始化进度条。

    输出：
    - `(graph, distance, distance_stats)`：标准化路网、卡车/无人机距离和初始化耗时。

    实现逻辑：
    1. 只读取并标准化一次目标路网。
    2. 只构造一次卡车与无人机全点对距离。
    3. 将准备结果交给不同客户规模的实验共享，避免重复初始化大型矩阵。
    """
    # 地图对象与距离矩阵共同组成一张地图的可复用准备结果。
    graph = manhattan(graph_path)
    distance, distance_stats = _pairwise_distance(
        graph,
        return_stats=True,
        show_progress=show_distance_progress,
    )
    return graph, distance, distance_stats


def sample_multiagent_instances(graph, num, depots, cities, seed=0):
    """
    在已准备好的路网上采样多仓库、多客户实例，不重复构造距离矩阵。

    输入：
    - graph: 已读取并标准化的路网。
    - num: 需要生成的实例数量。
    - depots: 每个实例的仓库数量。
    - cities: 每个实例的客户数量。
    - seed: 随机种子；默认保持原实验的确定性采样语义。

    输出：
    - `(depot_list, city_list)`：按实例索引配对的仓库集合与客户集合。

    实现逻辑：
    1. 为当前客户规模创建独立随机数生成器。
    2. 每个实例无放回采样仓库与客户节点。
    3. 打乱采样结果后按仓库数切分，保持原有实验的数据生成方式。
    """

    # 使用局部随机数生成器，避免不同地图或客户规模之间互相污染全局随机状态。
    random_state = np.random.RandomState(seed)
    graph_nodes = list(graph.nodes)
    depot_list, city_list = [], []
    for _ in range(num):
        # 每个实例内部保证仓库和客户节点不重复。
        locations = random_state.choice(
            graph_nodes,
            depots + cities,
            replace=False,
        )
        random_state.shuffle(locations)
        depot_list.append(locations[:depots])
        city_list.append(locations[depots:])
    return depot_list, city_list


def multiagent_instance_on_manhattan(
    num,
    depots,
    cities,
    graph_path=None,
    return_distance_stats=False,
    show_distance_progress=True,
):
    """
    在指定的 Manhattan/NYC 路网上生成多仓库随机实例。

    输入：
    - num: 实例数量。
    - depots: 每个实例中的仓库数。
    - cities: 每个实例中的客户数。
    - graph_path: 可选 GraphML 路径；省略时使用现有默认 Manhattan 地图。
    - return_distance_stats: 是否额外返回全点对距离初始化耗时。
    - show_distance_progress: 是否显示全点对距离初始化进度条。

    输出：
    - 默认返回 `(graph, depot_list, city_list, distance)`。
    - 请求初始化统计时，在末尾追加 `distance_stats`。

    实现逻辑：
    1. 读取调用方指定的 Manhattan/NYC 路网，或沿用默认 Manhattan 路网。
    2. 构造图上统一的全对距离矩阵。
    3. 重复采样仓库和客户节点。
    """
    # 地图和距离的准备逻辑独立后，旧接口仍然保持一次调用完成全部工作的行为。
    graph, distance, distance_stats = prepare_manhattan_road_network(
        graph_path,
        show_distance_progress=show_distance_progress,
    )
    # 客户规模变化时只重新采样实例，不再重复构造全点对距离。
    _depots, _cities = sample_multiagent_instances(
        graph,
        num,
        depots,
        cities,
    )
    # 默认保持历史四元组接口；实验计时模式额外返回初始化统计。
    if return_distance_stats:
        return graph, _depots, _cities, distance, distance_stats
    return graph, _depots, _cities, distance


def multiagent_instance_on_cambridge(num, depots, cities):
    """
    在 Cambridge 路网上生成多仓库随机实例，并复用缓存距离矩阵。

    输入：
    - num: 实例数量。
    - depots: 每个实例中的仓库数。
    - cities: 每个实例中的客户数。

    输出：
    - `(graph, depot_list, city_list, distance)`。

    实现逻辑：
    1. 读取 Cambridge 路网。
    2. 优先从缓存加载距离矩阵。
    3. 若缓存缺失或与当前图规模不一致，则重新计算并写回缓存。
    4. 重复采样生成多个随机实例。
    """
    np.random.seed(0)
    graph = cambridge()
    _ensure_datasets_dir()
    distance = _load_pairwise_cache(CAMBRIDGE_CACHE, graph)
    if distance is None:
        print('Preparing Cambridge/Boston pairwise distance cache.')
        distance = _pairwise_distance(graph)
        _save_pairwise_cache(CAMBRIDGE_CACHE, graph, distance)
    _depots, _cities = [], []
    for _ in range(num):
        locations = np.random.choice(graph.nodes, depots + cities, replace=False)
        _depots.append(locations[:depots])
        _cities.append(locations[depots:])
    return graph, _depots, _cities, distance
