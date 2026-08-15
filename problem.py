"""
本文件负责构造实验实例与读取/生成路网数据。

主要内容：
1. 读取 Manhattan 与 Cambridge 路网。
2. 在缺少真实数据时自动生成合成网格路网。
3. 预计算卡车路网距离与无人机直线距离。
4. 生成论文实验所需的小规模、多仓库随机实例。
"""

# 导入 `json`，用于保存 Manhattan 的距离缓存。
import json
# 导入 `networkx`，用于图构建与最短路计算。
import networkx as nx
# 导入 `numpy`，用于随机采样和数组处理。
import numpy as np
import matplotlib
matplotlib.use("Agg")


# 导入 `osmnx`，用于从 OpenStreetMap 下载真实路网。   
import osmnx as ox
# 导入 `pickle`，用于缓存 Cambridge 的距离矩阵。
import pickle
# 导入 `lru_cache`，避免重复读取大型 GraphML 文件。
from functools import lru_cache
# 导入 `Path`，用于校验调用方显式传入的 GraphML 文件路径。
from pathlib import Path
# 导入球面距离函数，用于根据经纬度构造边权。
from utils import haversine
from config import (
    ALLOW_OSM_DOWNLOAD,
    DATASETS_DIR,
    OSM_CENTER_POINT,
    OSM_DIST_METERS,
    OSM_MAX_NODES,
    OSM_TIMEOUT,
    OVERPASS_ENDPOINTS,
    PROJECT_ROOT,
    REFRESH_OSM,
)

# 定义 Manhattan 缓存文件路径。
MANHATTAN_CACHE = DATASETS_DIR / 'manhattan.json'
# 定义 Cambridge 缓存文件路径。
CAMBRIDGE_CACHE = DATASETS_DIR / 'cambridge_all_pair_road_distance.pkl'
# 定义 Manhattan 图文件可能出现的位置。优先使用论文示意图对应的曼哈顿子图。
MANHATTAN_GRAPH_CANDIDATES = (
    PROJECT_ROOT / 'manhatten.graphml',
    DATASETS_DIR / 'manhatten.graphml',
    PROJECT_ROOT / 'manhattan.graphml',
    DATASETS_DIR / 'manhattan.graphml',
    PROJECT_ROOT / 'nyc.graphml',
    DATASETS_DIR / 'nyc.graphml',
)
# 定义 Cambridge/Boston 图文件可能出现的位置。
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
    # 新建一个标准化后的多重有向图。
    road_graph = nx.MultiDiGraph()
    # 如果没有指定节点子集，就保留全部节点。
    selected_nodes = list(graph.nodes if nodes is None else nodes)
    # 为每个旧节点分配一个新的连续整数编号。
    index = {node: i for i, node in enumerate(selected_nodes)}
    # 遍历保留节点，拷贝坐标。
    for node in selected_nodes:
        # 将坐标保存到统一的 `pos` 字段中。
        road_graph.add_node(index[node], pos=[float(graph.nodes[node][x_key]), float(graph.nodes[node][y_key])])
    # 遍历原图中的边。
    for start, end, _ in graph.edges(data=True):
        # 只保留两个端点都在选定集合中的边。
        if start in index and end in index:
            # 读取起点坐标。
            start_pos = road_graph.nodes[index[start]]['pos']
            # 读取终点坐标。
            end_pos = road_graph.nodes[index[end]]['pos']
            # 用球面距离作为边权建立标准化边。
            road_graph.add_edge(index[start], index[end], weight=haversine(start_pos, end_pos))
    # 返回标准化后的图。
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
    # 新建一个多重有向图。
    graph = nx.MultiDiGraph()
    # 逐行生成网格节点。
    for row in range(rows):
        # 遍历当前行中的每一列。
        for col in range(cols):
            # 将二维网格坐标编码为一维节点编号。
            node = row * cols + col
            # 计算该节点的经度。
            lon = origin[0] + col * step
            # 计算该节点的纬度。
            lat = origin[1] + row * step
            # 将节点加入图中，并保存位置。
            graph.add_node(node, pos=[lon, lat])
    # 再次遍历网格，用于添加边。
    for row in range(rows):
        # 遍历当前行的每一列。
        for col in range(cols):
            # 计算当前节点编号。
            node = row * cols + col
            # 如果右侧还有节点，则连一条横向双向边。
            if col + 1 < cols:
                # 右邻节点编号。
                neighbor = node + 1
                # 计算当前节点与右邻节点之间的距离。
                weight = haversine(graph.nodes[node]['pos'], graph.nodes[neighbor]['pos'])
                # 添加从当前节点到右邻节点的边。
                graph.add_edge(node, neighbor, weight=weight)
                # 添加从右邻节点回当前节点的边。
                graph.add_edge(neighbor, node, weight=weight)
            # 如果下方还有节点，则连一条纵向双向边。
            if row + 1 < rows:
                # 下邻节点编号。
                neighbor = node + cols
                # 计算当前节点与下邻节点之间的距离。
                weight = haversine(graph.nodes[node]['pos'], graph.nodes[neighbor]['pos'])
                # 添加从当前节点到下邻节点的边。
                graph.add_edge(node, neighbor, weight=weight)
                # 添加从下邻节点回当前节点的边。
                graph.add_edge(neighbor, node, weight=weight)
    # 返回生成好的网格图。
    return graph


def _pairwise_distance(graph):
    """
    为给定路网一次性构造卡车与无人机的全对距离矩阵。

    输入：
    - graph: 路网图。

    输出：
    - 一个字典：
      - `truck`: 路网最短路距离。
      - `drone`: 节点间球面直线距离。

    实现逻辑：
    1. 用 Dijkstra 计算卡车在路网上的全对最短路。
    2. 用 `haversine` 计算无人机的节点间直线距离。
    """
    # 返回包含卡车距离和无人机距离的字典。
    return {'truck': dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight')),
            'drone': {i: {j: haversine(graph.nodes[i]['pos'], graph.nodes[j]['pos']) for j in graph.nodes}
                      for i in graph.nodes}}


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
    if selected_graph_path is None:
        # 打印提示信息。
        print('Manhattan GraphML not found, using a synthetic Manhattan-style grid instance instead.')
        # 生成一个合成 Manhattan 风格网格图。
        manhattan_graph = _synthetic_grid_graph(20, 20, origin=(-73.99, 40.75), step=0.002)
    else:
        # 读取真实 GraphML 文件。
        g = nx.MultiDiGraph(nx.read_graphml(selected_graph_path))
        # 将真实图标准化为统一编号与坐标结构。
        x_key, y_key = _coordinate_keys(g)
        nodes = _largest_strong_component(g)
        manhattan_graph = _normalize_graph(g, x_key, y_key, nodes=nodes)
    # 确保缓存目录存在。
    _ensure_datasets_dir()
    # 如果 Manhattan 距离缓存不存在，就创建它。
    if (
        graph_path is None
        and not MANHATTAN_CACHE.is_file()
        and manhattan_graph.number_of_nodes() <= 5000
    ):
        # 打印缓存构造提示。
        print('=============preparing pairwise data=================')
        # 计算所有节点对之间的最短路距离。
        lengths = dict(nx.all_pairs_dijkstra_path_length(manhattan_graph, weight='weight'))
        # 将距离字典改写成二维列表，便于 JSON 保存。
        pairwise_distances = [[lengths[i][j] for j in manhattan_graph.nodes] for i in manhattan_graph.nodes]
        # 以写入方式打开缓存文件。
        with MANHATTAN_CACHE.open('w') as f:
            # 将距离矩阵写入 JSON。
            json.dump(pairwise_distances, f)
    # 返回路网图。
    return manhattan_graph


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
            # 从 OSM 下载小范围可驾驶路网，避免整座 Boston 过大导致 Overpass 失败。
            graph = _download_boston_graph()
            graph = _limit_nodes_near_center(graph, max_nodes, center_point)
            # 取最大强连通分量，确保最短路普遍可达。
            nodes = _largest_strong_component(graph)
            # 返回标准化后的真实图。
            return _normalize_graph(graph, 'x', 'y', nodes=nodes)
        except Exception as exc:
            # 下载失败时给出提示。
            print(f'Unable to download the Boston road network: {type(exc).__name__}: {exc}')
            print('Using a synthetic Cambridge-style grid instead.')
    else:
        # 若未开启联网下载，则直接提示走离线路径。
        print('boston.graphml/cambridge.graphml not found, using a synthetic Cambridge-style grid instance instead.')
    # 返回合成 Cambridge 网格图。
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
    # 固定随机种子，保证结果可复现。
    np.random.seed(0)
    # 检查节点总数是否足够完成采样。
    assert len(graph.nodes) > num_depots + num_destinations, \
        f"impossible to sample {num_depots + num_destinations} locations from {len(graph.nodes)} nodes"
    # 检查仓库数必须大于 1，因为该函数面向多仓库问题。
    assert num_depots > 1, f"fewer than 2 depots, try to use random_instance function to generate for single agent"
    # 一次性从图中采样仓库与客户节点。
    locations = np.random.choice(graph.nodes, size=num_depots + num_destinations)
    # 返回图对象、仓库集合和客户集合。
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
    # 固定随机种子，保证可复现。
    np.random.seed(0)
    # 读取完整 Manhattan 路网。
    graph = manhattan()
    # 随机选择一个节点作为 BFS 起点。
    node = np.random.choice(graph.nodes, 1).item()
    # 用队列保存待扩张节点。
    _nodes = [node]
    # 新建一个有向子图。
    subgraph = nx.DiGraph()
    # 将起点加入子图并拷贝坐标。
    subgraph.add_node(node, pos=graph.nodes[node]['pos'])
    # 当子图规模未达到目标节点数时继续扩张。
    while subgraph.number_of_nodes() < nodes:
        # 取出队首节点并获得其邻居。
        neighbors = graph.neighbors(_nodes.pop(0))
        # 遍历这些邻居。
        for n in neighbors:
            # 如果当前邻居尚未加入子图，则将其加入。
            if not subgraph.has_node(n):
                # 将邻居加入 BFS 队列。
                _nodes.append(n)
                # 将邻居加入子图并拷贝坐标。
                subgraph.add_node(n, pos=graph.nodes[n]['pos'])
            # 如果规模已足够，则结束当前扩张层。
            if subgraph.number_of_nodes() >= nodes:
                break
    # 断言子图节点数正确。
    assert subgraph.number_of_nodes() == nodes, 'wrong number of nodes, check the code'
    # 为子图补齐原图中对应的边。
    for start in subgraph.nodes:
        # 遍历子图中的所有终点。
        for end in subgraph.nodes:
            # 若原图中存在 `start -> end` 边，则将其加入子图。
            if graph.has_edge(start, end):
                # 添加正向边。
                subgraph.add_edge(start, end, weight=graph.edges[start, end, 0]['weight'])
                # 同时补一条反向边，增强可达性。
                subgraph.add_edge(end, start, weight=graph.edges[start, end, 0]['weight'])
    # 在子图上构造卡车和无人机距离矩阵。
    distance = {'truck': dict(nx.all_pairs_dijkstra_path_length(subgraph, weight='weight')),
                'drone': {i: {j: haversine(subgraph.nodes[i]['pos'], subgraph.nodes[j]['pos'])
                              for j in subgraph.nodes} for i in subgraph.nodes}}
    # 用于保存多个实例的仓库采样。
    _depots, _cities = [], []
    # 重复采样 `num` 次。
    for _ in range(num):
        # 在子图节点中无放回采样仓库和客户。
        locations = np.random.choice(subgraph.nodes, depots + cities, replace=False)
        # 保存当前实例的仓库集合。
        _depots.append(locations[:depots])
        # 保存当前实例的客户集合。
        _cities.append(locations[depots:])
    # 返回子图、实例采样与距离矩阵。
    return subgraph, _depots, _cities, distance


def multiagent_instance_on_manhattan(num, depots, cities, graph_path=None):
    """
    在指定的 Manhattan/NYC 路网上生成多仓库随机实例。

    输入：
    - num: 实例数量。
    - depots: 每个实例中的仓库数。
    - cities: 每个实例中的客户数。
    - graph_path: 可选 GraphML 路径；省略时使用现有默认 Manhattan 地图。

    输出：
    - `(graph, depot_list, city_list, distance)`。

    实现逻辑：
    1. 读取调用方指定的 Manhattan/NYC 路网，或沿用默认 Manhattan 路网。
    2. 构造图上统一的全对距离矩阵。
    3. 重复采样仓库和客户节点。
    """
    # 固定随机种子。
    np.random.seed(0)
    # 读取 Manhattan 路网。
    graph = manhattan(graph_path)
    # 预计算卡车和无人机距离。
    distance = _pairwise_distance(graph)
    # 保存每个实例的仓库集合。
    _depots, _cities = [], []
    # 重复采样 `num` 次。
    for _ in range(num):
        # 无放回采样仓库与客户。
        locations = np.random.choice(graph.nodes, depots + cities, replace=False)
        # 打乱次序，避免仓库/客户分配有结构性偏差。
        np.random.shuffle(locations)
        # 保存当前实例的仓库集合。
        _depots.append(locations[:depots])
        # 保存当前实例的客户集合。
        _cities.append(locations[depots:])
    # 返回路网与采样结果。
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
    # 固定随机种子。
    np.random.seed(0)
    # 读取 Cambridge 路网。
    graph = cambridge()
    # 确保缓存目录存在。
    _ensure_datasets_dir()
    # 如果距离缓存存在且与当前图签名一致，则优先读取它。
    distance = _load_pairwise_cache(CAMBRIDGE_CACHE, graph)
    if distance is None:
        # 若缓存不存在或与当前 Boston/Cambridge 图不一致，则直接计算距离矩阵。
        print('Preparing Cambridge/Boston pairwise distance cache.')
        distance = _pairwise_distance(graph)
        # 将计算结果写入缓存。
        _save_pairwise_cache(CAMBRIDGE_CACHE, graph, distance)
    # 保存每个实例的仓库采样。
    _depots, _cities = [], []
    # 重复采样 `num` 次。
    for _ in range(num):
        # 无放回采样仓库与客户。
        locations = np.random.choice(graph.nodes, depots + cities, replace=False)
        # 保存当前实例的仓库集合。
        _depots.append(locations[:depots])
        # 保存当前实例的客户集合。
        _cities.append(locations[depots:])
    # 返回路网、采样结果与距离矩阵。
    return graph, _depots, _cities, distance
