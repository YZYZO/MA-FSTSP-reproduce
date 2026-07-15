"""
本文件负责构造实验实例与读取/生成路网数据。

主要内容：
1. 读取 Manhattan 与 Cambridge 路网。
2. 按显式路径标准化真实路网，并对 55k 本机运行进行硬保护。
3. 通过统一距离工厂提供 eager/H2H 卡车距离和按需无人机距离。
4. 生成论文实验所需的小规模、多仓库随机实例。
"""

# 导入 `networkx`，用于图构建与最短路计算。
import networkx as nx
# 导入 `numpy`，用于随机采样和数组处理。
import numpy as np
import matplotlib
matplotlib.use("Agg")


# 导入 `osmnx`，用于从 OpenStreetMap 下载真实路网。   
import osmnx as ox
# 导入 `lru_cache`，避免重复读取大型 GraphML 文件。
from functools import lru_cache
# 导入 `Path`，用于显式地图路径解析与错误报告。
from pathlib import Path
# 导入球面距离函数，用于根据经纬度构造边权。
from utils import haversine
from distance_oracle import build_distance_provider
from h2h_backend import enforce_local_graph_guard, graph_fingerprint
from config import (
    ALLOW_OSM_DOWNLOAD,
    ALLOW_GRAPH_PATH_FALLBACK,
    ALLOW_SYNTHETIC_GRAPH_FALLBACK,
    BOSTON_GRAPH_PATH,
    DATASETS_DIR,
    MANHATTAN_BASELINE_GRAPH_PATH,
    MANHATTAN_GRAPH_PATH,
    OSM_CENTER_POINT,
    OSM_DIST_METERS,
    OSM_MAX_NODES,
    OSM_TIMEOUT,
    OVERPASS_ENDPOINTS,
    PROJECT_ROOT,
    REFRESH_OSM,
)

# 旧全对缓存只用于迁移提示，不再读取、写入或自动删除。
MANHATTAN_CACHE = DATASETS_DIR / 'manhattan.json'
CAMBRIDGE_CACHE = DATASETS_DIR / 'cambridge_all_pair_road_distance.pkl'
# 仅当 `ALLOW_GRAPH_PATH_FALLBACK=True` 时才检查以下历史候选路径。
MANHATTAN_GRAPH_CANDIDATES = (
    PROJECT_ROOT / 'manhatten.graphml',
    DATASETS_DIR / 'manhatten.graphml',
    PROJECT_ROOT / 'manhattan.graphml',
    DATASETS_DIR / 'manhattan.graphml',
)
# 定义 Cambridge/Boston 图文件可能出现的位置。
CAMBRIDGE_GRAPH_CANDIDATES = (
    PROJECT_ROOT / 'boston.graphml',
    DATASETS_DIR / 'boston.graphml',
    PROJECT_ROOT / 'cambridge.graphml',
    DATASETS_DIR / 'cambridge.graphml',
)
BOSTON_GRAPH_CACHE = BOSTON_GRAPH_PATH


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


def _distance_provider(graph, dataset_name=None):
    """
    为已标准化路网构造统一的 eager/H2H 与按需无人机距离提供器。

    输入：
    - graph: 路网图。
    - dataset_name: 缓存目录使用的数据集名称。

    输出：
    - 保持 `distance['truck'][u][v]` / `distance['drone'][u][v]` 的字典。

    实现逻辑：
    - 小图由配置选择 eager；更大图构建或加载 H2H，绝不静默回退 Dijkstra。
    """
    source_path = graph.graph.get('source_path')
    label = dataset_name or graph.graph.get('dataset_name') or 'road-network'
    return build_distance_provider(
        graph,
        dataset_name=label,
        graph_path=source_path,
    )


def _pairwise_distance(graph):
    """
    保留旧私有函数名的兼容别名，但不再承诺或执行全对物化。

    输入：标准化路网图。
    输出：统一距离提供器。
    """
    return _distance_provider(graph)


@lru_cache(maxsize=4)
def manhattan(graph_path=None):
    """
    从显式路径读取并标准化 Manhattan/NYC 路网。

    输入：
    - graph_path: 可选 GraphML；省略时严格使用 `MANHATTAN_GRAPH_PATH`。

    输出：
    - 带连续整数节点、`pos` 和 haversine 边权的 `MultiDiGraph`。

    实现逻辑：
    1. 在任何文件读取前对默认 55k NYC 路径执行本机硬保护。
    2. 显式路径不存在时默认报错；只有配置开启才检查历史候选或合成图。
    3. 记录原始规模、最大强连通分量规模和标准化图哈希。
    4. 地图加载不再创建或读取 `manhattan.json` 全对缓存。
    """
    selected_path = Path(MANHATTAN_GRAPH_PATH if graph_path is None else graph_path).expanduser()
    enforce_local_graph_guard(graph_path=selected_path)
    if not selected_path.is_file() and ALLOW_GRAPH_PATH_FALLBACK:
        selected_path = next(
            (path for path in MANHATTAN_GRAPH_CANDIDATES if path.is_file()),
            selected_path,
        )
    if not selected_path.is_file():
        if not ALLOW_SYNTHETIC_GRAPH_FALLBACK:
            raise FileNotFoundError(
                f'Manhattan/NYC GraphML 不存在：{selected_path.resolve()}。'
                '如需检查历史路径，请显式设置 ALLOW_GRAPH_PATH_FALLBACK=True。'
            )
        manhattan_graph = _synthetic_grid_graph(
            20, 20, origin=(-73.99, 40.75), step=0.002
        )
        manhattan_graph.graph['dataset_name'] = 'synthetic-manhattan'
        print(
            f'Loaded synthetic Manhattan graph: nodes={manhattan_graph.number_of_nodes()}, '
            f'edges={manhattan_graph.number_of_edges()}, h2h_hash={graph_fingerprint(manhattan_graph)}.'
        )
        return manhattan_graph

    raw_graph = nx.MultiDiGraph(nx.read_graphml(selected_path))
    raw_nodes = raw_graph.number_of_nodes()
    raw_edges = raw_graph.number_of_edges()
    x_key, y_key = _coordinate_keys(raw_graph)
    nodes = _largest_strong_component(raw_graph)
    manhattan_graph = _normalize_graph(raw_graph, x_key, y_key, nodes=nodes)
    manhattan_graph.graph['source_path'] = str(selected_path.resolve())
    manhattan_graph.graph['dataset_name'] = selected_path.stem
    graph_hash = graph_fingerprint(manhattan_graph)
    print(
        f'Loaded Manhattan/NYC graph from {selected_path.resolve()}: '
        f'raw_nodes={raw_nodes}, raw_edges={raw_edges}, largest_scc_nodes={len(nodes)}, '
        f'normalized_edges={manhattan_graph.number_of_edges()}, h2h_hash={graph_hash}.'
    )
    if MANHATTAN_CACHE.is_file():
        print(f'Ignoring legacy all-pairs cache {MANHATTAN_CACHE}; it is not loaded or deleted.')
    return manhattan_graph


@lru_cache(maxsize=4)
def cambridge(graph_path=None):
    """
    从显式路径读取并标准化 Cambridge/Boston 路网。

    输入：
    - graph_path: 可选 GraphML；省略时严格使用 `BOSTON_GRAPH_PATH`。

    输出：
    - 最大强连通分量标准化后的 `MultiDiGraph`。

    实现逻辑：
    1. 默认只读显式本地文件；联网刷新、历史候选和合成图均需单独配置授权。
    2. 记录原始节点/边数、最大强连通分量和 H2H 图哈希。
    3. 旧 1.65 GB pairwise pickle 只提示，不打开、不覆盖、不删除。
    """
    selected_path = Path(BOSTON_GRAPH_PATH if graph_path is None else graph_path).expanduser()
    refresh_osm = REFRESH_OSM and ALLOW_OSM_DOWNLOAD and graph_path is None
    if refresh_osm:
        raw_graph = nx.MultiDiGraph(_download_boston_graph())
        selected_path = Path(BOSTON_GRAPH_CACHE)
    else:
        if not selected_path.is_file() and ALLOW_GRAPH_PATH_FALLBACK:
            selected_path = next(
                (path for path in CAMBRIDGE_GRAPH_CANDIDATES if path.is_file()),
                selected_path,
            )
        if not selected_path.is_file():
            if not ALLOW_SYNTHETIC_GRAPH_FALLBACK:
                raise FileNotFoundError(
                    f'Boston/Cambridge GraphML 不存在：{selected_path.resolve()}。'
                    '如需联网、历史候选或合成图，请分别显式开启对应配置。'
                )
            synthetic = _synthetic_grid_graph(
                14, 14, origin=(-71.11, 42.37), step=0.003
            )
            synthetic.graph['dataset_name'] = 'synthetic-boston'
            print(
                f'Loaded synthetic Boston graph: nodes={synthetic.number_of_nodes()}, '
                f'edges={synthetic.number_of_edges()}, h2h_hash={graph_fingerprint(synthetic)}.'
            )
            return synthetic
        raw_graph = _read_osm_graphml(selected_path)

    raw_nodes = raw_graph.number_of_nodes()
    raw_edges = raw_graph.number_of_edges()
    x_key, y_key = _coordinate_keys(raw_graph)
    if x_key == 'x' and y_key == 'y':
        raw_graph = _limit_nodes_near_center(
            raw_graph, OSM_MAX_NODES, _boston_center_point()
        )
    nodes = _largest_strong_component(raw_graph)
    graph = _normalize_graph(raw_graph, x_key, y_key, nodes=nodes)
    graph.graph['source_path'] = str(selected_path.resolve())
    graph.graph['dataset_name'] = selected_path.stem
    graph_hash = graph_fingerprint(graph)
    print(
        f'Loaded Boston/Cambridge graph from {selected_path.resolve()}: '
        f'raw_nodes={raw_nodes}, raw_edges={raw_edges}, largest_scc_nodes={len(nodes)}, '
        f'normalized_edges={graph.number_of_edges()}, h2h_hash={graph_hash}.'
    )
    if CAMBRIDGE_CACHE.is_file():
        print(f'Ignoring legacy all-pairs cache {CAMBRIDGE_CACHE}; it is not loaded or deleted.')
    return graph


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
    # 小实例使用已冻结的 4,333 节点基线图，不触发默认 55k NYC 目标路径。
    graph = manhattan(MANHATTAN_BASELINE_GRAPH_PATH)
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
    # 小图通过统一距离工厂运行：卡车使用 eager 基线，无人机距离按需计算。
    distance = build_distance_provider(subgraph, backend='eager', dataset_name='small-instance')
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


def multiagent_instance_on_manhattan(num, depots, cities):
    """
    在 Manhattan 路网上生成多仓库随机实例。

    输入：
    - num: 实例数量。
    - depots: 每个实例中的仓库数。
    - cities: 每个实例中的客户数。

    输出：
    - `(graph, depot_list, city_list, distance)`。

    实现逻辑：
    1. 读取 Manhattan 路网。
    2. 通过统一工厂加载/构建 H2H 与按需无人机距离。
    3. 重复采样仓库和客户节点。
    """
    # 固定随机种子。
    np.random.seed(0)
    # 读取 Manhattan 路网。
    graph = manhattan()
    # 统一距离工厂会在本机读取默认 55k 图前拦截，服务器则加载 H2H 缓存。
    distance = _distance_provider(graph, dataset_name=graph.graph.get('dataset_name', 'nyc'))
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
    在 Cambridge/Boston 路网上生成多仓库随机实例，并复用 H2H 索引缓存。

    输入：
    - num: 实例数量。
    - depots: 每个实例中的仓库数。
    - cities: 每个实例中的客户数。

    输出：
    - `(graph, depot_list, city_list, distance)`。

    实现逻辑：
    1. 读取 Cambridge 路网。
    2. 检测旧 pairwise pickle 但不打开；统一工厂按图哈希加载或构建 H2H。
    3. 无人机距离始终按需计算，不写全对缓存。
    4. 重复采样生成多个随机实例。
    """
    # 固定随机种子。
    np.random.seed(0)
    # 读取 Cambridge 路网。
    graph = cambridge()
    # 旧 1.65 GB pickle 绝不反序列化；距离工厂只使用版本化 H2H 缓存。
    distance = _distance_provider(graph, dataset_name=graph.graph.get('dataset_name', 'boston'))
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
