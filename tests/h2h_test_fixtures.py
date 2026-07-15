"""为距离抽象和后续 H2H 阶段提供可重复的小图测试夹具。"""

from __future__ import annotations

import random

import networkx as nx
import numpy as np

from utils import haversine


# 当前本地真实图的基线规模；算法实际使用最大强连通分量规模。
EXPECTED_LOCAL_GRAPH_STATS = {
    'manhatten.graphml': {
        'raw_nodes': 4426,
        'raw_edges': 9626,
        'largest_scc_nodes': 4333,
    },
    'boston.graphml': {
        'raw_nodes': 8412,
        'raw_edges': 18517,
        'largest_scc_nodes': 8313,
    },
}


def _add_geographic_edge(graph: nx.MultiDiGraph, source: int, target: int, scale: float = 1.0) -> None:
    """
    按项目当前距离语义向测试图添加一条有向边。

    输入：
    - graph: 已包含 `pos` 坐标的多重有向图。
    - source: 起点。
    - target: 终点。
    - scale: 仅用于构造较差平行边的权重倍率。

    输出：
    - 无显式返回值；边直接加入 `graph`。
    """
    weight = haversine(graph.nodes[source]['pos'], graph.nodes[target]['pos']) * scale
    graph.add_edge(source, target, weight=weight)


def build_fixed_20_node_graph() -> nx.MultiDiGraph:
    """
    构造固定的 20 节点强连通有向路网。

    输入：无。

    输出：
    - 含方向差异、平行边、自环和跨行捷径的 `MultiDiGraph`。

    实现逻辑：
    1. 以 4×5 网格生成经纬度坐标。
    2. 用单向 Hamilton 环保证强连通。
    3. 增加部分反向边和有向捷径，使 `d(u,v)` 与 `d(v,u)` 可不同。
    4. 增加一条较差平行边和一个零权自环，覆盖边界数据结构。
    """
    graph = nx.MultiDiGraph(name='fixed-directed-20')
    for node in range(20):
        row, column = divmod(node, 5)
        graph.add_node(
            node,
            pos=[-73.99 + column * 0.01, 40.75 + row * 0.01],
        )

    # 单向环是测试图强连通性的基础，不依赖后续随机或捷径边。
    for source in range(20):
        _add_geographic_edge(graph, source, (source + 1) % 20)

    # 局部反向道路与跨网格捷径用于制造真实的有向距离差异。
    for source, target in (
        (1, 0), (4, 3), (6, 5), (9, 8), (11, 10), (14, 13), (16, 15), (19, 18),
        (0, 5), (5, 10), (10, 15), (15, 0), (2, 7), (7, 12), (12, 17), (17, 2),
        (4, 9), (9, 14), (14, 19), (19, 4),
    ):
        _add_geographic_edge(graph, source, target)

    # 该平行边比已有 0→1 边更差，最短路语义应自然选择较小边。
    _add_geographic_edge(graph, 0, 1, scale=1.5)
    graph.add_edge(7, 7, weight=0.0)
    return graph


def build_random_strongly_connected_digraph(
    node_count: int,
    seed: int,
    extra_edge_probability: float = 0.12,
) -> nx.MultiDiGraph:
    """
    构造可复现的随机稀疏强连通有向图。

    输入：
    - node_count: 节点数量，必须大于 0。
    - seed: 独立随机种子。
    - extra_edge_probability: 每个非环有向边被加入的概率。

    输出：
    - 节点编号为 `[0, node_count)` 的 `MultiDiGraph`。

    实现逻辑：
    - 先建立覆盖全部节点的单向环，再按概率添加正权随机边；因此结果必然强连通。
    """
    if node_count <= 0:
        raise ValueError('node_count 必须大于 0。')
    if not 0.0 <= extra_edge_probability <= 1.0:
        raise ValueError('extra_edge_probability 必须位于 [0, 1]。')

    random_state = random.Random(seed)
    graph = nx.MultiDiGraph(name=f'random-directed-{node_count}-{seed}')
    for node in range(node_count):
        graph.add_node(node, pos=[-73.0 + node * 0.001, 40.0 + (node % 7) * 0.001])

    for source in range(node_count):
        target = (source + 1) % node_count
        graph.add_edge(source, target, weight=1.0 + random_state.random())

    for source in range(node_count):
        for target in range(node_count):
            if source == target or target == (source + 1) % node_count:
                continue
            if random_state.random() < extra_edge_probability:
                graph.add_edge(source, target, weight=0.1 + 9.9 * random_state.random())
    return graph


def build_h2h_acceptance_graphs() -> dict[str, nx.Graph]:
    """
    构造 Python 参考实现和 C++ 原生核心共用的人工正确性图。

    输入：无。
    输出：名称到强连通有向图的映射，覆盖单点、非对称、环、zero 和 shortcut。
    """
    def add_nodes(graph: nx.DiGraph, count: int) -> nx.DiGraph:
        """向人工图加入 `[0, count)` 连续节点和占位坐标，并返回原图。"""
        graph.add_nodes_from((node, {'pos': [float(node), 0.0]}) for node in range(count))
        return graph

    single = add_nodes(nx.DiGraph(), 1)

    asymmetric_two = add_nodes(nx.DiGraph(), 2)
    asymmetric_two.add_edge(0, 1, weight=1.25)
    asymmetric_two.add_edge(1, 0, weight=4.75)

    directed_cycle = add_nodes(nx.DiGraph(), 7)
    for node in range(7):
        directed_cycle.add_edge(node, (node + 1) % 7, weight=1.0 + node / 10.0)

    # 0 -> 3 的最短路必须经 1、2，消元后依靠 shortcut 才能保持距离 6。
    shortcut = add_nodes(nx.DiGraph(), 4)
    shortcut.add_weighted_edges_from((
        (0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0), (3, 0, 4.0),
        (1, 0, 8.0), (2, 1, 7.0), (3, 2, 6.0), (0, 3, 20.0),
        (0, 2, 9.0), (2, 0, 9.5), (1, 3, 8.5), (3, 1, 8.0),
    ))

    zero_weight = add_nodes(nx.DiGraph(), 3)
    zero_weight.add_weighted_edges_from((
        (0, 1, 0.0), (1, 2, 1.0), (2, 0, 2.0),
        (1, 0, 3.0), (2, 1, 4.0), (0, 2, 5.0),
    ))

    return {
        'single': single,
        'asymmetric-two': asymmetric_two,
        'directed-cycle': directed_cycle,
        'shortcut-required': shortcut,
        'zero-weight': zero_weight,
        'parallel-fixed-20': build_fixed_20_node_graph(),
    }


def build_legacy_distance(graph: nx.MultiDiGraph) -> dict[str, dict]:
    """
    按改造前实现物化卡车和无人机全对距离，作为冻结基线。

    输入：带边权与 `pos` 坐标的图。

    输出：
    - 包含 `truck` 和 `drone` 两个嵌套字典的旧式距离对象。
    """
    return {
        'truck': dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight')),
        'drone': {
            source: {
                target: haversine(graph.nodes[source]['pos'], graph.nodes[target]['pos'])
                for target in graph.nodes
            }
            for source in graph.nodes
        },
    }


def build_fixed_algorithm_case():
    """
    返回所有算法共用的固定 20 节点小实例。

    输入：无。

    输出：
    - `(graph, depots, cities)`；包含 2 个仓库和 2 个客户。
    实现逻辑：
    - 距离层夹具保留 MultiDiGraph 的平行边与自环；算法层则按当前 LRMP 的边访问
      约定转换成简单 DiGraph，同一方向只保留最小权重。
    - 仓库和客户使用 NumPy 数组，兼容现有 MST 分组中的向量比较逻辑。
    """
    multi_graph = build_fixed_20_node_graph()
    graph = nx.DiGraph(name='fixed-algorithm-directed-20')
    graph.add_nodes_from(multi_graph.nodes(data=True))
    for source, target, edge_data in multi_graph.edges(data=True):
        weight = float(edge_data['weight'])
        if not graph.has_edge(source, target) or weight < graph.edges[source, target]['weight']:
            graph.add_edge(source, target, weight=weight)

    # 现有算法用 NumPy 向量定位节点所属仓库分组，不能传普通 Python 列表。
    depots = np.asarray([0, 10], dtype=int)
    cities = np.asarray([3, 16], dtype=int)
    return graph, depots, cities
