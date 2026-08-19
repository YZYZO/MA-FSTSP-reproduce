"""提供显式对称化的 MST 和超级根多仓库生成森林分区。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx
import numpy as np

from utils import mst_partition


def _combine_directions(forward: float, backward: float, strategy: str) -> float:
    """
    将两个方向的集合转移代价合成为无向边权。

    输入：正向代价、反向代价和合并策略。
    输出：无向边权。

    `sum` 只用于验证它与 `mean` 的常数缩放等价性，正式实验默认使用 `mean`。
    """
    forward = float(forward)
    backward = float(backward)
    if strategy == 'mean':
        return (forward + backward) / 2.0
    if strategy == 'sum':
        return forward + backward
    if strategy == 'max':
        return max(forward, backward)
    if strategy == 'min':
        return min(forward, backward)
    raise ValueError(f'不支持的有向代价对称化策略：{strategy!r}。')


def build_symmetric_graph(
    nodes: Sequence,
    directed_costs: Mapping,
    strategy: str = 'mean',
) -> nx.Graph:
    """
    从完整有向代价矩阵构造无向完全图。

    输入：节点序列、有向代价矩阵和对称化策略。
    输出：带 `weight` 属性的 `networkx.Graph`。
    """
    ordered_nodes = list(nodes)
    graph = nx.Graph()
    graph.add_nodes_from(ordered_nodes)
    for index, start in enumerate(ordered_nodes):
        for end in ordered_nodes[index + 1:]:
            graph.add_edge(
                start,
                end,
                weight=_combine_directions(
                    directed_costs[start][end],
                    directed_costs[end][start],
                    strategy,
                ),
            )
    return graph


def partition_corrected_mst(
    directed_costs: Mapping,
    depots: Sequence,
    cities: Sequence,
    strategy: str = 'mean',
) -> dict[Any, list[Any]]:
    """
    使用显式对称化边权运行原论文的 MST+树形 DP 分区。

    输入：完整有向代价、仓库、客户和对称化策略。
    输出：仓库到客户列表的映射。

    该函数复用原 `mst_partition`，目的是只隔离“有向覆盖修正”的影响。
    """
    depot_array = np.asarray(list(depots))
    city_array = np.asarray(list(cities))
    graph = build_symmetric_graph(
        list(depot_array) + list(city_array),
        directed_costs,
        strategy,
    )
    return mst_partition(graph, depot_array, city_array)


def partition_rooted_msf(
    directed_costs: Mapping,
    depots: Sequence,
    cities: Sequence,
    strategy: str = 'mean',
) -> dict[Any, list[Any]]:
    """
    使用超级根构造每个分量恰含一个仓库的最小生成森林。

    输入：完整有向代价、仓库、客户和对称化策略。
    输出：仓库到客户列表的映射。

    实现逻辑：
    1. 对仓库—客户、客户—客户边进行显式对称化；不加入仓库—仓库边。
    2. 用严格小于普通边的常数连接超级根与全部仓库。
    3. 求一次 MST 并删除超级根，得到多仓库根森林。
    4. 强制校验每个分量恰好包含一个仓库。
    """
    depot_list = list(depots)
    city_list = list(cities)
    if not depot_list:
        raise ValueError('超级根 MSF 至少需要一个仓库。')

    nodes = depot_list + city_list
    symmetric = build_symmetric_graph(nodes, directed_costs, strategy)

    # 仓库之间通过超级根连接；删除直接仓库边，防止其参与森林目标。
    for index, depot in enumerate(depot_list):
        for other in depot_list[index + 1:]:
            if symmetric.has_edge(depot, other):
                symmetric.remove_edge(depot, other)

    ordinary_weights = [
        float(data['weight'])
        for _, _, data in symmetric.edges(data=True)
    ]
    minimum_weight = min(ordinary_weights, default=0.0)
    root_weight = min(0.0, minimum_weight) - 1.0
    super_root = object()
    symmetric.add_node(super_root)
    for depot in depot_list:
        symmetric.add_edge(super_root, depot, weight=root_weight)

    tree = nx.minimum_spanning_tree(symmetric, weight='weight', algorithm='kruskal')
    forest = tree.copy()
    forest.remove_node(super_root)

    depot_set = set(depot_list)
    city_to_depot: dict[Any, Any] = {}
    for component in nx.connected_components(forest):
        component_depots = depot_set.intersection(component)
        if len(component_depots) != 1:
            raise RuntimeError(
                '超级根 MSF 生成了不合法分量：'
                f'节点={list(component)!r}，仓库={list(component_depots)!r}。'
            )
        component_depot = next(iter(component_depots))
        for city in component:
            if city not in depot_set:
                city_to_depot[city] = component_depot

    if set(city_to_depot) != set(city_list):
        missing = set(city_list) - set(city_to_depot)
        raise RuntimeError(f'超级根 MSF 未覆盖全部客户：{list(missing)!r}。')

    # 按原客户顺序输出，减少下游 Set-TSP 的非必要索引差异。
    groups = {depot: [] for depot in depot_list}
    for city in city_list:
        groups[city_to_depot[city]].append(city)
    return groups

