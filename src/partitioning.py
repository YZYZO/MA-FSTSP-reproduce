"""
为学习型客户划分提供独立的 MST 构图与分区实现。

本模块不修改论文原始 `MultiAgentFlyingSidekickTSP.set_mst`，而是提供：
1. 与原实现一致的 `legacy` 构图，用于回归对照；
2. 对有向道路距离显式对称化的 `mean/min/max` 构图；
3. 可直接供后续学习策略调用的客户划分入口。
"""

import math

import networkx as nx
import numpy as np

from utils import mst_partition


SYMMETRIC_EDGE_MODES = ("mean", "min", "max")


def _stable_node_key(node):
    """
    为仓库和客户节点生成稳定排序键。

    输入：任意可哈希节点编号。
    输出：由类型名和文本表示组成的元组。
    逻辑：不同输入顺序使用相同节点插入顺序，减少 MST 并列边的顺序影响。
    """
    return type(node).__name__, repr(node)


def _service_penalty(entity, candidate, depots, drone_distance, speed, coefficient):
    """
    计算实体通过候选道路节点接受无人机服务的近似代价。

    输入：实体、候选节点、仓库集合、无人机距离、速度和论文系数。
    输出：仓库返回零，客户返回飞行时间近似值。
    逻辑：仓库本身就是道路节点；客户需要计入候选点与客户之间的无人机航程。
    """
    if entity in depots:
        return 0.0
    return drone_distance[entity][candidate] / speed * coefficient


def _directed_pair_weight(
    start,
    end,
    regions,
    depots,
    truck_distance,
    drone_distance,
    speed,
    coefficient,
):
    """
    计算从一个仓库/客户实体到另一个实体的有向集合近似距离。

    输入：起止实体、候选区域、仓库集合及两类距离参数。
    输出：该方向上的最小近似代价。
    逻辑：先保留实体节点之间的纯卡车距离，再枚举两端候选点并加入无人机服务代价。
    """
    best_weight = float(truck_distance[start][end])
    for start_candidate in regions[start]:
        start_penalty = _service_penalty(
            start,
            start_candidate,
            depots,
            drone_distance,
            speed,
            coefficient,
        )
        for end_candidate in regions[end]:
            candidate_weight = (
                truck_distance[start_candidate][end_candidate]
                + start_penalty
                + _service_penalty(
                    end,
                    end_candidate,
                    depots,
                    drone_distance,
                    speed,
                    coefficient,
                )
            )
            best_weight = min(best_weight, float(candidate_weight))
    return best_weight


def _combine_directional_weights(forward_weight, backward_weight, edge_mode):
    """
    将两个方向的道路代价合并为无向 MST 边权。

    输入：正向代价、反向代价和 `mean/min/max` 模式。
    输出：合并后的浮点边权。
    逻辑：显式选择对称化方式，避免 `nx.Graph.add_edge` 隐式覆盖边权。
    """
    if edge_mode == "mean":
        return (forward_weight + backward_weight) / 2.0
    if edge_mode == "min":
        return min(forward_weight, backward_weight)
    if edge_mode == "max":
        return max(forward_weight, backward_weight)
    raise ValueError(f"unsupported symmetric edge mode: {edge_mode}")


def _build_legacy_partition_graph(
    depots,
    cities,
    convex_sets,
    truck_distance,
    drone_distance,
    speed,
    coefficient,
):
    """
    按论文仓库当前代码的循环顺序复现原始 MST 完全图。

    输入：仓库、客户、候选区域、距离、无人机速度和近似系数。
    输出：与原 `set_mst` 构图行为一致的 `networkx.Graph`。
    逻辑：保留重复添加无向边导致后写方向覆盖前写方向的历史语义，仅用于基线对照。
    """
    graph = nx.Graph()
    for depot in depots:
        graph.add_node(depot)
    for city in cities:
        graph.add_node(city)

    for depot in depots:
        for city in cities:
            weight = truck_distance[depot][city]
            for node in convex_sets[city]:
                weight = min(
                    weight,
                    truck_distance[depot][node]
                    + drone_distance[node][city] / speed * coefficient,
                )
            graph.add_edge(depot, city, weight=float(weight))
        for other_depot in depots:
            graph.add_edge(
                depot,
                other_depot,
                weight=float(truck_distance[depot][other_depot]),
            )

    for city in cities:
        for other_city in cities:
            weight = truck_distance[city][other_city]
            for node in convex_sets[city]:
                for other_node in convex_sets[other_city]:
                    weight = min(
                        weight,
                        truck_distance[node][other_node]
                        + drone_distance[city][node] / speed * coefficient
                        + drone_distance[other_city][other_node] / speed * coefficient,
                    )
            graph.add_edge(city, other_city, weight=float(weight))
    return graph


def _build_symmetric_partition_graph(
    depots,
    cities,
    convex_sets,
    truck_distance,
    drone_distance,
    speed,
    coefficient,
    edge_mode,
):
    """
    构造对客户输入顺序不敏感的有向距离对称化完全图。

    输入：仓库、客户、候选区域、距离参数和对称化模式。
    输出：不含自环、每对实体只添加一次的无向完全图。
    逻辑：分别计算每对实体的两个方向，再用 `mean/min/max` 显式合并。
    """
    ordered_depots = sorted(depots, key=_stable_node_key)
    ordered_cities = sorted(cities, key=_stable_node_key)
    entities = ordered_depots + ordered_cities

    # 每个仓库只有自身一个候选点；客户使用第一阶段生成的边界候选集合。
    regions = {
        **{depot: [depot] for depot in ordered_depots},
        **{city: list(convex_sets[city]) for city in ordered_cities},
    }
    depot_set = set(ordered_depots)

    graph = nx.Graph()
    graph.add_nodes_from(entities)
    for index, start in enumerate(entities):
        for end in entities[index + 1:]:
            forward_weight = _directed_pair_weight(
                start,
                end,
                regions,
                depot_set,
                truck_distance,
                drone_distance,
                speed,
                coefficient,
            )
            backward_weight = _directed_pair_weight(
                end,
                start,
                regions,
                depot_set,
                truck_distance,
                drone_distance,
                speed,
                coefficient,
            )
            graph.add_edge(
                start,
                end,
                weight=_combine_directional_weights(
                    forward_weight,
                    backward_weight,
                    edge_mode,
                ),
                forward_weight=forward_weight,
                backward_weight=backward_weight,
            )
    return graph


def build_partition_graph(
    depots,
    cities,
    convex_sets,
    truck_distance,
    drone_distance,
    speed,
    edge_mode="mean",
    coefficient=math.sqrt(2),
):
    """
    根据指定边权模式构造供 MST 客户划分使用的完全图。

    输入：仓库、客户、候选集合、卡车/无人机距离、速度、边权模式和近似系数。
    输出：带 `weight` 边属性的无向图。
    逻辑：`legacy` 复现原实现，其余模式显式合并双向道路代价。
    """
    if edge_mode == "legacy":
        return _build_legacy_partition_graph(
            depots,
            cities,
            convex_sets,
            truck_distance,
            drone_distance,
            speed,
            coefficient,
        )
    if edge_mode not in SYMMETRIC_EDGE_MODES:
        raise ValueError(
            f"edge_mode must be legacy or one of {SYMMETRIC_EDGE_MODES}, got {edge_mode}"
        )
    return _build_symmetric_partition_graph(
        depots,
        cities,
        convex_sets,
        truck_distance,
        drone_distance,
        speed,
        coefficient,
        edge_mode,
    )


def partition_customers(
    depots,
    cities,
    convex_sets,
    truck_distance,
    drone_distance,
    speed,
    edge_mode="mean",
    coefficient=math.sqrt(2),
):
    """
    使用指定 MST 边权模式完成客户到仓库的划分。

    输入：构图所需全部实例数据以及 `legacy/mean/min/max` 模式。
    输出：`{depot: [customers]}` 字典。
    逻辑：先构造完全图，再调用原仓库的 `mst_partition`，最后稳定排序各组客户。
    """
    graph = build_partition_graph(
        depots,
        cities,
        convex_sets,
        truck_distance,
        drone_distance,
        speed,
        edge_mode=edge_mode,
        coefficient=coefficient,
    )
    # 原仓库 `mst_partition` 使用 NumPy 向量比较定位仓库下标；这里统一转换，
    # 使学习模块既能接收项目实验中的数组，也能接收策略环境中的普通列表。
    depot_array = np.asarray(list(depots))
    city_array = np.asarray(list(cities))
    groups = mst_partition(graph, depot_array, city_array)
    return {
        depot: sorted(groups[depot], key=_stable_node_key)
        for depot in depots
    }
