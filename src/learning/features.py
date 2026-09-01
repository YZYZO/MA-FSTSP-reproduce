"""客户和客户组的求解器感知特征。"""

import math

import networkx as nx
import numpy as np


CUSTOMER_FEATURE_NAMES = (
    "x",
    "y",
    "candidate_count",
    "nearest_depot_out_distance",
    "second_depot_out_distance",
    "nearest_depot_in_distance",
    "nearest_depot_drone_distance",
    "mean_road_asymmetry",
)

GROUP_FEATURE_NAMES = (
    "customer_count",
    "candidate_total",
    "candidate_square_sum",
    "set_tsp_complexity_proxy",
    "candidate_mean",
    "candidate_max",
    "candidate_variance",
    "truck_out_mean",
    "truck_out_max",
    "truck_in_mean",
    "truck_in_max",
    "road_asymmetry_mean",
    "drone_distance_mean",
    "serviceable_customer_ratio",
    "symmetric_mst_length",
    "centroid_dispersion",
    "bounding_box_diagonal",
    "depot_to_centroid",
)


def _mean(values):
    """
    返回数值序列均值，空序列返回零。

    输入：数值序列。
    输出：浮点均值或零。
    逻辑：统一处理空客户组，避免在特征调用处重复分支。
    """
    return float(np.mean(values)) if len(values) > 0 else 0.0


def _max(values):
    """
    返回数值序列最大值，空序列返回零。

    输入：数值序列。
    输出：浮点最大值或零。
    逻辑：为所有客户组生成固定长度特征。
    """
    return float(np.max(values)) if len(values) > 0 else 0.0


def _node_position(graph, node):
    """
    读取路网节点的二维坐标。

    输入：路网图和节点编号。
    输出：长度为 2 的 NumPy 浮点数组。
    逻辑：沿用仓库现有统一 `pos=[x,y]` 节点属性。
    """
    return np.asarray(graph.nodes[node]["pos"], dtype=float)


def extract_customer_feature_dict(graph, distance, city, depots, convex_sets):
    """
    提取一个客户相对于全部仓库的基础学习特征。

    输入：路网、距离字典、客户、仓库集合和客户候选区域。
    输出：按 `CUSTOMER_FEATURE_NAMES` 命名的特征字典。
    逻辑：同时保留进出道路距离和不对称程度，避免丢失有向路网信息。
    """
    position = _node_position(graph, city)
    outgoing = sorted(float(distance["truck"][depot][city]) for depot in depots)
    incoming = sorted(float(distance["truck"][city][depot]) for depot in depots)
    drone_distances = sorted(float(distance["drone"][depot][city]) for depot in depots)
    asymmetries = [
        abs(distance["truck"][depot][city] - distance["truck"][city][depot])
        / max(
            (distance["truck"][depot][city] + distance["truck"][city][depot]) / 2.0,
            1e-12,
        )
        for depot in depots
    ]
    second_outgoing = outgoing[1] if len(outgoing) > 1 else outgoing[0]
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "candidate_count": float(len(convex_sets[city])),
        "nearest_depot_out_distance": outgoing[0],
        "second_depot_out_distance": second_outgoing,
        "nearest_depot_in_distance": incoming[0],
        "nearest_depot_drone_distance": drone_distances[0],
        "mean_road_asymmetry": _mean(asymmetries),
    }


def extract_customer_feature_matrix(graph, distance, cities, depots, convex_sets):
    """
    按输入客户顺序生成固定列顺序的客户特征矩阵。

    输入：路网、距离、客户、仓库和候选集合。
    输出：形状为 `[客户数, 特征数]` 的 NumPy 数组。
    逻辑：逐客户提取命名特征，再按公开常量列顺序组装。
    """
    rows = []
    for city in cities:
        feature_dict = extract_customer_feature_dict(
            graph,
            distance,
            city,
            depots,
            convex_sets,
        )
        rows.append([feature_dict[name] for name in CUSTOMER_FEATURE_NAMES])
    return np.asarray(rows, dtype=float).reshape(-1, len(CUSTOMER_FEATURE_NAMES))


def _symmetric_mst_length(depot, customers, truck_distance):
    """
    计算仓库和客户节点上的双向均值 MST 长度。

    输入：仓库、客户列表和有向卡车距离。
    输出：无向均值图的最小生成树总长度。
    逻辑：每对节点只添加一次，边权取两个方向平均值。
    """
    nodes = [depot] + list(customers)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for index, start in enumerate(nodes):
        for end in nodes[index + 1:]:
            graph.add_edge(
                start,
                end,
                weight=(
                    truck_distance[start][end]
                    + truck_distance[end][start]
                )
                / 2.0,
            )
    tree = nx.minimum_spanning_tree(graph, weight="weight")
    return float(tree.size(weight="weight"))


def extract_group_feature_dict(graph, distance, depot, customers, convex_sets):
    """
    提取一个仓库客户组的结构、路网、几何和规模特征。

    输入：路网、距离、仓库、客户列表和候选集合。
    输出：按 `GROUP_FEATURE_NAMES` 命名的浮点特征字典。
    逻辑：重点计算候选节点组合规模代理，并保留有向距离不对称性。
    """
    customers = list(customers)
    customer_count = len(customers)
    candidate_sizes = [len(convex_sets[city]) for city in customers]
    candidate_total = sum(candidate_sizes)
    candidate_square_sum = sum(size ** 2 for size in candidate_sizes)

    # 该规模代理对应 Set-TSP 的集合层、集合内和集合间变量数量主项。
    complexity_proxy = (
        (customer_count + 1) ** 2
        + candidate_square_sum
        + candidate_total ** 2
    )

    truck_out = [float(distance["truck"][depot][city]) for city in customers]
    truck_in = [float(distance["truck"][city][depot]) for city in customers]
    road_asymmetry = [
        abs(outgoing - incoming) / max((outgoing + incoming) / 2.0, 1e-12)
        for outgoing, incoming in zip(truck_out, truck_in)
    ]
    drone_distances = [float(distance["drone"][depot][city]) for city in customers]

    depot_position = _node_position(graph, depot)
    customer_positions = np.asarray(
        [_node_position(graph, city) for city in customers],
        dtype=float,
    ).reshape(-1, 2)
    if customer_count > 0:
        centroid = customer_positions.mean(axis=0)
        centroid_dispersion = float(
            np.linalg.norm(customer_positions - centroid, axis=1).mean()
        )
        bounding_box_diagonal = float(
            np.linalg.norm(
                customer_positions.max(axis=0) - customer_positions.min(axis=0)
            )
        )
        depot_to_centroid = float(np.linalg.norm(depot_position - centroid))
    else:
        centroid_dispersion = 0.0
        bounding_box_diagonal = 0.0
        depot_to_centroid = 0.0

    return {
        "customer_count": float(customer_count),
        "candidate_total": float(candidate_total),
        "candidate_square_sum": float(candidate_square_sum),
        "set_tsp_complexity_proxy": float(complexity_proxy),
        "candidate_mean": _mean(candidate_sizes),
        "candidate_max": _max(candidate_sizes),
        "candidate_variance": float(np.var(candidate_sizes)) if candidate_sizes else 0.0,
        "truck_out_mean": _mean(truck_out),
        "truck_out_max": _max(truck_out),
        "truck_in_mean": _mean(truck_in),
        "truck_in_max": _max(truck_in),
        "road_asymmetry_mean": _mean(road_asymmetry),
        "drone_distance_mean": _mean(drone_distances),
        "serviceable_customer_ratio": (
            sum(size > 0 for size in candidate_sizes) / customer_count
            if customer_count > 0
            else 0.0
        ),
        "symmetric_mst_length": _symmetric_mst_length(
            depot,
            customers,
            distance["truck"],
        ),
        "centroid_dispersion": centroid_dispersion,
        "bounding_box_diagonal": bounding_box_diagonal,
        "depot_to_centroid": depot_to_centroid,
    }


def extract_group_feature_vector(graph, distance, depot, customers, convex_sets):
    """
    将命名客户组特征转换成固定顺序向量。

    输入：与 `extract_group_feature_dict` 相同。
    输出：长度为 `len(GROUP_FEATURE_NAMES)` 的 NumPy 数组。
    逻辑：代理模型只依赖公开列顺序，便于数据集和训练代码复用。
    """
    feature_dict = extract_group_feature_dict(
        graph,
        distance,
        depot,
        customers,
        convex_sets,
    )
    return np.asarray([feature_dict[name] for name in GROUP_FEATURE_NAMES], dtype=float)

