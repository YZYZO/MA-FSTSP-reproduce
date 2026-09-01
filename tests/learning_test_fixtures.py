"""学习型客户划分测试使用的小型确定性路网夹具。"""

import math

import networkx as nx
import numpy as np

from src.fstsp import MultiAgentFlyingSidekickTSP


def build_directed_road_fixture():
    """
    构造一个带明显方向不对称性的六节点完全路网。

    输入：无。
    输出：`(graph, depots, cities, distance, convex_sets)`。
    逻辑：卡车距离按节点方向加入不同倍率，无人机距离保持欧氏对称。
    """
    positions = {
        0: (0.0, 0.0),
        1: (1.6, 0.1),
        2: (0.3, 0.5),
        3: (1.1, 0.7),
        4: (0.8, 1.3),
        5: (1.7, 1.4),
    }
    graph = nx.DiGraph()
    for node, position in positions.items():
        graph.add_node(node, pos=list(position))

    truck_distance = {start: {} for start in positions}
    drone_distance = {start: {} for start in positions}
    for start, start_position in positions.items():
        for end, end_position in positions.items():
            euclidean_distance = math.dist(start_position, end_position)
            if start == end:
                truck_value = 0.0
            else:
                # 节点编号方向用于制造稳定且明显的有向道路差异。
                direction_factor = 1.0 if start < end else 1.45
                truck_value = euclidean_distance * direction_factor + 0.03 * start
                graph.add_edge(start, end, weight=truck_value)
            truck_distance[start][end] = truck_value
            drone_distance[start][end] = euclidean_distance

    # 原项目实例生成器返回 NumPy 数组，回归测试保持相同输入类型。
    depots = np.asarray([0, 1])
    cities = np.asarray([2, 3, 4, 5])
    convex_sets = {city: [city] for city in cities}
    distance = {"truck": truck_distance, "drone": drone_distance}
    return graph, depots, cities, distance, convex_sets


def build_test_model(city_count=3):
    """
    基于确定性路网创建一个小型原 MA-FSTSP 模型。

    输入：模型包含的客户数量。
    输出：`(model, convex_sets)`。
    逻辑：使用一架无人机和足够覆盖邻近节点的航程，供真实小型 DP 测试使用。
    """
    graph, depots, cities, distance, convex_sets = build_directed_road_fixture()
    selected_cities = cities[:city_count]
    model = MultiAgentFlyingSidekickTSP(
        graph,
        depots,
        selected_cities,
        distance,
        drone=1,
        limit=2.0,
        speed=1.6,
        theta=(0.5, 0.5),
    )
    return model, {city: convex_sets[city] for city in selected_cities}


def build_singleton_set_tsp_inputs(number_of_sets=4):
    """
    构造每个集合只包含一个候选节点的小型 Set-TSP 输入。

    输入：集合数量，其中集合 0 表示仓库。
    输出：`(convex_sets, distance, internal_distance)`。
    逻辑：使用带轻微方向差异的环形距离，便于快速获得可行整数解。
    """
    convex_sets = [[index] for index in range(number_of_sets)]
    distance = []
    for start in range(number_of_sets):
        start_rows = []
        for end in range(number_of_sets):
            if start == end:
                value = 0.0
            else:
                value = float(abs(start - end) + 1 + 0.1 * start)
            start_rows.append([[value]])
        distance.append(start_rows)
    internal_distance = [[[0.0]] for _ in range(number_of_sets)]
    return convex_sets, distance, internal_distance
