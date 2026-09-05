"""分区修复测试专用的小型人工有向路网。"""

import math

import networkx as nx
import numpy as np

from src.fstsp import MultiAgentFlyingSidekickTSP


def tiny_model(customer_count=6, depot_count=3):
    """输入客户/仓库数，生成小型完整有向图及精确距离，输出不依赖外部地图的模型。"""
    count = customer_count + depot_count
    graph = nx.MultiDiGraph()
    for node in range(count):
        graph.add_node(node, pos=(node % 4 * .002, node // 4 * .002))
    drone = {first: {} for first in graph}
    for first in graph:
        for second in graph:
            a, b = graph.nodes[first]['pos'], graph.nodes[second]['pos']
            flight = math.hypot(a[0] - b[0], a[1] - b[1]) * 100
            drone[first][second] = flight
            if first != second:
                graph.add_edge(first, second, weight=flight * (1.2 if first < second else 1.5))
    truck = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
    return MultiAgentFlyingSidekickTSP(
        graph, np.arange(depot_count), np.arange(depot_count, count),
        {'truck': truck, 'drone': drone}, drone=2, limit=.3,
    )
