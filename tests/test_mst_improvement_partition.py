"""测试有向集合代价对称化和超级根 MSF 的独立语义。"""

import unittest

import networkx as nx
import numpy as np

from src.mst_improvement.partition import (
    build_symmetric_graph,
    partition_corrected_mst,
    partition_rooted_msf,
)


def _directed_fixture():
    """
    构造两个仓库、两个客户的完整非对称代价矩阵。

    客户 2 明显靠近仓库 0，客户 3 明显靠近仓库 1，且不存在并列最优边。
    """
    nodes = [0, 1, 2, 3]
    costs = {start: {end: 0.0 for end in nodes} for start in nodes}
    undirected_base = {
        (0, 1): 7.0,
        (0, 2): 1.0,
        (0, 3): 9.0,
        (1, 2): 8.0,
        (1, 3): 1.5,
        (2, 3): 6.0,
    }
    for (start, end), value in undirected_base.items():
        costs[start][end] = value
        costs[end][start] = value + 0.2
    return costs


class PartitionTests(unittest.TestCase):
    """验证新分区函数不依赖无向图的覆盖顺序。"""

    def test_networkx_legacy_edge_is_overwritten(self):
        """证明原实现向无向图写入反向边时，后写入权重会覆盖前值。"""
        graph = nx.Graph()
        graph.add_edge(2, 3, weight=1.0)
        graph.add_edge(3, 2, weight=9.0)
        self.assertEqual(graph[2][3]['weight'], 9.0)

    def test_sum_and_mean_only_differ_by_constant_scale(self):
        """验证 sum 与 mean 的全部边权相差固定倍数，分区结果相同。"""
        costs = _directed_fixture()
        nodes = [0, 1, 2, 3]
        mean_graph = build_symmetric_graph(nodes, costs, 'mean')
        sum_graph = build_symmetric_graph(nodes, costs, 'sum')
        for start, end in mean_graph.edges:
            self.assertAlmostEqual(
                sum_graph[start][end]['weight'],
                2.0 * mean_graph[start][end]['weight'],
            )

        depots = np.asarray([0, 1])
        cities = np.asarray([2, 3])
        self.assertEqual(
            partition_rooted_msf(costs, depots, cities, 'mean'),
            partition_rooted_msf(costs, depots, cities, 'sum'),
        )

    def test_rooted_msf_assigns_every_city_once(self):
        """验证超级根森林的每个客户只属于一个预期仓库。"""
        groups = partition_rooted_msf(
            _directed_fixture(),
            np.asarray([0, 1]),
            np.asarray([2, 3]),
            'mean',
        )
        self.assertEqual(groups, {0: [2], 1: [3]})
        assigned = [city for cities in groups.values() for city in cities]
        self.assertCountEqual(assigned, [2, 3])

    def test_corrected_mst_is_stable_under_city_input_order(self):
        """验证显式对称化后，反转客户输入顺序不会改变客户归属。"""
        costs = _directed_fixture()
        forward = partition_corrected_mst(
            costs, np.asarray([0, 1]), np.asarray([2, 3]), 'mean'
        )
        reversed_order = partition_corrected_mst(
            costs, np.asarray([0, 1]), np.asarray([3, 2]), 'mean'
        )
        normalized_forward = {depot: set(cities) for depot, cities in forward.items()}
        normalized_reversed = {
            depot: set(cities) for depot, cities in reversed_order.items()
        }
        self.assertEqual(normalized_forward, normalized_reversed)


if __name__ == '__main__':
    unittest.main()

