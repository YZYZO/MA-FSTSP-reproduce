"""验证阶段 0 固定图夹具与本地真实图规模基线。"""

import unittest
from pathlib import Path

import networkx as nx

from config import DATASETS_DIR
from tests.h2h_test_fixtures import (
    EXPECTED_LOCAL_GRAPH_STATS,
    build_fixed_20_node_graph,
    build_random_strongly_connected_digraph,
)


class GraphFixtureBaselineTests(unittest.TestCase):
    """冻结后续 H2H 正确性测试依赖的图结构和真实图规模。"""

    def test_fixed_graph_is_directed_and_strongly_connected(self):
        """固定夹具必须包含 20 个节点，并保持有向强连通。"""
        graph = build_fixed_20_node_graph()
        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertEqual(graph.number_of_nodes(), 20)
        self.assertTrue(nx.is_strongly_connected(graph))
        self.assertGreater(graph.number_of_edges(0, 1), 1)
        self.assertTrue(graph.has_edge(7, 7))

    def test_fixed_graph_contains_asymmetric_shortest_distances(self):
        """固定图至少应有一个节点对满足正反方向最短距离不同。"""
        graph = build_fixed_20_node_graph()
        forward = nx.dijkstra_path_length(graph, 0, 10, weight='weight')
        backward = nx.dijkstra_path_length(graph, 10, 0, weight='weight')
        self.assertNotAlmostEqual(forward, backward, places=12)

    def test_random_fixture_is_reproducible_and_strongly_connected(self):
        """相同种子必须生成相同边表，且随机夹具始终强连通。"""
        first = build_random_strongly_connected_digraph(30, seed=260715)
        second = build_random_strongly_connected_digraph(30, seed=260715)
        first_edges = list(first.edges(keys=True, data='weight'))
        second_edges = list(second.edges(keys=True, data='weight'))
        self.assertEqual(first_edges, second_edges)
        self.assertTrue(nx.is_strongly_connected(first))

    def test_local_graph_sizes_match_recorded_baseline(self):
        """若本地真实图存在，记录并验证原始规模与最大强连通分量规模。"""
        missing = [name for name in EXPECTED_LOCAL_GRAPH_STATS if not (DATASETS_DIR / name).is_file()]
        if missing:
            self.skipTest(f'本地真实图不存在：{missing}')

        for name, expected in EXPECTED_LOCAL_GRAPH_STATS.items():
            with self.subTest(dataset=name):
                graph_path = Path(DATASETS_DIR) / name
                graph = nx.MultiDiGraph(nx.read_graphml(graph_path))
                largest_scc = max(nx.strongly_connected_components(graph), key=len)
                self.assertEqual(graph.number_of_nodes(), expected['raw_nodes'])
                self.assertEqual(graph.number_of_edges(), expected['raw_edges'])
                self.assertEqual(len(largest_scc), expected['largest_scc_nodes'])


if __name__ == '__main__':
    unittest.main()
