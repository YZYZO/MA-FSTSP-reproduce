"""对 Python 有向 H2H 参考实现执行小图全对最短路验收。"""

import unittest

import networkx as nx
import numpy as np

from config import H2H_REFERENCE_MAX_NODES
from h2h_reference import DirectedH2HReference
from tests.h2h_test_fixtures import (
    build_fixed_20_node_graph,
    build_h2h_acceptance_graphs,
    build_random_strongly_connected_digraph,
)


class H2HReferenceCorrectnessTests(unittest.TestCase):
    """验证有向标签公式、分解树不变量和参考实现的范围保护。"""

    def assert_all_pairs_match_dijkstra(self, graph: nx.Graph) -> DirectedH2HReference:
        """
        对一张图逐项比较参考 H2H 与 NetworkX Dijkstra。

        输入：不超过参考上限的强连通有向图。
        输出：构建完成的参考索引，便于调用方继续检查统计。
        """
        index = DirectedH2HReference(graph)
        expected = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
        for source in graph.nodes:
            for target in graph.nodes:
                self.assertAlmostEqual(
                    index.query(source, target),
                    expected[source][target],
                    places=10,
                    msg=f'节点对 {source} -> {target} 不一致',
                )
        return index

    def test_required_handcrafted_graphs_match_all_pairs(self):
        """单点、非对称、环、平行边、零权和 shortcut 图必须全部逐项一致。"""
        for name, graph in build_h2h_acceptance_graphs().items():
            with self.subTest(graph=name):
                index = self.assert_all_pairs_match_dijkstra(graph)
                self.assertEqual(index.stats.node_count, graph.number_of_nodes())
                self.assertEqual(len(index.parent), graph.number_of_nodes())
                self.assertEqual(sum(node == parent for node, parent in enumerate(index.parent)), 1)

    def test_random_sparse_directed_graphs_match_all_pairs(self):
        """多个随机种子和规模下的稀疏强连通图必须通过全对验证。"""
        for node_count in (10, 30, 60):
            for seed in (260715, 260716, 260717):
                with self.subTest(node_count=node_count, seed=seed):
                    graph = build_random_strongly_connected_digraph(node_count, seed=seed)
                    self.assert_all_pairs_match_dijkstra(graph)

    def test_decomposition_bags_map_to_ancestor_chain(self):
        """每个 bag 位置必须唯一映射到本节点的祖先链，父 rank 必须更高。"""
        index = DirectedH2HReference(build_fixed_20_node_graph())
        for node in range(index.node_count):
            self.assertEqual(index.positions[node], sorted(set(index.positions[node])))
            self.assertIn(index.depth[node], index.positions[node])
            if node != index.root:
                self.assertGreater(index.rank[index.parent[node]], index.rank[node])
            for bag_node in [node] + index.bag_neighbors[node]:
                cursor = node
                while index.depth[cursor] > index.depth[bag_node]:
                    cursor = index.parent[cursor]
                self.assertEqual(cursor, bag_node)

    def test_query_accepts_numpy_integer_and_batch(self):
        """标量与批量接口应返回 Python float，并接受 NumPy 整数节点。"""
        index = DirectedH2HReference(build_fixed_20_node_graph())
        value = index.query(np.int64(0), np.int32(10))
        self.assertIs(type(value), float)
        batch = index.query_batch([0, 10, 7], [10, 0, 7])
        self.assertEqual(len(batch), 3)
        self.assertTrue(all(type(item) is float for item in batch))
        with self.assertRaisesRegex(ValueError, '长度必须一致'):
            index.query_batch([0], [1, 2])

    def test_reference_size_limit_is_checked_before_build(self):
        """超过 200 节点时必须立即拒绝，不能开始消元或退化成大图实现。"""
        graph = nx.DiGraph()
        graph.add_nodes_from(range(H2H_REFERENCE_MAX_NODES + 1))
        with self.assertRaisesRegex(ValueError, '最多允许'):
            DirectedH2HReference(graph)

    def test_invalid_graphs_report_clear_errors(self):
        """非连续编号、负权和非强连通输入必须给出明确异常。"""
        non_contiguous = nx.DiGraph()
        non_contiguous.add_weighted_edges_from(((1, 2, 1.0), (2, 1, 1.0)))
        with self.assertRaisesRegex(ValueError, '连续编号'):
            DirectedH2HReference(non_contiguous)

        negative = nx.DiGraph()
        negative.add_nodes_from((node, {'pos': [float(node), 0.0]}) for node in range(2))
        negative.add_weighted_edges_from(((0, 1, -1.0), (1, 0, 1.0)))
        with self.assertRaisesRegex(ValueError, '负权'):
            DirectedH2HReference(negative)

        disconnected = nx.DiGraph()
        disconnected.add_nodes_from((node, {'pos': [float(node), 0.0]}) for node in range(2))
        disconnected.add_edge(0, 1, weight=1.0)
        with self.assertRaisesRegex(ValueError, '强连通'):
            DirectedH2HReference(disconnected)


if __name__ == '__main__':
    unittest.main()
