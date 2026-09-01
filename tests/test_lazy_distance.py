"""大型路网按需距离矩阵的独立兼容性测试。"""

import unittest

import networkx as nx

from src.learning.lazy_distance import build_lazy_distance
from utils import haversine


class LazyDistanceTest(unittest.TestCase):
    """验证按需道路距离、无人机距离和 LRU 行缓存。"""

    def _build_graph(self):
        """
        构造三节点带方向差异的强连通图。

        输入：无。
        输出：带 `pos` 和 `weight` 的 `DiGraph`。
        逻辑：0→2 的最短路应经过节点 1，便于验证 Dijkstra 语义。
        """
        graph = nx.DiGraph()
        graph.add_node(0, pos=[0.0, 0.0])
        graph.add_node(1, pos=[0.01, 0.0])
        graph.add_node(2, pos=[0.02, 0.0])
        graph.add_weighted_edges_from(
            [
                (0, 1, 1.0),
                (1, 2, 2.0),
                (0, 2, 10.0),
                (1, 0, 1.5),
                (2, 1, 2.5),
                (2, 0, 8.0),
            ]
        )
        return graph

    def test_lazy_matrices_match_networkx_and_haversine(self):
        """
        验证嵌套索引结果与原距离定义一致。

        输入：三节点测试图。
        输出：卡车 0→2 为 3，无人机距离等于 `utils.haversine`。
        逻辑：直接使用现有算法要求的 `distance[type][u][v]` 语法。
        """
        graph = self._build_graph()
        distance = build_lazy_distance(graph, max_cached_truck_rows=2)
        self.assertAlmostEqual(distance["truck"][0][2], 3.0)
        self.assertAlmostEqual(
            distance["drone"][0][2],
            haversine(graph.nodes[0]["pos"], graph.nodes[2]["pos"]),
        )

    def test_truck_rows_use_bounded_lru_cache(self):
        """
        验证查询超过上限的源节点时只保留指定行数。

        输入：最大两行缓存，依次查询三个不同源。
        输出：Dijkstra 调用三次，缓存行数保持二。
        逻辑：这是避免 55K 路网再次退化成 N² 内存的关键保证。
        """
        distance = build_lazy_distance(self._build_graph(), max_cached_truck_rows=2)
        _ = distance["truck"][0][2]
        _ = distance["truck"][1][2]
        _ = distance["truck"][2][0]
        telemetry = distance["truck"].telemetry()
        self.assertEqual(telemetry["dijkstra_call_count"], 3)
        self.assertEqual(telemetry["cached_row_count"], 2)
        distance["truck"].clear_cache()
        telemetry = distance["truck"].telemetry()
        self.assertEqual(telemetry["dijkstra_call_count"], 0)
        self.assertEqual(telemetry["cached_row_count"], 0)


if __name__ == "__main__":
    unittest.main()
