"""验证阶段 1 距离代理、eager 后端、按需无人机后端与工厂行为。"""

import unittest

import networkx as nx
import numpy as np

from config import EAGER_DISTANCE_MAX_NODES
from distance_oracle import (
    EagerDistanceMatrix,
    GeographicDistanceMatrix,
    UnsupportedDistanceOperation,
    build_distance_provider,
)
from tests.h2h_test_fixtures import build_fixed_20_node_graph, build_legacy_distance


class DistanceOracleTests(unittest.TestCase):
    """逐项比较新距离接口与改造前嵌套字典的结果。"""

    @classmethod
    def setUpClass(cls):
        """为本测试类只构造一次固定图、旧距离和新距离提供器。"""
        cls.graph = build_fixed_20_node_graph()
        cls.legacy = build_legacy_distance(cls.graph)
        cls.provider = build_distance_provider(cls.graph, backend='auto', dataset_name='fixed-20')

    def test_eager_truck_distances_match_legacy_for_all_pairs(self):
        """eager 卡车后端必须与旧全对 Dijkstra 在所有节点对上逐项一致。"""
        truck = self.provider['truck']
        self.assertIsInstance(truck, EagerDistanceMatrix)
        for source in self.graph.nodes:
            for target in self.graph.nodes:
                self.assertAlmostEqual(
                    truck[source][target],
                    self.legacy['truck'][source][target],
                    places=12,
                )

    def test_geographic_distances_match_legacy_without_pairwise_storage(self):
        """无人机后端必须逐项一致，并且内部只保存 O(n) 坐标。"""
        drone = self.provider['drone']
        self.assertIsInstance(drone, GeographicDistanceMatrix)
        self.assertEqual(len(drone._coordinates), self.graph.number_of_nodes())
        self.assertFalse(hasattr(drone, '_distances'))
        for source in self.graph.nodes:
            for target in self.graph.nodes:
                self.assertAlmostEqual(
                    drone[source][target],
                    self.legacy['drone'][source][target],
                    places=12,
                )

    def test_numpy_integer_nodes_return_python_float(self):
        """NumPy 整数节点编号应可查询，公开结果必须是 Python `float`。"""
        truck_value = self.provider['truck'][np.int64(0)][np.int32(10)]
        drone_value = self.provider['drone'][np.int64(0)][np.int32(10)]
        self.assertIs(type(truck_value), float)
        self.assertIs(type(drone_value), float)

    def test_unknown_nodes_raise_clear_key_error(self):
        """未知源点或终点不能进入后端，应直接抛出包含节点编号的错误。"""
        with self.assertRaisesRegex(KeyError, '999'):
            _ = self.provider['truck'][999][0]
        with self.assertRaisesRegex(KeyError, '999'):
            _ = self.provider['drone'][0][999]

    def test_distance_objects_are_read_only_and_not_iterable(self):
        """赋值、删除、items 和遍历都必须明确拒绝，不能隐式物化全矩阵。"""
        truck = self.provider['truck']
        row = truck[0]
        with self.assertRaises(UnsupportedDistanceOperation):
            row[1] = 1.0
        with self.assertRaises(UnsupportedDistanceOperation):
            del row[1]
        with self.assertRaises(UnsupportedDistanceOperation):
            row.items()
        with self.assertRaises(UnsupportedDistanceOperation):
            iter(truck)

    def test_disconnected_eager_query_reports_unreachable_pair(self):
        """eager 后端遇到不可达节点对时必须给出明确异常。"""
        graph = nx.DiGraph()
        graph.add_node(0, pos=[0.0, 0.0])
        graph.add_node(1, pos=[1.0, 1.0])
        matrix = EagerDistanceMatrix.from_graph(graph)
        with self.assertRaisesRegex(ValueError, '无法到达'):
            _ = matrix[0][1]

    def test_factory_validates_backend_and_large_graph_policy(self):
        """工厂应拒绝未知后端，并阻止大图 eager 或静默 Dijkstra 回退。"""
        with self.assertRaisesRegex(ValueError, '未知距离后端'):
            build_distance_provider(self.graph, backend='unknown')

        large_graph = nx.DiGraph()
        for node in range(EAGER_DISTANCE_MAX_NODES + 1):
            large_graph.add_node(node, pos=[float(node), 0.0])
        with self.assertRaisesRegex(ValueError, '超过 eager 上限'):
            build_distance_provider(large_graph, backend='eager', dataset_name='too-large')
        # 阶段 4 已接入 H2H；该人工大图不强连通，应在 builder 前明确拒绝，
        # 同时证明不会静默执行逐次 Dijkstra。
        with self.assertRaisesRegex(ValueError, '强连通'):
            build_distance_provider(large_graph, backend='auto', dataset_name='needs-h2h')


if __name__ == '__main__':
    unittest.main()
