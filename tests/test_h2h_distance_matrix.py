"""验证生产 H2HDistanceMatrix、批量接口、统计与距离工厂。"""

import tempfile
import unittest

import networkx as nx
import numpy as np

from distance_oracle import UnsupportedDistanceOperation, build_distance_provider
from h2h_backend import H2HDistanceMatrix, ensure_h2h_index
from tests.h2h_native_test_utils import ensure_native_built
from tests.h2h_test_fixtures import build_fixed_20_node_graph


class H2HDistanceMatrixTests(unittest.TestCase):
    """逐项确认双下标兼容层不会物化标签或放宽只读语义。"""

    @classmethod
    def setUpClass(cls):
        """确保 Release 原生产物存在。"""
        cls.builder_path, cls.library_path = ensure_native_built('release')

    def test_scalar_batch_stats_close_and_read_only(self):
        """标量、批量、统计、关闭后重开和只读异常必须同时工作。"""
        graph = build_fixed_20_node_graph()
        expected = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
        with tempfile.TemporaryDirectory(prefix='h2h-matrix-') as temp:
            cache = ensure_h2h_index(
                graph, 'matrix', index_dir=temp, builder_path=self.builder_path
            )
            matrix = H2HDistanceMatrix(
                cache.index_path,
                self.library_path,
                cache.node_count,
                cache.graph_hash,
                stats_enabled=True,
            )
            self.assertIsNone(matrix._handle)
            value = matrix[np.int64(0)][np.int32(10)]
            self.assertIs(type(value), float)
            self.assertAlmostEqual(value, expected[0][10], places=10)
            self.assertIsNotNone(matrix._handle)

            sources = [0, 10, 7, 19]
            targets = [10, 0, 7, 3]
            batch = matrix.query_batch(sources, targets)
            for source, target, result in zip(sources, targets, batch):
                self.assertAlmostEqual(result, expected[source][target], places=10)
            self.assertEqual(matrix.statistics['query_count'], 5)

            with self.assertRaises(UnsupportedDistanceOperation):
                matrix[0][1] = 1.0
            with self.assertRaises(UnsupportedDistanceOperation):
                matrix.items()
            with self.assertRaisesRegex(KeyError, '999'):
                _ = matrix[999][0]

            matrix.close()
            self.assertIsNone(matrix._handle)
            self.assertAlmostEqual(matrix[10][0], expected[10][0], places=10)
            matrix.close()

    def test_factory_h2h_returns_native_truck_and_on_demand_drone(self):
        """显式 h2h 工厂应返回原生卡车矩阵和 O(n) 无人机坐标矩阵。"""
        graph = build_fixed_20_node_graph()
        with tempfile.TemporaryDirectory(prefix='h2h-factory-') as temp:
            provider = build_distance_provider(
                graph,
                backend='h2h',
                dataset_name='factory',
                index_dir=temp,
                builder_path=str(self.builder_path),
                library_path=str(self.library_path),
            )
            self.assertIsInstance(provider['truck'], H2HDistanceMatrix)
            self.assertFalse(hasattr(provider['drone'], '_distances'))
            expected = nx.dijkstra_path_length(graph, 0, 10, weight='weight')
            self.assertAlmostEqual(provider['truck'][0][10], expected, places=10)
            provider['truck'].close()


if __name__ == '__main__':
    unittest.main()
