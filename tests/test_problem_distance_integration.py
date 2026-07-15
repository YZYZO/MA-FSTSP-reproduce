"""验证 `problem.small_instance` 已接入统一距离工厂且不会触碰真实数据集。"""

import math
import unittest
from unittest.mock import patch

from distance_oracle import EagerDistanceMatrix, GeographicDistanceMatrix
from problem import small_instance
from tests.h2h_test_fixtures import build_fixed_20_node_graph


class ProblemDistanceIntegrationTests(unittest.TestCase):
    """覆盖阶段 1 在现有问题生成入口中的最小接入范围。"""

    def test_small_instance_returns_distance_provider(self):
        """
        用内存固定图替换 Manhattan 加载，并检查小图入口返回新的只读距离对象。

        输入：无外部文件；通过 mock 注入 20 节点固定图。
        输出：无显式返回值；断言卡车、无人机后端类型与查询结果。
        """
        fixed_graph = build_fixed_20_node_graph()
        with patch('problem.manhattan', return_value=fixed_graph):
            graph, depots, cities, distance = small_instance(1, 10, 2, 2)

        self.assertEqual(graph.number_of_nodes(), 10)
        self.assertEqual(len(depots), 1)
        self.assertEqual(len(cities), 1)
        self.assertIsInstance(distance['truck'], EagerDistanceMatrix)
        self.assertIsInstance(distance['drone'], GeographicDistanceMatrix)

        source, target = list(graph.nodes)[:2]
        self.assertTrue(math.isfinite(distance['truck'][source][target]))
        self.assertTrue(math.isfinite(distance['drone'][source][target]))


if __name__ == '__main__':
    unittest.main()
