"""阶段 6 算法验收：在冻结成本之外检查分组、端点、航程和有限解。"""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from collections import Counter

import numpy as np

from distance_oracle import build_distance_provider
from src.fstsp import MultiAgentFlyingSidekickTSP
from src.hc_vns import HillClimbingVariableNeighborhoodSearch
from src.lp import LinearProgramming
from src.lrmp import LinearRelaxedMasterProblem
from tests.h2h_native_test_utils import ensure_native_built
from tests.h2h_test_fixtures import build_fixed_algorithm_case


REGRESSION_RANDOM_SEED = 20260715


class H2HAlgorithmFeasibilityTests(unittest.TestCase):
    """确认 H2H 后端下的公开算法结果满足可检查的结构性约束。"""

    @classmethod
    def setUpClass(cls):
        """构建一份四种算法共享的生产 H2H 距离提供器。"""
        builder_path, library_path = ensure_native_built('release')
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix='h2h-feasibility-')
        graph, _, _ = build_fixed_algorithm_case()
        cls.distance = build_distance_provider(
            graph,
            backend='h2h',
            dataset_name='algorithm-feasibility',
            index_dir=cls.temporary_directory.name,
            builder_path=str(builder_path),
            library_path=str(library_path),
        )

    @classmethod
    def tearDownClass(cls):
        """关闭原生 mmap 并清理临时缓存。"""
        cls.distance['truck'].close()
        cls.temporary_directory.cleanup()

    def _assert_route_structure(self, solution, depots, limit):
        """
        检查每条公开路线的仓库端点和全部无人机 sortie 航程。

        输入：统一路线列表、仓库数组和无人机航程限制。
        输出：无；任一结构或航程违规时由 unittest 断言失败。
        """
        self.assertEqual(len(solution), len(depots))
        for depot, route in zip(depots, solution):
            truck_route = route['truck']
            self.assertGreaterEqual(len(truck_route), 2)
            self.assertEqual(int(truck_route[0]), int(depot))
            self.assertEqual(int(truck_route[-1]), int(depot))
            for drone_routes in route['drone']:
                for sortie in drone_routes:
                    self.assertEqual(len(sortie), 3)
                    launch, city, recovery = map(int, sortie)
                    flight_distance = (
                        self.distance['drone'][launch][city]
                        + self.distance['drone'][city][recovery]
                    )
                    self.assertLessEqual(flight_distance, limit + 1e-10)

    def test_main_and_hc_assign_each_customer_once_and_remain_feasible(self):
        """主算法与 HC 必须恰好分配客户一次并返回仓库闭合可行路线。"""
        for algorithm_class in (
            MultiAgentFlyingSidekickTSP,
            HillClimbingVariableNeighborhoodSearch,
        ):
            with self.subTest(algorithm=algorithm_class.__name__):
                random.seed(REGRESSION_RANDOM_SEED)
                np.random.seed(REGRESSION_RANDOM_SEED)
                graph, depots, cities = build_fixed_algorithm_case()
                keyword_arguments = {'rounds': 10} if (
                    algorithm_class is HillClimbingVariableNeighborhoodSearch
                ) else {}
                model = algorithm_class(
                    graph, depots, cities, self.distance, 1, **keyword_arguments
                )
                solution, cost = model.solve()
                self.assertTrue(math.isfinite(float(cost)))
                self._assert_route_structure(solution, depots, model.limit)

                assigned = [
                    int(city)
                    for depot in depots
                    for city in model.groups[depot]
                ]
                expected = Counter(map(int, cities))
                self.assertEqual(Counter(assigned), expected)

    @unittest.expectedFailure
    def test_lrmp_should_enforce_total_drone_sortie_range(self):
        """
        记录既有 LRMP 缺陷：去程加回程必须不超过 limit。

        修复 `src/lrmp.py` 后本测试会变成 unexpected success，提醒移除预期失败标记、
        重新冻结成本并把阶段 6 算法航程退出项改为通过。
        """
        random.seed(REGRESSION_RANDOM_SEED)
        np.random.seed(REGRESSION_RANDOM_SEED)
        graph, depots, cities = build_fixed_algorithm_case()
        model = LinearRelaxedMasterProblem(
            graph, depots, cities, self.distance, 1
        )
        solution, cost = model.solve()
        self.assertTrue(math.isfinite(float(cost)))
        assigned = [int(city) for route in solution for city in route['group'][1:]]
        self.assertEqual(Counter(assigned), Counter(map(int, cities)))
        self._assert_route_structure(solution, depots, model.limit)

    def test_linear_programming_returns_finite_feasible_objective(self):
        """LP 当前只公开目标值；必须有 Gurobi 可行解且成本有限、非负。"""
        random.seed(REGRESSION_RANDOM_SEED)
        np.random.seed(REGRESSION_RANDOM_SEED)
        graph, depots, cities = build_fixed_algorithm_case()
        solution, cost = LinearProgramming(
            graph, depots, cities, self.distance, 1
        ).solve()
        self.assertEqual(solution, [])
        self.assertIsNotNone(cost)
        self.assertTrue(math.isfinite(float(cost)))
        self.assertGreaterEqual(float(cost), 0.0)


if __name__ == '__main__':
    unittest.main()
