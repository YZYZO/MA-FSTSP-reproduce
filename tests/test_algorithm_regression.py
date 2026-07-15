"""比较旧嵌套距离字典与阶段 1 距离代理下的小图算法结果。"""

import math
import random
import unittest

import numpy as np

from distance_oracle import build_distance_provider
from src.fstsp import MultiAgentFlyingSidekickTSP
from src.hc_vns import HillClimbingVariableNeighborhoodSearch
from src.lp import LinearProgramming
from src.lrmp import LinearRelaxedMasterProblem
from tests.h2h_test_fixtures import build_fixed_algorithm_case, build_legacy_distance


# 固定随机种子与旧 eager 成本共同构成阶段 0 的可重复算法基线。
REGRESSION_RANDOM_SEED = 20260715
EXPECTED_EAGER_COSTS = {
    MultiAgentFlyingSidekickTSP: 19.801939117443776,
    HillClimbingVariableNeighborhoodSearch: 19.801939117443776,
    LinearRelaxedMasterProblem: 14.580934885493017,
    LinearProgramming: 8.962128837952594,
}


class AlgorithmDistanceRegressionTests(unittest.TestCase):
    """确认距离抽象层没有改变主算法和三个基线的成本。"""

    def _solve(self, algorithm_class, use_provider: bool):
        """
        使用旧距离或新距离接口求解同一个固定实例。

        输入：
        - algorithm_class: 待运行算法类。
        - use_provider: `True` 使用阶段 1 距离工厂，否则使用旧嵌套字典。

        输出：
        - `(solution, cost)` 算法结果。
        """
        # 每次求解前重置两套随机源，保证旧字典与新代理经历相同的搜索过程。
        random.seed(REGRESSION_RANDOM_SEED)
        np.random.seed(REGRESSION_RANDOM_SEED)
        graph, depots, cities = build_fixed_algorithm_case()
        if use_provider:
            distance = build_distance_provider(graph, backend='eager', dataset_name='algorithm-regression')
        else:
            distance = build_legacy_distance(graph)

        keyword_arguments = {}
        if algorithm_class is HillClimbingVariableNeighborhoodSearch:
            keyword_arguments['rounds'] = 10
        model = algorithm_class(graph, depots, cities, distance, 1, **keyword_arguments)
        return model.solve()

    def test_all_algorithms_keep_cost_with_distance_proxy(self):
        """主算法、HC、LRMP、LP 的新旧距离接口成本必须在浮点容差内一致。"""
        algorithms = (
            MultiAgentFlyingSidekickTSP,
            HillClimbingVariableNeighborhoodSearch,
            LinearRelaxedMasterProblem,
            LinearProgramming,
        )
        for algorithm_class in algorithms:
            with self.subTest(algorithm=algorithm_class.__name__):
                legacy_solution, legacy_cost = self._solve(algorithm_class, use_provider=False)
                provider_solution, provider_cost = self._solve(algorithm_class, use_provider=True)
                self.assertIsNotNone(legacy_solution)
                self.assertIsNotNone(provider_solution)
                self.assertTrue(math.isfinite(float(legacy_cost)))
                self.assertTrue(math.isfinite(float(provider_cost)))
                self.assertAlmostEqual(
                    float(legacy_cost),
                    EXPECTED_EAGER_COSTS[algorithm_class],
                    places=10,
                )
                self.assertAlmostEqual(float(provider_cost), float(legacy_cost), places=10)


if __name__ == '__main__':
    unittest.main()
