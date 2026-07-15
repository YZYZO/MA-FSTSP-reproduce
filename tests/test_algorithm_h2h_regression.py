"""验证四类算法通过生产 H2HDistanceMatrix 时保持阶段 0 冻结成本。"""

import math
import random
import tempfile
import unittest

import numpy as np

from distance_oracle import build_distance_provider
from src.fstsp import MultiAgentFlyingSidekickTSP
from src.hc_vns import HillClimbingVariableNeighborhoodSearch
from src.lp import LinearProgramming
from src.lrmp import LinearRelaxedMasterProblem
from tests.h2h_native_test_utils import ensure_native_built
from tests.h2h_test_fixtures import build_fixed_algorithm_case


REGRESSION_RANDOM_SEED = 20260715
EXPECTED_H2H_COSTS = {
    MultiAgentFlyingSidekickTSP: 19.801939117443776,
    HillClimbingVariableNeighborhoodSearch: 19.801939117443776,
    LinearRelaxedMasterProblem: 14.580934885493017,
    LinearProgramming: 8.962128837952594,
}


class AlgorithmH2HRegressionTests(unittest.TestCase):
    """确认阶段 5 业务算法只依赖双下标协议，可直接使用 mmap 原生索引。"""

    @classmethod
    def setUpClass(cls):
        """构建一份类级临时 H2H 缓存，四个算法共享同一只读索引。"""
        cls.builder_path, cls.library_path = ensure_native_built('release')
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix='h2h-algorithm-')
        graph, _, _ = build_fixed_algorithm_case()
        cls.distance = build_distance_provider(
            graph,
            backend='h2h',
            dataset_name='algorithm-h2h',
            index_dir=cls.temporary_directory.name,
            builder_path=str(cls.builder_path),
            library_path=str(cls.library_path),
        )

    @classmethod
    def tearDownClass(cls):
        """关闭 mmap 句柄并删除类级临时缓存。"""
        cls.distance['truck'].close()
        cls.temporary_directory.cleanup()

    def test_all_algorithms_keep_frozen_cost_with_h2h(self):
        """主算法、HC、LRMP、LP 的 H2H 成本必须与冻结 eager 成本一致。"""
        algorithms = (
            MultiAgentFlyingSidekickTSP,
            HillClimbingVariableNeighborhoodSearch,
            LinearRelaxedMasterProblem,
            LinearProgramming,
        )
        for algorithm_class in algorithms:
            with self.subTest(algorithm=algorithm_class.__name__):
                random.seed(REGRESSION_RANDOM_SEED)
                np.random.seed(REGRESSION_RANDOM_SEED)
                graph, depots, cities = build_fixed_algorithm_case()
                keyword_arguments = {}
                if algorithm_class is HillClimbingVariableNeighborhoodSearch:
                    keyword_arguments['rounds'] = 10
                solution, cost = algorithm_class(
                    graph, depots, cities, self.distance, 1, **keyword_arguments
                ).solve()
                self.assertIsNotNone(solution)
                self.assertTrue(math.isfinite(float(cost)))
                self.assertAlmostEqual(
                    float(cost), EXPECTED_H2H_COSTS[algorithm_class], places=10
                )


if __name__ == '__main__':
    unittest.main()
