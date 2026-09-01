"""单客户组真实评估器的小型集成测试。"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from src.learning.cache import EvaluationCache
from src.learning.evaluator import GroupEvaluator
from src.learning.settings import SetTSPSolverSettings
from tests.learning_test_fixtures import build_test_model


class GroupEvaluatorTest(unittest.TestCase):
    """验证评估结果、原模型无副作用和缓存命中。"""

    def test_evaluation_does_not_mutate_original_groups_or_cost(self):
        """
        验证单组标签求解不会改变原模型的分组和累计目标值。

        输入：一个仓库和两个客户的小型真实求解。
        输出：有限成本、完整路线、可行遥测以及不变的原模型状态。
        逻辑：学习数据生成必须能反复评估邻域而不污染基线模型。
        """
        model, convex_sets = build_test_model(city_count=3)
        original_groups = deepcopy(model.groups)
        original_cost = model.cost

        tests_directory = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_directory) as temporary_directory:
            cache_path = Path(temporary_directory) / "group.sqlite3"
            with EvaluationCache(cache_path) as cache:
                evaluator = GroupEvaluator(
                    model,
                    convex_sets,
                    map_id="fixture-map",
                    cache=cache,
                    solver_settings=SetTSPSolverSettings(time_limit_seconds=5.0),
                )
                first = evaluator.evaluate(depot=0, customers=[3, 2])
                second = evaluator.evaluate(depot=0, customers=[2, 3])

                self.assertEqual(first.customers, (2, 3))
                self.assertEqual(first.visit_route[0], 0)
                self.assertEqual(first.visit_route[-1], 0)
                self.assertGreaterEqual(first.final_cost, 0.0)
                self.assertFalse(first.cache_hit)
                self.assertTrue(second.cache_hit)
                self.assertEqual(first.final_cost, second.final_cost)
                self.assertEqual(cache.count(), 1)

        self.assertEqual(model.groups, original_groups)
        self.assertEqual(model.cost, original_cost)


if __name__ == "__main__":
    unittest.main()
