"""带时间预算的 Set-TSP 求解器测试。"""

import unittest

from gurobipy import GRB

from src.set_tsp_solver import SetTSPSolveResult, solve_set_tsp_with_telemetry
from tests.learning_test_fixtures import build_singleton_set_tsp_inputs


class SetTSPSolverTelemetryTest(unittest.TestCase):
    """验证正常求解、零时间预算和结果序列化。"""

    def test_small_instance_returns_complete_sequence_and_metrics(self):
        """
        验证小型 Set-TSP 能返回完整环路和基础遥测。

        输入：四个单节点集合与五秒时间预算。
        输出：首尾为零、覆盖全部集合的顺序和非负运行指标。
        逻辑：这是学习标签求解器的最小真实 Gurobi 冒烟测试。
        """
        convex_sets, distance, internal_distance = build_singleton_set_tsp_inputs(4)
        result = solve_set_tsp_with_telemetry(
            convex_sets,
            distance,
            internal_distance,
            time_limit_seconds=5.0,
            fallback_sequence=[0, 1, 2, 3, 0],
        )
        self.assertEqual(result.sequence[0], 0)
        self.assertEqual(result.sequence[-1], 0)
        self.assertEqual(set(result.sequence[:-1]), {0, 1, 2, 3})
        self.assertTrue(result.has_incumbent)
        self.assertGreaterEqual(result.runtime_seconds, 0.0)
        self.assertGreaterEqual(result.solution_count, 1)

    def test_zero_time_limit_returns_fallback_without_crashing(self):
        """
        验证零时间预算下没有 incumbent 也能安全返回后备顺序。

        输入：六个集合、零秒预算和显式后备顺序。
        输出：状态为超时；无 incumbent 时使用完整后备顺序。
        逻辑：覆盖原代码读取不存在 `.X` 值会报错的风险。
        """
        convex_sets, distance, internal_distance = build_singleton_set_tsp_inputs(6)
        fallback = [0, 1, 2, 3, 4, 5, 0]
        result = solve_set_tsp_with_telemetry(
            convex_sets,
            distance,
            internal_distance,
            time_limit_seconds=0.0,
            fallback_sequence=fallback,
        )
        self.assertEqual(result.status, GRB.TIME_LIMIT)
        self.assertTrue(result.timed_out)
        if not result.has_incumbent:
            self.assertTrue(result.fallback_used)
            self.assertEqual(result.sequence, tuple(fallback))

    def test_result_round_trip_preserves_fields(self):
        """
        验证遥测结果写入缓存后可以无损恢复。

        输入：只有仓库的显然实例结果。
        输出：字典往返后的 dataclass 完全相等。
        逻辑：后续 SQLite 缓存依赖稳定序列化。
        """
        result = solve_set_tsp_with_telemetry(
            [[0]],
            [[[[0.0]]]],
            [[[0.0]]],
        )
        restored = SetTSPSolveResult.from_dict(result.to_dict())
        self.assertEqual(result, restored)
        self.assertEqual(result.sequence, (0, 0))


if __name__ == "__main__":
    unittest.main()

