"""固定分区入口与真实 Gurobi/DP 集成测试。"""

from copy import deepcopy
import itertools
import unittest
from unittest.mock import patch

from experiment_results import _build_stsp_result_arrays, _solve_model_with_process_data
from src.partition_repair.evaluator import evaluate_group, evaluate_partition, fixed_boundary
from src.partition_repair.settings import SolverOptions
from tests.partition_repair_fixtures import tiny_model


class EvaluatorTests(unittest.TestCase):
    """验证下游评价确实使用指定分区，并且不会累积旧状态。"""

    def setUp(self):
        """为每个测试建立新的人工图、固定分区和同一求解预算。"""
        self.model = tiny_model()
        self.partition = {0: [3, 4, 5], 1: [6, 7, 8], 2: []}
        self.options = SolverOptions(time_limit=5)

    def test_explicit_partition_both_entries_and_repeat(self):
        """两个求解入口在显式分区时不得调用 MST，重复调用的路线和成本保持一致。"""
        with patch.object(self.model, 'set_mst', side_effect=AssertionError('不应重新分区')):
            first = self.model.solve(partition=self.partition, solver_options=self.options)
            second = self.model.solve(partition=self.partition, solver_options=self.options)
            third = _solve_model_with_process_data(self.model, partition=self.partition, solver_options=self.options)
        self.assertEqual(first, second)
        self.assertEqual(first, third[:2])
        self.assertEqual(self.model.groups, self.partition)
        self.assertEqual(len(first[0]), 3)

    def test_group_evaluation_does_not_modify_model(self):
        """给定组评价只返回结果，不修改共享模型分区、路线和累计成本。"""
        self.model.cost = 123
        before = deepcopy((self.model.groups, self.model.solution, self.model.cost))
        result = evaluate_group(self.model, 0, [3, 4], fixed_boundary(self.model), self.options)
        self.assertTrue(result['complete'])
        self.assertEqual(before, (self.model.groups, self.model.solution, self.model.cost))

    def test_normal_flow_matches_evaluation_of_its_partition(self):
        """正常 MST 流程产生分区后，固定该分区再次评价应得到相同配送成本。"""
        _, expected = self.model.solve(solver_options=self.options)
        actual = evaluate_partition(self.model, self.model.groups, fixed_boundary(self.model), self.options)
        self.assertAlmostEqual(expected, actual['final_delivery_cost'])

    def test_existing_npz_builder_preserves_solver_telemetry(self):
        """原有结果序列化入口可接收统一记录，并保存详细遥测和实例构造时间。"""
        solution, cost, process = _solve_model_with_process_data(
            self.model, partition=self.partition, solver_options=self.options,
        )
        arrays = _build_stsp_result_arrays(
            [(cost, process['solve_seconds'], solution, process)],
            [self.model.depots], [self.model.cities], self.model.distance, self.model.drone,
            drone_limit=self.model.limit, drone_speed=self.model.speed,
        )
        self.assertEqual(arrays['phase2_time'].shape, (1, 3))
        self.assertIn('phase2_input_seconds', arrays['solver_telemetry_json'][0])
        self.assertGreaterEqual(arrays['instance_initialization_time'][0], 0)

    def test_set_tsp_matches_enumeration_for_singleton_sets(self):
        """单点边界时，用枚举所有客户顺序的精确 TSP 值核对真实 MIP 目标。"""
        group = [3, 4, 5]
        sequence, info = self.model.get_seq(0, [[0], [3], [4], [5]], cities=group,
                                           solver_options=self.options, return_info=True)
        distance = self.model.distance['truck']
        optimal = min(sum(distance[a][b] for a, b in zip([0] + list(order), list(order) + [0]))
                      for order in itertools.permutations(group))
        self.assertAlmostEqual(info['set_tsp_objective'], optimal)
        self.assertEqual(sorted(sequence[1:-1]), [1, 2, 3])
        self.assertEqual(info['num_binary'], 36)

    def test_real_time_limit_produces_complete_downstream_result(self):
        """真实限时允许优化器已找到可行解，两种分支都应产生完整下游结果。"""
        result = evaluate_group(self.model, 0, [3, 4, 5], fixed_boundary(self.model), SolverOptions(time_limit=0))
        self.assertTrue(result['timeout'])
        self.assertEqual(result['fallback_used'], not result['has_incumbent'])
        self.assertEqual(sorted(result['visit_route'][1:-1]), [3, 4, 5])
        self.assertTrue(result['complete'])

    def test_no_incumbent_uses_directed_fallback(self):
        """固定模拟无可行解状态，验证回退触发、顺序和计时字段，避免依赖机器上的限时偶然性。"""
        with patch.object(self.model, 'set_tsp', return_value=(None, {'status': 9, 'timeout': True})):
            result = evaluate_group(self.model, 0, [3, 4, 5], fixed_boundary(self.model), self.options)
        self.assertTrue(result['fallback_used'])
        self.assertFalse(result['has_incumbent'])
        self.assertEqual(result['set_tsp_sequence'], self.model.nearest_neighbor_sequence(0, [3, 4, 5]))
        self.assertGreaterEqual(result['phase2_fallback_seconds'], 0)

    def test_empty_group_does_not_build_mip(self):
        """空车组返回零成本闭环，不创建 MIP，也不执行 DP。"""
        with patch.object(self.model, 'set_tsp', side_effect=AssertionError('空组不应建模')):
            result = evaluate_group(self.model, 2, [], fixed_boundary(self.model), self.options)
        self.assertEqual(result['final_delivery_cost'], 0)
        self.assertEqual(result['num_vars'], 0)
        self.assertEqual(result['visit_route'], [2, 2])

    def test_empty_boundary_has_explicit_fallback_status(self):
        """空边界不隐式改变全局集合，而是按共同回退规则生成顺序。"""
        sequence, info = self.model.get_seq(0, [[0], []], cities=[3], return_info=True)
        self.assertEqual(sequence, [0, 1, 0])
        self.assertEqual(info['status'], 'EMPTY_BOUNDARY')
        self.assertTrue(info['fallback_used'])


if __name__ == '__main__':
    unittest.main()
