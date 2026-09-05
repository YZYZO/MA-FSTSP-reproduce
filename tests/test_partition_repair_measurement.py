"""计时边界、完整缓存上下文及多组增量测试。"""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src.partition_repair.evaluator import evaluate_group, evaluate_partition, fixed_boundary
from src.partition_repair.settings import SolverOptions
from src.partition_repair.storage import context_fingerprint, group_cache_key
from tests.partition_repair_fixtures import tiny_model


class MeasurementTests(unittest.TestCase):
    """验证统计口径而非真实墙钟的偶然大小。"""

    def test_phase2_wall_includes_setup_and_recovery(self):
        """注入单调虚拟时钟，确认完整第二阶段包含顺序求解前后开销。"""
        model = tiny_model()
        boundary = fixed_boundary(model)
        model.theta = (0.5, 0.0)
        ticks = iter(float(i) for i in range(100))
        with patch('time.perf_counter', side_effect=lambda: next(ticks)):
            record = evaluate_group(model, 0, [3], boundary)
        parts = sum(record[f'phase2_{name}_seconds'] for name in
                    ('input', 'distance', 'build', 'optimize', 'extract', 'fallback', 'other'))
        self.assertEqual(parts, record['phase2_wall_seconds'])
        self.assertGreater(record['phase2_wall_seconds'], record['phase2_optimize_seconds'])
        self.assertGreater(record['phase2_input_seconds'], 0)
        self.assertGreater(record['phase2_extract_seconds'], 0)

    def test_context_covers_global_sets_parameters_and_solver(self):
        """同一组在不同全局边界、区域、车辆或求解策略下不能命中相同缓存。"""
        model = tiny_model()
        boundary = fixed_boundary(model)
        first = context_fingerprint(model, boundary, 'graph', {'source': 'a'})
        changed = deepcopy(boundary)
        changed[8] = [7, 8]
        second = context_fingerprint(model, changed, 'graph', {'source': 'a'})
        self.assertNotEqual(first, second)
        model.speed += .1
        self.assertNotEqual(first, context_fingerprint(model, boundary, 'graph', {'source': 'a'}))
        a = group_cache_key(first, 0, [3, 4], SolverOptions())
        self.assertEqual(a, group_cache_key(first, 0, [4, 3], SolverOptions()))
        self.assertNotEqual(a, group_cache_key(first, 0, [3, 4], SolverOptions(seed=1)))
        self.assertNotEqual(a, group_cache_key(first, 0, [3, 4], SolverOptions(), repeat=1))

    def test_incremental_evaluation_handles_more_than_two_changed_groups(self):
        """三组都改变时重算三组，其汇总成本与整实例真实重算一致。"""
        model = tiny_model()
        boundary = fixed_boundary(model)
        baseline = {0: [3, 4, 5, 6], 1: [7, 8], 2: []}
        changed = {0: [3, 4], 1: [5, 7], 2: [6, 8]}
        old = evaluate_partition(model, baseline, boundary)
        reused = {r['depot_node']: r for r in old['depot_records']}
        calls = []

        def provider(depot, customers):
            """仅当当前成员与原成员完全一致时复用；否则记录并执行真实组评价。"""
            if customers == baseline[depot]:
                return reused[depot]
            calls.append(depot)
            return evaluate_group(model, depot, customers, boundary)

        incremental = evaluate_partition(model, changed, boundary, group_provider=provider)
        fresh = evaluate_partition(model, changed, boundary)
        self.assertEqual(calls, [0, 1, 2])
        self.assertAlmostEqual(incremental['final_delivery_cost'], fresh['final_delivery_cost'])
        self.assertAlmostEqual(sum(r['final_delivery_cost'] for r in fresh['depot_records']), fresh['final_delivery_cost'])


if __name__ == '__main__':
    unittest.main()
