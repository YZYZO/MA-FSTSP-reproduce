"""总体成本约束、逐实例退化和缺失记录的统计测试。"""

from pathlib import Path
import tempfile
import unittest

from src.partition_repair.reporting import aggregate_pairs, evaluation_report, oracle_select
from src.partition_repair.storage import RecordTable, save_json


class ReportingTests(unittest.TestCase):
    """用可手算数据验证报告，不依赖历史实验 CSV。"""

    def test_totals_ratio_is_not_mean_ratio(self):
        """总量之比达标时仍可能存在个体明显恶化，报告应同时保留两种信息。"""
        pairs = [dict(instance_id='a', size=50, baseline_cost=100, cost=110, baseline_time=10, time=9),
                 dict(instance_id='b', size=150, baseline_cost=900, cost=935, baseline_time=90, time=63)]
        stats = aggregate_pairs(pairs, bootstrap=20)
        self.assertAlmostEqual(stats['cost_change'], .045)
        self.assertAlmostEqual(stats['phase2_saving'], .28)
        self.assertNotAlmostEqual(stats['cost_change'], stats['mean_instance_cost_change'])
        self.assertEqual(stats['cost_over_5_percent_count'], 1)
        self.assertTrue(stats['passes_point_thresholds'])

    def test_aggregate_oracle_differs_from_per_instance_constraint(self):
        """总体预算允许某实例超过5%，另一个实例的成本改善可以抵消它。"""
        grouped = {}
        for key, cost, elapsed in [('a', 110, 1), ('b', 90, 9)]:
            grouped[key] = [dict(instance_id=key, size=50, name='stay', final_delivery_cost=100, phase2_wall_seconds=10),
                            dict(instance_id=key, size=50, name='repair', final_delivery_cost=cost, phase2_wall_seconds=elapsed)]
        aggregate = aggregate_pairs(oracle_select(grouped, .05), bootstrap=0)
        individual = aggregate_pairs(oracle_select(grouped, .05, per_instance=True), bootstrap=0)
        self.assertAlmostEqual(aggregate['phase2_saving'], .5)
        self.assertAlmostEqual(individual['phase2_saving'], .05)

    def test_incomplete_instance_prevents_passing(self):
        """清单中未完成实例不能从分母消失后让方法被标为通过。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_json(root / 'evaluation_config.json', {'configuration': {
                'expected_instances': ['a', 'b'], 'methods': ['symmetric_mst', 'handcrafted'], 'repeats': 1}})
            table = RecordTable(root, 'evaluation_runs')
            for method, elapsed in [('symmetric_mst', 10), ('handcrafted', 5)]:
                table.put(dict(id=method, instance_id='a', method=method, size=50, repeat=0, complete=True,
                               final_delivery_cost=100, phase2_wall_seconds=elapsed, phase3_seconds=1,
                               online_seconds=elapsed + 2, timeout_groups=0, fallback_groups=0))
            report = evaluation_report(root, root / 'report', bootstrap=0)
            stats = report['methods']['handcrafted']
            self.assertFalse(stats['passes_point_thresholds'])
            self.assertEqual(stats['incomplete_instances'], ['b'])


if __name__ == '__main__':
    unittest.main()
