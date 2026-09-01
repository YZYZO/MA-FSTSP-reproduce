"""里程碑 D 汇总指标与 1% 成本约束的独立测试。"""

from pathlib import Path
import tempfile
import unittest

from src.learning.evaluation import (
    summarize_partition_evaluations,
    write_partition_evaluation_outputs,
)


def _evaluation_record(method, final_cost, set_tsp_times, timeout_flags):
    """
    构造一个最小但完整的实例评估记录。

    输入：方法、成本、各组 Set-TSP 时间和超时标记。
    输出：可交给正式汇总器的字典。
    逻辑：两客户组足以验证 P90、最大值、超时率和真实调用统计。
    """
    group_records = []
    for index, (wall_time, timeout) in enumerate(zip(set_tsp_times, timeout_flags)):
        group_records.append(
            {
                "customer_count": 5 + index,
                "set_tsp_wall_seconds": wall_time,
                "phase3_seconds": 1.0,
                "downstream_seconds": wall_time + 1.0,
                "set_tsp_timeout": timeout,
            }
        )
    return {
        "map_label": "fixture",
        "customer_size": 10,
        "instance_index": 0,
        "method": method,
        "final_cost": final_cost,
        "cost_has_approximate_group": int(any(timeout_flags)),
        "partition_strategy_seconds": 0.2,
        "end_to_end_seconds": sum(set_tsp_times) + 2.2,
        "strategy_real_solver_calls": 0,
        "downstream_real_solver_calls": 2,
        "total_real_solver_calls": 2,
        "group_records": group_records,
    }


class PartitionEvaluationTest(unittest.TestCase):
    """验证约束判断、时间改善和三种输出文件。"""

    def test_joint_acceptance_uses_one_percent_cost_and_runtime(self):
        """
        验证成本上升 0.9% 且 P90 下降时联合验收通过。

        输入：symmetric 成本 100、HRL 成本 100.9 及更短时间。
        输出：成本约束、时间改善和联合结论均为真。
        逻辑：直接覆盖用户批准的正式验收规则边界。
        """
        records = [
            _evaluation_record("symmetric_mst", 100.0, [10.0, 20.0], [0, 0]),
            _evaluation_record("solver_aware_hrl", 100.9, [5.0, 8.0], [0, 0]),
        ]
        summary = summarize_partition_evaluations(records, cost_tolerance=0.01)
        hrl = next(
            row for row in summary["summary_rows"]
            if row["method"] == "solver_aware_hrl"
        )
        self.assertAlmostEqual(hrl["cost_change_vs_symmetric"], 0.009)
        self.assertTrue(hrl["cost_constraint_pass"])
        self.assertTrue(hrl["runtime_or_timeout_improved"])
        self.assertTrue(hrl["joint_acceptance_pass"])

    def test_output_writer_creates_csv_json_and_markdown(self):
        """
        验证详细、摘要和人类可读报告可独立生成。

        输入：一条 symmetric 夹具记录和临时目录。
        输出：三个非空文件及一行摘要。
        逻辑：确保独立实验脚本的最终交付文件不依赖内存状态。
        """
        records = [
            _evaluation_record("symmetric_mst", 100.0, [1.0, 2.0], [0, 0])
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = write_partition_evaluation_outputs(
                records,
                Path(directory),
                run_id="fixture",
            )
            for path in paths[:3]:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(len(paths[3]["summary_rows"]), 1)


if __name__ == "__main__":
    unittest.main()

