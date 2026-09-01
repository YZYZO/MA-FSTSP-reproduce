"""小规模难度探测模块的独立测试。"""

import json
from pathlib import Path
import tempfile
import unittest

from src.learning.cache import EvaluationCache
from src.learning.probe import (
    ProbeSettings,
    probe_instance,
    summarize_probe_records,
    write_probe_outputs,
)
from tests.learning_test_fixtures import build_directed_road_fixture


class SetTSPDifficultyProbeTest(unittest.TestCase):
    """验证探测记录完整性、统计汇总和文件输出。"""

    def _run_fixture_probe(self, temporary_directory):
        """
        在六节点夹具上运行一个两仓库三客户的真实探测。

        输入：测试临时目录。
        输出：两条客户组记录。
        逻辑：使用五秒预算，覆盖 MST、Gurobi、第三阶段 DP 和缓存链路。
        """
        graph, depots, cities, distance, _ = build_directed_road_fixture()
        cache_path = Path(temporary_directory) / "probe.sqlite3"
        with EvaluationCache(cache_path) as cache:
            return probe_instance(
                graph,
                depots,
                cities[:3],
                distance,
                map_id="fixture-map",
                customer_size=3,
                instance_index=0,
                cache=cache,
                settings=ProbeSettings(
                    drone_count=1,
                    set_tsp_time_limit_seconds=5.0,
                ),
            )

    def test_probe_generates_one_record_per_depot(self):
        """
        验证每个仓库生成一条带特征、遥测和最终成本的记录。

        输入：一个小型真实实例。
        输出：两条记录，客户总数为三，且关键字段存在。
        逻辑：确保正式 CSV 的行粒度正是代理模型所需的单客户组。
        """
        tests_directory = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_directory) as temporary_directory:
            records = self._run_fixture_probe(temporary_directory)
        self.assertEqual(len(records), 2)
        self.assertEqual(sum(record["customer_count"] for record in records), 3)
        for record in records:
            self.assertIn("set_tsp_runtime_seconds", record)
            self.assertIn("set_tsp_complexity_proxy", record)
            self.assertIn("final_cost", record)
            self.assertEqual(record["mst_edge_mode"], "mean")

    def test_summary_and_output_files_are_readable(self):
        """
        验证探测汇总、CSV 和 JSON 文件可以直接读取。

        输入：夹具探测记录和测试临时目录。
        输出：总体客户组数正确，两个输出文件存在且 JSON 内容一致。
        逻辑：防止长实验结束后因序列化错误丢失结果。
        """
        tests_directory = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_directory) as temporary_directory:
            records = self._run_fixture_probe(temporary_directory)
            summary = summarize_probe_records(records)
            csv_path, summary_path, written_summary = write_probe_outputs(
                records,
                Path(temporary_directory) / "outputs",
                run_id="test",
            )
            self.assertEqual(summary["overall"]["group_count"], 2)
            self.assertIn("downstream_total_seconds", summary["overall"])
            self.assertIn(
                "complexity_proxy_vs_log_downstream_time_spearman",
                summary["overall"],
            )
            self.assertEqual(written_summary["overall"]["group_count"], 2)
            self.assertTrue(csv_path.is_file())
            self.assertTrue(summary_path.is_file())
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded_summary["overall"]["group_count"], 2)


if __name__ == "__main__":
    unittest.main()
