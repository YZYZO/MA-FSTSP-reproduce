"""Set-MST 邻域学习数据生成的独立测试。"""

from pathlib import Path
import tempfile
import unittest

from src.learning.cache import EvaluationCache
from src.learning.dataset import (
    DatasetGenerationSettings,
    PartitionAction,
    apply_partition_action,
    assign_instance_split,
    generate_instance_dataset_records,
    generate_neighbor_actions,
    summarize_learning_dataset,
    write_learning_dataset,
)
from tests.learning_test_fixtures import build_directed_road_fixture


class LearningDatasetTest(unittest.TestCase):
    """验证实例级拆分、邻域动作、真实标签和数据输出。"""

    def test_instance_split_keeps_five_instance_pattern(self):
        """
        验证每个客户规模的五个实例按 3/1/1 划分。

        输入：实例编号 0 到 4。
        输出：三个 train、一个 validation、一个 test。
        逻辑：所有相邻划分继承同一原始实例拆分，避免信息泄漏。
        """
        splits = [assign_instance_split(index, 5) for index in range(5)]
        self.assertEqual(splits, ["train", "train", "train", "validation", "test"])

    def test_relocate_and_swap_preserve_customer_partition(self):
        """
        验证 relocate 和 swap 后客户仍恰好出现一次。

        输入：两个仓库、四个客户的手工分区。
        输出：动作前后客户多重集合相同。
        逻辑：邻域生成不能丢失或重复服务客户。
        """
        groups = {0: [2, 3], 1: [4, 5]}
        relocate = PartitionAction("relocate", 0, 1, 2, None, 0.0)
        swap = PartitionAction("swap", 0, 1, 3, 4, 0.0)
        for action in (relocate, swap):
            updated = apply_partition_action(groups, action)
            self.assertEqual(
                sorted(updated[0] + updated[1]),
                [2, 3, 4, 5],
            )

    def test_fixture_generation_contains_both_neighbor_types(self):
        """
        验证一个小型实例可以生成基准、relocate 和 swap 标签。

        输入：六节点夹具、两类邻域和五秒 Set-TSP 上限。
        输出：去重记录、无拆分泄漏摘要以及可读 CSV/JSON。
        逻辑：覆盖里程碑 B 数据链路的最小真实 Gurobi 集成场景。
        """
        graph, depots, cities, distance, _ = build_directed_road_fixture()
        settings = DatasetGenerationSettings(
            drone_count=1,
            relocate_samples_per_instance=2,
            swap_samples_per_instance=1,
            set_tsp_time_limit_seconds=5.0,
            solver_seed=7,
        )
        tests_directory = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_directory) as temporary_directory:
            with EvaluationCache(Path(temporary_directory) / "dataset.sqlite3") as cache:
                records = generate_instance_dataset_records(
                    graph,
                    depots,
                    cities[:3],
                    distance,
                    map_id="fixture-map",
                    customer_size=3,
                    instance_index=0,
                    instances_per_size=5,
                    cache=cache,
                    settings=settings,
                )
            action_types = {record["action_type"] for record in records}
            self.assertIn("base", action_types)
            self.assertIn("relocate", action_types)
            self.assertIn("swap", action_types)
            self.assertEqual(len({record["sample_key"] for record in records}), len(records))
            self.assertTrue(all(record["split"] == "train" for record in records))

            summary = summarize_learning_dataset(records, settings)
            self.assertTrue(summary["instance_split_leakage_free"])
            csv_path, json_path, _ = write_learning_dataset(
                records,
                Path(temporary_directory) / "outputs",
                settings,
                run_id="test",
            )
            self.assertTrue(csv_path.is_file())
            self.assertTrue(json_path.is_file())

    def test_neighbor_action_counts_follow_request(self):
        """
        验证候选充足时返回指定数量的 relocate 和 swap 动作。

        输入：夹具分区和请求数量 3/1。
        输出：恰好三个 relocate 与一个 swap。
        逻辑：正式数据集规模由命令行参数可预测控制。
        """
        _, depots, _, distance, _ = build_directed_road_fixture()
        groups = {depots[0]: [2, 3], depots[1]: [4, 5]}
        actions = generate_neighbor_actions(
            groups,
            depots,
            distance["truck"],
            relocate_count=3,
            swap_count=1,
            seed=0,
        )
        self.assertEqual(sum(action.action_type == "relocate" for action in actions), 3)
        self.assertEqual(sum(action.action_type == "swap" for action in actions), 1)


if __name__ == "__main__":
    unittest.main()

