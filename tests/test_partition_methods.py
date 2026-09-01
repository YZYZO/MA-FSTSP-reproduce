"""里程碑 D 划分选项与均衡基线的独立测试。"""

from collections import Counter
import unittest

from src.learning.partition_methods import (
    PRODUCTION_PARTITION_OPTIONS,
    balance_mst_groups,
    build_partition_method_context,
    run_partition_method,
)
from tests.learning_test_fixtures import build_test_model


class PartitionMethodsTest(unittest.TestCase):
    """验证正式选项、传统方法客户守恒和容量均衡性质。"""

    def test_production_options_are_exactly_the_three_public_modes(self):
        """
        验证正式接入入口只暴露用户批准的三个模式。

        输入：模块常量。
        输出：legacy、symmetric 和 solver-aware HRL 三项。
        逻辑：实验消融方法不混入最终生产选项。
        """
        self.assertEqual(
            PRODUCTION_PARTITION_OPTIONS,
            ("legacy_mst", "symmetric_mst", "solver_aware_hrl"),
        )

    def test_traditional_methods_preserve_all_customers(self):
        """
        验证 legacy、symmetric 和 balanced MST 都保持客户恰好出现一次。

        输入：两仓库四客户确定性夹具。
        输出：三种方法的客户多重集合都与原实例一致。
        逻辑：通过统一上下文运行真实分区适配层，而不是直接测试内部辅助函数。
        """
        model, _ = build_test_model(city_count=4)
        context = build_partition_method_context(
            model.graph,
            model.depots,
            model.cities,
            model.distance,
            model.drone,
            drone_limit=model.limit,
            drone_speed=model.speed,
            theta=model.theta,
        )
        expected = Counter(model.cities.tolist())
        for method in ("legacy_mst", "symmetric_mst", "balanced_mst"):
            result = run_partition_method(context, method)
            actual = Counter(
                customer
                for customers in result.groups.values()
                for customer in customers
            )
            self.assertEqual(actual, expected)
            self.assertGreaterEqual(result.partition_strategy_seconds, 0.0)

    def test_balance_repair_limits_group_size_difference_to_one(self):
        """
        验证容量均衡修复能处理极端 4/0 初始分区。

        输入：两仓库、四客户全部位于首组的人工分区。
        输出：两个组均为两名客户且客户集合不变。
        逻辑：亲和力仅决定移动谁，不改变 floor/ceil 容量约束。
        """
        model, _ = build_test_model(city_count=4)
        initial = {model.depots[0]: [2, 3, 4, 5], model.depots[1]: []}
        balanced = balance_mst_groups(
            initial,
            model.depots,
            model.distance["truck"],
        )
        sizes = [len(balanced[depot]) for depot in model.depots]
        self.assertEqual(sorted(sizes), [2, 2])
        self.assertEqual(
            Counter(customer for group in balanced.values() for customer in group),
            Counter([2, 3, 4, 5]),
        )


if __name__ == "__main__":
    unittest.main()

