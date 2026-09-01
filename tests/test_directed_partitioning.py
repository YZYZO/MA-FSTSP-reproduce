"""有向道路距离 MST 构图与划分的回归测试。"""

import unittest

from src.partitioning import build_partition_graph, partition_customers
from tests.learning_test_fixtures import (
    build_directed_road_fixture,
    build_test_model,
)


def _edge_weights(graph):
    """
    将无向图边权转换成与边插入方向无关的字典。

    输入：NetworkX 无向图。
    输出：以节点不可变集合为键的边权字典。
    逻辑：测试只比较实体对和最终 MST 权重，不比较内部插入顺序。
    """
    return {
        frozenset((start, end)): float(data["weight"])
        for start, end, data in graph.edges(data=True)
        if start != end
    }


class DirectedPartitioningTest(unittest.TestCase):
    """验证 legacy 兼容性和显式对称化的不变性。"""

    def setUp(self):
        """
        为每个测试准备同一个确定性有向路网。

        输入：无。
        输出：测试实例字段。
        逻辑：所有断言共享相同距离，避免随机性影响。
        """
        (
            self.graph,
            self.depots,
            self.cities,
            self.distance,
            self.convex_sets,
        ) = build_directed_road_fixture()

    def test_legacy_partition_matches_original_method(self):
        """
        验证独立 legacy 分区与原模型 `set_mst` 输出一致。

        输入：固定路网与客户顺序。
        输出：各仓库客户集合完全一致。
        逻辑：新学习模块不能改变官方论文基线。
        """
        model, model_convex_sets = build_test_model(city_count=4)
        model.set_mst(model_convex_sets)
        legacy_groups = partition_customers(
            model.depots,
            model.cities,
            model_convex_sets,
            model.distance["truck"],
            model.distance["drone"],
            model.speed,
            edge_mode="legacy",
            coefficient=model.const,
        )
        for depot in model.depots:
            self.assertEqual(
                set(model.groups[depot]),
                set(legacy_groups[depot]),
            )

    def test_mean_graph_is_invariant_to_customer_order(self):
        """
        验证 mean 模式的边权和划分不受客户输入顺序影响。

        输入：正序和倒序客户列表。
        输出：完全相同的边权和客户集合划分。
        逻辑：显式双向合并并稳定插入节点，应消除原覆盖问题。
        """
        forward_graph = build_partition_graph(
            self.depots,
            self.cities,
            self.convex_sets,
            self.distance["truck"],
            self.distance["drone"],
            speed=1.6,
            edge_mode="mean",
        )
        backward_graph = build_partition_graph(
            self.depots,
            list(reversed(self.cities)),
            self.convex_sets,
            self.distance["truck"],
            self.distance["drone"],
            speed=1.6,
            edge_mode="mean",
        )
        self.assertEqual(_edge_weights(forward_graph), _edge_weights(backward_graph))

        forward_groups = partition_customers(
            self.depots,
            self.cities,
            self.convex_sets,
            self.distance["truck"],
            self.distance["drone"],
            speed=1.6,
            edge_mode="mean",
        )
        backward_groups = partition_customers(
            self.depots,
            list(reversed(self.cities)),
            self.convex_sets,
            self.distance["truck"],
            self.distance["drone"],
            speed=1.6,
            edge_mode="mean",
        )
        self.assertEqual(forward_groups, backward_groups)

    def test_depot_edge_uses_explicit_directional_mean(self):
        """
        验证两个仓库之间的边权是正反道路距离均值。

        输入：固定有向距离。
        输出：仓库边权等于手工计算值。
        逻辑：仓库没有无人机服务惩罚，便于直接核验对称化公式。
        """
        graph = build_partition_graph(
            self.depots,
            self.cities,
            self.convex_sets,
            self.distance["truck"],
            self.distance["drone"],
            speed=1.6,
            edge_mode="mean",
        )
        expected = (
            self.distance["truck"][0][1] + self.distance["truck"][1][0]
        ) / 2.0
        self.assertAlmostEqual(graph.edges[0, 1]["weight"], expected)

    def test_min_and_max_bound_mean_weight(self):
        """
        验证 min、mean、max 三种消融模式满足自然大小关系。

        输入：同一个方向不对称实体对。
        输出：`min <= mean <= max`。
        逻辑：防止模式分派错误或意外再次采用覆盖语义。
        """
        graphs = {
            mode: build_partition_graph(
                self.depots,
                self.cities,
                self.convex_sets,
                self.distance["truck"],
                self.distance["drone"],
                speed=1.6,
                edge_mode=mode,
            )
            for mode in ("min", "mean", "max")
        }
        weights = [graphs[mode].edges[2, 3]["weight"] for mode in ("min", "mean", "max")]
        self.assertLessEqual(weights[0], weights[1])
        self.assertLessEqual(weights[1], weights[2])


if __name__ == "__main__":
    unittest.main()

