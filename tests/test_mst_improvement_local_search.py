"""使用可控评价器测试 relocate/swap 搜索，不调用 Set-TSP。"""

import unittest

from src.mst_improvement.local_search import (
    PartitionSearchConfig,
    improve_partition,
)


def _complete_costs(nodes, default=10.0):
    """构造对角线为零、其余边使用统一默认值的有向代价矩阵。"""
    return {
        start: {
            end: 0.0 if start == end else float(default)
            for end in nodes
        }
        for start in nodes
    }


class LocalSearchTests(unittest.TestCase):
    """验证搜索只接受能改善下游评价目标的操作。"""

    def test_relocate_uses_exact_evaluator_and_preserves_input(self):
        """客户 3 应从仓库 0 移至仓库 1，且初始字典保持不变。"""
        costs = _complete_costs([0, 1, 2, 3])
        for start, end, value in [
            (0, 2, 1), (2, 0, 1), (0, 3, 9), (3, 0, 9),
            (1, 2, 9), (2, 1, 9), (1, 3, 1), (3, 1, 1),
        ]:
            costs[start][end] = float(value)

        assignment_penalty = {
            (0, 2): 1.0, (1, 2): 20.0,
            (0, 3): 20.0, (1, 3): 1.0,
        }

        def evaluate(depot, cities):
            """返回客户归属惩罚之和，模拟昂贵的下游路线评价器。"""
            return sum(assignment_penalty[(depot, city)] for city in cities)

        initial = {0: [2, 3], 1: []}
        result = improve_partition(
            initial,
            costs,
            evaluate,
            PartitionSearchConfig(
                enable_relocate=True,
                enable_swap=False,
                exact_candidate_count=10,
                max_iterations=5,
                time_limit_seconds=5.0,
            ),
        )
        self.assertEqual(initial, {0: [2, 3], 1: []})
        self.assertEqual(result.groups, {0: [2], 1: [3]})
        self.assertEqual(result.accepted_relocates, 1)

    def test_swap_can_improve_when_relocate_is_disabled(self):
        """两个客户都在错误仓库时，swap 应交换其归属。"""
        costs = _complete_costs([0, 1, 2, 3])
        for start, end, value in [
            (0, 2, 9), (2, 0, 9), (1, 2, 1), (2, 1, 1),
            (0, 3, 1), (3, 0, 1), (1, 3, 9), (3, 1, 9),
        ]:
            costs[start][end] = float(value)

        assignment_penalty = {
            (0, 2): 10.0, (1, 2): 1.0,
            (0, 3): 1.0, (1, 3): 10.0,
        }

        def evaluate(depot, cities):
            """返回客户归属惩罚之和，用于确认交换后的真实增量。"""
            return sum(assignment_penalty[(depot, city)] for city in cities)

        result = improve_partition(
            {0: [2], 1: [3]},
            costs,
            evaluate,
            PartitionSearchConfig(
                enable_relocate=False,
                enable_swap=True,
                exact_candidate_count=5,
                max_iterations=3,
                time_limit_seconds=5.0,
            ),
        )
        self.assertEqual(result.groups, {0: [3], 1: [2]})
        self.assertEqual(result.accepted_swaps, 1)


if __name__ == '__main__':
    unittest.main()

