"""对称距离、候选守恒、连通小簇及选择接口测试。"""

from collections import Counter
import unittest

import networkx as nx

from src.partition_repair.candidates import connected_cluster, count_repair_path, generate_candidates, symmetric_set_graph
from src.partition_repair.evaluator import fixed_boundary
from src.partition_repair.features import FeatureContext, binary_count, partition_key
from src.partition_repair.selector import select_candidate
from src.partition_repair.settings import RepairOptions
from tests.partition_repair_fixtures import tiny_model


class CandidateTests(unittest.TestCase):
    """用小型有向图检查候选构造约束和道路方向。"""

    def setUp(self):
        """构建明显不均衡的分区，以覆盖部分修复和空车组接收客户。"""
        self.model = tiny_model(customer_count=12)
        self.boundary = fixed_boundary(self.model)
        self.context = FeatureContext(self.model, self.boundary)
        self.baseline = {0: list(range(3, 15)), 1: [], 2: []}

    def test_candidates_conserve_customers_and_have_no_duplicates(self):
        """生成候选均包含全部客户且只归属一次，预算含 stay，输入分区保持原状。"""
        candidates = generate_candidates(self.context, self.baseline)
        self.assertEqual(candidates[0].name, 'stay')
        self.assertLessEqual(len(candidates), 12)
        keys = [partition_key(c.partition, self.context.depots) for c in candidates]
        self.assertEqual(len(set(keys)), len(keys))
        for candidate in candidates:
            self.assertEqual(Counter(c for group in candidate.partition.values() for c in group), Counter(range(3, 15)))
        self.assertEqual(self.baseline[0], list(range(3, 15)))
        self.assertIn('count', {c.kind for c in candidates})
        self.assertIn('cluster', {c.kind for c in candidates})

    def test_count_path_finishes_balanced_and_changes_multiple_groups(self):
        """完整路径达到 floor/ceil 容量，且一个前缀可以同时改变两个以上仓库。"""
        path = count_repair_path(self.context, self.baseline, RepairOptions())
        self.assertEqual([len(path[-1][d]) for d in self.context.depots], [4, 4, 4])
        self.assertEqual(sum(path[-1][d] != self.baseline[d] for d in self.context.depots), 3)

    def test_symmetry_uses_both_directions(self):
        """仓库和客户均为单点时，对称集合权重必须等于真实双向道路距离均值。"""
        graph = symmetric_set_graph(self.model, self.boundary)
        for first, second in ((0, 1), (0, 4), (4, 5)):
            expected = (self.model.distance['truck'][first][second] + self.model.distance['truck'][second][first]) / 2
            self.assertAlmostEqual(graph[first][second]['weight'], expected)

    def test_cluster_is_connected(self):
        """小簇通过已有树边扩展，保持连通和指定客户数量。"""
        tree = self.context.customer_tree(self.baseline[0])
        for size in (2, 4, 8):
            cluster = connected_cluster(tree, 3, size)
            self.assertEqual(len(cluster), size)
            self.assertTrue(nx.is_connected(tree.subgraph(cluster)))

    def test_candidate_budget_and_selection_are_deterministic(self):
        """相同预算和上下文给出稳定候选与选择；stay 并列时优先。"""
        options = RepairOptions(max_candidates=4)
        first = generate_candidates(self.context, self.baseline, options)
        second = generate_candidates(self.context, self.baseline, options)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 4)
        self.assertEqual(select_candidate(self.context, self.baseline, first, 'symmetric_mst').name, 'stay')
        self.assertEqual(select_candidate(self.context, self.baseline, first, 'random', seed=7),
                         select_candidate(self.context, self.baseline, first, 'random', seed=7))

    def test_binary_formula_handles_unequal_boundaries_and_empty_groups(self):
        """边界点数不同造成的建模负担应被区分，空组实际变量为零。"""
        boundary = {3: [3, 4], 5: [5, 6, 7]}
        self.assertEqual(binary_count([3, 5], boundary), 9 + 1 + 4 + 9 + 36)
        self.assertEqual(binary_count([], boundary), 0)


if __name__ == '__main__':
    unittest.main()
