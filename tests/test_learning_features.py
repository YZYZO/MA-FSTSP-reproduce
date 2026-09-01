"""求解器感知客户与客户组特征测试。"""

import unittest

import numpy as np

from src.learning.features import (
    CUSTOMER_FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
    extract_customer_feature_matrix,
    extract_group_feature_dict,
    extract_group_feature_vector,
)
from tests.learning_test_fixtures import build_directed_road_fixture


class LearningFeatureTest(unittest.TestCase):
    """验证规模代理、固定维度、空客户组和有向路网特征。"""

    def setUp(self):
        """
        准备固定路网，并让两个客户拥有不同候选集合规模。

        输入：无。
        输出：测试实例字段。
        逻辑：候选规模 2 和 1 便于手工核验复杂度公式。
        """
        (
            self.graph,
            self.depots,
            self.cities,
            self.distance,
            self.convex_sets,
        ) = build_directed_road_fixture()
        self.convex_sets[2] = [2, 4]
        self.convex_sets[3] = [3]

    def test_complexity_proxy_matches_formula(self):
        """
        验证候选集合规模代理严格匹配设计公式。

        输入：两个客户，候选集合规模分别为 2 和 1。
        输出：`(2+1)^2 + (2^2+1^2) + (2+1)^2 = 23`。
        逻辑：该值是后续预测 Set-TSP 运行难度的核心结构特征。
        """
        features = extract_group_feature_dict(
            self.graph,
            self.distance,
            depot=0,
            customers=[2, 3],
            convex_sets=self.convex_sets,
        )
        self.assertEqual(features["customer_count"], 2.0)
        self.assertEqual(features["candidate_total"], 3.0)
        self.assertEqual(features["candidate_square_sum"], 5.0)
        self.assertEqual(features["set_tsp_complexity_proxy"], 23.0)

    def test_feature_vectors_have_fixed_finite_shape(self):
        """
        验证客户和客户组特征维度固定且数值有限。

        输入：两个客户及其仓库。
        输出：符合公开列名长度的有限 NumPy 数组。
        逻辑：代理模型训练依赖稳定列顺序和无 NaN 输入。
        """
        customer_matrix = extract_customer_feature_matrix(
            self.graph,
            self.distance,
            cities=[2, 3],
            depots=self.depots,
            convex_sets=self.convex_sets,
        )
        group_vector = extract_group_feature_vector(
            self.graph,
            self.distance,
            depot=0,
            customers=[2, 3],
            convex_sets=self.convex_sets,
        )
        self.assertEqual(customer_matrix.shape, (2, len(CUSTOMER_FEATURE_NAMES)))
        self.assertEqual(group_vector.shape, (len(GROUP_FEATURE_NAMES),))
        self.assertTrue(np.isfinite(customer_matrix).all())
        self.assertTrue(np.isfinite(group_vector).all())

    def test_empty_group_has_finite_features(self):
        """
        验证允许空客户组时仍能生成有效特征。

        输入：仓库零和空客户列表。
        输出：固定长度有限向量，客户数为零。
        逻辑：保持原问题允许车辆空闲的可行域。
        """
        vector = extract_group_feature_vector(
            self.graph,
            self.distance,
            depot=0,
            customers=[],
            convex_sets=self.convex_sets,
        )
        self.assertEqual(vector[GROUP_FEATURE_NAMES.index("customer_count")], 0.0)
        self.assertTrue(np.isfinite(vector).all())


if __name__ == "__main__":
    unittest.main()

