"""分层客户分区环境的独立单元测试。"""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from src.learning.features import GROUP_FEATURE_NAMES, extract_group_feature_vector
from src.learning.partition_env import (
    GroupPrediction,
    PartitionEnvironmentConfig,
    PartitionRepairEnvironment,
)
from tests.learning_test_fixtures import build_test_model


class DeterministicGroupScorer:
    """
    为环境测试提供无需加载 PyTorch 检查点的确定性评分器。

    输入：路网实例和是否输出高不确定性。
    输出：按客户组规模平方生成成本和时间的预测器。
    逻辑：规模平方使不平衡分区更差，便于验证 relocate 奖励方向。
    """

    def __init__(self, graph, distance, convex_sets, high_uncertainty=False):
        """
        初始化测试评分器和与正式模型同形状的归一化参数。

        输入：路网、距离、候选集合和高不确定性开关。
        输出：评分器对象。
        逻辑：所有均值为零、标准差为一，使测试观察易于解释。
        """
        self.graph = graph
        self.distance = distance
        self.convex_sets = convex_sets
        self.high_uncertainty = high_uncertainty
        self.normalization = {
            "feature_mean": [0.0] * len(GROUP_FEATURE_NAMES),
            "feature_std": [1.0] * len(GROUP_FEATURE_NAMES),
            "cost_mean": 0.0,
            "cost_std": 1.0,
            "time_mean": 0.0,
            "time_std": 1.0,
        }
        self.cache = {}

    @property
    def group_observation_dim(self):
        """返回正式评分器相同的客户组观察维数。"""
        return len(GROUP_FEATURE_NAMES) + 5

    @staticmethod
    def _key(depot, customers):
        """
        生成与正式环境一致的整数节点客户组键。

        输入：仓库和客户列表。
        输出：仓库与排序客户元组。
        逻辑：测试夹具节点均为整数，因此普通排序即可稳定复现。
        """
        return depot, tuple(sorted(customers))

    def predict_many(self, requests):
        """
        批量返回按组规模平方构造的预测。

        输入：客户组请求列表。
        输出：按稳定组键索引的 `GroupPrediction` 字典。
        逻辑：缓存真实覆盖结果，未见组按固定公式生成。
        """
        results = {}
        for depot, customers in requests:
            key = self._key(depot, customers)
            if key not in self.cache:
                features = extract_group_feature_vector(
                    self.graph,
                    self.distance,
                    depot,
                    customers,
                    self.convex_sets,
                )
                group_size = len(customers)
                wall_time = float(group_size ** 2)
                self.cache[key] = GroupPrediction(
                    features=tuple(features),
                    cost_mean=float(group_size ** 2),
                    cost_std=3.0 if self.high_uncertainty else 0.1,
                    log_time_mean=math.log1p(wall_time),
                    log_time_std=1.0 if self.high_uncertainty else 0.1,
                    wall_time_mean=wall_time,
                    timeout_probability=0.8 if group_size >= 4 else 0.0,
                )
            results[key] = self.cache[key]
        return results

    def group_observation(self, prediction):
        """
        将测试预测转换为固定维观察。

        输入：客户组预测。
        输出：结构特征加五个预测量的数组。
        逻辑：测试归一化为单位变换，仅裁剪过大的复杂度值。
        """
        extra = np.asarray(
            [
                prediction.cost_mean,
                prediction.log_time_mean,
                prediction.timeout_probability,
                prediction.cost_std,
                prediction.log_time_std,
            ]
        )
        return np.clip(np.concatenate((prediction.features, extra)), -10.0, 10.0)

    def set_real_prediction(self, depot, customers, evaluation):
        """
        用伪真实结果覆盖一个测试客户组。

        输入：仓库、客户和伪评估对象。
        输出：零不确定性的真实来源预测。
        逻辑：复现正式评分器覆盖缓存的接口，用于测试复核预算。
        """
        key = self._key(depot, customers)
        old = self.predict_many([(depot, customers)])[key]
        prediction = GroupPrediction(
            features=old.features,
            cost_mean=evaluation.final_cost,
            cost_std=0.0,
            log_time_mean=math.log1p(evaluation.set_tsp_wall_seconds),
            log_time_std=0.0,
            wall_time_mean=evaluation.set_tsp_wall_seconds,
            timeout_probability=float(evaluation.set_tsp_result.timed_out),
            source="real",
        )
        self.cache[key] = prediction
        return prediction


class FakeGroupEvaluator:
    """记录真实复核调用次数并返回固定伪标签。"""

    def __init__(self):
        """初始化空调用记录。"""
        self.calls = []

    def evaluate(self, depot, customers):
        """
        返回一个满足正式覆盖接口的伪评估结果。

        输入：仓库和客户集合。
        输出：包含成本、墙钟时间和超时标记的简单对象。
        逻辑：成本仍按规模平方，重点验证环境是否遵守总调用预算。
        """
        self.calls.append((depot, tuple(customers)))
        return SimpleNamespace(
            final_cost=float(len(customers) ** 2),
            set_tsp_wall_seconds=0.5,
            set_tsp_result=SimpleNamespace(timed_out=False),
        )


def build_partition_test_environment(config=None, high_uncertainty=False, evaluator=None):
    """
    构造一个明显不平衡的两仓库四客户修复环境。

    输入：可选环境配置、不确定性开关和伪真实评估器。
    输出：`PartitionRepairEnvironment`。
    逻辑：初始客户数为 3/1，规模平方目标应偏好把一个客户从大组移到小组。
    """
    model, _ = build_test_model(city_count=4)
    convex_sets = model.get_boundary_convex_sets(model.theta[0])
    initial_groups = {
        model.depots[0]: [2, 3, 4],
        model.depots[1]: [5],
    }
    scorer = DeterministicGroupScorer(
        model.graph,
        model.distance,
        convex_sets,
        high_uncertainty=high_uncertainty,
    )
    return PartitionRepairEnvironment(
        model.graph,
        model.distance,
        model.depots,
        convex_sets,
        initial_groups,
        scorer,
        config=config or PartitionEnvironmentConfig(max_steps=3),
        group_evaluator=evaluator,
    )


class PartitionEnvironmentTest(unittest.TestCase):
    """验证动作掩码、客户守恒、奖励和真实复核预算。"""

    def test_preview_and_relocate_preserve_customer_partition(self):
        """
        验证下层只生成合法动作且 relocate 后客户不丢失不重复。

        输入：3/1 初始分区。
        输出：执行一个大组到小组的移动后规模为 2/2，奖励为正。
        逻辑：从动作预览中按仓库方向定位 relocate，不依赖候选数组固定下标。
        """
        environment = build_partition_test_environment()
        previews = environment.preview_pair(0)
        self.assertEqual(previews[-1].action, None)
        selected_index = next(
            index
            for index, preview in enumerate(previews)
            if preview.action is not None
            and preview.action.action_type == "relocate"
            and preview.action.source_depot == environment.depots[0]
        )
        _, reward, done, info = environment.step(0, selected_index)
        flattened = [
            customer
            for depot in environment.depots
            for customer in environment.groups[depot]
        ]
        self.assertEqual(sorted(flattened), [2, 3, 4, 5])
        self.assertEqual(len(set(flattened)), 4)
        self.assertEqual(
            sorted(len(environment.groups[depot]) for depot in environment.depots),
            [2, 2],
        )
        self.assertGreater(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(info["action_type"], "relocate")

    def test_teacher_selects_improvement_then_stop(self):
        """
        验证一步代理教师先平衡分区，达到局部最优后停止。

        输入：不平衡测试环境。
        输出：首次选择有效移动，执行后选择上层停止索引。
        逻辑：规模平方成本在 2/2 时最低，任何后续 relocate 都不应通过最小改善阈值。
        """
        environment = build_partition_test_environment()
        upper_index, lower_index, reward = environment.best_predicted_action()
        self.assertEqual(upper_index, 0)
        self.assertIsNotNone(lower_index)
        self.assertGreater(reward, 0.0)
        environment.step(upper_index, lower_index)
        upper_index, lower_index, reward = environment.best_predicted_action()
        self.assertEqual(upper_index, len(environment.upper_pairs()))
        self.assertIsNone(lower_index)
        self.assertEqual(reward, 0.0)

    def test_high_uncertainty_verification_respects_lifetime_budget(self):
        """
        验证高不确定性动作只调用配置数量的真实客户组评估。

        输入：真实复核总预算为 1 的高不确定性环境。
        输出：执行动作时恰好一次复核，重置后预算仍不会恢复。
        逻辑：预算属于环境生命周期，避免每个训练 episode 重新消耗昂贵标签。
        """
        evaluator = FakeGroupEvaluator()
        config = PartitionEnvironmentConfig(
            max_steps=2,
            real_verification_budget=1,
            uncertainty_verification_threshold=0.1,
        )
        environment = build_partition_test_environment(
            config=config,
            high_uncertainty=True,
            evaluator=evaluator,
        )
        upper_index, lower_index, _ = environment.best_predicted_action()
        environment.step(upper_index, lower_index)
        self.assertEqual(len(evaluator.calls), 1)
        self.assertEqual(environment.total_real_solver_calls, 1)
        environment.reset()
        upper_index, lower_index, _ = environment.best_predicted_action()
        environment.step(upper_index, lower_index)
        self.assertEqual(len(evaluator.calls), 1)


if __name__ == "__main__":
    unittest.main()

