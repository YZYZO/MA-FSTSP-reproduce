"""单层 RL 消融策略的独立训练与推理测试。"""

import unittest

import numpy as np

from src.learning.partition_env import PartitionEnvironmentConfig
from src.learning.single_level_policy import (
    SingleLevelActorCritic,
    SingleLevelPolicyConfig,
    SingleLevelTrainingConfig,
    enumerate_flat_actions,
    evaluate_single_level_policy,
    rollout_single_level,
    train_single_level_policy,
)
from tests.test_partition_environment import build_partition_test_environment


class SingleLevelPolicyTest(unittest.TestCase):
    """验证扁平动作空间和单层 Actor-Critic 的完整闭环。"""

    def _build_policy(self, environment, seed=0):
        """
        根据测试环境建立小型单层策略。

        输入：环境和随机种子。
        输出：隐藏层缩小的 `SingleLevelActorCritic`。
        逻辑：输入维数等于上层仓库对特征与下层动作特征之和。
        """
        return SingleLevelActorCritic(
            SingleLevelPolicyConfig(
                action_feature_dim=(
                    environment.upper_pair_feature_dim
                    + environment.lower_action_feature_dim
                ),
                hidden_dim=24,
                embedding_dim=16,
                seed=seed,
            )
        )

    def test_flat_actions_include_improving_moves_and_one_stop(self):
        """
        验证扁平空间只保留改善动作并恰好包含一个全局停止。

        输入：3/1 不平衡测试环境。
        输出：至少一个有效移动、最后一项停止且全部特征维数固定。
        逻辑：单层策略与 HRL 共享相同代理最小改善安全门。
        """
        environment = build_partition_test_environment()
        candidates = enumerate_flat_actions(environment)
        self.assertGreater(len(candidates), 1)
        self.assertIsNone(candidates[-1].upper_index)
        self.assertTrue(
            all(candidate.predicted_reward >= environment.config.minimum_improvement
                for candidate in candidates[:-1])
        )
        expected_dim = (
            environment.upper_pair_feature_dim
            + environment.lower_action_feature_dim
        )
        self.assertTrue(all(len(candidate.features) == expected_dim for candidate in candidates))

    def test_short_training_produces_finite_safe_rollout(self):
        """
        验证短模仿和强化学习训练可以完成并保持代理目标不恶化。

        输入：各一个训练/验证环境和很小训练预算。
        输出：有限指标、三行历史和安全结束轨迹。
        逻辑：不要求随机小模型固定获益，只检查完整梯度链和动作安全门。
        """
        environment_config = PartitionEnvironmentConfig(max_steps=2)
        training_environment = build_partition_test_environment(config=environment_config)
        validation_environment = build_partition_test_environment(config=environment_config)
        policy = self._build_policy(training_environment, seed=11)
        policy, artifacts = train_single_level_policy(
            policy,
            [training_environment],
            [validation_environment],
            config=SingleLevelTrainingConfig(
                imitation_epochs=1,
                reinforcement_episodes=2,
                learning_rate=1e-3,
                evaluation_interval=1,
                seed=11,
            ),
        )
        metrics = evaluate_single_level_policy(policy, [validation_environment])
        rollout = rollout_single_level(policy, validation_environment, deterministic=True)
        self.assertEqual(len(artifacts["history"]), 3)
        self.assertTrue(np.isfinite(metrics["mean_final_score"]))
        self.assertLessEqual(
            rollout["final_snapshot"].score,
            rollout["initial_snapshot"].score + 1e-8,
        )
        self.assertLessEqual(rollout["move_count"], environment_config.max_steps)


if __name__ == "__main__":
    unittest.main()

