"""分层客户分区策略网络和训练流程的独立测试。"""

import unittest

import numpy as np
import torch

from src.learning.hierarchical_policy import (
    ActorCriticTrainingConfig,
    HierarchicalPartitionPolicy,
    HierarchicalPolicyConfig,
    evaluate_hierarchical_policy,
    rollout_policy,
    train_hierarchical_policy,
)
from src.learning.partition_env import PartitionEnvironmentConfig
from tests.test_partition_environment import build_partition_test_environment


class HierarchicalPolicyTest(unittest.TestCase):
    """验证可变动作输出、完整轨迹和小型训练闭环。"""

    def _build_policy(self, environment, seed=0):
        """
        按测试环境维数建立一个小型分层策略。

        输入：环境和随机种子。
        输出：`HierarchicalPartitionPolicy`。
        逻辑：缩小隐藏层以加快单元测试，不改变正式模型接口。
        """
        return HierarchicalPartitionPolicy(
            HierarchicalPolicyConfig(
                group_observation_dim=environment.group_observation_dim,
                action_feature_dim=environment.lower_action_feature_dim,
                group_embedding_dim=16,
                hidden_dim=24,
                seed=seed,
            )
        )

    def test_network_outputs_match_valid_dynamic_actions(self):
        """
        验证上下层 logits 数量与环境当前合法候选数量一致。

        输入：两仓库测试环境及其唯一仓库对动作预览。
        输出：上层有“一个仓库对+停止”，下层每个预览对应一个 logit。
        逻辑：动态候选数量就是动作掩码后的空间，网络不输出非法动作位置。
        """
        environment = build_partition_test_environment()
        policy = self._build_policy(environment)
        group_observations = torch.as_tensor(
            environment.group_observation_matrix(),
            dtype=torch.float32,
        )
        pair_indices = torch.tensor([[0, 1]], dtype=torch.long)
        upper_logits, upper_value = policy.upper(group_observations, pair_indices)
        previews = environment.preview_pair(0)
        action_features = torch.as_tensor(
            np.asarray([preview.features for preview in previews]),
            dtype=torch.float32,
        )
        lower_logits, lower_value = policy.lower(action_features)
        self.assertEqual(tuple(upper_logits.shape), (2,))
        self.assertEqual(tuple(lower_logits.shape), (len(previews),))
        self.assertEqual(upper_value.ndim, 0)
        self.assertEqual(lower_value.ndim, 0)

    def test_small_actor_critic_training_and_rollout_complete(self):
        """
        验证模仿加 Actor-Critic 能在小实例上完成训练与推理。

        输入：各一个训练和验证环境、两轮模仿与两轮强化学习。
        输出：有限验证指标、非空历史，以及在最大步数内结束的轨迹。
        逻辑：测试完整梯度链和最佳权重恢复，不要求随机小网络达到固定提升幅度。
        """
        config = PartitionEnvironmentConfig(max_steps=2)
        training_environment = build_partition_test_environment(config=config)
        validation_environment = build_partition_test_environment(config=config)
        policy = self._build_policy(training_environment, seed=7)
        training_config = ActorCriticTrainingConfig(
            imitation_epochs=2,
            reinforcement_episodes=2,
            learning_rate=1e-3,
            evaluation_interval=1,
            seed=7,
        )
        policy, artifacts = train_hierarchical_policy(
            policy,
            [training_environment],
            [validation_environment],
            training_config=training_config,
        )
        metrics = evaluate_hierarchical_policy(policy, [validation_environment])
        rollout = rollout_policy(policy, validation_environment, deterministic=True)
        self.assertEqual(len(artifacts["history"]), 4)
        self.assertTrue(np.isfinite(metrics["mean_final_score"]))
        self.assertLessEqual(
            rollout["final_snapshot"].score,
            rollout["initial_snapshot"].score + 1e-8,
        )
        self.assertLessEqual(rollout["move_count"], config.max_steps)
        self.assertTrue(validation_environment.done)
        self.assertIn(
            rollout["stop_reason"],
            ("upper_stop", "lower_stop", "max_steps"),
        )


if __name__ == "__main__":
    unittest.main()
