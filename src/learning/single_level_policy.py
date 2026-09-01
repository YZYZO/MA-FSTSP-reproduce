"""用于层次策略消融对比的单层 Actor-Critic 客户分区策略。"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class SingleLevelPolicyConfig:
    """
    保存单层策略网络结构。

    输入：扁平动作特征维数、隐藏层宽度和随机种子。
    输出：不可变配置对象。
    逻辑：每个动作同时包含仓库对特征和组内操作特征。
    """

    action_feature_dim: int
    hidden_dim: int = 128
    embedding_dim: int = 64
    seed: int = 0


@dataclass(frozen=True)
class SingleLevelTrainingConfig:
    """
    保存单层策略的模仿和 Actor-Critic 训练参数。

    输入：训练轮数、学习率、折扣率、熵和价值损失权重。
    输出：不可变配置对象。
    逻辑：与分层策略保持相近训练预算，使消融比较更公平。
    """

    imitation_epochs: int = 6
    reinforcement_episodes: int = 40
    learning_rate: float = 3e-4
    gamma: float = 0.95
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    max_gradient_norm: float = 5.0
    evaluation_interval: int = 10
    seed: int = 0


@dataclass(frozen=True)
class FlatActionCandidate:
    """
    描述一个扁平化后的有效分区动作。

    输入：可选上/下层索引、固定维特征和代理预计奖励。
    输出：不可变候选对象；索引均为空表示全局停止。
    逻辑：消融模型不使用层次决策，但执行时仍复用同一个环境动作定义。
    """

    upper_index: int | None
    lower_index: int | None
    features: tuple
    predicted_reward: float


class SingleLevelActorCritic(nn.Module):
    """
    对所有有效仓库对操作一次性评分的单层 Actor-Critic。

    输入：可变数量的扁平动作特征。
    输出：每个动作的 logit 和候选集合的状态价值。
    逻辑：网络规模与分层策略相近，但没有先选组再选客户的结构偏置。
    """

    def __init__(self, config):
        """
        建立共享动作编码器、策略头和值函数头。

        输入：`SingleLevelPolicyConfig`。
        输出：初始化后的 PyTorch 模型。
        逻辑：固定随机种子并逐动作共享编码参数。
        """
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(config.embedding_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, action_features):
        """
        计算扁平有效动作的 logits 和当前状态价值。

        输入：`[动作数, action_feature_dim]` 张量。
        输出：一维 logits 和标量价值。
        逻辑：动作嵌入均值表示当前完整候选集合。
        """
        encoded = self.action_encoder(action_features)
        logits = self.action_head(encoded).squeeze(-1)
        value = self.value_head(encoded.mean(dim=0)).squeeze(-1)
        return logits, value


def enumerate_flat_actions(environment):
    """
    枚举当前分区全部有改善的扁平动作和一个全局停止动作。

    输入：`PartitionRepairEnvironment`。
    输出：`FlatActionCandidate` 列表，最后一项固定为停止。
    逻辑：拼接上层仓库对特征和下层操作特征，并沿用相同最小改善安全门。
    """
    pair_features = environment.upper_pair_feature_matrix()
    candidates = []
    for upper_index in range(len(environment.upper_pairs())):
        previews = environment.preview_pair(upper_index)
        for lower_index, preview in enumerate(previews):
            if preview.action is None:
                continue
            if preview.predicted_reward < environment.config.minimum_improvement:
                continue
            features = np.concatenate(
                (pair_features[upper_index], np.asarray(preview.features, dtype=np.float32))
            )
            candidates.append(
                FlatActionCandidate(
                    upper_index=upper_index,
                    lower_index=lower_index,
                    features=tuple(float(value) for value in features),
                    predicted_reward=preview.predicted_reward,
                )
            )

    # 全局停止不属于任何仓库对；其特征只打开下层特征中的 stop 标记。
    stop_features = np.zeros(
        environment.upper_pair_feature_dim + environment.lower_action_feature_dim,
        dtype=np.float32,
    )
    stop_features[environment.upper_pair_feature_dim + 2] = 1.0
    candidates.append(
        FlatActionCandidate(
            upper_index=None,
            lower_index=None,
            features=tuple(float(value) for value in stop_features),
            predicted_reward=0.0,
        )
    )
    return candidates


def _candidate_tensor(candidates):
    """
    将扁平动作列表转换为模型输入张量。

    输入：`FlatActionCandidate` 列表。
    输出：二维 CPU float32 张量。
    逻辑：环境始终提供停止动作，因此张量至少包含一行。
    """
    return torch.as_tensor(
        np.asarray([candidate.features for candidate in candidates], dtype=np.float32),
        dtype=torch.float32,
    )


def _select(logits, deterministic):
    """
    从单层动作分布中选择一个候选。

    输入：动作 logits 和确定性开关。
    输出：动作索引、对数概率和熵。
    逻辑：训练时采样，评估时选择最大 logit。
    """
    distribution = Categorical(logits=logits)
    index = torch.argmax(logits) if deterministic else distribution.sample()
    return int(index.item()), distribution.log_prob(index), distribution.entropy()


def rollout_single_level(policy, environment, deterministic=True, keep_gradients=False):
    """
    在一个实例上运行单层策略直到停止或达到最大步数。

    输入：策略、环境、确定性开关和是否保留梯度。
    输出：轨迹、初末快照、最终分区和调用统计。
    逻辑：扁平索引映射回环境的上/下层索引执行，保证与 HRL 使用同一邻域。
    """
    environment.reset()
    initial_snapshot = environment.initial_snapshot
    transitions = []
    context = torch.enable_grad() if keep_gradients else torch.no_grad()
    with context:
        while not environment.done:
            candidates = enumerate_flat_actions(environment)
            logits, value = policy(_candidate_tensor(candidates))
            candidate_index, log_probability, entropy = _select(logits, deterministic)
            candidate = candidates[candidate_index]
            if candidate.upper_index is None:
                upper_index = len(environment.upper_pairs())
                lower_index = None
            else:
                upper_index = candidate.upper_index
                lower_index = candidate.lower_index
            _, reward, done, info = environment.step(upper_index, lower_index)
            transitions.append(
                {
                    "log_probability": log_probability,
                    "entropy": entropy,
                    "value": value,
                    "reward": float(reward),
                    "info": info,
                }
            )
            if done:
                break
    return {
        "transitions": transitions,
        "initial_snapshot": initial_snapshot,
        "final_snapshot": environment.current_snapshot,
        "groups": {
            depot: list(customers)
            for depot, customers in environment.groups.items()
        },
        "move_count": sum(
            transition["info"]["action_type"] != "stop"
            for transition in transitions
        ),
        "stop_reason": environment.stop_reason,
        "real_solver_calls": (
            environment.total_real_solver_calls
            - environment.episode_real_solver_calls_start
        ),
    }


def _teacher_candidate_index(environment, candidates):
    """
    将一步代理教师动作映射到扁平候选索引。

    输入：环境及当前扁平候选列表。
    输出：教师候选下标。
    逻辑：教师停止时选择最后一项，否则按上/下层索引精确匹配。
    """
    upper_target, lower_target, _ = environment.best_predicted_action()
    if upper_target == len(environment.upper_pairs()):
        return len(candidates) - 1
    return next(
        index
        for index, candidate in enumerate(candidates)
        if candidate.upper_index == upper_target
        and candidate.lower_index == lower_target
    )


def _imitation_pretrain(policy, environments, config, optimizer, progress_callback):
    """
    使用同一个一步代理教师预训练单层策略。

    输入：策略、训练环境、配置、优化器和进度回调。
    输出：模仿阶段历史行。
    逻辑：保证单层与分层模型看到相同教师，差异只来自策略结构。
    """
    history = []
    for epoch in range(1, config.imitation_epochs + 1):
        losses = []
        for environment in environments:
            environment.reset()
            while not environment.done:
                candidates = enumerate_flat_actions(environment)
                target_index = _teacher_candidate_index(environment, candidates)
                logits, _ = policy(_candidate_tensor(candidates))
                loss = nn.functional.cross_entropy(
                    logits.reshape(1, -1),
                    torch.tensor([target_index], dtype=torch.long),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_gradient_norm)
                optimizer.step()
                losses.append(float(loss.detach()))
                target = candidates[target_index]
                if target.upper_index is None:
                    environment.step(len(environment.upper_pairs()), None)
                else:
                    environment.step(target.upper_index, target.lower_index)
        row = {
            "stage": "imitation",
            "iteration": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
        }
        history.append(row)
        if progress_callback is not None:
            progress_callback(row)
    return history


def _actor_critic_loss(rollout, config):
    """
    计算一条单层策略轨迹的 Actor-Critic 损失。

    输入：保留计算图的轨迹和训练配置。
    输出：总损失、策略损失、价值损失和熵。
    逻辑：按折扣回报计算优势，并用状态价值作为基线。
    """
    transitions = rollout["transitions"]
    returns = []
    running_return = 0.0
    for transition in reversed(transitions):
        running_return = transition["reward"] + config.gamma * running_return
        returns.append(running_return)
    returns.reverse()
    policy_losses = []
    value_losses = []
    entropies = []
    for transition, return_value in zip(transitions, returns):
        target = torch.tensor(return_value, dtype=torch.float32)
        advantage = target - transition["value"]
        policy_losses.append(-transition["log_probability"] * advantage.detach())
        value_losses.append(advantage.pow(2))
        entropies.append(transition["entropy"])
    policy_loss = torch.stack(policy_losses).mean()
    value_loss = torch.stack(value_losses).mean()
    entropy = torch.stack(entropies).mean()
    total = (
        policy_loss
        + config.value_weight * value_loss
        - config.entropy_weight * entropy
    )
    return total, policy_loss, value_loss, entropy


def evaluate_single_level_policy(policy, environments):
    """
    在多个实例上贪心评估单层策略。

    输入：策略和环境列表。
    输出：平均初末分数、改善、动作数和逐实例明细。
    逻辑：指标口径与分层策略评估一致，便于直接消融比较。
    """
    policy.eval()
    details = []
    for index, environment in enumerate(environments):
        rollout = rollout_single_level(policy, environment, deterministic=True)
        initial = rollout["initial_snapshot"]
        final = rollout["final_snapshot"]
        details.append(
            {
                "instance_index": index,
                "initial_score": initial.score,
                "final_score": final.score,
                "score_improvement": initial.score - final.score,
                "initial_cost_sum": initial.cost_sum,
                "final_cost_sum": final.cost_sum,
                "move_count": rollout["move_count"],
                "stop_reason": rollout["stop_reason"],
            }
        )
    return {
        "instance_count": len(details),
        "mean_initial_score": float(np.mean([row["initial_score"] for row in details])),
        "mean_final_score": float(np.mean([row["final_score"] for row in details])),
        "mean_score_improvement": float(
            np.mean([row["score_improvement"] for row in details])
        ),
        "mean_initial_cost_sum": float(
            np.mean([row["initial_cost_sum"] for row in details])
        ),
        "mean_final_cost_sum": float(
            np.mean([row["final_cost_sum"] for row in details])
        ),
        "mean_move_count": float(np.mean([row["move_count"] for row in details])),
        "details": details,
    }


def train_single_level_policy(
    policy,
    training_environments,
    validation_environments,
    config=None,
    progress_callback=None,
):
    """
    训练单层策略并按验证集改善选择最佳权重。

    输入：策略、训练/验证环境、配置和可选回调。
    输出：最佳策略及训练历史和验证指标。
    逻辑：先模仿预训练，再 Actor-Critic 微调；不让微调破坏最佳验证模型。
    """
    config = config or SingleLevelTrainingConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    history = _imitation_pretrain(
        policy,
        training_environments,
        config,
        optimizer,
        progress_callback,
    )
    selection_environments = validation_environments or training_environments
    best_metrics = evaluate_single_level_policy(policy, selection_environments)
    best_improvement = best_metrics["mean_score_improvement"]
    best_state = deepcopy(policy.state_dict())

    policy.train()
    for episode in range(1, config.reinforcement_episodes + 1):
        environment = training_environments[(episode - 1) % len(training_environments)]
        rollout = rollout_single_level(
            policy,
            environment,
            deterministic=False,
            keep_gradients=True,
        )
        total_loss, policy_loss, value_loss, entropy = _actor_critic_loss(rollout, config)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), config.max_gradient_norm)
        optimizer.step()
        row = {
            "stage": "actor_critic",
            "iteration": episode,
            "loss": float(total_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "episode_reward": float(sum(item["reward"] for item in rollout["transitions"])),
            "move_count": rollout["move_count"],
        }
        should_evaluate = (
            episode % config.evaluation_interval == 0
            or episode == config.reinforcement_episodes
        )
        if should_evaluate:
            validation_metrics = evaluate_single_level_policy(
                policy,
                selection_environments,
            )
            row["validation_score_improvement"] = validation_metrics[
                "mean_score_improvement"
            ]
            if row["validation_score_improvement"] > best_improvement:
                best_improvement = row["validation_score_improvement"]
                best_metrics = validation_metrics
                best_state = deepcopy(policy.state_dict())
            policy.train()
        history.append(row)
        if progress_callback is not None:
            progress_callback(row)
    policy.load_state_dict(best_state)
    policy.eval()
    return policy, {
        "history": history,
        "best_validation_metrics": best_metrics,
        "best_validation_score_improvement": best_improvement,
    }


def single_level_checkpoint_payload(
    policy,
    training_config,
    environment_config,
    surrogate_checkpoint,
    artifacts,
    metrics,
):
    """
    构造单层 RL 基线的可复现检查点。

    输入：策略、训练/环境配置、代理路径、训练产物和指标。
    输出：包含权重及全部推理元数据的字典。
    逻辑：与分层检查点格式分开命名，避免评估脚本错误加载。
    """
    return {
        "policy_state_dict": policy.state_dict(),
        "policy_config": asdict(policy.config),
        "training_config": asdict(training_config),
        "environment_config": asdict(environment_config),
        "surrogate_checkpoint": str(surrogate_checkpoint),
        "best_validation_score_improvement": artifacts[
            "best_validation_score_improvement"
        ],
        "metrics": metrics,
    }

