"""客户分区修复使用的分层 Actor-Critic 策略与训练流程。"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class HierarchicalPolicyConfig:
    """
    保存上层选组和下层选动作网络的结构参数。

    输入：观察维数、动作维数、隐藏层大小和随机种子。
    输出：不可变策略配置对象。
    逻辑：上层共享编码每个客户组，下层共享编码可变数量的有效动作。
    """

    group_observation_dim: int
    action_feature_dim: int
    group_embedding_dim: int = 48
    hidden_dim: int = 96
    seed: int = 0


@dataclass(frozen=True)
class ActorCriticTrainingConfig:
    """
    保存模仿预训练和 Actor-Critic 微调参数。

    输入：预训练轮数、强化学习轮数、优化率、折扣和损失权重。
    输出：不可变训练配置对象。
    逻辑：先模仿一步代理局部搜索，减少随机策略在大动作空间中的冷启动风险。
    """

    imitation_epochs: int = 8
    reinforcement_episodes: int = 60
    learning_rate: float = 3e-4
    gamma: float = 0.95
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    max_gradient_norm: float = 5.0
    evaluation_interval: int = 10
    seed: int = 0


class UpperActorCritic(nn.Module):
    """
    上层策略：共享编码客户组后选择一个仓库对或全局停止。

    输入：客户组观察矩阵和仓库对索引矩阵。
    输出：有效仓库对加停止动作的 logits，以及当前状态价值。
    逻辑：仓库对表示由两个组嵌入及其绝对差异组成，能够表达组间不平衡。
    """

    def __init__(self, config):
        """
        根据策略配置建立上层共享编码器、动作头和价值头。

        输入：`HierarchicalPolicyConfig`。
        输出：初始化后的 PyTorch 模块。
        逻辑：所有仓库共享同一个组编码器，使策略能迁移到相同特征口径的新实例。
        """
        super().__init__()
        self.group_encoder = nn.Sequential(
            nn.Linear(config.group_observation_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.group_embedding_dim),
            nn.SiLU(),
        )
        pair_dim = config.group_embedding_dim * 3
        self.pair_head = nn.Sequential(
            nn.Linear(pair_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(config.group_embedding_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.group_embedding_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, group_observations, pair_indices):
        """
        计算当前分区的上层动作 logits 和状态价值。

        输入：`[组数, 组特征维数]` 观察及 `[仓库对数, 2]` 索引。
        输出：长度为 `仓库对数 + 1` 的 logits 和标量价值。
        逻辑：最后一个 logit 固定对应全局停止，池化组嵌入用于停止与价值判断。
        """
        encoded_groups = self.group_encoder(group_observations)
        pooled = encoded_groups.mean(dim=0)
        if len(pair_indices) > 0:
            source = encoded_groups[pair_indices[:, 0]]
            target = encoded_groups[pair_indices[:, 1]]
            pair_embeddings = torch.cat(
                (source, target, torch.abs(source - target)),
                dim=-1,
            )
            pair_logits = self.pair_head(pair_embeddings).squeeze(-1)
        else:
            pair_logits = torch.empty(0, dtype=group_observations.dtype)
        stop_logit = self.stop_head(pooled).reshape(1)
        logits = torch.cat((pair_logits, stop_logit), dim=0)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value


class LowerActorCritic(nn.Module):
    """
    下层策略：在选定仓库对内选择 relocate、swap 或停止。

    输入：当前仓库对的有效动作特征矩阵。
    输出：每个有效动作的 logit 和下层状态价值。
    逻辑：只对环境枚举出的合法候选评分，因此可变动作列表天然实现动作掩码。
    """

    def __init__(self, config):
        """
        根据策略配置建立下层动作编码器、动作头和价值头。

        输入：`HierarchicalPolicyConfig`。
        输出：初始化后的 PyTorch 模块。
        逻辑：同一个 MLP 逐行动作评分，可处理不同仓库对产生的不同候选数量。
        """
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.group_embedding_dim),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(config.group_embedding_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(config.group_embedding_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, action_features):
        """
        计算当前有效下层动作的 logits 和状态价值。

        输入：`[动作数, 动作特征维数]` 浮点矩阵。
        输出：长度为动作数的 logits 和标量价值。
        逻辑：动作嵌入均值表示当前下层候选集合，用于估计状态价值。
        """
        encoded_actions = self.action_encoder(action_features)
        logits = self.action_head(encoded_actions).squeeze(-1)
        value = self.value_head(encoded_actions.mean(dim=0)).squeeze(-1)
        return logits, value


class HierarchicalPartitionPolicy(nn.Module):
    """
    组合上层和下层 Actor-Critic 的完整客户分区策略。

    输入：`HierarchicalPolicyConfig`。
    输出：包含 `upper` 与 `lower` 两个可独立调用子网络的模型。
    逻辑：两个层次分别建模选组和组内操作，优化时共享同一个总回报。
    """

    def __init__(self, config):
        """
        初始化完整分层策略。

        输入：策略结构配置。
        输出：初始化后的策略模型。
        逻辑：固定 PyTorch 随机种子，便于小数据训练复现。
        """
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.upper = UpperActorCritic(config)
        self.lower = LowerActorCritic(config)


def _pair_index_tensor(group_count):
    """
    按环境仓库顺序构造无序仓库对索引。

    输入：客户组数量。
    输出：形状为 `[pair_count, 2]` 的长整型张量。
    逻辑：顺序与 `PartitionRepairEnvironment.upper_pairs()` 完全一致。
    """
    pairs = [
        (source, target)
        for source in range(group_count)
        for target in range(source + 1, group_count)
    ]
    return torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2)


def _upper_inputs(environment):
    """
    将环境上层观察转换为 PyTorch 张量。

    输入：分区修复环境。
    输出：客户组观察张量和仓库对索引张量。
    逻辑：保持 CPU float32，与当前小型策略网络和代理模型部署方式一致。
    """
    observations = torch.as_tensor(
        environment.group_observation_matrix(),
        dtype=torch.float32,
    )
    return observations, _pair_index_tensor(len(observations))


def _lower_inputs(previews):
    """
    将一个仓库对的动作预览转换为 PyTorch 张量。

    输入：`ActionPreview` 列表。
    输出：二维动作特征张量。
    逻辑：预览末尾始终包含下层停止，因此动作矩阵不会为空。
    """
    return torch.as_tensor(
        np.asarray([preview.features for preview in previews], dtype=np.float32),
        dtype=torch.float32,
    )


def _select_from_logits(logits, deterministic):
    """
    从一组动作 logits 中确定性或随机选择动作。

    输入：一维 logits 和是否确定性选择。
    输出：动作索引、对数概率及分布熵。
    逻辑：训练用分类分布采样，评估用最大 logit，二者共享概率计算。
    """
    distribution = Categorical(logits=logits)
    if deterministic:
        index = torch.argmax(logits)
    else:
        index = distribution.sample()
    return int(index.item()), distribution.log_prob(index), distribution.entropy()


def rollout_policy(policy, environment, deterministic=True, keep_gradients=False):
    """
    在一个实例上运行完整分层策略轨迹。

    输入：策略、环境、是否贪心和是否保留训练计算图。
    输出：每步动作概率/价值/奖励以及初末分区指标。
    逻辑：上层停止或下层停止都会结束，移动动作最多执行环境配置的步数。
    """
    environment.reset()
    initial_snapshot = environment.initial_snapshot
    transitions = []
    context = torch.enable_grad() if keep_gradients else torch.no_grad()
    with context:
        while not environment.done:
            group_observations, pair_indices = _upper_inputs(environment)
            upper_logits, upper_value = policy.upper(group_observations, pair_indices)
            # 上层只允许选择至少含一个有效改善动作的仓库对，并始终保留全局 stop。
            # 同时预览所有仓库对还能让后续下层直接复用，避免两级掩码口径不一致。
            pair_previews = [
                environment.preview_pair(pair_index)
                for pair_index in range(len(environment.upper_pairs()))
            ]
            valid_pairs = torch.as_tensor(
                [
                    any(
                        preview.action is not None
                        and preview.predicted_reward >= environment.config.minimum_improvement
                        for preview in previews
                    )
                    for previews in pair_previews
                ]
                + [True],
                dtype=torch.bool,
            )
            upper_logits = upper_logits.masked_fill(~valid_pairs, -torch.inf)
            upper_index, upper_log_probability, upper_entropy = _select_from_logits(
                upper_logits,
                deterministic,
            )
            pair_count = len(environment.upper_pairs())
            lower_log_probability = None
            lower_entropy = None
            lower_value = None
            lower_index = None
            if upper_index < pair_count:
                previews = pair_previews[upper_index]
                lower_logits, lower_value = policy.lower(_lower_inputs(previews))
                # 下层动态掩码只保留达到最低代理改善的动作与 stop。
                # 这道安全门避免探索策略执行已知会恶化目标或回到旧分区的动作。
                valid_actions = torch.as_tensor(
                    [
                        preview.action is None
                        or preview.predicted_reward >= environment.config.minimum_improvement
                        for preview in previews
                    ],
                    dtype=torch.bool,
                )
                lower_logits = lower_logits.masked_fill(~valid_actions, -torch.inf)
                lower_index, lower_log_probability, lower_entropy = _select_from_logits(
                    lower_logits,
                    deterministic,
                )
            _, reward, done, info = environment.step(upper_index, lower_index)
            transitions.append(
                {
                    "upper_log_probability": upper_log_probability,
                    "upper_entropy": upper_entropy,
                    "upper_value": upper_value,
                    "lower_log_probability": lower_log_probability,
                    "lower_entropy": lower_entropy,
                    "lower_value": lower_value,
                    "reward": float(reward),
                    "upper_index": upper_index,
                    "lower_index": lower_index,
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
        "stop_reason": environment.stop_reason,
        "move_count": sum(
            transition["info"]["action_type"] != "stop"
            for transition in transitions
        ),
        "real_solver_calls": (
            environment.total_real_solver_calls
            - environment.episode_real_solver_calls_start
        ),
    }


def imitation_pretrain(
    policy,
    environments,
    training_config,
    optimizer,
    progress_callback=None,
):
    """
    使用一步代理局部搜索教师预训练上下层策略。

    输入：策略、训练环境、训练配置、优化器和可选进度回调。
    输出：逐轮模仿损失历史。
    逻辑：教师每步枚举预测改善最大的动作；改善不足时把停止作为标签。
    """
    history = []
    for epoch in range(1, training_config.imitation_epochs + 1):
        epoch_losses = []
        teacher_rewards = []
        for environment in environments:
            environment.reset()
            while not environment.done:
                upper_target, lower_target, teacher_reward = environment.best_predicted_action()
                group_observations, pair_indices = _upper_inputs(environment)
                upper_logits, _ = policy.upper(group_observations, pair_indices)
                loss = nn.functional.cross_entropy(
                    upper_logits.reshape(1, -1),
                    torch.tensor([upper_target], dtype=torch.long),
                )
                if lower_target is not None:
                    previews = environment.preview_pair(upper_target)
                    lower_logits, _ = policy.lower(_lower_inputs(previews))
                    loss = loss + nn.functional.cross_entropy(
                        lower_logits.reshape(1, -1),
                        torch.tensor([lower_target], dtype=torch.long),
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), training_config.max_gradient_norm)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
                teacher_rewards.append(float(teacher_reward))
                environment.step(upper_target, lower_target)
        row = {
            "stage": "imitation",
            "iteration": epoch,
            "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "mean_teacher_reward": (
                float(np.mean(teacher_rewards)) if teacher_rewards else 0.0
            ),
        }
        history.append(row)
        if progress_callback is not None:
            progress_callback(row)
    return history


def _actor_critic_loss(rollout, training_config):
    """
    根据一条分层轨迹计算 Actor-Critic 损失。

    输入：保留计算图的 rollout 和训练配置。
    输出：总损失以及策略、价值、熵三个标量。
    逻辑：上下层共享同一折扣回报，各自使用自己的价值基线降低方差。
    """
    transitions = rollout["transitions"]
    returns = []
    running_return = 0.0
    for transition in reversed(transitions):
        running_return = transition["reward"] + training_config.gamma * running_return
        returns.append(running_return)
    returns.reverse()

    policy_losses = []
    value_losses = []
    entropies = []
    for transition, return_value in zip(transitions, returns):
        target = torch.tensor(return_value, dtype=torch.float32)
        upper_advantage = target - transition["upper_value"]
        policy_losses.append(
            -transition["upper_log_probability"] * upper_advantage.detach()
        )
        value_losses.append(upper_advantage.pow(2))
        entropies.append(transition["upper_entropy"])
        if transition["lower_log_probability"] is not None:
            lower_advantage = target - transition["lower_value"]
            policy_losses.append(
                -transition["lower_log_probability"] * lower_advantage.detach()
            )
            value_losses.append(lower_advantage.pow(2))
            entropies.append(transition["lower_entropy"])

    policy_loss = torch.stack(policy_losses).mean()
    value_loss = torch.stack(value_losses).mean()
    entropy = torch.stack(entropies).mean()
    total_loss = (
        policy_loss
        + training_config.value_weight * value_loss
        - training_config.entropy_weight * entropy
    )
    return total_loss, policy_loss, value_loss, entropy


def evaluate_hierarchical_policy(policy, environments):
    """
    在若干实例上贪心评估分层策略。

    输入：策略和环境列表。
    输出：聚合指标及每个实例的轨迹摘要。
    逻辑：报告代理目标的前后变化、动作数和停止原因，不调用训练反向传播。
    """
    policy.eval()
    details = []
    for index, environment in enumerate(environments):
        rollout = rollout_policy(
            policy,
            environment,
            deterministic=True,
            keep_gradients=False,
        )
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
                "initial_time_budget_risk": initial.time_budget_risk,
                "final_time_budget_risk": final.time_budget_risk,
                "initial_timeout_risk": initial.timeout_risk,
                "final_timeout_risk": final.timeout_risk,
                "move_count": rollout["move_count"],
                "stop_reason": rollout["stop_reason"],
                "real_solver_calls": rollout["real_solver_calls"],
            }
        )
    return {
        "instance_count": len(details),
        "mean_initial_score": float(np.mean([item["initial_score"] for item in details])),
        "mean_final_score": float(np.mean([item["final_score"] for item in details])),
        "mean_score_improvement": float(
            np.mean([item["score_improvement"] for item in details])
        ),
        "mean_initial_cost_sum": float(
            np.mean([item["initial_cost_sum"] for item in details])
        ),
        "mean_final_cost_sum": float(
            np.mean([item["final_cost_sum"] for item in details])
        ),
        "mean_move_count": float(np.mean([item["move_count"] for item in details])),
        "total_real_solver_calls": int(
            sum(item["real_solver_calls"] for item in details)
        ),
        "details": details,
    }


def train_hierarchical_policy(
    policy,
    training_environments,
    validation_environments,
    training_config=None,
    progress_callback=None,
):
    """
    完成模仿预训练、Actor-Critic 微调和验证集模型选择。

    输入：策略、训练/验证环境、训练配置和可选进度回调。
    输出：最佳策略、完整历史和最佳验证指标。
    逻辑：每次验证按平均分数改善选择权重，避免强化学习阶段破坏预训练策略。
    """
    training_config = training_config or ActorCriticTrainingConfig()
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=training_config.learning_rate,
        weight_decay=1e-4,
    )
    history = imitation_pretrain(
        policy,
        training_environments,
        training_config,
        optimizer,
        progress_callback=progress_callback,
    )

    selection_environments = validation_environments or training_environments
    best_metrics = evaluate_hierarchical_policy(policy, selection_environments)
    best_improvement = best_metrics["mean_score_improvement"]
    best_state = deepcopy(policy.state_dict())

    policy.train()
    for episode in range(1, training_config.reinforcement_episodes + 1):
        environment = training_environments[(episode - 1) % len(training_environments)]
        rollout = rollout_policy(
            policy,
            environment,
            deterministic=False,
            keep_gradients=True,
        )
        total_loss, policy_loss, value_loss, entropy = _actor_critic_loss(
            rollout,
            training_config,
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), training_config.max_gradient_norm)
        optimizer.step()
        row = {
            "stage": "actor_critic",
            "iteration": episode,
            "loss": float(total_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "episode_reward": float(
                sum(item["reward"] for item in rollout["transitions"])
            ),
            "move_count": rollout["move_count"],
        }

        should_evaluate = (
            episode % training_config.evaluation_interval == 0
            or episode == training_config.reinforcement_episodes
        )
        if should_evaluate:
            validation_metrics = evaluate_hierarchical_policy(
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


def policy_checkpoint_payload(
    policy,
    training_config,
    environment_config,
    surrogate_checkpoint,
    training_artifacts,
    metrics,
):
    """
    构造可直接用于里程碑 D 推理的策略检查点。

    输入：策略、两类配置、代理路径、训练产物和评估指标。
    输出：包含权重和全部必要元数据的普通字典。
    逻辑：策略检查点只引用代理检查点，不复制代理权重，避免两个模型版本混淆。
    """
    return {
        "policy_state_dict": policy.state_dict(),
        "policy_config": asdict(policy.config),
        "training_config": asdict(training_config),
        "environment_config": asdict(environment_config),
        "surrogate_checkpoint": str(surrogate_checkpoint),
        "best_validation_score_improvement": training_artifacts[
            "best_validation_score_improvement"
        ],
        "metrics": metrics,
    }
