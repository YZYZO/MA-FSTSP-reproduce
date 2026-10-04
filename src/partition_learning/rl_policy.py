"""面向可变长度局部动作集合的轻量级 PPO 策略。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class PPOConfig:
    """保存 Actor–Critic 网络与 PPO 更新超参数。"""

    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_weight: float = 0.50
    entropy_weight: float = 0.02
    max_grad_norm: float = 1.0
    update_epochs: int = 4


@dataclass(frozen=True)
class TrajectoryStep:
    """保存一次策略交互所需的状态、动作集合和旧策略统计量。"""

    state_features: np.ndarray
    action_features: np.ndarray
    action_index: int
    old_log_probability: float
    old_value: float
    reward: float
    done: bool


@dataclass(frozen=True)
class PPOExperience:
    """保存完成 GAE 计算后可直接用于 PPO 更新的一条经验。"""

    state_features: np.ndarray
    action_features: np.ndarray
    action_index: int
    old_log_probability: float
    advantage: float
    return_value: float


class PartitionActorCritic(nn.Module):
    """
    对可变数量局部动作逐个评分，并用固定长度状态估计价值。

    Actor 共享动作编码器，因此同一个网络可以处理不同客户规模和不同动作数；Critic
    只读取与节点编号无关的分组统计状态，避免依赖固定客户数量。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state_features: Tensor,
        action_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """输入单状态和其可变长度动作矩阵，输出动作 logits 与状态价值。"""
        state_embedding = self.state_encoder(state_features)
        action_embeddings = self.action_encoder(action_features)
        repeated_state = state_embedding.unsqueeze(0).expand(
            len(action_embeddings),
            -1,
        )
        logits = self.actor(
            torch.cat((repeated_state, action_embeddings), dim=1)
        ).squeeze(1)
        value = self.critic(state_embedding).squeeze(0)
        return logits, value

    @torch.no_grad()
    def select_action(
        self,
        state_features: np.ndarray,
        action_features: np.ndarray,
        *,
        deterministic: bool = False,
        device: torch.device | str = "cpu",
    ) -> tuple[int, float, float, float]:
        """根据当前策略采样或贪心选择动作，并返回索引、对数概率、价值和熵。"""
        state = torch.as_tensor(
            state_features,
            dtype=torch.float32,
            device=device,
        )
        actions = torch.as_tensor(
            action_features,
            dtype=torch.float32,
            device=device,
        )
        logits, value = self(state, actions)
        distribution = Categorical(logits=logits)
        action = (
            torch.argmax(logits)
            if deterministic
            else distribution.sample()
        )
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(value.item()),
            float(distribution.entropy().item()),
        )


def finish_trajectory(
    steps: list[TrajectoryStep],
    *,
    gamma: float,
    gae_lambda: float,
) -> list[PPOExperience]:
    """输入一条完整轨迹，反向计算 GAE 优势与价值回报。"""
    experiences: list[PPOExperience] = []
    advantages = [0.0] * len(steps)
    returns = [0.0] * len(steps)
    next_value = 0.0
    generalized_advantage = 0.0
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        continuation = 0.0 if step.done else 1.0
        temporal_difference = (
            step.reward
            + gamma * next_value * continuation
            - step.old_value
        )
        generalized_advantage = (
            temporal_difference
            + gamma
            * gae_lambda
            * continuation
            * generalized_advantage
        )
        advantages[index] = generalized_advantage
        returns[index] = generalized_advantage + step.old_value
        next_value = step.old_value
    for step, advantage, return_value in zip(steps, advantages, returns):
        experiences.append(PPOExperience(
            state_features=step.state_features,
            action_features=step.action_features,
            action_index=step.action_index,
            old_log_probability=step.old_log_probability,
            advantage=float(advantage),
            return_value=float(return_value),
        ))
    return experiences


def ppo_update(
    policy: PartitionActorCritic,
    optimizer: torch.optim.Optimizer,
    experiences: list[PPOExperience],
    config: PPOConfig,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """使用一批可变动作经验执行多轮 PPO 裁剪更新并返回训练统计。"""
    if not experiences:
        return {
            "loss": 0.0,
            "actor_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
    advantages = np.asarray(
        [experience.advantage for experience in experiences],
        dtype=np.float32,
    )
    if len(advantages) > 1 and float(np.std(advantages)) > 1e-8:
        advantages = (advantages - np.mean(advantages)) / np.std(advantages)

    epoch_rows: list[dict[str, float]] = []
    for _ in range(config.update_epochs):
        actor_losses, value_losses, entropies = [], [], []
        for experience, advantage in zip(experiences, advantages):
            state = torch.as_tensor(
                experience.state_features,
                dtype=torch.float32,
                device=device,
            )
            actions = torch.as_tensor(
                experience.action_features,
                dtype=torch.float32,
                device=device,
            )
            logits, value = policy(state, actions)
            distribution = Categorical(logits=logits)
            action = torch.tensor(
                experience.action_index,
                dtype=torch.long,
                device=device,
            )
            new_log_probability = distribution.log_prob(action)
            ratio = torch.exp(
                new_log_probability
                - torch.tensor(
                    experience.old_log_probability,
                    dtype=torch.float32,
                    device=device,
                )
            )
            advantage_tensor = torch.tensor(
                float(advantage),
                dtype=torch.float32,
                device=device,
            )
            unclipped = ratio * advantage_tensor
            clipped = torch.clamp(
                ratio,
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            ) * advantage_tensor
            actor_losses.append(-torch.minimum(unclipped, clipped))
            target_value = torch.tensor(
                experience.return_value,
                dtype=torch.float32,
                device=device,
            )
            value_losses.append((value - target_value).pow(2))
            entropies.append(distribution.entropy())

        actor_loss = torch.stack(actor_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()
        loss = (
            actor_loss
            + config.value_weight * value_loss
            - config.entropy_weight * entropy
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            policy.parameters(),
            config.max_grad_norm,
        )
        optimizer.step()
        epoch_rows.append({
            "loss": float(loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
        })
    return {
        name: float(np.mean([row[name] for row in epoch_rows]))
        for name in epoch_rows[0]
    }


def save_rl_policy(
    path: str | Path,
    policy: PartitionActorCritic,
    *,
    ppo_config: PPOConfig,
    metadata: dict[str, Any],
) -> None:
    """保存可独立加载的策略参数、结构配置和训练元数据。"""
    torch.save(
        {
            "kind": "partition_surrogate_ppo",
            "state_dim": policy.state_dim,
            "action_dim": policy.action_dim,
            "hidden_dim": policy.hidden_dim,
            "ppo_config": asdict(ppo_config),
            "metadata": metadata,
            "model_state_dict": policy.state_dict(),
        },
        Path(path),
    )


def load_rl_policy(
    path: str | Path,
) -> tuple[PartitionActorCritic, dict[str, Any]]:
    """加载策略检查点，返回评估模式的 CPU 模型和完整元数据。"""
    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    policy = PartitionActorCritic(
        checkpoint["state_dim"],
        checkpoint["action_dim"],
        checkpoint["hidden_dim"],
    )
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy, checkpoint
