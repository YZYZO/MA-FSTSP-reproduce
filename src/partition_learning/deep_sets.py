"""用层次化 Deep Sets 预测客户划分的二三阶段时间、成本与超时风险。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import REGRESSION_TARGETS, TARGET_INDEX


# 正值目标在数据集中使用 log1p 编码，降低长尾求解时间对训练的支配作用。
POSITIVE_TARGET_INDICES = tuple(
    TARGET_INDEX[name]
    for name in (
        "phase2_serial_seconds",
        "phase3_seconds",
        "downstream_total_seconds",
        "solver_work_sum",
        "final_cost",
    )
)


@dataclass
class DeepSetBatch:
    """
    保存一批候选划分的打包层次集合张量。

    输入由边界节点、客户、仓库组和实例四层组成；索引张量把下一层元素映射到上一层。
    输出供 HierarchicalDeepSets 和损失函数直接使用，不要求每个实例具有相同客户数或仓库数。
    """

    boundary_features: Tensor
    boundary_customer: Tensor
    customer_static_features: Tensor
    baseline_assignment_features: Tensor
    candidate_assignment_features: Tensor
    baseline_customer_group: Tensor
    candidate_customer_group: Tensor
    baseline_group_features: Tensor
    candidate_group_features: Tensor
    baseline_group_sample: Tensor
    candidate_group_sample: Tensor
    global_features: Tensor
    targets: Tensor
    target_mask: Tensor
    right_censored: Tensor
    cost_feasible: Tensor
    cost_feasible_mask: Tensor
    group_log_phase2: Tensor
    group_time_mask: Tensor
    group_right_censored: Tensor
    instance_index: Tensor
    instance_ids: tuple[str, ...]
    candidate_names: tuple[str, ...]

    def to(self, device: torch.device | str) -> "DeepSetBatch":
        """输入目标设备，输出所有张量均迁移后的新批次对象。"""
        values = {
            name: value.to(device) if isinstance(value, Tensor) else value
            for name, value in self.__dict__.items()
        }
        return DeepSetBatch(**values)

    @property
    def batch_size(self) -> int:
        """返回当前批次包含的候选划分数量。"""
        return int(self.global_features.shape[0])


@dataclass(frozen=True)
class DeepSetsModelConfig:
    """保存层次化 Deep Sets 的输入维度和网络宽度。"""

    boundary_feature_dim: int
    customer_static_dim: int
    assignment_feature_dim: int
    group_feature_dim: int
    global_feature_dim: int
    hidden_dim: int = 64
    dropout: float = 0.10


@dataclass(frozen=True)
class DeepSetsLossConfig:
    """保存多任务损失权重，便于训练报告完整记录实验配置。"""

    regression: float = 1.0
    group_time: float = 0.35
    right_censored: float = 0.20
    group_right_censored: float = 0.15
    cost_feasible: float = 0.20
    ranking: float = 0.30
    sum_consistency: float = 0.15


class _MLP(nn.Module):
    """输入固定维度向量，输出带 LayerNorm 的两层非线性表示。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        """输入二维特征矩阵，输出逐行编码后的表示。"""
        return self.network(values)


def segment_statistics(values: Tensor, segment_index: Tensor, segment_count: int) -> Tensor:
    """
    按无序集合分段计算和、均值和最大值。

    输入元素表示、元素所属集合下标和集合数量；输出每个集合的三种统计拼接。
    空集合输出全零，使没有客户的仓库组也能参与后续分区聚合。
    """
    feature_dim = int(values.shape[1])
    sums = values.new_zeros((segment_count, feature_dim))
    expanded_index = segment_index[:, None].expand(-1, feature_dim)
    sums.scatter_add_(0, expanded_index, values)

    counts = values.new_zeros((segment_count, 1))
    counts.scatter_add_(0, segment_index[:, None], values.new_ones((len(values), 1)))
    means = sums / counts.clamp_min(1.0)

    maxima = values.new_full((segment_count, feature_dim), float("-inf"))
    maxima.scatter_reduce_(0, expanded_index, values, reduce="amax", include_self=True)
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    return torch.cat((sums, means, maxima), dim=1)


class HierarchicalDeepSets(nn.Module):
    """
    编码“道路节点集合—客户—仓库组—完整划分”的两分支共享网络。

    输入为 MST 基线与候选划分共享的客户集合，以及各自的客户归属和组特征；输出七个
    回归目标、划分/分组超时概率、成本可行概率和候选组的 Phase 2 时间。
    """

    def __init__(self, config: DeepSetsModelConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        dropout = config.dropout

        # 第一层 Deep Sets：边界道路节点聚合为客户集合表示。
        self.boundary_phi = _MLP(config.boundary_feature_dim, hidden, hidden, dropout)
        self.boundary_rho = _MLP(3 * hidden, hidden, hidden, dropout)

        # 基线和候选共用客户、仓库组与分区编码器，避免两个分支学习不一致的坐标系。
        customer_input = hidden + config.customer_static_dim + config.assignment_feature_dim
        self.customer_phi = _MLP(customer_input, hidden, hidden, dropout)
        group_input = 3 * hidden + config.group_feature_dim
        self.group_rho = _MLP(group_input, hidden, hidden, dropout)
        self.partition_rho = _MLP(3 * hidden, hidden, hidden, dropout)

        if config.global_feature_dim:
            self.global_encoder: nn.Module | None = _MLP(
                config.global_feature_dim, hidden, hidden, dropout
            )
            global_output_dim = hidden
        else:
            self.global_encoder = None
            global_output_dim = 0

        decision_input = 3 * hidden + global_output_dim
        self.decision_encoder = _MLP(decision_input, hidden, hidden, dropout)
        self.regression_head = nn.Linear(hidden, len(REGRESSION_TARGETS))
        self.right_censored_head = nn.Linear(hidden, 1)
        self.cost_feasible_head = nn.Linear(hidden, 1)
        self.group_log_phase2_head = nn.Linear(hidden, 1)
        self.group_right_censored_head = nn.Linear(hidden, 1)

    def _encode_partition(
        self,
        customer_set_embeddings: Tensor,
        customer_static_features: Tensor,
        assignment_features: Tensor,
        customer_group: Tensor,
        group_features: Tensor,
        group_sample: Tensor,
        sample_count: int,
    ) -> tuple[Tensor, Tensor]:
        """
        把客户集合表示按当前归属聚合为仓库组和完整划分表示。

        输入客户静态/归属特征与客户到组、组到样本的索引；输出逐组表示和逐样本表示。
        """
        customer_input = torch.cat(
            (customer_set_embeddings, customer_static_features, assignment_features), dim=1
        )
        customer_embeddings = self.customer_phi(customer_input)
        group_statistics = segment_statistics(
            customer_embeddings, customer_group, int(group_features.shape[0])
        )
        group_embeddings = self.group_rho(torch.cat((group_statistics, group_features), dim=1))
        partition_statistics = segment_statistics(group_embeddings, group_sample, sample_count)
        partition_embeddings = self.partition_rho(partition_statistics)
        return group_embeddings, partition_embeddings

    def forward(self, batch: DeepSetBatch) -> dict[str, Tensor]:
        """
        输入打包批次，输出多任务预测字典。

        回归正值目标位于 log1p 空间；调用方在生成可读报告时负责执行 expm1 逆变换。
        """
        boundary_embeddings = self.boundary_phi(batch.boundary_features)
        customer_set_statistics = segment_statistics(
            boundary_embeddings,
            batch.boundary_customer,
            int(batch.customer_static_features.shape[0]),
        )
        customer_set_embeddings = self.boundary_rho(customer_set_statistics)

        _, baseline_partition = self._encode_partition(
            customer_set_embeddings,
            batch.customer_static_features,
            batch.baseline_assignment_features,
            batch.baseline_customer_group,
            batch.baseline_group_features,
            batch.baseline_group_sample,
            batch.batch_size,
        )
        candidate_groups, candidate_partition = self._encode_partition(
            customer_set_embeddings,
            batch.customer_static_features,
            batch.candidate_assignment_features,
            batch.candidate_customer_group,
            batch.candidate_group_features,
            batch.candidate_group_sample,
            batch.batch_size,
        )

        decision_parts = [
            baseline_partition,
            candidate_partition,
            candidate_partition - baseline_partition,
        ]
        if self.global_encoder is not None:
            decision_parts.append(self.global_encoder(batch.global_features))
        decision_embedding = self.decision_encoder(torch.cat(decision_parts, dim=1))

        return {
            "regression": self.regression_head(decision_embedding),
            "right_censored_logit": self.right_censored_head(decision_embedding).squeeze(1),
            "cost_feasible_logit": self.cost_feasible_head(decision_embedding).squeeze(1),
            "group_log_phase2": self.group_log_phase2_head(candidate_groups).squeeze(1),
            "group_right_censored_logit": self.group_right_censored_head(candidate_groups).squeeze(1),
        }


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """输入逐项损失和布尔掩码，输出有效项均值；无有效项时返回可求导零值。"""
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _pairwise_ranking_loss(outputs: Mapping[str, Tensor], batch: DeepSetBatch) -> Tensor:
    """
    在同一实例的真实成本可行候选之间计算成对时间排序损失。

    输入模型输出和批次标签；输出 RankNet 风格损失，时间近似相等的候选不构造比较对。
    """
    predicted = outputs["regression"][:, TARGET_INDEX["downstream_total_seconds"]]
    actual = batch.targets[:, TARGET_INDEX["downstream_total_seconds"]]
    exact = batch.target_mask[:, TARGET_INDEX["downstream_total_seconds"]]
    eligible = exact & batch.cost_feasible_mask & (batch.cost_feasible > 0.5)
    losses: list[Tensor] = []
    for instance in torch.unique(batch.instance_index):
        indices = torch.nonzero(
            (batch.instance_index == instance) & eligible, as_tuple=False
        ).flatten()
        if len(indices) < 2:
            continue
        first, second = torch.triu_indices(len(indices), len(indices), offset=1, device=indices.device)
        first_indices = indices[first]
        second_indices = indices[second]
        actual_difference = actual[second_indices] - actual[first_indices]
        informative = actual_difference.abs() > 1e-5
        if not torch.any(informative):
            continue
        sign = torch.sign(actual_difference[informative])
        predicted_difference = (
            predicted[second_indices[informative]] - predicted[first_indices[informative]]
        )
        losses.append(F.softplus(-sign * predicted_difference).mean())
    if not losses:
        return predicted.sum() * 0.0
    return torch.stack(losses).mean()


def hierarchical_deepsets_loss(
    outputs: Mapping[str, Tensor],
    batch: DeepSetBatch,
    config: DeepSetsLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    计算时间、成本、超时、排序和分组一致性的联合损失。

    输入模型输出、批次及可选权重；输出总损失和各组成项，便于训练日志定位问题。
    """
    weights = config or DeepSetsLossConfig()
    regression_values = F.smooth_l1_loss(
        outputs["regression"], batch.targets, reduction="none"
    )
    regression_targets = []
    for target_index in range(regression_values.shape[1]):
        regression_targets.append(_masked_mean(
            regression_values[:, target_index], batch.target_mask[:, target_index]
        ))
    regression_loss = torch.stack(regression_targets).mean()

    group_time_loss = _masked_mean(
        F.smooth_l1_loss(
            outputs["group_log_phase2"], batch.group_log_phase2, reduction="none"
        ),
        batch.group_time_mask,
    )
    right_censored_loss = F.binary_cross_entropy_with_logits(
        outputs["right_censored_logit"], batch.right_censored
    )
    group_right_censored_loss = F.binary_cross_entropy_with_logits(
        outputs["group_right_censored_logit"], batch.group_right_censored
    )
    cost_feasible_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["cost_feasible_logit"], batch.cost_feasible, reduction="none"
        ),
        batch.cost_feasible_mask,
    )
    ranking_loss = _pairwise_ranking_loss(outputs, batch)

    # 串行 Phase 2 的分区预测应与各仓库组预测之和一致。
    predicted_group_seconds = torch.expm1(outputs["group_log_phase2"].clamp(max=20.0)).clamp_min(0.0)
    predicted_group_sum = predicted_group_seconds.new_zeros(batch.batch_size)
    predicted_group_sum.scatter_add_(0, batch.candidate_group_sample, predicted_group_seconds)
    predicted_total_log = torch.log1p(predicted_group_sum)
    phase2_index = TARGET_INDEX["phase2_serial_seconds"]
    consistency_loss = _masked_mean(
        F.smooth_l1_loss(
            predicted_total_log,
            outputs["regression"][:, phase2_index],
            reduction="none",
        ),
        batch.target_mask[:, phase2_index],
    )

    components = {
        "regression": regression_loss,
        "group_time": group_time_loss,
        "right_censored": right_censored_loss,
        "group_right_censored": group_right_censored_loss,
        "cost_feasible": cost_feasible_loss,
        "ranking": ranking_loss,
        "sum_consistency": consistency_loss,
    }
    total = (
        weights.regression * regression_loss
        + weights.group_time * group_time_loss
        + weights.right_censored * right_censored_loss
        + weights.group_right_censored * group_right_censored_loss
        + weights.cost_feasible * cost_feasible_loss
        + weights.ranking * ranking_loss
        + weights.sum_consistency * consistency_loss
    )
    return total, components


def inverse_regression_targets(values: Tensor) -> Tensor:
    """输入模型回归输出，返回把正值目标从 log1p 空间还原后的副本。"""
    result = values.clone()
    for index in POSITIVE_TARGET_INDICES:
        result[:, index] = torch.expm1(result[:, index]).clamp_min(0.0)
    return result

