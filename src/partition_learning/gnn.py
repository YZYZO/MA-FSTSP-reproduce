"""用有向客户道路图和仓库分配关系预测候选划分性能。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .deep_sets import COST_FEASIBILITY_THRESHOLDS, _MLP, segment_statistics
from .gnn_data import PartitionGraphBatch
from .models import REGRESSION_TARGETS, TARGET_INDEX


GNN_MODEL_VARIANTS = ("gnn_only", "gnn_fused")
MESSAGE_OPERATORS = ("edge_mlp", "edge_attention", "multiscale")


@dataclass(frozen=True)
class PartitionGNNConfig:
    """保存候选划分GNN的输入维度、层数和消融结构。"""

    boundary_feature_dim: int
    customer_static_dim: int
    assignment_feature_dim: int
    group_feature_dim: int
    road_edge_feature_dim: int
    global_feature_dim: int
    hidden_dim: int = 64
    message_layers: int = 2
    dropout: float = 0.10
    model_variant: str = "gnn_fused"
    separate_ranking_heads: bool = False
    time_quantile_heads: bool = False
    cost_risk_heads: bool = False
    message_operator: str = "edge_mlp"


def monotone_feasibility_logits(raw_logits: Tensor) -> Tensor:
    """
    把四个原始输出转换为随成本阈值单调不减的可行性logit。

    输入形状为[候选数, 4]；第一列是0%阈值logit，后三列表示非负增量；
    输出保证P(cost<=0%)不大于P(cost<=5%)、P(cost<=7%)和P(cost<=10%)。
    """
    first = raw_logits[:, :1]
    increments = torch.nn.functional.softplus(raw_logits[:, 1:])
    return torch.cat((first, first + torch.cumsum(increments, dim=1)), dim=1)


class DirectedPartitionGraphLayer(nn.Module):
    """在客户道路边与客户—仓库分配边上执行一轮关系消息传递。"""

    def __init__(
        self,
        hidden_dim: int,
        road_edge_dim: int,
        assignment_dim: int,
        dropout: float,
        message_operator: str = "edge_mlp",
    ):
        super().__init__()
        if message_operator not in MESSAGE_OPERATORS:
            raise ValueError(f"未知道路消息算子：{message_operator}")
        hidden = hidden_dim
        self.message_operator = message_operator
        # 道路消息显式读取源/目标表示、道路边属性及候选同组标志。
        self.road_message = _MLP(
            2 * hidden + road_edge_dim + 1, hidden, hidden, dropout
        )
        self.road_attention = nn.Sequential(
            _MLP(2 * hidden + road_edge_dim + 1, hidden, hidden, dropout),
            nn.Linear(hidden, 1),
        ) if message_operator in {"edge_attention", "multiscale"} else None
        self.multiscale_fusion = _MLP(
            6 * hidden, 3 * hidden, 3 * hidden, dropout
        ) if message_operator == "multiscale" else None
        self.customer_to_group = _MLP(
            hidden + assignment_dim, hidden, hidden, dropout
        )
        self.group_update = _MLP(4 * hidden, hidden, hidden, dropout)
        self.group_to_customer = _MLP(
            hidden + assignment_dim, hidden, hidden, dropout
        )
        self.customer_update = _MLP(5 * hidden, hidden, hidden, dropout)
        self.customer_norm = nn.LayerNorm(hidden)
        self.group_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def _attention_context(
        self,
        edge_inputs: Tensor,
        road_messages: Tensor,
        target: Tensor,
        customer_count: int,
    ) -> Tensor:
        """输入道路边表示，按目标客户执行数值稳定的边注意力聚合。"""
        assert self.road_attention is not None
        logits = self.road_attention(edge_inputs).squeeze(1)
        maxima = logits.new_full((customer_count,), float("-inf"))
        maxima.scatter_reduce_(0, target, logits, reduce="amax", include_self=True)
        weights = torch.exp(logits - maxima[target])
        denominators = logits.new_zeros(customer_count)
        denominators.scatter_add_(0, target, weights)
        weights = weights / denominators[target].clamp_min(1e-9)
        attended = road_messages.new_zeros((customer_count, road_messages.shape[1]))
        attended.scatter_add_(
            0,
            target[:, None].expand_as(road_messages),
            road_messages * weights[:, None],
        )
        # 保持与sum/mean/max统计相同的三倍宽度，使不同算子的其余参数可公平比较。
        return torch.cat((attended, attended, attended), dim=1)

    def forward(
        self,
        customers: Tensor,
        groups: Tensor,
        assignment_features: Tensor,
        customer_group: Tensor,
        road_edge_index: Tensor,
        road_edge_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        输入客户/仓库组表示、当前归属和道路边，输出更新后的两类节点表示。

        道路消息按目标客户聚合；客户消息按所属仓库聚合，随后仓库上下文再广播回客户。
        """
        source, target = road_edge_index
        same_group = (customer_group[source] == customer_group[target]).to(
            customers.dtype
        )[:, None]
        edge_inputs = torch.cat((
            customers[source],
            customers[target],
            road_edge_features,
            same_group,
        ), dim=1)
        road_messages = self.road_message(edge_inputs)
        statistics_context = segment_statistics(road_messages, target, len(customers))
        if self.message_operator == "edge_mlp":
            road_context = statistics_context
        else:
            attention_context = self._attention_context(
                edge_inputs, road_messages, target, len(customers)
            )
            if self.message_operator == "edge_attention":
                road_context = attention_context
            else:
                assert self.multiscale_fusion is not None
                road_context = self.multiscale_fusion(torch.cat((
                    statistics_context, attention_context,
                ), dim=1))

        group_messages = self.customer_to_group(torch.cat((
            customers, assignment_features,
        ), dim=1))
        group_context = segment_statistics(group_messages, customer_group, len(groups))
        group_delta = self.group_update(torch.cat((groups, group_context), dim=1))
        groups = self.group_norm(groups + self.dropout(group_delta))

        assigned_group = groups[customer_group]
        group_broadcast = self.group_to_customer(torch.cat((
            assigned_group, assignment_features,
        ), dim=1))
        customer_delta = self.customer_update(torch.cat((
            customers, road_context, group_broadcast,
        ), dim=1))
        customers = self.customer_norm(customers + self.dropout(customer_delta))
        return customers, groups


class PartitionPerformanceGNN(nn.Module):
    """
    编码客户道路集合、客户近邻图和仓库分配图的候选划分性能预测器。

    MST基线与候选划分共享全部参数；输出接口与Deep Sets一致，可复用现有多任务损失和评估。
    """

    def __init__(self, config: PartitionGNNConfig):
        super().__init__()
        if config.model_variant not in GNN_MODEL_VARIANTS:
            raise ValueError(f"未知GNN结构：{config.model_variant}")
        if config.message_layers < 1:
            raise ValueError("GNN至少需要一层消息传递。")
        self.config = config
        self.uses_deep_sets = True
        self.uses_global_features = config.model_variant == "gnn_fused"
        hidden = config.hidden_dim
        dropout = config.dropout

        # 客户道路节点集合仍使用顺序不变的Deep Sets作为初始表示。
        self.boundary_phi = _MLP(config.boundary_feature_dim, hidden, hidden, dropout)
        self.boundary_rho = _MLP(3 * hidden, hidden, hidden, dropout)
        self.customer_encoder = _MLP(
            hidden + config.customer_static_dim, hidden, hidden, dropout
        )
        self.group_encoder = _MLP(config.group_feature_dim, hidden, hidden, dropout)
        self.graph_layers = nn.ModuleList([
            DirectedPartitionGraphLayer(
                hidden,
                config.road_edge_feature_dim,
                config.assignment_feature_dim,
                dropout,
                config.message_operator,
            )
            for _ in range(config.message_layers)
        ])
        self.partition_rho = _MLP(3 * hidden, hidden, hidden, dropout)

        if self.uses_global_features:
            if config.global_feature_dim <= 0:
                raise ValueError("gnn_fused必须提供全局人工特征。")
            self.global_encoder: nn.Module | None = _MLP(
                config.global_feature_dim, hidden, hidden, dropout
            )
            global_dim = hidden
        else:
            self.global_encoder = None
            global_dim = 0

        self.decision_encoder = _MLP(3 * hidden + global_dim, hidden, hidden, dropout)
        self.regression_head = nn.Linear(hidden, len(REGRESSION_TARGETS))
        # 独立排序头避免Top-K优化直接扭曲需要保持量纲的时间与成本回归值。
        self.ranking_head = (
            nn.Linear(hidden, 2) if config.separate_ranking_heads else None
        )
        self.right_censored_head = nn.Linear(hidden, 1)
        self.cost_feasible_head = nn.Linear(hidden, 1)
        # 多阈值分类与成本分位数仅在新版风险模型中启用，旧检查点结构保持不变。
        self.cost_threshold_head = (
            nn.Linear(hidden, len(COST_FEASIBILITY_THRESHOLDS))
            if config.cost_risk_heads else None
        )
        self.cost_quantile_head = nn.Linear(hidden, 2) if config.cost_risk_heads else None
        self.group_log_phase2_head = nn.Linear(hidden, 1)
        self.group_right_censored_head = nn.Linear(hidden, 1)
        # 分位数头输出P50与非负增量，保证P90始终不小于P50。
        self.time_quantile_head = (
            nn.Linear(hidden, 2) if config.time_quantile_heads else None
        )
        self.group_time_quantile_head = (
            nn.Linear(hidden, 2) if config.time_quantile_heads else None
        )

    def _encode_partition(
        self,
        customer_embeddings: Tensor,
        assignment_features: Tensor,
        customer_group: Tensor,
        group_features: Tensor,
        group_sample: Tensor,
        batch: PartitionGraphBatch,
    ) -> tuple[Tensor, Tensor]:
        """输入一个基线或候选归属，输出仓库组表示和完整划分表示。"""
        customers = customer_embeddings
        groups = self.group_encoder(group_features)
        for layer in self.graph_layers:
            customers, groups = layer(
                customers,
                groups,
                assignment_features,
                customer_group,
                batch.road_edge_index,
                batch.road_edge_features,
            )
        partition_statistics = segment_statistics(groups, group_sample, batch.batch_size)
        return groups, self.partition_rho(partition_statistics)

    def forward(self, batch: PartitionGraphBatch) -> dict[str, Tensor]:
        """输入打包候选图，输出时间、成本、超时和逐仓库组时间预测。"""
        boundary_embeddings = self.boundary_phi(batch.boundary_features)
        boundary_statistics = segment_statistics(
            boundary_embeddings,
            batch.boundary_customer,
            len(batch.customer_static_features),
        )
        customer_set_embeddings = self.boundary_rho(boundary_statistics)
        customer_embeddings = self.customer_encoder(torch.cat((
            customer_set_embeddings,
            batch.customer_static_features,
        ), dim=1))

        _, baseline_partition = self._encode_partition(
            customer_embeddings,
            batch.baseline_assignment_features,
            batch.baseline_customer_group,
            batch.baseline_group_features,
            batch.baseline_group_sample,
            batch,
        )
        candidate_groups, candidate_partition = self._encode_partition(
            customer_embeddings,
            batch.candidate_assignment_features,
            batch.candidate_customer_group,
            batch.candidate_group_features,
            batch.candidate_group_sample,
            batch,
        )
        decision_parts = [
            baseline_partition,
            candidate_partition,
            candidate_partition - baseline_partition,
        ]
        if self.global_encoder is not None:
            decision_parts.append(self.global_encoder(batch.global_features))
        decision = self.decision_encoder(torch.cat(decision_parts, dim=1))
        regression = self.regression_head(decision)
        ranking_scores = (
            self.ranking_head(decision)
            if self.ranking_head is not None
            else torch.stack((
                regression[:, TARGET_INDEX["downstream_total_seconds"]],
                regression[:, TARGET_INDEX["cost_change_ratio"]],
            ), dim=1)
        )
        outputs = {
            "regression": regression,
            "ranking_scores": ranking_scores,
            "right_censored_logit": self.right_censored_head(decision).squeeze(1),
            "cost_feasible_logit": self.cost_feasible_head(decision).squeeze(1),
            "group_log_phase2": self.group_log_phase2_head(candidate_groups).squeeze(1),
            "group_right_censored_logit": self.group_right_censored_head(
                candidate_groups
            ).squeeze(1),
        }
        if self.time_quantile_head is not None and self.group_time_quantile_head is not None:
            raw_time_quantiles = self.time_quantile_head(decision)
            raw_group_quantiles = self.group_time_quantile_head(candidate_groups)
            outputs["time_quantiles"] = torch.stack((
                raw_time_quantiles[:, 0],
                raw_time_quantiles[:, 0] + torch.nn.functional.softplus(
                    raw_time_quantiles[:, 1]
                ),
            ), dim=1)
            outputs["group_time_quantiles"] = torch.stack((
                raw_group_quantiles[:, 0],
                raw_group_quantiles[:, 0] + torch.nn.functional.softplus(
                    raw_group_quantiles[:, 1]
                ),
            ), dim=1)
        if self.cost_threshold_head is not None and self.cost_quantile_head is not None:
            threshold_logits = monotone_feasibility_logits(
                self.cost_threshold_head(decision)
            )
            raw_cost_quantiles = self.cost_quantile_head(decision)
            outputs["cost_feasible_threshold_logits"] = threshold_logits
            # 10%列兼容原有单阈值接口，使旧的策略代码也会读取新版保守分类头。
            outputs["cost_feasible_logit"] = threshold_logits[:, -1]
            outputs["cost_change_quantiles"] = torch.stack((
                raw_cost_quantiles[:, 0],
                raw_cost_quantiles[:, 0] + torch.nn.functional.softplus(
                    raw_cost_quantiles[:, 1]
                ),
            ), dim=1)
        return outputs
