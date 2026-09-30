"""用有向客户道路图和仓库分配关系预测候选划分性能。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .deep_sets import _MLP, segment_statistics
from .gnn_data import PartitionGraphBatch
from .models import REGRESSION_TARGETS, TARGET_INDEX


GNN_MODEL_VARIANTS = ("gnn_only", "gnn_fused")


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


class DirectedPartitionGraphLayer(nn.Module):
    """在客户道路边与客户—仓库分配边上执行一轮关系消息传递。"""

    def __init__(
        self,
        hidden_dim: int,
        road_edge_dim: int,
        assignment_dim: int,
        dropout: float,
    ):
        super().__init__()
        hidden = hidden_dim
        # 道路消息显式读取源/目标表示、道路边属性及候选同组标志。
        self.road_message = _MLP(
            2 * hidden + road_edge_dim + 1, hidden, hidden, dropout
        )
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
        road_messages = self.road_message(torch.cat((
            customers[source],
            customers[target],
            road_edge_features,
            same_group,
        ), dim=1))
        road_context = segment_statistics(road_messages, target, len(customers))

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
        self.group_log_phase2_head = nn.Linear(hidden, 1)
        self.group_right_censored_head = nn.Linear(hidden, 1)

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
        return {
            "regression": regression,
            "ranking_scores": ranking_scores,
            "right_censored_logit": self.right_censored_head(decision).squeeze(1),
            "cost_feasible_logit": self.cost_feasible_head(decision).squeeze(1),
            "group_log_phase2": self.group_log_phase2_head(candidate_groups).squeeze(1),
            "group_right_censored_logit": self.group_right_censored_head(
                candidate_groups
            ).squeeze(1),
        }
