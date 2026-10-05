"""把任意客户划分编码后交给已训练GNN，供后续强化学习快速查询。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .candidates import (
    Partition,
    PartitionCandidate,
    canonical_partition,
    moved_customer_count,
)
from .dataset import load_instances
from .deep_sets import COST_FEASIBILITY_NAMES, inverse_regression_targets
from .deep_sets_data import _partition_features, _static_instance_tensors
from .features import action_delta_features
from .gnn_data import collate_partition_graphs
from .gnn_training import load_partition_gnn_model
from .models import REGRESSION_TARGETS
from .road import build_road_csr, load_road_graph


class PartitionGNNPredictor:
    """
    为一个固定路网实例提供任意客户划分的批量监督模型预测。

    初始化输入GNN缓存、原始NPZ目录、实例编号和一个或多个模型检查点；predict_many
    输入完整客户划分，输出时间、成本、可行概率、排序分数和跨模型不确定性。
    """

    def __init__(
        self,
        cache: dict[str, Any],
        result_root: str | Path,
        instance_id: str,
        checkpoint_paths: Sequence[str | Path],
        *,
        distance_batch_size: int = 128,
    ):
        if not checkpoint_paths:
            raise ValueError("至少需要一个GNN检查点。")
        self.cache = cache
        self.instance_id = str(instance_id)
        self.static = cache["instances"][self.instance_id]
        npz_paths = {path.stem: path for path in Path(result_root).rglob("*.npz")}
        source_name = str(self.static["source_name"])
        if source_name not in npz_paths:
            raise FileNotFoundError(f"未找到实例来源NPZ：{source_name}")
        self.instance = load_instances(
            npz_paths[source_name], [int(self.static["instance_index"])]
        )[0]
        self.graph = load_road_graph(self.instance.graph_path)
        road_csr = build_road_csr(self.graph)
        _, self.context = _static_instance_tensors(
            self.instance,
            self.graph,
            road_csr,
            distance_batch_size=distance_batch_size,
        )
        # 模型与各自归一化参数绑定；允许多个随机种子直接形成推理集成。
        self.models = [load_partition_gnn_model(path) for path in checkpoint_paths]

    def _candidate(
        self,
        partition: dict[int, Iterable[int]],
        *,
        name: str,
        kind: str,
        strength: float,
    ) -> PartitionCandidate:
        """输入任意可迭代分区，输出仓库/客户顺序稳定的候选对象。"""
        canonical = canonical_partition(partition, self.instance.depots)
        return PartitionCandidate(
            name=name,
            kind=kind,
            strength=float(strength),
            partition=canonical,
            moved_customers=moved_customer_count(self.instance.partition, canonical),
        )

    def _sample(
        self,
        candidate: PartitionCandidate,
        checkpoint: dict[str, Any],
    ) -> dict[str, Any]:
        """输入候选及模型归一化元数据，输出可直接打包的单候选图样本。"""
        assignment, customer_group, group_features = _partition_features(
            candidate.partition, self.instance.partition, self.context
        )
        feature_values = action_delta_features(
            self.instance.partition,
            candidate,
            self.instance.boundary_sizes,
            self.context["truck"].query,
            self.context["drone"].query,
            graph_nodes=len(self.graph),
            graph_edges=self.graph.number_of_edges(),
        )
        selected_names = tuple(checkpoint["global_feature_names"])
        raw_global = np.asarray(
            [feature_values.get(name, 0.0) for name in selected_names], dtype=np.float32
        )
        mean = np.asarray(checkpoint["global_feature_mean"], dtype=np.float32)
        scale = np.asarray(checkpoint["global_feature_scale"], dtype=np.float32)
        depot_count = len(self.instance.depots)
        return {
            **self.static,
            "instance_id": self.instance_id,
            "candidate_name": candidate.name,
            "candidate_assignment_features": assignment,
            "candidate_customer_group": customer_group,
            "candidate_group_features": group_features,
            "global_features": (raw_global - mean) / scale,
            # 推理样本不包含真实标签；占位张量仅用于复用统一的批次结构。
            "targets": np.zeros(len(REGRESSION_TARGETS), dtype=np.float32),
            "target_mask": np.zeros(len(REGRESSION_TARGETS), dtype=bool),
            "right_censored": 0.0,
            "cost_feasible": 0.0,
            "cost_feasible_thresholds": np.zeros(
                len(COST_FEASIBILITY_NAMES), dtype=np.float32
            ),
            "cost_feasible_mask": False,
            "group_log_phase2": np.zeros(depot_count, dtype=np.float32),
            "group_time_mask": np.zeros(depot_count, dtype=bool),
            "group_right_censored": np.zeros(depot_count, dtype=np.float32),
        }

    @torch.no_grad()
    def predict_many(
        self,
        partitions: Sequence[dict[int, Iterable[int]]],
        *,
        kinds: Sequence[str] | None = None,
        strengths: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        批量预测多个完整客户划分。

        输入分区及可选生成算法/动作强度；输出与输入同序的集成均值、标准差和逐模型值。
        """
        kinds = kinds or ["active"] * len(partitions)
        strengths = strengths or [0.0] * len(partitions)
        candidates = [
            self._candidate(
                partition,
                name=f"online_{index:04d}",
                kind=str(kinds[index]),
                strength=float(strengths[index]),
            )
            for index, partition in enumerate(partitions)
        ]
        member_outputs: list[list[dict[str, Any]]] = []
        for model, checkpoint in self.models:
            samples = [self._sample(candidate, checkpoint) for candidate in candidates]
            batch = collate_partition_graphs(samples)
            outputs = model(batch)
            regression = inverse_regression_targets(outputs["regression"]).cpu().numpy()
            ranking = outputs["ranking_scores"].cpu().numpy()
            feasible = torch.sigmoid(outputs["cost_feasible_logit"]).cpu().numpy()
            threshold_probabilities = None
            if "cost_feasible_threshold_logits" in outputs:
                threshold_probabilities = torch.sigmoid(
                    outputs["cost_feasible_threshold_logits"]
                ).cpu().numpy()
            censored = torch.sigmoid(outputs["right_censored_logit"]).cpu().numpy()
            quantiles = None
            if "time_quantiles" in outputs:
                quantiles = torch.expm1(outputs["time_quantiles"]).clamp_min(0).cpu().numpy()
            cost_quantiles = (
                outputs["cost_change_quantiles"].cpu().numpy()
                if "cost_change_quantiles" in outputs else None
            )
            member = []
            for index in range(len(candidates)):
                item: dict[str, Any] = {
                    **{
                        name: float(regression[index, target_index])
                        for target_index, name in enumerate(REGRESSION_TARGETS)
                    },
                    "time_rank_score": float(ranking[index, 0]),
                    "cost_rank_score": float(ranking[index, 1]),
                    "cost_feasible_probability": float(feasible[index]),
                    "right_censored_probability": float(censored[index]),
                }
                if quantiles is not None:
                    item["downstream_time_p50"] = float(quantiles[index, 0])
                    item["downstream_time_p90"] = float(quantiles[index, 1])
                if threshold_probabilities is not None:
                    for threshold_index, threshold_name in enumerate(
                        COST_FEASIBILITY_NAMES
                    ):
                        item[f"cost_feasible_probability_{threshold_name}"] = float(
                            threshold_probabilities[index, threshold_index]
                        )
                if cost_quantiles is not None:
                    item["cost_change_p50"] = float(cost_quantiles[index, 0])
                    item["cost_change_p90"] = float(cost_quantiles[index, 1])
                member.append(item)
            member_outputs.append(member)

        results: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            members = [output[candidate_index] for output in member_outputs]
            numeric_names = tuple(members[0])
            means = {
                name: float(np.mean([member[name] for member in members]))
                for name in numeric_names
            }
            standard_deviations = {
                name: float(np.std([member[name] for member in members]))
                for name in numeric_names
            }
            results.append({
                "candidate_name": candidate.name,
                "partition": candidate.partition,
                "prediction": means,
                "uncertainty": standard_deviations,
                "members": members,
            })
        return results

    def predict(
        self,
        partition: dict[int, Iterable[int]],
        *,
        kind: str = "active",
        strength: float = 0.0,
    ) -> dict[str, Any]:
        """输入一个完整客户划分，输出与批量接口相同的单候选预测。"""
        return self.predict_many(
            [partition], kinds=[kind], strengths=[strength]
        )[0]
