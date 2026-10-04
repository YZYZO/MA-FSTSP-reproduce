"""统一加载数值、排序和风险 GNN 集成，供上层强化学习查询。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .gnn_data import load_partition_gnn_cache
from .gnn_inference import PartitionGNNPredictor


REQUIRED_SURROGATE_ROLES = ("numeric", "ranking", "risk")


def _resolve_bundle_path(bundle_directory: Path, value: str) -> Path:
    """输入部署清单目录和相对或绝对路径，输出规范化的本地路径。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (bundle_directory / path).resolve()


def combine_surrogate_outputs(
    numeric: dict[str, Any],
    ranking: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, Any]:
    """合并三个角色对同一划分的预测，输出上层策略可直接使用的字段。"""
    numeric_prediction = numeric["prediction"]
    ranking_prediction = ranking["prediction"]
    risk_prediction = risk["prediction"]
    return {
        "candidate_name": numeric["candidate_name"],
        "partition": numeric["partition"],
        "prediction": {
            # balanced 数值头作为强化学习奖励和约束的主要点预测。
            "phase2_serial_seconds": numeric_prediction["phase2_serial_seconds"],
            "phase3_seconds": numeric_prediction["phase3_seconds"],
            "downstream_total_seconds": numeric_prediction["downstream_total_seconds"],
            "final_cost": numeric_prediction["final_cost"],
            "time_saving_ratio": numeric_prediction["time_saving_ratio"],
            "cost_change_ratio": numeric_prediction["cost_change_ratio"],
            # target-weighted 头只承担同一实例内部的候选排序。
            "time_rank_score": ranking_prediction["time_rank_score"],
            "cost_rank_score": ranking_prediction["cost_rank_score"],
            "cost_feasible_probability": ranking_prediction[
                "cost_feasible_probability"
            ],
            # robust 头提供风险信号，不替代主点预测。
            "downstream_time_p50": risk_prediction.get(
                "downstream_time_p50",
                risk_prediction["downstream_total_seconds"],
            ),
            "downstream_time_p90": risk_prediction.get(
                "downstream_time_p90",
                risk_prediction["downstream_total_seconds"],
            ),
            "right_censored_probability": risk_prediction[
                "right_censored_probability"
            ],
        },
        "uncertainty": {
            "numeric": numeric["uncertainty"],
            "ranking": ranking["uncertainty"],
            "risk": risk["uncertainty"],
        },
        "members": {
            "numeric": numeric["members"],
            "ranking": ranking["members"],
            "risk": risk["members"],
        },
    }


class PartitionSurrogateBundle:
    """
    为一个路网实例提供统一的三角色监督代理模型。

    初始化时读取可移植部署清单并加载三个集成；predict_many 输入任意完整客户
    划分，输出数值预测、实例内排序、删失风险和三类集成不确定性。
    """

    def __init__(
        self,
        manifest_path: str | Path,
        result_root: str | Path,
        instance_id: str,
        *,
        distance_batch_size: int = 128,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        bundle_directory = self.manifest_path.parent
        roles = self.manifest["roles"]
        missing = [role for role in REQUIRED_SURROGATE_ROLES if role not in roles]
        if missing:
            raise ValueError(f"部署清单缺少角色：{missing}")

        # 每个角色可选择不同邻居数，因此分别持有与检查点匹配的图缓存。
        self.predictors: dict[str, PartitionGNNPredictor] = {}
        for role in REQUIRED_SURROGATE_ROLES:
            specification = roles[role]
            cache_path = _resolve_bundle_path(bundle_directory, specification["cache"])
            checkpoints = [
                _resolve_bundle_path(bundle_directory, value)
                for value in specification["checkpoints"]
            ]
            self.predictors[role] = PartitionGNNPredictor(
                load_partition_gnn_cache(cache_path),
                result_root,
                instance_id,
                checkpoints,
                distance_batch_size=distance_batch_size,
            )

    @property
    def instance(self) -> Any:
        """返回数值模型持有的原始问题实例，便于上层构造初始划分。"""
        return self.predictors["numeric"].instance

    def predict_many(
        self,
        partitions: Sequence[dict[int, Iterable[int]]],
        *,
        kinds: Sequence[str] | None = None,
        strengths: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """批量预测任意完整划分，按输入顺序返回统一三角色结果。"""
        outputs = {
            role: predictor.predict_many(
                partitions,
                kinds=kinds,
                strengths=strengths,
            )
            for role, predictor in self.predictors.items()
        }
        return [
            combine_surrogate_outputs(
                outputs["numeric"][index],
                outputs["ranking"][index],
                outputs["risk"][index],
            )
            for index in range(len(partitions))
        ]

    def predict(
        self,
        partition: dict[int, Iterable[int]],
        *,
        kind: str = "active",
        strength: float = 0.0,
    ) -> dict[str, Any]:
        """预测单个完整划分，返回与批量接口相同的统一结果。"""
        return self.predict_many(
            [partition],
            kinds=[kind],
            strengths=[strength],
        )[0]
