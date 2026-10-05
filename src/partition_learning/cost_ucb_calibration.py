"""用实例级折外集成预测校准相对MST成本变化上界。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .models import TARGET_INDEX


def load_oof_cost_records(
    cross_validation_root: str | Path,
    *,
    model_name: str = "balanced",
    baseline_candidate_name: str = "stay",
    sigma_floor: float = 0.005,
) -> list[dict[str, Any]]:
    """
    从五折多随机种子NPZ读取逐候选相对MST成本预测。

    输入交叉验证根目录、模型名、基线候选名和标准差下限；输出每个折外候选的
    集成均值、标准差、真实成本变化与时间。相对变化按成员内配对计算，保留
    候选与MST预测误差的相关性。
    """
    root = Path(cross_validation_root)
    paths = sorted(root.glob(f"fold_*/seed_*/{model_name}/test_predictions.npz"))
    if not paths:
        raise FileNotFoundError(f"未找到折外预测：{root} / {model_name}")
    by_fold: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        fold_name = next(part for part in path.parts if part.startswith("fold_"))
        by_fold[fold_name].append(path)

    final_cost_index = TARGET_INDEX["final_cost"]
    cost_change_index = TARGET_INDEX["cost_change_ratio"]
    time_index = TARGET_INDEX["downstream_total_seconds"]
    records: list[dict[str, Any]] = []
    for fold_name, member_paths in sorted(by_fold.items()):
        members = [np.load(path, allow_pickle=False) for path in sorted(member_paths)]
        first = members[0]
        instance_ids = first["instance_ids"].astype(str)
        candidate_names = first["candidate_names"].astype(str)
        for member in members[1:]:
            if not (
                np.array_equal(member["instance_ids"].astype(str), instance_ids)
                and np.array_equal(member["candidate_names"].astype(str), candidate_names)
            ):
                raise ValueError(f"{fold_name} 的随机种子预测顺序不一致。")
        predicted = np.stack([member["predicted"] for member in members])
        actual = first["actual"]
        target_mask = first["target_mask"]
        for instance_id in np.unique(instance_ids):
            indices = np.flatnonzero(instance_ids == instance_id)
            baseline_indices = indices[
                candidate_names[indices] == baseline_candidate_name
            ]
            if len(baseline_indices) != 1:
                raise ValueError(
                    f"{fold_name}/{instance_id} 未找到唯一{baseline_candidate_name}基线。"
                )
            baseline_index = int(baseline_indices[0])
            member_baseline_cost = predicted[:, baseline_index, final_cost_index]
            actual_baseline_time = float(actual[baseline_index, time_index])
            for index in indices:
                if not bool(target_mask[index, cost_change_index]):
                    continue
                member_change = (
                    predicted[:, index, final_cost_index] - member_baseline_cost
                ) / np.maximum(np.abs(member_baseline_cost), 1e-6)
                records.append({
                    "fold": fold_name,
                    "instance_id": str(instance_id),
                    "candidate_name": str(candidate_names[index]),
                    "is_baseline": int(index) == baseline_index,
                    "predicted_cost_change_mean": float(np.mean(member_change)),
                    "predicted_cost_change_std": max(
                        float(np.std(member_change)), float(sigma_floor)
                    ),
                    "actual_cost_change": float(actual[index, cost_change_index]),
                    "predicted_time_mean": float(np.mean(
                        predicted[:, index, time_index]
                    )),
                    "actual_time": float(actual[index, time_index]),
                    "actual_baseline_time": actual_baseline_time,
                })
    return records


def conformal_cost_kappa(
    records: Iterable[dict[str, Any]],
    *,
    coverage: float = 0.95,
) -> float:
    """输入校准候选，输出达到目标单侧覆盖率的非负标准差乘数κ。"""
    values = np.asarray([
        (
            float(row["actual_cost_change"])
            - float(row["predicted_cost_change_mean"])
        ) / max(float(row["predicted_cost_change_std"]), 1e-9)
        for row in records
        if not bool(row["is_baseline"])
    ], dtype=np.float64)
    if not len(values):
        raise ValueError("没有非基线候选可用于成本UCB校准。")
    try:
        quantile = float(np.quantile(values, coverage, method="higher"))
    except TypeError:
        quantile = float(np.quantile(values, coverage, interpolation="higher"))
    return max(0.0, quantile)


def evaluate_cost_ucb(
    records: Iterable[dict[str, Any]],
    *,
    kappa: float,
    cost_limit: float = 0.10,
) -> dict[str, Any]:
    """输入折外候选和κ，输出候选级覆盖及实例级选择安全性。"""
    rows = list(records)
    non_baseline = [row for row in rows if not bool(row["is_baseline"])]
    for row in rows:
        row["cost_ucb"] = (
            float(row["predicted_cost_change_mean"])
            + float(kappa) * float(row["predicted_cost_change_std"])
        )
    covered = [
        float(row["actual_cost_change"]) <= float(row["cost_ucb"])
        for row in non_baseline
    ]
    predicted_feasible = [
        row for row in non_baseline if float(row["cost_ucb"]) <= cost_limit
    ]
    false_feasible = [
        row for row in predicted_feasible
        if float(row["actual_cost_change"]) > cost_limit
    ]
    actual_feasible = [
        row for row in non_baseline
        if float(row["actual_cost_change"]) <= cost_limit
    ]
    retained_actual_feasible = [
        row for row in actual_feasible if float(row["cost_ucb"]) <= cost_limit
    ]

    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_instance[str(row["instance_id"])].append(row)
    selections = []
    for instance_id, instance_rows in by_instance.items():
        baseline = next(row for row in instance_rows if bool(row["is_baseline"]))
        feasible_pool = [
            row for row in instance_rows
            if bool(row["is_baseline"]) or float(row["cost_ucb"]) <= cost_limit
        ]
        selected = min(
            feasible_pool, key=lambda row: float(row["predicted_time_mean"])
        )
        actual_violation = max(
            0.0, float(selected["actual_cost_change"]) - cost_limit
        )
        selections.append({
            "instance_id": instance_id,
            "candidate_name": selected["candidate_name"],
            "returned_baseline": bool(selected["is_baseline"]),
            "actual_cost_feasible": float(selected["actual_cost_change"]) <= cost_limit,
            "actual_cost_change": float(selected["actual_cost_change"]),
            "actual_constraint_violation": actual_violation,
            "actual_time_saving": (
                float(baseline["actual_time"]) - float(selected["actual_time"])
            ) / max(float(baseline["actual_time"]), 1e-9),
        })

    return {
        "kappa": float(kappa),
        "candidate_count": len(non_baseline),
        "upper_bound_coverage": float(np.mean(covered)) if covered else 0.0,
        "predicted_feasible_count": len(predicted_feasible),
        "false_feasible_count": len(false_feasible),
        "false_feasible_rate_given_predicted_feasible": (
            len(false_feasible) / max(len(predicted_feasible), 1)
        ),
        "actual_feasible_recall": (
            len(retained_actual_feasible) / max(len(actual_feasible), 1)
        ),
        "selected_true_cost_violation_fraction": float(np.mean([
            not row["actual_cost_feasible"] for row in selections
        ])),
        "selected_mean_constraint_violation": float(np.mean([
            row["actual_constraint_violation"] for row in selections
        ])),
        "selected_mean_time_saving": float(np.mean([
            row["actual_time_saving"] for row in selections
        ])),
        "returned_baseline_fraction": float(np.mean([
            row["returned_baseline"] for row in selections
        ])),
        "selections": selections,
    }


def cross_fitted_cost_calibration(
    records: Iterable[dict[str, Any]],
    *,
    coverage: float = 0.95,
    cost_limit: float = 0.10,
) -> dict[str, Any]:
    """每折使用其余折校准κ并在留出折评价，最后给出全OOF部署κ。"""
    rows = list(records)
    folds = sorted({str(row["fold"]) for row in rows})
    fold_reports = []
    for fold in folds:
        calibration = [row.copy() for row in rows if str(row["fold"]) != fold]
        evaluation = [row.copy() for row in rows if str(row["fold"]) == fold]
        kappa = conformal_cost_kappa(calibration, coverage=coverage)
        fold_reports.append({
            "fold": fold,
            **evaluate_cost_ucb(
                evaluation, kappa=kappa, cost_limit=cost_limit
            ),
        })
    deployment_kappa = conformal_cost_kappa(rows, coverage=coverage)
    return {
        "target_upper_coverage": float(coverage),
        "cost_limit": float(cost_limit),
        "deployment_kappa": deployment_kappa,
        "folds": fold_reports,
        "cross_fitted": {
            "mean_kappa": float(np.mean([row["kappa"] for row in fold_reports])),
            "mean_upper_bound_coverage": float(np.mean([
                row["upper_bound_coverage"] for row in fold_reports
            ])),
            "total_predicted_feasible_count": int(sum(
                row["predicted_feasible_count"] for row in fold_reports
            )),
            "total_false_feasible_count": int(sum(
                row["false_feasible_count"] for row in fold_reports
            )),
            "false_feasible_rate_given_predicted_feasible": (
                sum(row["false_feasible_count"] for row in fold_reports)
                / max(sum(row["predicted_feasible_count"] for row in fold_reports), 1)
            ),
            "mean_selected_true_cost_violation_fraction": float(np.mean([
                row["selected_true_cost_violation_fraction"] for row in fold_reports
            ])),
            "mean_selected_time_saving": float(np.mean([
                row["selected_mean_time_saving"] for row in fold_reports
            ])),
            "mean_returned_baseline_fraction": float(np.mean([
                row["returned_baseline_fraction"] for row in fold_reports
            ])),
        },
        "comparisons": {
            "kappa_0": evaluate_cost_ucb(
                [row.copy() for row in rows], kappa=0.0, cost_limit=cost_limit
            ),
            "kappa_1": evaluate_cost_ucb(
                [row.copy() for row in rows], kappa=1.0, cost_limit=cost_limit
            ),
            "deployment_kappa": evaluate_cost_ucb(
                [row.copy() for row in rows],
                kappa=deployment_kappa,
                cost_limit=cost_limit,
            ),
        },
    }
