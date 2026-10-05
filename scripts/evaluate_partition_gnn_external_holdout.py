"""用训练阶段未见过的新实例评价多个 GNN 检查点集成。"""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.deep_sets import (  # noqa: E402
    COST_FEASIBILITY_NAMES,
    COST_FEASIBILITY_THRESHOLDS,
)
from src.partition_learning.deep_sets_training import (  # noqa: E402
    _classification_metrics,
    _regression_metrics,
    _within_instance_metrics,
    evaluate_candidate_predictions,
)
from src.partition_learning.gnn_data import load_partition_gnn_cache  # noqa: E402
from src.partition_learning.gnn_training import predict_partition_gnn  # noqa: E402
from src.partition_learning.models import TARGET_INDEX  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """解析缓存、新实例清单、检查点集成和报告输出目录。"""
    parser = argparse.ArgumentParser(description="评价 GNN 在独立主动学习实例上的泛化性能")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--instance-manifest", type=Path, required=True)
    parser.add_argument(
        "--ensemble",
        action="append",
        required=True,
        help="格式：名称=交叉验证根目录::模型子目录，例如 balanced=stage1::balanced",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instances-per-batch", type=int, default=2)
    return parser.parse_args()


def parse_ensemble_specification(value: str) -> tuple[str, Path, str]:
    """输入“名称=根目录::模型名”，输出集成名称、根目录与模型子目录。"""
    name, expression = value.split("=", 1)
    root, model_name = expression.rsplit("::", 1)
    return name.strip(), Path(root.strip()), model_name.strip()


def load_manifest_instance_ids(path: Path) -> tuple[str, ...]:
    """读取 Stage 7 实例清单，按清单顺序返回独立实例编号。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(str(item["instance_id"]) for item in payload["instances"])


def find_checkpoints(root: Path, model_name: str) -> tuple[Path, ...]:
    """在五折目录中查找指定模型的全部随机种子检查点。"""
    checkpoints = tuple(sorted(
        root.glob(f"fold_*/seed_*/{model_name}/partition_gnn_model.pt")
    ))
    if not checkpoints:
        raise FileNotFoundError(f"未找到检查点：{root} / {model_name}")
    return checkpoints


def aggregate_member_predictions(
    members: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """输入多个成员的同序预测，输出均值、标准差以及首成员携带的真实标签。"""
    first = members[0]
    result = {
        "actual": first["actual"],
        "target_mask": first["target_mask"],
        "instance_ids": first["instance_ids"],
        "candidate_names": first["candidate_names"],
        "predicted": np.mean([item["predicted"] for item in members], axis=0),
        "predicted_std": np.std([item["predicted"] for item in members], axis=0),
        "ranking_scores": np.mean([item["ranking_scores"] for item in members], axis=0),
        "ranking_scores_std": np.std(
            [item["ranking_scores"] for item in members], axis=0
        ),
        "cost_feasible_probability": np.mean(
            [item["cost_feasible_probability"] for item in members], axis=0
        ),
        "right_censored_probability": np.mean(
            [item["right_censored_probability"] for item in members], axis=0
        ),
    }
    if all("time_quantiles" in item for item in members):
        result["time_quantiles"] = np.mean(
            [item["time_quantiles"] for item in members], axis=0
        )
    if all("cost_change_quantiles" in item for item in members):
        result["cost_change_quantiles"] = np.mean(
            [item["cost_change_quantiles"] for item in members], axis=0
        )
    if all("cost_feasible_threshold_probability" in item for item in members):
        result["cost_feasible_threshold_probability"] = np.mean(
            [item["cost_feasible_threshold_probability"] for item in members],
            axis=0,
        )
    return result


def _uncertainty_summary(
    actual: np.ndarray,
    predicted: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, float]:
    """输入真实值、集成均值和成员标准差，输出不确定性与绝对误差的关系。"""
    absolute_error = np.abs(predicted - actual)
    correlation = (
        spearmanr(uncertainty, absolute_error).statistic
        if len(actual) >= 2
        and float(np.ptp(uncertainty)) > 0.0
        and float(np.ptp(absolute_error)) > 0.0
        else float("nan")
    )
    return {
        "uncertainty_error_spearman": float(correlation),
        "normal_95_coverage": float(np.mean(absolute_error <= 1.96 * uncertainty)),
        "mean_uncertainty": float(np.mean(uncertainty)),
    }


def _record_labels(
    cache: dict[str, Any],
    instance_ids: np.ndarray,
    candidate_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按预测顺序连接缓存记录，返回删失、成本可行真值与可行标签掩码。"""
    lookup = {
        (str(record["instance_id"]), str(record["candidate_name"])): record
        for record in cache["records"]
    }
    rows = [
        lookup[(str(instance_id), str(candidate_name))]
        for instance_id, candidate_name in zip(instance_ids, candidate_names)
    ]
    return (
        np.asarray([row["right_censored"] for row in rows], dtype=np.float64),
        np.asarray([row["cost_feasible"] for row in rows], dtype=np.float64),
        np.asarray([row["cost_feasible_mask"] for row in rows], dtype=bool),
    )


def evaluate_ensemble(
    cache: dict[str, Any],
    instance_ids: tuple[str, ...],
    checkpoints: tuple[Path, ...],
    *,
    instances_per_batch: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """运行一个检查点集成，返回模型级报告和逐候选明细。"""
    members = [
        predict_partition_gnn(
            cache,
            checkpoint,
            instance_ids,
            instances_per_batch=instances_per_batch,
        )
        for checkpoint in checkpoints
    ]
    ensemble = aggregate_member_predictions(members)
    actual = ensemble["actual"]
    predicted = ensemble["predicted"]
    masks = ensemble["target_mask"]
    ids = ensemble["instance_ids"].astype(str)
    names = ensemble["candidate_names"].astype(str)
    ranking = ensemble["ranking_scores"]
    censored, feasible, feasible_mask = _record_labels(cache, ids, names)

    report = evaluate_candidate_predictions(
        actual,
        predicted,
        masks,
        ids,
        names,
        cost_limit=float(cache["cost_limit"]),
    )
    time_index = TARGET_INDEX["downstream_total_seconds"]
    final_cost_index = TARGET_INDEX["final_cost"]
    time_valid = masks[:, time_index]
    exact_time = time_valid & (censored < 0.5)
    cost_valid = masks[:, final_cost_index]
    report["member_count"] = len(checkpoints)
    report["candidate_count"] = len(ids)
    report["instance_count"] = len(set(ids))
    report["ranking_heads"] = {
        "downstream_total_seconds": _within_instance_metrics(
            actual[time_valid, time_index],
            ranking[time_valid, 0],
            ids[time_valid],
        ),
        "final_cost": _within_instance_metrics(
            actual[cost_valid, final_cost_index],
            ranking[cost_valid, 1],
            ids[cost_valid],
        ),
    }
    report["exact_downstream_time"] = {
        **_regression_metrics(
            actual[exact_time, time_index],
            predicted[exact_time, time_index],
        ),
        **_within_instance_metrics(
            actual[exact_time, time_index],
            ranking[exact_time, 0],
            ids[exact_time],
        ),
        "count": int(exact_time.sum()),
    }
    report["right_censored"] = _classification_metrics(
        censored, ensemble["right_censored_probability"]
    )
    report["cost_feasible"] = _classification_metrics(
        feasible[feasible_mask],
        ensemble["cost_feasible_probability"][feasible_mask],
    )
    report["time_uncertainty"] = _uncertainty_summary(
        actual[time_valid, time_index],
        predicted[time_valid, time_index],
        ensemble["predicted_std"][time_valid, time_index],
    )
    report["cost_uncertainty"] = _uncertainty_summary(
        actual[cost_valid, final_cost_index],
        predicted[cost_valid, final_cost_index],
        ensemble["predicted_std"][cost_valid, final_cost_index],
    )
    if "time_quantiles" in ensemble:
        quantiles = ensemble["time_quantiles"]
        report["downstream_time_quantiles"] = {
            "p50_mae_exact": float(np.mean(np.abs(
                quantiles[exact_time, 0] - actual[exact_time, time_index]
            ))),
            "p90_coverage_exact": float(np.mean(
                actual[exact_time, time_index] <= quantiles[exact_time, 1]
            )),
            "p90_censored_lower_bound_satisfaction": float(np.mean(
                quantiles[censored > 0.5, 1] >= actual[censored > 0.5, time_index]
            )),
        }

    cost_change_index = TARGET_INDEX["cost_change_ratio"]
    relative_cost_valid = masks[:, cost_change_index]
    if "cost_change_quantiles" in ensemble:
        cost_quantiles = ensemble["cost_change_quantiles"]
        report["cost_change_quantiles"] = {
            "p50_mae": float(np.mean(np.abs(
                cost_quantiles[relative_cost_valid, 0]
                - actual[relative_cost_valid, cost_change_index]
            ))),
            "p90_coverage": float(np.mean(
                actual[relative_cost_valid, cost_change_index]
                <= cost_quantiles[relative_cost_valid, 1]
            )),
        }
    if "cost_feasible_threshold_probability" in ensemble:
        threshold_probability = ensemble["cost_feasible_threshold_probability"]
        report["cost_feasible_by_limit"] = {
            name: _classification_metrics(
                (
                    actual[relative_cost_valid, cost_change_index] <= threshold
                ).astype(np.float64),
                threshold_probability[relative_cost_valid, index],
            )
            for index, (name, threshold) in enumerate(zip(
                COST_FEASIBILITY_NAMES, COST_FEASIBILITY_THRESHOLDS
            ))
        }

    rows = []
    for index, (instance_id, candidate_name) in enumerate(zip(ids, names)):
        rows.append({
            "instance_id": instance_id,
            "candidate_name": candidate_name,
            "right_censored": int(censored[index] > 0.5),
            "actual_time": float(actual[index, time_index]),
            "predicted_time": float(predicted[index, time_index]),
            "time_uncertainty": float(ensemble["predicted_std"][index, time_index]),
            "time_rank_score": float(ranking[index, 0]),
            "actual_cost": float(actual[index, final_cost_index]),
            "predicted_cost": float(predicted[index, final_cost_index]),
            "cost_uncertainty": float(ensemble["predicted_std"][index, final_cost_index]),
            "cost_rank_score": float(ranking[index, 1]),
            "cost_feasible_probability": float(
                ensemble["cost_feasible_probability"][index]
            ),
            "right_censored_probability": float(
                ensemble["right_censored_probability"][index]
            ),
        })
    return report, rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """把逐候选外部预测写成可直接分析的 UTF-8 CSV。"""
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, reports: dict[str, dict[str, Any]]) -> None:
    """生成无需外部依赖的外部留出核心指标表。"""
    body = []
    for name, report in reports.items():
        time = report["regression"]["downstream_total_seconds"]
        exact = report["exact_downstream_time"]
        time_rank = report["ranking_heads"]["downstream_total_seconds"]
        cost = report["regression"]["final_cost"]
        cost_rank = report["ranking_heads"]["final_cost"]
        body.append(
            "<tr>"
            f"<td>{escape(name)}</td><td>{report['member_count']}</td>"
            f"<td>{time['r2']:.3f}</td><td>{exact['r2']:.3f}</td>"
            f"<td>{time_rank['mean_within_instance_spearman']:.3f}</td>"
            f"<td>{time_rank['true_fastest_top3_hit_fraction']:.1%}</td>"
            f"<td>{time_rank['predicted_fastest_mean_regret_ratio']:.3f}</td>"
            f"<td>{cost['r2']:.3f}</td>"
            f"<td>{cost_rank['mean_within_instance_spearman']:.3f}</td>"
            f"<td>{cost_rank['true_fastest_top3_hit_fraction']:.1%}</td>"
            f"<td>{cost_rank['predicted_fastest_mean_regret_ratio']:.3f}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Stage 7 外部留出评估</title>
<style>body{{font-family:system-ui;margin:32px;background:#f7f8fa;color:#17202a}}
table{{border-collapse:collapse;background:white}}th,td{{padding:10px 12px;border:1px solid #d9dee7}}
th{{background:#eaf0f8}}small{{color:#536171}}</style></head>
<body><h1>Stage 7 外部留出评估</h1>
<p>6 个训练阶段未见实例，共 72 个候选；19 个时间标签为右删失。</p>
<table><thead><tr><th>模型</th><th>成员</th><th>时间 R²（含删失下界）</th>
<th>时间 R²（仅精确）</th><th>时间实例内 ρ</th><th>时间 Top-3</th><th>时间后悔</th>
<th>成本 R²</th><th>成本实例内 ρ</th><th>成本 Top-3</th><th>成本后悔</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>
<p><small>新实例未参与这些模型训练，因此该表是独立的前瞻性泛化检验。</small></p>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    """执行全部集成的外部留出评价并写出 JSON、CSV 与 HTML。"""
    arguments = parse_arguments()
    cache = load_partition_gnn_cache(arguments.cache.resolve())
    instance_ids = load_manifest_instance_ids(arguments.instance_manifest.resolve())
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict[str, Any]] = {}
    for value in arguments.ensemble:
        name, root, model_name = parse_ensemble_specification(value)
        checkpoints = find_checkpoints(root.resolve(), model_name)
        print(f"[external] {name}: {len(checkpoints)} 个检查点", flush=True)
        report, rows = evaluate_ensemble(
            cache,
            instance_ids,
            checkpoints,
            instances_per_batch=arguments.instances_per_batch,
        )
        reports[name] = report
        write_rows(output_dir / f"{name}_predictions.csv", rows)
    payload = {
        "kind": "partition_gnn_external_holdout",
        "cache": str(arguments.cache.resolve()),
        "instance_ids": list(instance_ids),
        "reports": reports,
    }
    (output_dir / "external_holdout_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_html(output_dir / "external_holdout_report.html", reports)
    print(f"外部留出评估完成：{output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
