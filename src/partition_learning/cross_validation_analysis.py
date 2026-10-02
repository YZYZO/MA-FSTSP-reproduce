"""汇总候选划分GNN的折外误差、随机种子集成与不确定性。"""

from __future__ import annotations

from collections import defaultdict
import csv
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score

from .models import TARGET_INDEX


def _finite(values: Iterable[float]) -> np.ndarray:
    """输入数值序列，输出仅包含有限值的一维数组。"""
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _record_lookup(cache: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """输入GNN缓存，输出由实例编号和候选名定位标签记录的字典。"""
    return {
        (str(record["instance_id"]), str(record["candidate_name"])): record
        for record in cache["records"]
    }


def _feature_value(cache: dict[str, Any], record: dict[str, Any], name: str) -> float:
    """输入缓存记录和全局特征名，输出该候选对应的原始特征值。"""
    index = cache["feature_names"]["global"].index(name)
    return float(np.asarray(record["global_features"])[index])


def load_cross_validation_rows(
    cache: dict[str, Any],
    cross_validation_dir: str | Path,
) -> list[dict[str, Any]]:
    """
    读取交叉验证目录下全部折外NPZ，并与候选元数据连接。

    输入GNN缓存和第一阶段输出目录；输出每个模型、折次、随机种子、候选一行的字典。
    """
    directory = Path(cross_validation_dir)
    lookup = _record_lookup(cache)
    rows: list[dict[str, Any]] = []
    prediction_paths = sorted(directory.glob("fold_*/seed_*/*/test_predictions.npz"))
    for path in prediction_paths:
        fold = int(path.parents[2].name.split("_")[-1])
        seed = int(path.parents[1].name.split("_")[-1])
        model_name = path.parent.name
        with np.load(path, allow_pickle=False) as data:
            predicted = np.asarray(data["predicted"], dtype=np.float64)
            actual = np.asarray(data["actual"], dtype=np.float64)
            target_mask = np.asarray(data["target_mask"], dtype=bool)
            ranking = np.asarray(data["ranking_scores"], dtype=np.float64)
            feasible_probability = np.asarray(
                data["cost_feasible_probability"], dtype=np.float64
            )
            censored_probability = np.asarray(
                data["right_censored_probability"], dtype=np.float64
            )
            instance_ids = data["instance_ids"].astype(str)
            candidate_names = data["candidate_names"].astype(str)
        for index, (instance_id, candidate_name) in enumerate(
            zip(instance_ids, candidate_names)
        ):
            record = lookup[(instance_id, candidate_name)]
            time_index = TARGET_INDEX["downstream_total_seconds"]
            cost_change_index = TARGET_INDEX["cost_change_ratio"]
            cost_index = TARGET_INDEX["final_cost"]
            # 这些复杂度代理均在求解前可获得，用于定位模型在哪类划分上失准。
            rows.append({
                "fold": fold,
                "seed": seed,
                "model": model_name,
                "instance_id": instance_id,
                "candidate_name": candidate_name,
                "candidate_kind": str(record["candidate_kind"]),
                "graph_name": str(record["graph_name"]),
                "customer_count": int(record["customer_count"]),
                "right_censored": bool(record["right_censored"]),
                "actual_time": float(actual[index, time_index]),
                "predicted_time": float(predicted[index, time_index]),
                "time_mask": bool(target_mask[index, time_index]),
                "time_rank_score": float(ranking[index, 0]),
                "actual_cost_change": float(actual[index, cost_change_index]),
                "actual_cost": float(actual[index, cost_index]),
                "predicted_cost": float(predicted[index, cost_index]),
                "cost_mask": bool(target_mask[index, cost_index]),
                "cost_rank_score": float(ranking[index, 1]),
                "cost_feasible_probability": float(feasible_probability[index]),
                "right_censored_probability": float(censored_probability[index]),
                "candidate_variable_max": _feature_value(
                    cache, record, "candidate_variable_max"
                ),
                "candidate_size_gini": _feature_value(
                    cache, record, "candidate_size_gini"
                ),
                "candidate_boundary_sum": _feature_value(
                    cache, record, "candidate_boundary_sum"
                ),
                "moved_fraction": _feature_value(cache, record, "moved_fraction"),
            })
    if not rows:
        raise FileNotFoundError(f"未找到交叉验证预测：{directory}")
    return rows


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """输入真实值和预测值，在有效样本不少于两个时返回R²。"""
    mask = np.isfinite(actual) & np.isfinite(predicted)
    return float(r2_score(actual[mask], predicted[mask])) if mask.sum() >= 2 else float("nan")


def _error_stats(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """输入一组逐候选记录，输出时间与成本的误差、偏差和R²。"""
    actual_time = np.asarray([row["actual_time"] for row in rows], dtype=np.float64)
    predicted_time = np.asarray([row["predicted_time"] for row in rows], dtype=np.float64)
    time_mask = np.asarray([row["time_mask"] for row in rows], dtype=bool)
    actual_cost = np.asarray([row["actual_cost"] for row in rows], dtype=np.float64)
    predicted_cost = np.asarray([row["predicted_cost"] for row in rows], dtype=np.float64)
    cost_mask = np.asarray([row["cost_mask"] for row in rows], dtype=bool)
    time_error = predicted_time[time_mask] - actual_time[time_mask]
    cost_error = predicted_cost[cost_mask] - actual_cost[cost_mask]
    return {
        "count": len(rows),
        "time_count": int(time_mask.sum()),
        "time_mae": float(np.mean(np.abs(time_error))) if time_error.size else float("nan"),
        "time_bias": float(np.mean(time_error)) if time_error.size else float("nan"),
        "time_r2": _r2(actual_time[time_mask], predicted_time[time_mask]),
        "cost_count": int(cost_mask.sum()),
        "cost_mae": float(np.mean(np.abs(cost_error))) if cost_error.size else float("nan"),
        "cost_bias": float(np.mean(cost_error)) if cost_error.size else float("nan"),
        "cost_r2": _r2(actual_cost[cost_mask], predicted_cost[cost_mask]),
    }


def build_error_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """输入折外逐候选记录，输出按路网、规模、算法族和难度分层的误差报告。"""
    actual_times = _finite(row["actual_time"] for row in rows if row["time_mask"])
    quantiles = np.quantile(actual_times, [0.25, 0.50, 0.75])
    for row in rows:
        row["time_bin"] = int(np.searchsorted(quantiles, row["actual_time"], side="right") + 1)

    dimensions = {
        "overall": (),
        "graph": ("graph_name",),
        "customer_count": ("customer_count",),
        "candidate_kind": ("candidate_kind",),
        "right_censored": ("right_censored",),
        "time_quartile": ("time_bin",),
        "graph_and_size": ("graph_name", "customer_count"),
    }
    report: dict[str, Any] = {"time_quartile_boundaries": quantiles.tolist(), "models": {}}
    for model_name in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model_name]
        model_report: dict[str, Any] = {}
        for dimension, keys in dimensions.items():
            groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in model_rows:
                groups[tuple(row[key] for key in keys)].append(row)
            model_report[dimension] = [
                {
                    "group": " / ".join(str(value) for value in group_key) or "all",
                    **_error_stats(group_rows),
                }
                for group_key, group_rows in sorted(groups.items(), key=lambda item: str(item[0]))
            ]
        report["models"][model_name] = model_report
    return report


def _selection_metrics(
    rows: list[dict[str, Any]],
    *,
    actual_name: str,
    score_name: str,
    mask_name: str,
) -> dict[str, float]:
    """输入集成候选行和字段名，输出实例内Spearman、Top-3命中与选择后悔值。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row[mask_name]:
            groups[row["instance_id"]].append(row)
    correlations, hits, regrets = [], [], []
    for group_rows in groups.values():
        actual = np.asarray([row[actual_name] for row in group_rows], dtype=np.float64)
        score = np.asarray([row[score_name] for row in group_rows], dtype=np.float64)
        mask = np.isfinite(actual) & np.isfinite(score)
        if mask.sum() < 2:
            continue
        actual, score = actual[mask], score[mask]
        correlation = (
            spearmanr(actual, score).statistic
            if np.ptp(actual) > 0 and np.ptp(score) > 0
            else float("nan")
        )
        if np.isfinite(correlation):
            correlations.append(float(correlation))
        best_index = int(np.argmin(actual))
        top_indices = np.argsort(score)[: min(3, score.size)]
        hits.append(float(best_index in top_indices))
        selected_actual = actual[int(np.argmin(score))]
        best_actual = actual[best_index]
        regrets.append(float((selected_actual - best_actual) / max(abs(best_actual), 1e-6)))
    return {
        "mean_within_instance_spearman": float(np.mean(correlations)),
        "top3_hit_fraction": float(np.mean(hits)),
        "mean_regret_ratio": float(np.mean(regrets)),
    }


def _uncertainty_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    """输入集成候选行和目标前缀，输出不确定性与误差的相关性、覆盖率和分箱校准。"""
    actual = np.asarray([row[f"actual_{prefix}"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row[f"predicted_{prefix}"] for row in rows], dtype=np.float64)
    uncertainty = np.asarray([row[f"uncertainty_{prefix}"] for row in rows], dtype=np.float64)
    labelled = np.asarray([row[f"{prefix}_mask"] for row in rows], dtype=bool)
    mask = labelled & np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(uncertainty)
    actual, predicted, uncertainty = actual[mask], predicted[mask], uncertainty[mask]
    absolute_error = np.abs(predicted - actual)
    correlation = (
        spearmanr(uncertainty, absolute_error).statistic
        if actual.size >= 2 and np.ptp(uncertainty) > 0 and np.ptp(absolute_error) > 0
        else float("nan")
    )
    coverage = np.mean(absolute_error <= 1.96 * uncertainty) if actual.size else float("nan")
    bin_ids = np.searchsorted(
        np.quantile(uncertainty, [0.2, 0.4, 0.6, 0.8]), uncertainty, side="right"
    ) if actual.size else np.zeros(0, dtype=np.int64)
    bins = []
    for bin_index in range(5):
        selected = bin_ids == bin_index
        if selected.any():
            bins.append({
                "bin": bin_index + 1,
                "count": int(selected.sum()),
                "mean_uncertainty": float(np.mean(uncertainty[selected])),
                "mae": float(np.mean(absolute_error[selected])),
            })
    return {
        "uncertainty_error_spearman": float(correlation),
        "normal_95_coverage": float(coverage),
        "bins": bins,
    }


def _cross_fitted_uncertainty_calibration(
    rows: list[dict[str, Any]], prefix: str
) -> dict[str, Any]:
    """
    用其他测试折的误差校准当前折模型标准差，输出近似95%区间覆盖率。

    这种交叉拟合避免用同一候选同时估计放大系数和评价覆盖率；结果只用于风险尺度，
    不改变点预测与排序分数。
    """
    fold_reports = []
    all_covered: list[bool] = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        calibration = [
            row for row in rows
            if int(row["fold"]) != fold and row[f"{prefix}_mask"]
        ]
        evaluation = [
            row for row in rows
            if int(row["fold"]) == fold and row[f"{prefix}_mask"]
        ]
        if not calibration or not evaluation:
            continue
        calibration_uncertainty = np.asarray([
            row[f"uncertainty_{prefix}"] for row in calibration
        ], dtype=np.float64)
        positive = calibration_uncertainty[calibration_uncertainty > 0]
        floor = max(float(np.median(positive) * 0.05) if positive.size else 0.0, 1e-6)
        ratios = np.asarray([
            abs(row[f"predicted_{prefix}"] - row[f"actual_{prefix}"])
            / max(row[f"uncertainty_{prefix}"], floor)
            for row in calibration
        ])
        factor = float(np.quantile(ratios, 0.95))
        covered = [
            abs(row[f"predicted_{prefix}"] - row[f"actual_{prefix}"])
            <= factor * max(row[f"uncertainty_{prefix}"], floor)
            for row in evaluation
        ]
        all_covered.extend(covered)
        fold_reports.append({
            "fold": fold,
            "scale_factor": factor,
            "uncertainty_floor": floor,
            "coverage": float(np.mean(covered)),
            "count": len(covered),
        })
    return {
        "target_coverage": 0.95,
        "cross_fitted_coverage": float(np.mean(all_covered)) if all_covered else float("nan"),
        "folds": fold_reports,
    }


def _ensemble_policy_metrics(
    rows: list[dict[str, Any]],
    *,
    cost_limit: float,
    probability_threshold: float,
) -> dict[str, float | int]:
    """按成本可行概率过滤并按时间排序选择，输出集成模型的联合策略指标。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["time_mask"] and row["cost_mask"]:
            groups[row["instance_id"]].append(row)
    violations, regrets, savings = [], [], []
    fallback_count = 0
    for group_rows in groups.values():
        predicted_feasible = [
            row for row in group_rows
            if row["cost_feasible_probability"] >= probability_threshold
        ]
        if predicted_feasible:
            selected = min(predicted_feasible, key=lambda row: row["time_rank_score"])
        else:
            selected = max(group_rows, key=lambda row: row["cost_feasible_probability"])
            fallback_count += 1
        true_feasible = [
            row for row in group_rows if row["actual_cost_change"] <= cost_limit
        ]
        stays = [row for row in group_rows if row["candidate_name"] == "stay"]
        violation = selected["actual_cost_change"] > cost_limit
        violations.append(violation)
        if true_feasible and not violation:
            best = min(true_feasible, key=lambda row: row["actual_time"])
            regrets.append(max(
                0.0,
                (selected["actual_time"] - best["actual_time"])
                / max(abs(best["actual_time"]), 1e-6),
            ))
        if stays:
            baseline = max(abs(stays[0]["actual_time"]), 1e-6)
            savings.append((stays[0]["actual_time"] - selected["actual_time"]) / baseline)
    return {
        "probability_threshold": float(probability_threshold),
        "instance_count": len(violations),
        "fallback_count": fallback_count,
        "true_cost_violation_fraction": float(np.mean(violations)) if violations else float("nan"),
        "mean_feasible_time_regret_ratio": float(np.mean(regrets)) if regrets else float("nan"),
        "mean_selected_time_saving_vs_mst": float(np.mean(savings)) if savings else float("nan"),
    }


def build_seed_ensemble(
    rows: list[dict[str, Any]],
    *,
    cost_limit: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    将相同折次和模型的多个随机种子预测求均值与标准差。

    输出逐候选集成记录和模型级数值预测、排序、安全选择及不确定性报告。
    """
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["fold"], row["instance_id"], row["candidate_name"])
        groups[key].append(row)
    ensemble_rows: list[dict[str, Any]] = []
    for (model_name, fold, instance_id, candidate_name), group_rows in groups.items():
        first = group_rows[0]
        time_values = np.asarray([row["predicted_time"] for row in group_rows])
        cost_values = np.asarray([row["predicted_cost"] for row in group_rows])
        time_scores = np.asarray([row["time_rank_score"] for row in group_rows])
        cost_scores = np.asarray([row["cost_rank_score"] for row in group_rows])
        feasible_values = np.asarray([row["cost_feasible_probability"] for row in group_rows])
        ensemble_rows.append({
            "model": model_name,
            "fold": fold,
            "instance_id": instance_id,
            "candidate_name": candidate_name,
            "candidate_kind": first["candidate_kind"],
            "graph_name": first["graph_name"],
            "customer_count": first["customer_count"],
            "seed_count": len(group_rows),
            "actual_time": first["actual_time"],
            "time_mask": first["time_mask"],
            "predicted_time": float(np.mean(time_values)),
            "uncertainty_time": float(np.std(time_values)),
            "time_rank_score": float(np.mean(time_scores)),
            "actual_cost": first["actual_cost"],
            "actual_cost_change": first["actual_cost_change"],
            "cost_mask": first["cost_mask"],
            "predicted_cost": float(np.mean(cost_values)),
            "uncertainty_cost": float(np.std(cost_values)),
            "cost_rank_score": float(np.mean(cost_scores)),
            "cost_feasible_probability": float(np.mean(feasible_values)),
        })

    report: dict[str, Any] = {"cost_limit": cost_limit, "models": {}}
    for model_name in sorted({row["model"] for row in ensemble_rows}):
        model_rows = [row for row in ensemble_rows if row["model"] == model_name]
        actual_time = np.asarray([row["actual_time"] for row in model_rows])
        predicted_time = np.asarray([row["predicted_time"] for row in model_rows])
        actual_cost = np.asarray([row["actual_cost"] for row in model_rows])
        predicted_cost = np.asarray([row["predicted_cost"] for row in model_rows])
        time_mask = np.asarray([row["time_mask"] for row in model_rows], dtype=bool)
        cost_mask = np.asarray([row["cost_mask"] for row in model_rows], dtype=bool)
        actual_feasible = np.asarray([
            row["actual_cost_change"] <= cost_limit for row in model_rows
        ])
        feasible_probability = np.asarray([
            row["cost_feasible_probability"] for row in model_rows
        ])
        try:
            feasible_auc = float(roc_auc_score(
                actual_feasible[cost_mask], feasible_probability[cost_mask]
            ))
        except ValueError:
            feasible_auc = float("nan")
        report["models"][model_name] = {
            "candidate_count": len(model_rows),
            "time_r2": _r2(actual_time[time_mask], predicted_time[time_mask]),
            "cost_r2": _r2(actual_cost[cost_mask], predicted_cost[cost_mask]),
            "time_selection": _selection_metrics(
                model_rows,
                actual_name="actual_time",
                score_name="time_rank_score",
                mask_name="time_mask",
            ),
            "cost_selection": _selection_metrics(
                model_rows,
                actual_name="actual_cost",
                score_name="cost_rank_score",
                mask_name="cost_mask",
            ),
            "cost_feasible_auc": feasible_auc,
            "cost_feasible_brier": float(np.mean(
                (feasible_probability[cost_mask] - actual_feasible[cost_mask].astype(float)) ** 2
            )),
            "time_uncertainty": _uncertainty_metrics(model_rows, "time"),
            "cost_uncertainty": _uncertainty_metrics(model_rows, "cost"),
            "time_uncertainty_calibration": _cross_fitted_uncertainty_calibration(
                model_rows, "time"
            ),
            "cost_uncertainty_calibration": _cross_fitted_uncertainty_calibration(
                model_rows, "cost"
            ),
            "policy_by_threshold": {
                f"{threshold:.2f}": _ensemble_policy_metrics(
                    model_rows,
                    cost_limit=cost_limit,
                    probability_threshold=threshold,
                )
                for threshold in (0.50, 0.70, 0.80, 0.90)
            },
        }
    return ensemble_rows, report


def write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """输入同构字典行，将其写为便于后续统计的UTF-8 CSV文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_html(
    error_report: dict[str, Any],
    ensemble_report: dict[str, Any],
    path: str | Path,
) -> None:
    """输入误差和集成报告，输出无需外部资源即可浏览的简洁HTML总览。"""
    cards = []
    for model_name, metrics in ensemble_report["models"].items():
        time_selection = metrics["time_selection"]
        cost_selection = metrics["cost_selection"]
        cards.append(
            "<tr>"
            f"<td>{escape(model_name)}</td>"
            f"<td>{metrics['time_r2']:.3f}</td>"
            f"<td>{time_selection['mean_within_instance_spearman']:.3f}</td>"
            f"<td>{time_selection['top3_hit_fraction']:.1%}</td>"
            f"<td>{time_selection['mean_regret_ratio']:.3f}</td>"
            f"<td>{metrics['cost_r2']:.3f}</td>"
            f"<td>{cost_selection['mean_within_instance_spearman']:.3f}</td>"
            f"<td>{cost_selection['top3_hit_fraction']:.1%}</td>"
            f"<td>{cost_selection['mean_regret_ratio']:.3f}</td>"
            "</tr>"
        )
    worst_rows = []
    for model_name, sections in error_report["models"].items():
        kinds = sorted(sections["candidate_kind"], key=lambda item: item["time_mae"], reverse=True)
        for item in kinds[:5]:
            worst_rows.append(
                "<tr>"
                f"<td>{escape(model_name)}</td><td>{escape(item['group'])}</td>"
                f"<td>{item['count']}</td><td>{item['time_mae']:.2f}</td>"
                f"<td>{item['cost_mae']:.2f}</td></tr>"
            )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>监督模型交叉验证分析</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:32px;background:#f6f8fb;color:#17202a}}
table{{border-collapse:collapse;width:100%;background:white;margin:12px 0 28px}}th,td{{padding:9px;border:1px solid #d9e0e8;text-align:right}}th:first-child,td:first-child,td:nth-child(2){{text-align:left}}h1,h2{{color:#17365d}}.note{{color:#566573}}</style></head>
<body><h1>监督模型：折外预测、集成与误差分析</h1>
<p class="note">所有指标来自实例级交叉验证测试折；同一候选的多个随机种子预测取均值。</p>
<h2>随机种子集成效果</h2><table><thead><tr><th>模型</th><th>时间R²</th><th>时间ρ</th><th>时间Top-3</th><th>时间后悔</th><th>成本R²</th><th>成本ρ</th><th>成本Top-3</th><th>成本后悔</th></tr></thead><tbody>{''.join(cards)}</tbody></table>
<h2>各模型时间误差最大的五类划分</h2><table><thead><tr><th>模型</th><th>候选算法族</th><th>行数</th><th>时间MAE/秒</th><th>成本MAE</th></tr></thead><tbody>{''.join(worst_rows)}</tbody></table>
</body></html>"""
    Path(path).write_text(html, encoding="utf-8")


def save_analysis_reports(
    cache: dict[str, Any],
    cross_validation_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """执行误差分析和种子集成，并将CSV、JSON、HTML产物统一写入输出目录。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = load_cross_validation_rows(cache, cross_validation_dir)
    error_report = build_error_analysis(rows)
    ensemble_rows, ensemble_report = build_seed_ensemble(
        rows, cost_limit=float(cache["cost_limit"])
    )
    write_rows_csv(rows, directory / "candidate_errors.csv")
    write_rows_csv(ensemble_rows, directory / "ensemble_predictions.csv")
    (directory / "error_analysis.json").write_text(
        json.dumps(error_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "ensemble_report.json").write_text(
        json.dumps(ensemble_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_analysis_html(error_report, ensemble_report, directory / "analysis.html")
    return {"error_analysis": error_report, "ensemble": ensemble_report}
