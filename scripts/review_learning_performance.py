"""基于已有 CSV 独立核查耗时预测、规模基线和局部动作，不运行真实优化器。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def rank_correlation(expected, predicted):
    """输入真实量和排序分数，返回 Spearman 系数；常量或不足三条时返回空值。"""
    if len(expected) < 3 or len(set(expected)) < 2 or len(set(predicted)) < 2:
        return None
    return float(spearmanr(expected, predicted).statistic)


def audit_predictions(dataset_path, prediction_path):
    """输入单组标签和已保存测试预测，返回规模基线及无超时局部动作的核查结果。"""
    dataset = pd.read_csv(dataset_path)
    predictions = pd.read_csv(prediction_path)
    prediction_columns = [column for column in predictions if column.startswith("predicted_")]
    # 通过样本键关联，防止两个文件行顺序不同造成错误比较。
    frame = dataset.merge(predictions[["sample_key", *prediction_columns]], on="sample_key", validate="one_to_one")
    feature_scores = {
        "surrogate": "predicted_log_time_mean",
        "customer_count": "customer_count",
        "candidate_total": "candidate_total",
        "complexity_proxy": "set_tsp_complexity_proxy",
    }
    subsets = {"all_test": frame, "non_timeout_test": frame[frame.target_set_tsp_timeout == 0]}
    rankings = {
        subset_name: {
            "count": len(subset),
            **{name: rank_correlation(subset.target_log_set_tsp_time, subset[column])
               for name, column in feature_scores.items()},
        }
        for subset_name, subset in subsets.items()
    }

    actions = []
    for instance_id, instance in frame.groupby("base_instance_id"):
        base = instance[instance.action_type == "base"].set_index("depot_node")
        for partition_id, after in instance[instance.action_type != "base"].groupby("partition_id"):
            # 去重后若缺少动作的任一客户组，则不把不完整动作加入双组差值比较。
            if len(after) != 2:
                continue
            before = base.loc[after.depot_node]
            if before.target_set_tsp_timeout.any() or after.target_set_tsp_timeout.any():
                continue
            actual_gain = float(before.set_tsp_wall_seconds.sum() - after.set_tsp_wall_seconds.sum())
            predicted_gain = float(before.predicted_wall_time_mean.sum() - after.predicted_wall_time_mean.sum())
            actions.append({
                "instance_id": instance_id,
                "partition_id": partition_id,
                "actual_gain_seconds": actual_gain,
                "predicted_gain_seconds": predicted_gain,
                "direction_correct": bool(actual_gain * predicted_gain > 0.0),
            })
    material_actions = [action for action in actions if abs(action["actual_gain_seconds"]) >= 0.05]
    return {
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        "dataset_rows": len(dataset),
        "base_instances": int(dataset.base_instance_id.nunique()),
        "test_instances": int(frame.base_instance_id.nunique()),
        "constant_features": [name for name in feature_scores.values() if frame[name].nunique() == 1],
        "rankings_against_observed_wall_time": rankings,
        "local_actions_without_timeout": actions,
        "local_action_direction_accuracy": (
            sum(action["direction_correct"] for action in actions) / len(actions) if actions else None
        ),
        "actions_with_at_least_50ms_change": len(material_actions),
        "direction_accuracy_at_least_50ms": (
            sum(action["direction_correct"] for action in material_actions) / len(material_actions)
            if material_actions else None
        ),
        "note": "超时样本仅有删失下界；总体相关系数对照的是记录时间，不能作为真实长尾完成时间的精度。",
    }


def audit_partition_evaluations(path):
    """输入完整划分评估 CSV，按规模和方法汇总顺序执行的第二阶段总耗时及分组变化。"""
    frame = pd.read_csv(path)
    rows = []
    for (size, method), bucket in frame.groupby(["customer_size", "method"]):
        group_rows = [group for value in bucket.group_records for group in json.loads(value)]
        rows.append({
            "customer_size": int(size), "method": method, "instances": len(bucket),
            "phase2_mean_seconds": float(bucket.set_tsp_total_seconds.mean()),
            "phase2_instance_p90_seconds": float(bucket.set_tsp_total_seconds.quantile(0.9)),
            "strategy_mean_seconds": float(bucket.partition_strategy_seconds.mean()),
            "phase1_strategy_plus_phase2_mean_seconds": float((bucket.set_tsp_total_seconds + bucket.partition_strategy_seconds).mean()),
            "timeout_count": int(bucket.timeout_count.sum()),
            "group_p90_seconds": float(np.percentile([group["set_tsp_wall_seconds"] for group in group_rows], 90)),
            "max_group_customers": int(max(group["customer_count"] for group in group_rows)),
            "mean_final_cost": float(bucket.final_cost.mean()),
        })
    baseline = {row["customer_size"]: row for row in rows if row["method"] == "symmetric_mst"}
    for row in rows:
        reference = baseline[row["customer_size"]]
        row["phase2_change_vs_symmetric"] = row["phase2_mean_seconds"] / reference["phase2_mean_seconds"] - 1.0
        row["strategy_plus_phase2_change_vs_symmetric"] = row["phase1_strategy_plus_phase2_mean_seconds"] / reference["phase1_strategy_plus_phase2_mean_seconds"] - 1.0
        row["cost_change_vs_symmetric"] = row["mean_final_cost"] / reference["mean_final_cost"] - 1.0
        row["passes_5_percent_cost_and_20_percent_time_reduction_on_means"] = bool(
            row["cost_change_vs_symmetric"] <= 0.05 and row["phase2_change_vs_symmetric"] <= -0.20
        )
    return rows


def main():
    """发现两套完整预测产物及服务器评估文件，写出独立 JSON 核查报告并打印位置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/learning/review/performance_audit.json")
    args = parser.parse_args()
    # 直接读取保存的预测，不重新训练、选择检查点或用测试集调整参数。
    datasets = list((PROJECT_ROOT / "results").rglob("learning_dataset_*.csv"))
    reports = []
    for prediction_path in sorted((PROJECT_ROOT / "results").rglob("*_test_predictions.csv")):
        if "-final_" not in prediction_path.name and "linux-surrogate" not in prediction_path.name:
            continue
        directory = prediction_path.parent.parent
        matching = [path for path in datasets if path.parent.parent == directory]
        if len(matching) == 1:
            reports.append(audit_predictions(matching[0], prediction_path))
    evaluation_paths = list((PROJECT_ROOT / "results").rglob("partition_evaluation_linux-eval-pilot.csv"))
    payload = {
        "user_targets": {
            "maximum_cost_increase": 0.05,
            "minimum_phase2_wall_time_reduction": 0.20,
            "comparison": "相同规模下相对 symmetric_mst 的平均指标；逐实例约束仍需单独验收。",
        },
        "surrogate_audits": reports,
        "partition_audits": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "rows": audit_partition_evaluations(path)}
            for path in evaluation_paths
        ],
        "limitations": [
            "复核既有记录，没有重新计时；缓存记录及硬件差异会影响耗时解释。",
            "两套测试集各只有三个原始实例，动作方向样本更少，不能据此估计稳定泛化精度。",
            "局部动作仅保留前后两个客户组均有记录且均未超时的情况。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
