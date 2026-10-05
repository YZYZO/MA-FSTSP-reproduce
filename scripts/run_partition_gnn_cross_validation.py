"""对候选划分GNN执行实例级分层交叉验证，并导出折外预测。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.gnn_data import load_partition_gnn_cache  # noqa: E402
from src.partition_learning.gnn import MESSAGE_OPERATORS  # noqa: E402
from src.partition_learning.gnn_training import (  # noqa: E402
    PartitionGNNTrainingConfig,
    partition_loss_profile,
    predict_partition_gnn,
    stratified_instance_folds,
    train_partition_gnn,
)


MODEL_NAMES = (
    "balanced", "selection", "two_stage",
    "quantile_only", "target_weighted", "robust",
)
DEFAULT_MODEL_NAMES = ("balanced", "selection", "two_stage")
SUMMARY_METRIC_NAMES = (
    "time_r2", "time_spearman", "time_top3", "time_regret",
    "cost_r2", "cost_spearman", "cost_top3", "cost_regret",
    "policy_threshold", "cost_violation", "feasible_time_regret",
    "time_saving_vs_mst", "cost_p90_coverage",
    "cost_feasible_5pct_brier", "cost_feasible_7pct_brier",
    "cost_feasible_10pct_brier",
)


def parse_arguments() -> argparse.Namespace:
    """解析缓存、交叉验证规模、模型结构和训练预算。"""
    parser = argparse.ArgumentParser(
        description="实例级分层交叉验证候选划分GNN，并保存折外预测"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--fold-random-seed", type=int, default=260930)
    parser.add_argument(
        "--fold-indices",
        help="可选的一基折号列表，例如2,3；用于把不同折分配给独立本地进程。",
    )
    parser.add_argument("--seeds", default="260915,260916,260917")
    parser.add_argument("--models", default=",".join(DEFAULT_MODEL_NAMES))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--message-operator", choices=MESSAGE_OPERATORS, default="edge_mlp")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--instances-per-batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _parse_integer_list(value: str) -> tuple[int, ...]:
    """输入逗号分隔整数，输出去重且保持原顺序的随机种子。"""
    return tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))


def _parse_model_list(value: str) -> tuple[str, ...]:
    """输入逗号分隔模型名，输出经过允许列表校验的模型序列。"""
    names = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = sorted(set(names) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"未知交叉验证模型：{unknown}")
    return names


def _metric_summary(report: dict[str, Any]) -> dict[str, float]:
    """输入单次训练报告，输出衡量数值预测和候选选择的核心测试指标。"""
    evaluation = report["evaluation"]
    time_regression = evaluation["regression"]["downstream_total_seconds"]
    cost_regression = evaluation["regression"]["final_cost"]
    ranking = evaluation.get("ranking_heads", {})
    time_ranking = ranking.get("downstream_total_seconds", time_regression)
    cost_ranking = ranking.get("final_cost", cost_regression)
    validation_policies = report["validation_evaluation"].get(
        "ranking_policy_by_threshold", {}
    )
    test_policies = evaluation.get("ranking_policy_by_threshold", {})
    if validation_policies and test_policies:
        # 先满足成本违规不高于研究阈值，再在验证集上最小化可行时间后悔。
        selected_threshold = min(
            validation_policies,
            key=lambda threshold: (
                validation_policies[threshold]["true_cost_violation_fraction"] > 0.10,
                validation_policies[threshold]["true_cost_violation_fraction"],
                validation_policies[threshold]["mean_feasible_time_regret_ratio"],
                -validation_policies[threshold]["mean_selected_time_saving_vs_mst"],
            ),
        )
        policy = test_policies[selected_threshold]
        policy_threshold = float(selected_threshold)
    else:
        policy = evaluation["joint_policy"]
        policy_threshold = float("nan")
    cost_quantiles = evaluation.get("cost_change_quantiles", {})
    cost_thresholds = evaluation.get("cost_feasible_by_limit", {})
    return {
        "time_r2": float(time_regression["r2"]),
        "time_spearman": float(time_ranking["mean_within_instance_spearman"]),
        "time_top3": float(time_ranking["true_fastest_top3_hit_fraction"]),
        "time_regret": float(time_ranking["predicted_fastest_mean_regret_ratio"]),
        "cost_r2": float(cost_regression["r2"]),
        "cost_spearman": float(cost_ranking["mean_within_instance_spearman"]),
        "cost_top3": float(cost_ranking["true_fastest_top3_hit_fraction"]),
        "cost_regret": float(cost_ranking["predicted_fastest_mean_regret_ratio"]),
        "policy_threshold": policy_threshold,
        "cost_violation": float(policy["true_cost_violation_fraction"]),
        "feasible_time_regret": float(policy["mean_feasible_time_regret_ratio"]),
        "time_saving_vs_mst": float(policy["mean_selected_time_saving_vs_mst"]),
        "cost_p90_coverage": float(cost_quantiles.get("p90_coverage", float("nan"))),
        "cost_feasible_5pct_brier": float(
            cost_thresholds.get("5pct", {}).get("brier", float("nan"))
        ),
        "cost_feasible_7pct_brier": float(
            cost_thresholds.get("7pct", {}).get("brier", float("nan"))
        ),
        "cost_feasible_10pct_brier": float(
            cost_thresholds.get("10pct", {}).get("brier", float("nan"))
        ),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """输入所有折次结果，输出按模型分组的均值、标准差和有效样本数。"""
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    metric_names = SUMMARY_METRIC_NAMES
    for model_name in MODEL_NAMES:
        model_rows = [row for row in rows if row["model"] == model_name]
        if not model_rows:
            continue
        aggregate[model_name] = {}
        for metric_name in metric_names:
            values = np.asarray(
                [row[metric_name] for row in model_rows], dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            aggregate[model_name][metric_name] = {
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "std": float(np.std(finite)) if finite.size else float("nan"),
                "count": int(finite.size),
            }
    return aggregate


def _save_predictions(path: Path, predictions: dict[str, np.ndarray]) -> None:
    """输入逐候选预测字典，将其无损压缩保存为NPZ供后续分析与集成。"""
    np.savez_compressed(path, **predictions)


def _write_progress(
    path: Path,
    *,
    arguments: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> None:
    """输入当前已完成结果，原子性较弱但可读地刷新断点汇总文件。"""
    payload = {
        "configuration": {
            "cache": str(arguments.cache.resolve()),
            "fold_count": arguments.fold_count,
            "fold_random_seed": arguments.fold_random_seed,
            "fold_indices": arguments.fold_indices,
            "seeds": list(_parse_integer_list(arguments.seeds)),
            "models": list(_parse_model_list(arguments.models)),
            "hidden_dim": arguments.hidden_dim,
            "message_layers": arguments.message_layers,
            "message_operator": arguments.message_operator,
            "max_epochs": arguments.max_epochs,
            "patience": arguments.patience,
        },
        "runs": rows,
        "aggregate": _aggregate_rows(rows),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _train_or_resume(
    cache: dict[str, Any],
    directory: Path,
    *,
    training_config: PartitionGNNTrainingConfig,
    loss_profile: str,
    split: Any,
) -> dict[str, Any]:
    """输入一次运行定义；若结果完整则读取，否则训练并导出测试集逐候选预测。"""
    report_path = directory / "training_report.json"
    checkpoint_path = directory / "partition_gnn_model.pt"
    prediction_path = directory / "test_predictions.npz"
    if report_path.is_file() and checkpoint_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"[resume] {directory}", flush=True)
    else:
        _, report = train_partition_gnn(
            cache,
            directory,
            training_config=training_config,
            loss_config=partition_loss_profile(loss_profile),
            instance_split=split,
        )
    if not prediction_path.is_file():
        predictions = predict_partition_gnn(
            cache,
            checkpoint_path,
            split.test_ids,
            instances_per_batch=training_config.instances_per_batch,
        )
        _save_predictions(prediction_path, predictions)
    return report


def main() -> int:
    """依次执行全部折次与随机种子，并持续写入可恢复的汇总报告。"""
    arguments = parse_arguments()
    seeds = _parse_integer_list(arguments.seeds)
    models = _parse_model_list(arguments.models)
    if "two_stage" in models and "balanced" not in models:
        raise ValueError("two_stage需要同折同随机种子的balanced检查点作为第一阶段。")

    cache = load_partition_gnn_cache(arguments.cache.resolve())
    splits = stratified_instance_folds(
        cache,
        fold_count=arguments.fold_count,
        random_seed=arguments.fold_random_seed,
    )
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_folds = (
        {int(value) for value in arguments.fold_indices.split(",") if value.strip()}
        if arguments.fold_indices else set(range(1, len(splits) + 1))
    )
    suffix = (
        "_folds_" + "_".join(map(str, sorted(selected_folds)))
        if arguments.fold_indices else ""
    )
    report_path = output_dir / f"cross_validation_report{suffix}.json"
    rows: list[dict[str, Any]] = []

    for fold_index, split in enumerate(splits, start=1):
        if fold_index not in selected_folds:
            continue
        for seed in seeds:
            common = PartitionGNNTrainingConfig(
                random_seed=seed,
                hidden_dim=arguments.hidden_dim,
                message_layers=arguments.message_layers,
                message_operator=arguments.message_operator,
                dropout=arguments.dropout,
                learning_rate=arguments.learning_rate,
                weight_decay=arguments.weight_decay,
                max_epochs=arguments.max_epochs,
                patience=arguments.patience,
                instances_per_batch=arguments.instances_per_batch,
                model_variant="gnn_only",
                device=arguments.device,
            )
            balanced_checkpoint = (
                output_dir / f"fold_{fold_index}" / f"seed_{seed}" /
                "balanced" / "partition_gnn_model.pt"
            )
            for model_name in models:
                directory = (
                    output_dir / f"fold_{fold_index}" / f"seed_{seed}" / model_name
                )
                print(
                    f"[CV] fold={fold_index}/{len(splits)} seed={seed} model={model_name}",
                    flush=True,
                )
                if model_name == "balanced":
                    training_config = common
                    loss_profile = "balanced"
                elif model_name == "selection":
                    training_config = common
                    loss_profile = "selection"
                elif model_name == "two_stage":
                    training_config = replace(
                        common,
                        separate_ranking_heads=True,
                        warm_start_checkpoint=str(balanced_checkpoint),
                        train_selection_heads_only=True,
                    )
                    loss_profile = "dual_head"
                elif model_name == "quantile_only":
                    training_config = replace(common, time_quantile_heads=True)
                    loss_profile = "quantile_only"
                elif model_name == "target_weighted":
                    training_config = replace(common, separate_ranking_heads=True)
                    loss_profile = "target_weighted"
                else:
                    training_config = replace(
                        common,
                        separate_ranking_heads=True,
                        time_quantile_heads=True,
                        cost_risk_heads=True,
                    )
                    loss_profile = "robust_targets"
                report = _train_or_resume(
                    cache,
                    directory,
                    training_config=training_config,
                    loss_profile=loss_profile,
                    split=split,
                )
                row = {
                    "fold": fold_index,
                    "seed": seed,
                    "model": model_name,
                    "directory": str(directory),
                    "best_epoch": int(report["best_epoch"]),
                    **_metric_summary(report),
                }
                rows.append(row)
                _write_progress(report_path, arguments=arguments, rows=rows)

    print(f"交叉验证完成：{report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
