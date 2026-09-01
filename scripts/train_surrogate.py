"""训练并评估里程碑 B 的多任务客户组代理模型。"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

# 直接运行本脚本时使用项目根目录中的 `src` 和 `config` 模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR
from src.learning.features import GROUP_FEATURE_NAMES
from src.learning.surrogate import (
    SurrogateModelConfig,
    checkpoint_payload,
    evaluate_surrogate_model,
    train_surrogate_model,
)


DEFAULT_DATASET_DIRECTORY = RESULTS_DIR / "learning" / "dataset"
DEFAULT_OUTPUT_DIRECTORY = RESULTS_DIR / "learning" / "surrogate"


def _latest_dataset_path():
    """
    查找数据集目录中最新的学习 CSV。

    输入：无。
    输出：最新 CSV 路径；目录为空时返回 `None`。
    逻辑：允许完整里程碑 B 在数据生成后直接训练，无需手工复制文件名。
    """
    candidates = sorted(DEFAULT_DATASET_DIRECTORY.glob("learning_dataset_*.csv"))
    return candidates[-1] if candidates else None


def build_argument_parser():
    """
    构造代理模型训练参数。

    输入：无。
    输出：带小型表格模型默认值的参数解析器。
    逻辑：数据集可省略并自动选择最新文件，训练默认使用早停而非固定跑满。
    """
    parser = argparse.ArgumentParser(
        description="训练成本、Set-TSP 时间、超时和不确定性多任务代理模型。"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="学习数据 CSV；省略时使用结果目录中的最新文件。",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="模型检查点、指标、历史和预测输出目录。",
    )
    parser.add_argument("--run-id", default=None, help="可选输出文件标识。")
    parser.add_argument("--max-epochs", type=int, default=400, help="最大训练轮数。")
    parser.add_argument("--patience", type=int, default=50, help="验证损失早停耐心轮数。")
    parser.add_argument("--batch-size", type=int, default=32, help="训练批量大小。")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW 学习率。")
    parser.add_argument("--dropout", type=float, default=0.15, help="MC Dropout 比例。")
    parser.add_argument("--mc-samples", type=int, default=30, help="评估不确定性的重复前向次数。")
    parser.add_argument("--seed", type=int, default=0, help="训练随机种子。")
    return parser


def _frame_to_arrays(frame):
    """
    将一个 pandas 数据拆分转换为代理模型所需 NumPy 数组。

    输入：包含固定特征列和标签列的 DataFrame。
    输出：特征、回归目标、分类目标和删失标记字典。
    逻辑：超时标签同时作为时间右删失标记，近似成本标签用于损失降权。
    """
    return {
        "features": frame.loc[:, list(GROUP_FEATURE_NAMES)].to_numpy(dtype=np.float32),
        "cost": frame["target_final_cost"].to_numpy(dtype=np.float32),
        "log_time": frame["target_log_set_tsp_time"].to_numpy(dtype=np.float32),
        "timeout": frame["target_set_tsp_timeout"].to_numpy(dtype=np.float32),
        "time_censored": frame["target_set_tsp_timeout"].to_numpy(dtype=np.float32),
        "approximate_cost": frame["target_cost_is_approximate"].to_numpy(dtype=np.float32),
    }


def _instance_sets(frame):
    """
    按拆分返回原始实例标识集合。

    输入：完整数据 DataFrame。
    输出：`{split: set(instance_id)}`。
    逻辑：训练前再次确认相邻分区样本没有跨集合泄漏。
    """
    return {
        split: set(split_frame["base_instance_id"].unique())
        for split, split_frame in frame.groupby("split")
    }


def _quality_gate(validation_metrics, test_metrics):
    """
    根据排序和超时识别能力判断代理模型能否进入强化学习。

    输入：验证集和测试集指标。
    输出：包含阈值、逐项通过状态和总判断的字典。
    逻辑：任何核心指标不可评估或低于阈值时，总判断为不通过。
    """
    criteria = {
        "validation_exact_cost_spearman": (validation_metrics["exact_cost_spearman"], 0.50),
        "test_exact_cost_spearman": (test_metrics["exact_cost_spearman"], 0.50),
        "validation_time_spearman": (validation_metrics["time_spearman"], 0.70),
        "test_time_spearman": (test_metrics["time_spearman"], 0.70),
        "validation_timeout_roc_auc": (validation_metrics["timeout_roc_auc"], 0.70),
        "test_timeout_roc_auc": (test_metrics["timeout_roc_auc"], 0.70),
        "validation_cost_two_sigma_coverage": (
            validation_metrics["cost_two_sigma_coverage"],
            0.80,
        ),
        "test_cost_two_sigma_coverage": (
            test_metrics["cost_two_sigma_coverage"],
            0.80,
        ),
        "validation_time_two_sigma_coverage": (
            validation_metrics["time_two_sigma_coverage"],
            0.80,
        ),
        "test_time_two_sigma_coverage": (
            test_metrics["time_two_sigma_coverage"],
            0.80,
        ),
    }
    checks = {
        name: {
            "value": value,
            "threshold": threshold,
            "passed": value is not None and value >= threshold,
        }
        for name, (value, threshold) in criteria.items()
    }
    return {
        "criteria": checks,
        "eligible_for_reinforcement_learning": all(
            check["passed"] for check in checks.values()
        ),
    }


def _print_training_progress(history_row):
    """
    每隔固定轮数打印训练与验证总损失。

    输入：一个轮次的历史字典。
    输出：无。
    逻辑：避免 400 行冗余输出，同时让长训练保持可观察。
    """
    epoch = history_row["epoch"]
    if epoch == 1 or epoch % 25 == 0:
        print(
            "epoch={:03d} train={:.4f} validation={:.4f}".format(
                epoch,
                history_row["train_total"],
                history_row["validation_total"],
            ),
            flush=True,
        )


def _write_history(path, history):
    """
    将逐轮训练历史写成 CSV。

    输入：输出路径和历史字典列表。
    输出：无。
    逻辑：保留训练过程，便于判断欠拟合、过拟合和早停位置。
    """
    with path.open("w", encoding="utf-8-sig", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main(argv=None):
    """
    读取实例级拆分数据、训练多任务代理模型并执行质量门槛判断。

    输入：可选命令行参数列表。
    输出：进程状态码，并写出模型、指标、训练历史和测试集预测。
    逻辑：即使质量门槛不通过也保存检查点，但明确禁止自动进入强化学习阶段。
    """
    args = build_argument_parser().parse_args(argv)
    dataset_path = args.dataset or _latest_dataset_path()
    if dataset_path is None:
        raise FileNotFoundError("No learning dataset CSV was found.")
    dataset_path = Path(dataset_path)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(dataset_path)
    split_frames = {
        split: frame.loc[frame["split"] == split].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    if any(len(split_frame) == 0 for split_frame in split_frames.values()):
        raise ValueError(
            f"train/validation/test must all be non-empty: "
            f"{ {name: len(value) for name, value in split_frames.items()} }"
        )

    instance_sets = _instance_sets(frame)
    leakage_free = (
        instance_sets["train"].isdisjoint(instance_sets["validation"])
        and instance_sets["train"].isdisjoint(instance_sets["test"])
        and instance_sets["validation"].isdisjoint(instance_sets["test"])
    )
    if not leakage_free:
        raise ValueError("Base-instance leakage detected between data splits.")

    arrays = {
        split: _frame_to_arrays(split_frame)
        for split, split_frame in split_frames.items()
    }
    config = SurrogateModelConfig(
        input_dim=len(GROUP_FEATURE_NAMES),
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        seed=args.seed,
    )

    print(
        "Training surrogate from {} with rows train/validation/test = {}/{}/{}".format(
            dataset_path,
            len(split_frames["train"]),
            len(split_frames["validation"]),
            len(split_frames["test"]),
        ),
        flush=True,
    )
    model, training_artifacts = train_surrogate_model(
        arrays["train"],
        arrays["validation"],
        config,
        progress_callback=_print_training_progress,
    )
    validation_metrics, validation_predictions = evaluate_surrogate_model(
        model,
        arrays["validation"],
        training_artifacts["normalization"],
        mc_samples=args.mc_samples,
    )
    test_metrics, test_predictions = evaluate_surrogate_model(
        model,
        arrays["test"],
        training_artifacts["normalization"],
        mc_samples=args.mc_samples,
    )
    gate = _quality_gate(validation_metrics, test_metrics)
    metrics = {
        "dataset_path": str(dataset_path),
        "row_counts": {
            split: len(split_frame)
            for split, split_frame in split_frames.items()
        },
        "base_instance_counts": {
            split: len(instances)
            for split, instances in instance_sets.items()
        },
        "instance_split_leakage_free": leakage_free,
        "best_epoch": training_artifacts["best_epoch"],
        "best_validation_loss": training_artifacts["best_validation_loss"],
        "timeout_positive_weight": training_artifacts["timeout_positive_weight"],
        "validation": validation_metrics,
        "test": test_metrics,
        "quality_gate": gate,
    }

    checkpoint_path = args.output_directory / f"surrogate_{run_id}.pt"
    metrics_path = args.output_directory / f"surrogate_{run_id}_metrics.json"
    history_path = args.output_directory / f"surrogate_{run_id}_history.csv"
    predictions_path = args.output_directory / f"surrogate_{run_id}_test_predictions.csv"
    torch.save(
        checkpoint_payload(
            model,
            config,
            GROUP_FEATURE_NAMES,
            training_artifacts,
            metrics,
        ),
        checkpoint_path,
    )
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
    _write_history(history_path, training_artifacts["history"])

    prediction_frame = split_frames["test"].loc[:, [
        "sample_key",
        "base_instance_id",
        "customer_size",
        "depot_node",
        "target_final_cost",
        "target_log_set_tsp_time",
        "target_set_tsp_timeout",
    ]].copy()
    for name, values in test_predictions.items():
        prediction_frame[f"predicted_{name}"] = np.asarray(values)
    prediction_frame.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    print(
        "Best epoch={} | validation cost/time rho={:.3f}/{:.3f} | "
        "test cost/time rho={:.3f}/{:.3f}".format(
            training_artifacts["best_epoch"],
            validation_metrics["cost_spearman"],
            validation_metrics["time_spearman"],
            test_metrics["cost_spearman"],
            test_metrics["time_spearman"],
        ),
        flush=True,
    )
    print(
        "Timeout ROC-AUC validation/test={}/{} | eligible_for_RL={}".format(
            validation_metrics["timeout_roc_auc"],
            test_metrics["timeout_roc_auc"],
            gate["eligible_for_reinforcement_learning"],
        ),
        flush=True,
    )
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    print(f"History: {history_path}", flush=True)
    print(f"Test predictions: {predictions_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
