"""比较邻居数、消息层数和道路消息算子的实例级交叉验证表现。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.gnn_data import load_partition_gnn_cache  # noqa: E402
from src.partition_learning.gnn_training import (  # noqa: E402
    PartitionGNNTrainingConfig,
    partition_loss_profile,
    stratified_instance_folds,
    train_partition_gnn,
)


def parse_arguments() -> argparse.Namespace:
    """解析三个k邻居缓存、交叉验证预算和结构消融参数。"""
    parser = argparse.ArgumentParser(description="候选划分GNN图结构消融")
    parser.add_argument(
        "--cache",
        action="append",
        required=True,
        help="格式为k=缓存路径，可重复传入，例如 --cache 8=data.pt。",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument(
        "--fold-indices",
        default=None,
        help="仅运行指定折，使用逗号分隔的一基索引，例如 1,3；默认运行全部折。",
    )
    parser.add_argument("--fold-random-seed", type=int, default=260930)
    parser.add_argument("--random-seed", type=int, default=260915)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _selected_fold_indices(value: str | None, fold_count: int) -> list[int]:
    """解析可选折索引，返回需要运行的一基折编号列表。"""
    if value is None:
        return list(range(1, fold_count + 1))
    selected = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    invalid = [index for index in selected if index < 1 or index > fold_count]
    if invalid:
        raise ValueError(f"折索引超出 1..{fold_count}：{invalid}")
    return selected


def _report_filename(selected_folds: list[int], fold_count: int) -> str:
    """根据运行折集合生成互不冲突的进度报告文件名。"""
    if selected_folds == list(range(1, fold_count + 1)):
        return "structure_ablation_report.json"
    suffix = "_".join(str(index) for index in selected_folds)
    return f"structure_ablation_report_folds_{suffix}.json"


def _cache_paths(values: list[str]) -> dict[int, Path]:
    """输入k=路径参数，输出按邻居数索引的绝对缓存路径。"""
    result = {}
    for value in values:
        key, path = value.split("=", 1)
        result[int(key)] = Path(path).resolve()
    return result


def _experiment_grid(cache_paths: dict[int, Path]) -> list[dict[str, Any]]:
    """输入可用缓存，输出去重后的单因素邻居、层数和算子消融网格。"""
    experiments: list[dict[str, Any]] = []
    for k_neighbors in sorted(cache_paths):
        experiments.append({
            "name": f"neighbors_k{k_neighbors}",
            "k_neighbors": k_neighbors,
            "message_layers": 2,
            "message_operator": "edge_mlp",
        })
    for layers in (1, 2, 3):
        experiments.append({
            "name": f"layers_{layers}",
            "k_neighbors": 8,
            "message_layers": layers,
            "message_operator": "edge_mlp",
        })
    for operator in ("edge_mlp", "edge_attention", "multiscale"):
        experiments.append({
            "name": f"operator_{operator}",
            "k_neighbors": 8,
            "message_layers": 2,
            "message_operator": operator,
        })
    unique: dict[tuple[int, int, str], dict[str, Any]] = {}
    for experiment in experiments:
        key = (
            experiment["k_neighbors"],
            experiment["message_layers"],
            experiment["message_operator"],
        )
        unique.setdefault(key, experiment)
    return list(unique.values())


def _metrics(report: dict[str, Any]) -> dict[str, float]:
    """输入单折训练报告，输出结构消融关注的测试指标。"""
    evaluation = report["evaluation"]
    time = evaluation["regression"]["downstream_total_seconds"]
    cost = evaluation["regression"]["final_cost"]
    return {
        "time_r2": float(time["r2"]),
        "time_spearman": float(time["mean_within_instance_spearman"]),
        "time_top3": float(time["true_fastest_top3_hit_fraction"]),
        "time_regret": float(time["predicted_fastest_mean_regret_ratio"]),
        "cost_r2": float(cost["r2"]),
        "cost_spearman": float(cost["mean_within_instance_spearman"]),
        "cost_top3": float(cost["true_fastest_top3_hit_fraction"]),
        "cost_regret": float(cost["predicted_fastest_mean_regret_ratio"]),
    }


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
) -> None:
    """输入已完成折次，刷新逐折结果和按结构汇总的均值标准差。"""
    aggregate = {}
    metric_names = (
        "time_r2", "time_spearman", "time_top3", "time_regret",
        "cost_r2", "cost_spearman", "cost_top3", "cost_regret",
    )
    for experiment in experiments:
        selected = [row for row in rows if row["experiment"] == experiment["name"]]
        if not selected:
            continue
        aggregate[experiment["name"]] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in selected])),
                "std": float(np.std([row[metric] for row in selected])),
            }
            for metric in metric_names
        }
    path.write_text(json.dumps({
        "experiments": experiments,
        "runs": rows,
        "aggregate": aggregate,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    """对七个单因素图结构配置逐折训练，支持依据已有训练报告断点续跑。"""
    arguments = parse_arguments()
    cache_paths = _cache_paths(arguments.cache)
    if 8 not in cache_paths:
        raise ValueError("层数和消息算子消融需要k=8缓存。")
    experiments = _experiment_grid(cache_paths)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # 每个并行进程只写自己负责的折报告，模型目录也按折隔离。
    selected_folds = _selected_fold_indices(arguments.fold_indices, arguments.fold_count)
    rows: list[dict[str, Any]] = []
    report_path = output_dir / _report_filename(selected_folds, arguments.fold_count)

    for experiment in experiments:
        cache = load_partition_gnn_cache(cache_paths[experiment["k_neighbors"]])
        splits = stratified_instance_folds(
            cache,
            fold_count=arguments.fold_count,
            random_seed=arguments.fold_random_seed,
        )
        config = PartitionGNNTrainingConfig(
            random_seed=arguments.random_seed,
            message_layers=experiment["message_layers"],
            message_operator=experiment["message_operator"],
            model_variant="gnn_only",
            max_epochs=arguments.max_epochs,
            patience=arguments.patience,
            device=arguments.device,
        )
        for fold_index, split in enumerate(splits, start=1):
            if fold_index not in selected_folds:
                continue
            directory = output_dir / experiment["name"] / f"fold_{fold_index}"
            saved_report = directory / "training_report.json"
            print(
                f"[结构消融] {experiment['name']} fold={fold_index}/{len(splits)}",
                flush=True,
            )
            if saved_report.is_file() and (directory / "partition_gnn_model.pt").is_file():
                report = json.loads(saved_report.read_text(encoding="utf-8"))
            else:
                _, report = train_partition_gnn(
                    cache,
                    directory,
                    training_config=config,
                    loss_config=partition_loss_profile("balanced"),
                    instance_split=split,
                )
            rows.append({
                "experiment": experiment["name"],
                "fold": fold_index,
                "configuration": asdict(config),
                "best_epoch": int(report["best_epoch"]),
                **_metrics(report),
            })
            _write_report(report_path, rows, experiments)
    print(f"图结构消融完成：{report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
