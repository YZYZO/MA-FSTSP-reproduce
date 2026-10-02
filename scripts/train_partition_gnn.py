"""构造客户道路图缓存并训练已知道路网内的候选划分GNN。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.deep_sets_data import load_deepsets_cache  # noqa: E402
from src.partition_learning.gnn import GNN_MODEL_VARIANTS, MESSAGE_OPERATORS  # noqa: E402
from src.partition_learning.gnn_data import (  # noqa: E402
    build_partition_gnn_cache,
    load_partition_gnn_cache,
)
from src.partition_learning.gnn_training import (  # noqa: E402
    PartitionGNNTrainingConfig,
    partition_loss_profile,
    run_partition_gnn_ablation,
    train_partition_gnn,
)


def build_loss_config(profile: str):
    """
    输入损失配置名称，输出对应的多任务权重。

    balanced保持历史实验设置；selection加强实例内相对预测、Top-3排序以及
    成本不可行候选的识别，用于直接改善候选选择质量。
    """
    return partition_loss_profile(profile)


def parse_arguments() -> argparse.Namespace:
    """解析集合缓存、道路图缓存、训练结构和优化参数。"""
    parser = argparse.ArgumentParser(
        description="训练客户道路近邻图—仓库分配图候选划分性能预测器"
    )
    parser.add_argument(
        "--deepsets-cache",
        type=Path,
        help="已有deepsets_dataset.pt；首次构造GNN缓存时必须提供。",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        help="包含历史NPZ的结果根目录；首次构造GNN缓存时必须提供。",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        help="GNN缓存路径；默认使用输出目录下的partition_gnn_dataset.pt。",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=260915)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--message-operator", choices=MESSAGE_OPERATORS, default="edge_mlp")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--instances-per-batch", type=int, default=2)
    parser.add_argument(
        "--loss-profile",
        choices=(
            "balanced", "selection", "dual_head", "dual_head_topk",
            "quantile_only", "target_weighted", "robust_targets",
        ),
        default="balanced",
        help=(
            "balanced复现实验；selection加强候选选择；dual_head配合独立排序头；"
            "dual_head_topk进一步强调前三名。"
        ),
    )
    parser.add_argument(
        "--separate-ranking-heads",
        action="store_true",
        help="为总时间和成本变化启用独立排序头，避免排序损失扭曲回归值。",
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        help="从已有GNN模型加载编码器和预测头参数。",
    )
    parser.add_argument(
        "--train-selection-heads-only",
        action="store_true",
        help="冻结编码器和回归头，仅训练独立排序头与成本可行头。",
    )
    parser.add_argument(
        "--model-variant",
        choices=GNN_MODEL_VARIANTS,
        default="gnn_fused",
    )
    parser.add_argument(
        "--ablation-suite",
        action="store_true",
        help="在同一切分上依次训练gnn_only和gnn_fused。",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu或cuda。")
    return parser.parse_args()


def main() -> int:
    """复用或构造GNN缓存，随后训练单模型或两组消融模型。"""
    arguments = parse_arguments()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        arguments.cache or (output_dir / "partition_gnn_dataset.pt")
    ).resolve()

    if arguments.rebuild_cache or not cache_path.is_file():
        if arguments.deepsets_cache is None or arguments.result_root is None:
            raise ValueError(
                "首次构造GNN缓存时必须提供--deepsets-cache和--result-root。"
            )
        deepsets_cache = load_deepsets_cache(arguments.deepsets_cache.resolve())
        cache = build_partition_gnn_cache(
            deepsets_cache,
            arguments.result_root.resolve(),
            cache_path,
            k_neighbors=arguments.k_neighbors,
            distance_batch_size=arguments.distance_batch_size,
        )
    else:
        cache = load_partition_gnn_cache(cache_path)

    if arguments.cache_only:
        print(f"候选划分GNN缓存已就绪：{cache_path}", flush=True)
        return 0

    training_config = PartitionGNNTrainingConfig(
        random_seed=arguments.random_seed,
        validation_fraction=arguments.validation_fraction,
        test_fraction=arguments.test_fraction,
        hidden_dim=arguments.hidden_dim,
        message_layers=arguments.message_layers,
        message_operator=arguments.message_operator,
        dropout=arguments.dropout,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        max_epochs=arguments.max_epochs,
        patience=arguments.patience,
        instances_per_batch=arguments.instances_per_batch,
        model_variant=arguments.model_variant,
        separate_ranking_heads=arguments.separate_ranking_heads,
        warm_start_checkpoint=(
            str(arguments.warm_start_checkpoint.resolve())
            if arguments.warm_start_checkpoint is not None
            else None
        ),
        train_selection_heads_only=arguments.train_selection_heads_only,
        device=arguments.device,
    )
    loss_config = build_loss_config(arguments.loss_profile)
    if arguments.ablation_suite:
        report = run_partition_gnn_ablation(
            cache,
            output_dir / "ablation",
            training_config=training_config,
            loss_config=loss_config,
        )
        print(
            f"GNN两组消融完成：{output_dir / 'ablation' / 'gnn_ablation_report.json'}，"
            f"模型数={len(report['models'])}",
            flush=True,
        )
        return 0

    _, report = train_partition_gnn(
        cache,
        output_dir / arguments.model_variant,
        training_config=training_config,
        loss_config=loss_config,
    )
    print(
        f"{arguments.model_variant}训练完成：best_epoch={report['best_epoch']}，"
        f"模型={output_dir / arguments.model_variant / 'partition_gnn_model.pt'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
