"""从现有候选划分构造层次集合缓存并训练 Deep Sets 性能预测器。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.deep_sets import (  # noqa: E402
    DeepSetsLossConfig,
    MODEL_VARIANTS,
)
from src.partition_learning.deep_sets_ablation import run_ablation_suite  # noqa: E402
from src.partition_learning.deep_sets_data import (  # noqa: E402
    build_deepsets_cache,
    load_deepsets_cache,
)
from src.partition_learning.deep_sets_training import (  # noqa: E402
    DeepSetsTrainingConfig,
    train_deepsets,
)
from src.partition_learning.reporting import (  # noqa: E402
    deduplicate_candidate_records,
    read_jsonl,
)


def parse_arguments() -> argparse.Namespace:
    """解析候选数据、历史NPZ、缓存、模型和损失参数，返回命令行配置。"""
    parser = argparse.ArgumentParser(
        description="训练道路节点集合—客户—仓库组—划分四层 Deep Sets 模型"
    )
    parser.add_argument(
        "--records",
        type=Path,
        nargs="+",
        help="构造新缓存时提供一个或多个candidate_records.jsonl；复用缓存时可省略。",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        help="包含原始历史NPZ的结果根目录；首次构造缓存时必须提供。",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        help="张量缓存路径；默认使用输出目录下的 deepsets_dataset.pt。",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--cost-limit", type=float, default=0.10)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=260915)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--instances-per-batch", type=int, default=2)
    parser.add_argument("--include-method-features", action="store_true")
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default="fused",
        help="单模型训练结构；消融模式会依次训练全部结构。",
    )
    parser.add_argument(
        "--ablation-suite",
        action="store_true",
        help="运行人工特征、纯Deep Sets、纯全局MLP和融合模型四组公平消融。",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu或cuda。")
    return parser.parse_args()


def _resolve_record_path(path: Path) -> Path:
    """输入候选JSONL或实验目录，输出实际读取的候选文件。"""
    if path.is_file():
        return path
    preferred = path / "candidate_records_selected.jsonl"
    if preferred.is_file():
        return preferred
    active = path / "candidate_records_active.jsonl"
    if active.is_file():
        return active
    return path / "candidate_records.jsonl"


def main() -> int:
    """复用或构造张量缓存，随后训练并保存层次化 Deep Sets。"""
    arguments = parse_arguments()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (arguments.cache or (output_dir / "deepsets_dataset.pt")).resolve()

    if arguments.rebuild_cache or not cache_path.is_file():
        if arguments.result_root is None:
            raise ValueError("首次构造 Deep Sets 缓存时必须提供 --result-root。")
        if not arguments.records:
            raise ValueError("首次构造 Deep Sets 缓存时必须提供 --records。")
        record_paths = [_resolve_record_path(path.resolve()) for path in arguments.records]
        records = deduplicate_candidate_records([
            row for path in record_paths for row in read_jsonl(path)
        ])
        cache = build_deepsets_cache(
            records,
            arguments.result_root.resolve(),
            cache_path,
            cost_limit=arguments.cost_limit,
            distance_batch_size=arguments.distance_batch_size,
        )
    else:
        cache = load_deepsets_cache(cache_path)

    if arguments.cache_only:
        print(f"Deep Sets 张量缓存已就绪：{cache_path}", flush=True)
        return 0

    training_config = DeepSetsTrainingConfig(
        random_seed=arguments.random_seed,
        validation_fraction=arguments.validation_fraction,
        test_fraction=arguments.test_fraction,
        hidden_dim=arguments.hidden_dim,
        dropout=arguments.dropout,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        max_epochs=arguments.max_epochs,
        patience=arguments.patience,
        instances_per_batch=arguments.instances_per_batch,
        include_method_features=arguments.include_method_features,
        model_variant=arguments.model_variant,
        device=arguments.device,
    )
    if arguments.ablation_suite:
        report = run_ablation_suite(
            cache,
            output_dir / "ablation",
            training_config=training_config,
            loss_config=DeepSetsLossConfig(),
        )
        print(
            f"四组消融完成：{output_dir / 'ablation' / 'ablation_report.json'}，"
            f"模型数={len(report['models'])}",
            flush=True,
        )
        return 0

    _, report = train_deepsets(
        cache,
        output_dir / "model",
        training_config=training_config,
        loss_config=DeepSetsLossConfig(),
    )
    print(
        f"{arguments.model_variant} 训练完成：best_epoch={report['best_epoch']}，"
        f"模型={output_dir / 'model' / 'deepsets_model.pt'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

