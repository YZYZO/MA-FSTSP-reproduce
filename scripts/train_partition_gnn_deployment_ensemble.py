"""用全部已标注实例训练可供强化学习查询的多随机种子GNN集成。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.deep_sets_data import InstanceSplit  # noqa: E402
from src.partition_learning.gnn import MESSAGE_OPERATORS  # noqa: E402
from src.partition_learning.gnn_data import load_partition_gnn_cache  # noqa: E402
from src.partition_learning.gnn_training import (  # noqa: E402
    PartitionGNNTrainingConfig,
    partition_loss_profile,
    stratified_instance_folds,
    train_partition_gnn,
)


def parse_arguments() -> argparse.Namespace:
    """解析缓存、部署模型结构、随机种子和训练预算。"""
    parser = argparse.ArgumentParser(description="训练在线划分预测部署集成")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="260915,260916,260917")
    parser.add_argument(
        "--loss-profile",
        choices=("balanced", "selection", "target_weighted", "robust_targets"),
        default="robust_targets",
    )
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--message-operator", choices=MESSAGE_OPERATORS, default="edge_mlp")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    """
    每个成员用一折作早停集，其余四折训练，并把早停折同时用作非独立诊断集。

    该产物只用于部署推理；论文泛化指标必须继续引用独立交叉验证，而不能引用这里的诊断值。
    """
    arguments = parse_arguments()
    seeds = tuple(int(value) for value in arguments.seeds.split(",") if value.strip())
    cache = load_partition_gnn_cache(arguments.cache.resolve())
    folds = stratified_instance_folds(cache, fold_count=5, random_seed=260930)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    members = []
    for member_index, seed in enumerate(seeds):
        validation_ids = folds[member_index % len(folds)].test_ids
        train_ids = tuple(sorted(set(cache["instances"]) - set(validation_ids)))
        # 部署训练没有独立测试集；复用早停集仅为了满足统一训练/报告接口。
        split = InstanceSplit(
            train_ids=train_ids,
            validation_ids=validation_ids,
            test_ids=validation_ids,
        )
        directory = output_dir / f"member_{member_index + 1}_seed_{seed}"
        report_path = directory / "training_report.json"
        if report_path.is_file() and (directory / "partition_gnn_model.pt").is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            robust = arguments.loss_profile == "robust_targets"
            weighted = arguments.loss_profile in {"selection", "target_weighted", "robust_targets"}
            config = PartitionGNNTrainingConfig(
                random_seed=seed,
                hidden_dim=arguments.hidden_dim,
                message_layers=arguments.message_layers,
                message_operator=arguments.message_operator,
                separate_ranking_heads=weighted,
                time_quantile_heads=robust,
                max_epochs=arguments.max_epochs,
                patience=arguments.patience,
                model_variant="gnn_only",
                device=arguments.device,
            )
            _, report = train_partition_gnn(
                cache,
                directory,
                training_config=config,
                loss_config=partition_loss_profile(arguments.loss_profile),
                instance_split=split,
            )
        members.append({
            "seed": seed,
            "checkpoint": str(directory / "partition_gnn_model.pt"),
            "train_instance_count": len(train_ids),
            "early_stop_instance_count": len(validation_ids),
            "best_epoch": int(report["best_epoch"]),
        })
    manifest = {
        "kind": "partition_gnn_deployment_ensemble",
        "warning": "训练报告中的test复用了早停折，不得作为论文泛化结果。",
        "cache": str(arguments.cache.resolve()),
        "loss_profile": arguments.loss_profile,
        "message_layers": arguments.message_layers,
        "message_operator": arguments.message_operator,
        "members": members,
    }
    (output_dir / "ensemble_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"部署集成完成：{output_dir / 'ensemble_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
