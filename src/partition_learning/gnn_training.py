"""训练、评估并比较已知道路网内的候选划分GNN。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .deep_sets import DeepSetsLossConfig
from .deep_sets_data import (
    DeepSetsCandidateDataset,
    InstanceBatchSampler,
    InstanceSplit,
    global_feature_normalizer,
    select_global_feature_indices,
    split_instance_ids_three_way,
)
from .deep_sets_training import (
    _resolve_device,
    _run_epoch,
    _seed_everything,
    evaluate_deepsets,
)
from .gnn import GNN_MODEL_VARIANTS, PartitionGNNConfig, PartitionPerformanceGNN
from .gnn_data import collate_partition_graphs
from .models import TARGET_INDEX


@dataclass(frozen=True)
class PartitionGNNTrainingConfig:
    """保存候选划分GNN的切分、网络与优化超参数。"""

    random_seed: int = 260915
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    hidden_dim: int = 64
    message_layers: int = 2
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 250
    patience: int = 30
    instances_per_batch: int = 2
    model_variant: str = "gnn_fused"
    separate_ranking_heads: bool = False
    warm_start_checkpoint: str | None = None
    train_selection_heads_only: bool = False
    device: str = "auto"


def _apply_warm_start(
    model: PartitionPerformanceGNN,
    checkpoint_path: str | Path,
) -> dict[str, list[str]]:
    """
    输入模型和已有GNN检查点，加载可复用参数并初始化新增排序头。

    输出缺失与多余参数名；当旧模型没有排序头时，使用总时间与成本变化回归头
    对应的权重初始化两个排序分数，使第二阶段训练从已有排序关系继续优化。
    """
    try:
        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if model.ranking_head is not None and any(
        name.startswith("ranking_head.") for name in missing
    ):
        time_index = TARGET_INDEX["downstream_total_seconds"]
        cost_index = TARGET_INDEX["cost_change_ratio"]
        with torch.no_grad():
            model.ranking_head.weight[0].copy_(model.regression_head.weight[time_index])
            model.ranking_head.bias[0].copy_(model.regression_head.bias[time_index])
            model.ranking_head.weight[1].copy_(model.regression_head.weight[cost_index])
            model.ranking_head.bias[1].copy_(model.regression_head.bias[cost_index])
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def _configure_trainable_parameters(
    model: PartitionPerformanceGNN,
    *,
    selection_heads_only: bool,
) -> list[torch.nn.Parameter]:
    """
    输入模型与训练阶段，输出交给优化器的参数。

    选择头阶段冻结道路编码器和回归头，仅更新独立排序头及成本可行分类头，
    从结构上保证第一阶段训练得到的绝对时间和成本预测不会被破坏。
    """
    if selection_heads_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if model.ranking_head is None:
            raise ValueError("仅训练选择头时必须启用独立排序头。")
        for module in (model.ranking_head, model.cost_feasible_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _build_loader(
    dataset: DeepSetsCandidateDataset,
    *,
    instances_per_batch: int,
    shuffle: bool,
    random_seed: int,
) -> tuple[InstanceBatchSampler, DataLoader]:
    """输入候选数据集和批次配置，输出实例完整批采样器与GNN加载器。"""
    sampler = InstanceBatchSampler(
        dataset,
        instances_per_batch=instances_per_batch,
        shuffle=shuffle,
        random_seed=random_seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_partition_graphs,
        num_workers=0,
    )
    return sampler, loader


def train_partition_gnn(
    cache: dict[str, Any],
    output_dir: str | Path,
    *,
    training_config: PartitionGNNTrainingConfig | None = None,
    loss_config: DeepSetsLossConfig | None = None,
    instance_split: InstanceSplit | None = None,
) -> tuple[PartitionPerformanceGNN, dict[str, Any]]:
    """
    在固定实例级训练/验证/测试切分上训练候选划分GNN。

    输入GNN缓存、输出目录与可选切分；输出验证集选出的模型及独立测试报告。
    """
    config = training_config or PartitionGNNTrainingConfig()
    losses = loss_config or DeepSetsLossConfig()
    if config.model_variant not in GNN_MODEL_VARIANTS:
        raise ValueError(f"未知GNN结构：{config.model_variant}")
    _seed_everything(config.random_seed)
    device = _resolve_device(config.device)
    split = instance_split or split_instance_ids_three_way(
        cache,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        random_seed=config.random_seed,
    )

    if config.model_variant == "gnn_only":
        feature_indices = np.zeros(0, dtype=np.int64)
        selected_feature_names: tuple[str, ...] = ()
    else:
        feature_indices, selected_feature_names = select_global_feature_indices(
            cache, include_method_features=False
        )
    feature_mean, feature_scale = global_feature_normalizer(
        cache, split.train_ids, feature_indices
    )
    train_dataset = DeepSetsCandidateDataset(
        cache, split.train_ids, feature_indices, feature_mean, feature_scale
    )
    validation_dataset = DeepSetsCandidateDataset(
        cache, split.validation_ids, feature_indices, feature_mean, feature_scale
    )
    test_dataset = DeepSetsCandidateDataset(
        cache, split.test_ids, feature_indices, feature_mean, feature_scale
    )
    train_sampler, train_loader = _build_loader(
        train_dataset,
        instances_per_batch=config.instances_per_batch,
        shuffle=True,
        random_seed=config.random_seed,
    )
    _, validation_loader = _build_loader(
        validation_dataset,
        instances_per_batch=config.instances_per_batch,
        shuffle=False,
        random_seed=config.random_seed,
    )
    _, test_loader = _build_loader(
        test_dataset,
        instances_per_batch=config.instances_per_batch,
        shuffle=False,
        random_seed=config.random_seed,
    )

    names = cache["feature_names"]
    model_config = PartitionGNNConfig(
        boundary_feature_dim=len(names["boundary"]),
        customer_static_dim=len(names["customer_static"]),
        assignment_feature_dim=len(names["assignment"]),
        group_feature_dim=len(names["group"]),
        road_edge_feature_dim=len(names["road_edge"]),
        global_feature_dim=len(selected_feature_names),
        hidden_dim=config.hidden_dim,
        message_layers=config.message_layers,
        dropout=config.dropout,
        model_variant=config.model_variant,
        separate_ranking_heads=config.separate_ranking_heads,
    )
    model = PartitionPerformanceGNN(model_config).to(device)
    warm_start_report: dict[str, list[str]] | None = None
    if config.warm_start_checkpoint is not None:
        warm_start_report = _apply_warm_start(
            model, config.warm_start_checkpoint
        )
    trainable_parameters = _configure_trainable_parameters(
        model,
        selection_heads_only=config.train_selection_heads_only,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = []
    best_validation = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    for epoch in range(config.max_epochs):
        train_sampler.set_epoch(epoch)
        train_losses = _run_epoch(model, train_loader, device, losses, optimizer)
        validation_losses = _run_epoch(model, validation_loader, device, losses, None)
        history.append({
            "epoch": epoch + 1,
            "train": train_losses,
            "validation": validation_losses,
        })
        validation_total = validation_losses["total"]
        if validation_total < best_validation - 1e-5:
            best_validation = validation_total
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"[{config.model_variant}] epoch={epoch + 1} "
                f"train={train_losses['total']:.4f} "
                f"validation={validation_total:.4f}",
                flush=True,
            )
        if stale_epochs >= config.patience:
            break

    model.load_state_dict(best_state)
    validation_evaluation = evaluate_deepsets(
        model, validation_loader, device, cost_limit=float(cache["cost_limit"])
    )
    test_evaluation = evaluate_deepsets(
        model, test_loader, device, cost_limit=float(cache["cost_limit"])
    )
    report = {
        "train_instance_count": len(split.train_ids),
        "validation_instance_count": len(split.validation_ids),
        "test_instance_count": len(split.test_ids),
        "train_candidate_count": len(train_dataset),
        "validation_candidate_count": len(validation_dataset),
        "test_candidate_count": len(test_dataset),
        "selected_global_feature_count": len(selected_feature_names),
        "selected_global_feature_names": list(selected_feature_names),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "training_config": asdict(config),
        "loss_config": asdict(losses),
        "model_config": asdict(model_config),
        "warm_start_report": warm_start_report,
        "trainable_parameter_count": int(sum(
            parameter.numel() for parameter in trainable_parameters
        )),
        "train_instance_ids": list(split.train_ids),
        "validation_instance_ids": list(split.validation_ids),
        "test_instance_ids": list(split.test_ids),
        "validation_evaluation": validation_evaluation,
        "evaluation": test_evaluation,
        "history": history,
    }

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_config": asdict(model_config),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "global_feature_names": selected_feature_names,
        "global_feature_indices": feature_indices,
        "global_feature_mean": feature_mean,
        "global_feature_scale": feature_scale,
        "cost_limit": float(cache["cost_limit"]),
    }
    torch.save(checkpoint, directory / "partition_gnn_model.pt")
    (directory / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model.cpu(), report


def _summary_row(model_name: str, report: dict[str, Any]) -> dict[str, Any]:
    """输入一个GNN报告，输出与Deep Sets消融表一致的核心测试指标。"""
    evaluation = report["evaluation"]
    phase2 = evaluation["regression"]["phase2_serial_seconds"]
    final_cost = evaluation["regression"]["final_cost"]
    policy = evaluation["joint_policy"]
    return {
        "model": model_name,
        "phase2_r2": phase2["r2"],
        "phase2_within_instance_spearman": phase2["mean_within_instance_spearman"],
        "phase2_top3_hit_fraction": phase2["true_fastest_top3_hit_fraction"],
        "phase2_selection_regret_ratio": phase2["predicted_fastest_mean_regret_ratio"],
        "final_cost_r2": final_cost["r2"],
        "final_cost_within_instance_spearman": final_cost[
            "mean_within_instance_spearman"
        ],
        "final_cost_top3_hit_fraction": final_cost["true_fastest_top3_hit_fraction"],
        "cost_violation_fraction": policy["true_cost_violation_fraction"],
        "selected_time_saving_vs_mst": policy["mean_selected_time_saving_vs_mst"],
    }


def run_partition_gnn_ablation(
    cache: dict[str, Any],
    output_dir: str | Path,
    *,
    training_config: PartitionGNNTrainingConfig | None = None,
    loss_config: DeepSetsLossConfig | None = None,
) -> dict[str, Any]:
    """在同一实例切分上训练纯GNN和GNN加人工特征两种模型。"""
    config = training_config or PartitionGNNTrainingConfig()
    losses = loss_config or DeepSetsLossConfig()
    split = split_instance_ids_three_way(
        cache,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        random_seed=config.random_seed,
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    reports = {}
    for variant in GNN_MODEL_VARIANTS:
        print(f"[GNN ablation] 训练 {variant}", flush=True)
        _, report = train_partition_gnn(
            cache,
            directory / variant,
            training_config=replace(config, model_variant=variant),
            loss_config=losses,
            instance_split=split,
        )
        reports[variant] = report
    result = {
        "training_config": asdict(config),
        "loss_config": asdict(losses),
        "split": asdict(split),
        "models": {name: _summary_row(name, report) for name, report in reports.items()},
    }
    (directory / "gnn_ablation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def load_partition_gnn_model(
    path: str | Path,
) -> tuple[PartitionPerformanceGNN, dict[str, Any]]:
    """输入可信检查点，输出评估模式的CPU候选划分GNN及归一化元数据。"""
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location="cpu")
    model = PartitionPerformanceGNN(PartitionGNNConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
