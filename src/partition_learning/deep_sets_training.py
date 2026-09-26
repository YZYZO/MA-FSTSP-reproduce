"""训练、评估和保存层次化 Deep Sets 划分性能预测器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader

from .deep_sets import (
    DeepSetsLossConfig,
    DeepSetsModelConfig,
    HierarchicalDeepSets,
    hierarchical_deepsets_loss,
    inverse_regression_targets,
)
from .deep_sets_data import (
    DeepSetsCandidateDataset,
    InstanceBatchSampler,
    collate_deepsets,
    global_feature_normalizer,
    select_global_feature_indices,
    split_instance_ids,
)
from .models import REGRESSION_TARGETS, TARGET_INDEX


@dataclass(frozen=True)
class DeepSetsTrainingConfig:
    """保存可复现实验所需的训练超参数。"""

    random_seed: int = 260915
    test_fraction: float = 0.20
    hidden_dim: int = 64
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 250
    patience: int = 30
    instances_per_batch: int = 2
    include_method_features: bool = False
    device: str = "auto"


def _seed_everything(seed: int) -> None:
    """输入随机种子，统一设置 Python、NumPy 和 PyTorch 随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    """输入设备配置，输出实际可用的 PyTorch 设备。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _average_losses(loss_rows: list[dict[str, float]]) -> dict[str, float]:
    """输入逐批损失字典，输出逐项算术平均。"""
    if not loss_rows:
        return {}
    return {
        name: float(np.mean([row[name] for row in loss_rows]))
        for name in loss_rows[0]
    }


def _run_epoch(
    model: HierarchicalDeepSets,
    loader: DataLoader,
    device: torch.device,
    loss_config: DeepSetsLossConfig,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """
    运行一个训练或验证轮次。

    optimizer 非空时更新参数，否则只计算验证损失；输出包含总损失和各任务组成项的均值。
    """
    model.train(optimizer is not None)
    rows: list[dict[str, float]] = []
    for batch in loader:
        batch = batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(batch)
            loss, components = hierarchical_deepsets_loss(outputs, batch, loss_config)
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        rows.append({
            "total": float(loss.detach().cpu()),
            **{name: float(value.detach().cpu()) for name, value in components.items()},
        })
    return _average_losses(rows)


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """输入真实值和预测值，输出MAE、R²及Spearman相关。"""
    if len(actual) == 0:
        return {"mae": 0.0, "r2": 0.0, "spearman": 0.0}
    if len(actual) < 2 or float(np.ptp(actual)) < 1e-12 or float(np.ptp(predicted)) < 1e-12:
        correlation = 0.0
    else:
        correlation = float(spearmanr(actual, predicted).statistic)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else 0.0,
        "spearman": correlation if np.isfinite(correlation) else 0.0,
    }


def _within_instance_metrics(
    actual: np.ndarray, predicted: np.ndarray, instance_ids: np.ndarray
) -> dict[str, float]:
    """输入候选数值和实例编号，输出实例内排序、Top-3命中率及选择后悔值。"""
    correlations, top3_hits, regrets = [], [], []
    for instance_id in np.unique(instance_ids):
        indices = np.flatnonzero(instance_ids == instance_id)
        if len(indices) < 2:
            continue
        actual_values = actual[indices]
        predicted_values = predicted[indices]
        if float(np.ptp(actual_values)) < 1e-12 or float(np.ptp(predicted_values)) < 1e-12:
            correlations.append(0.0)
        else:
            correlation = spearmanr(actual_values, predicted_values).statistic
            correlations.append(float(correlation) if np.isfinite(correlation) else 0.0)
        true_best = int(np.argmin(actual_values))
        predicted_order = np.argsort(predicted_values)
        top3_hits.append(true_best in set(map(int, predicted_order[: min(3, len(indices))])))
        selected = float(actual_values[int(predicted_order[0])])
        best = max(float(np.min(actual_values)), 1e-9)
        regrets.append((selected - best) / best)
    return {
        "mean_within_instance_spearman": float(np.mean(correlations)) if correlations else 0.0,
        "true_fastest_top3_hit_fraction": float(np.mean(top3_hits)) if top3_hits else 0.0,
        "predicted_fastest_mean_regret_ratio": float(np.mean(regrets)) if regrets else 0.0,
    }


def _classification_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """输入二分类真值和概率，输出Brier、阳性比例及可计算时的AUC。"""
    result = {
        "brier": float(brier_score_loss(actual, probability)),
        "positive_fraction": float(np.mean(actual)),
    }
    if len(np.unique(actual)) == 2:
        result["roc_auc"] = float(roc_auc_score(actual, probability))
    return result


def _joint_policy_metrics(
    actual_time: np.ndarray,
    actual_cost: np.ndarray,
    predicted_time: np.ndarray,
    predicted_cost: np.ndarray,
    instance_ids: np.ndarray,
    candidate_names: np.ndarray,
    cost_limit: float,
) -> dict[str, float | int]:
    """评价成本预测约束下选择最快候选的完整决策质量。"""
    violations, regrets, selected_savings, oracle_savings = [], [], [], []
    evaluated = 0
    for instance_id in np.unique(instance_ids):
        indices = np.flatnonzero(instance_ids == instance_id)
        true_feasible = indices[actual_cost[indices] <= cost_limit]
        if len(true_feasible) == 0:
            continue
        predicted_feasible = indices[predicted_cost[indices] <= cost_limit]
        selected = int(
            predicted_feasible[np.argmin(predicted_time[predicted_feasible])]
            if len(predicted_feasible)
            else indices[np.argmin(predicted_cost[indices])]
        )
        oracle = int(true_feasible[np.argmin(actual_time[true_feasible])])
        stays = indices[candidate_names[indices] == "stay"]
        if len(stays) == 0:
            continue
        baseline = max(float(actual_time[int(stays[0])]), 1e-9)
        violation = bool(actual_cost[selected] > cost_limit)
        violations.append(violation)
        if not violation:
            best = max(float(actual_time[oracle]), 1e-9)
            regrets.append(max(0.0, (float(actual_time[selected]) - best) / best))
        selected_savings.append((baseline - float(actual_time[selected])) / baseline)
        oracle_savings.append((baseline - float(actual_time[oracle])) / baseline)
        evaluated += 1
    return {
        "cost_limit": float(cost_limit),
        "instance_count": evaluated,
        "true_cost_violation_fraction": float(np.mean(violations)) if violations else 0.0,
        "feasible_selection_count": len(regrets),
        "mean_feasible_time_regret_ratio": float(np.mean(regrets)) if regrets else 0.0,
        "median_feasible_time_regret_ratio": float(np.median(regrets)) if regrets else 0.0,
        "mean_selected_time_saving_vs_mst": (
            float(np.mean(selected_savings)) if selected_savings else 0.0
        ),
        "mean_oracle_time_saving_vs_mst": (
            float(np.mean(oracle_savings)) if oracle_savings else 0.0
        ),
    }


@torch.no_grad()
def evaluate_deepsets(
    model: HierarchicalDeepSets,
    loader: DataLoader,
    device: torch.device,
    *,
    cost_limit: float,
) -> dict[str, Any]:
    """输入训练后的模型和测试加载器，输出与现有表格模型可比较的完整报告。"""
    model.eval()
    predicted_rows, actual_rows, mask_rows = [], [], []
    censored_probability, censored_actual = [], []
    feasible_probability, feasible_actual, feasible_mask = [], [], []
    instance_ids: list[str] = []
    candidate_names: list[str] = []
    group_predicted, group_actual, group_mask = [], [], []
    group_censored_probability, group_censored_actual = [], []

    for batch in loader:
        device_batch = batch.to(device)
        outputs = model(device_batch)
        predicted_rows.append(inverse_regression_targets(outputs["regression"]).cpu().numpy())
        actual_rows.append(inverse_regression_targets(device_batch.targets).cpu().numpy())
        mask_rows.append(device_batch.target_mask.cpu().numpy())
        censored_probability.append(torch.sigmoid(outputs["right_censored_logit"]).cpu().numpy())
        censored_actual.append(device_batch.right_censored.cpu().numpy())
        feasible_probability.append(torch.sigmoid(outputs["cost_feasible_logit"]).cpu().numpy())
        feasible_actual.append(device_batch.cost_feasible.cpu().numpy())
        feasible_mask.append(device_batch.cost_feasible_mask.cpu().numpy())
        instance_ids.extend(batch.instance_ids)
        candidate_names.extend(batch.candidate_names)
        group_predicted.append(torch.expm1(outputs["group_log_phase2"]).clamp_min(0).cpu().numpy())
        group_actual.append(torch.expm1(device_batch.group_log_phase2).cpu().numpy())
        group_mask.append(device_batch.group_time_mask.cpu().numpy())
        group_censored_probability.append(
            torch.sigmoid(outputs["group_right_censored_logit"]).cpu().numpy()
        )
        group_censored_actual.append(device_batch.group_right_censored.cpu().numpy())

    predicted = np.concatenate(predicted_rows)
    actual = np.concatenate(actual_rows)
    masks = np.concatenate(mask_rows)
    ids = np.asarray(instance_ids)
    names = np.asarray(candidate_names)
    report: dict[str, Any] = {"regression": {}}
    for target_index, target_name in enumerate(REGRESSION_TARGETS):
        mask = masks[:, target_index]
        metrics = _regression_metrics(actual[mask, target_index], predicted[mask, target_index])
        if target_name in {
            "phase2_serial_seconds", "phase3_seconds", "downstream_total_seconds",
            "solver_work_sum", "final_cost",
        }:
            metrics.update(_within_instance_metrics(
                actual[mask, target_index], predicted[mask, target_index], ids[mask]
            ))
        report["regression"][target_name] = metrics

    relative_mask = masks[:, TARGET_INDEX["time_saving_ratio"]] & masks[:, TARGET_INDEX["cost_change_ratio"]]
    report["joint_policy"] = _joint_policy_metrics(
        actual[relative_mask, TARGET_INDEX["downstream_total_seconds"]],
        actual[relative_mask, TARGET_INDEX["cost_change_ratio"]],
        predicted[relative_mask, TARGET_INDEX["downstream_total_seconds"]],
        predicted[relative_mask, TARGET_INDEX["cost_change_ratio"]],
        ids[relative_mask],
        names[relative_mask],
        cost_limit,
    )
    censor_y = np.concatenate(censored_actual)
    censor_p = np.concatenate(censored_probability)
    report["right_censored"] = _classification_metrics(censor_y, censor_p)
    feasible_y = np.concatenate(feasible_actual)
    feasible_p = np.concatenate(feasible_probability)
    feasible_valid = np.concatenate(feasible_mask)
    report["cost_feasible"] = _classification_metrics(
        feasible_y[feasible_valid], feasible_p[feasible_valid]
    )

    group_valid = np.concatenate(group_mask)
    group_y = np.concatenate(group_actual)
    group_p = np.concatenate(group_predicted)
    report["group_phase2_seconds"] = _regression_metrics(
        group_y[group_valid], group_p[group_valid]
    )
    group_censor_y = np.concatenate(group_censored_actual)
    group_censor_p = np.concatenate(group_censored_probability)
    report["group_right_censored"] = _classification_metrics(
        group_censor_y, group_censor_p
    )
    return report


def train_deepsets(
    cache: dict[str, Any],
    output_dir: str | Path,
    *,
    training_config: DeepSetsTrainingConfig | None = None,
    loss_config: DeepSetsLossConfig | None = None,
) -> tuple[HierarchicalDeepSets, dict[str, Any]]:
    """
    在实例级切分上训练层次化 Deep Sets，并保存模型与评估报告。

    输入张量缓存和输出目录；输出已用全部训练实例拟合的最佳模型及可序列化报告。
    """
    config = training_config or DeepSetsTrainingConfig()
    losses = loss_config or DeepSetsLossConfig()
    _seed_everything(config.random_seed)
    device = _resolve_device(config.device)
    train_ids, test_ids = split_instance_ids(
        cache, test_fraction=config.test_fraction, random_seed=config.random_seed
    )
    feature_indices, selected_feature_names = select_global_feature_indices(
        cache, include_method_features=config.include_method_features
    )
    feature_mean, feature_scale = global_feature_normalizer(
        cache, train_ids, feature_indices
    )
    train_dataset = DeepSetsCandidateDataset(
        cache, train_ids, feature_indices, feature_mean, feature_scale
    )
    test_dataset = DeepSetsCandidateDataset(
        cache, test_ids, feature_indices, feature_mean, feature_scale
    )
    train_sampler = InstanceBatchSampler(
        train_dataset,
        instances_per_batch=config.instances_per_batch,
        shuffle=True,
        random_seed=config.random_seed,
    )
    test_sampler = InstanceBatchSampler(
        test_dataset,
        instances_per_batch=config.instances_per_batch,
        shuffle=False,
        random_seed=config.random_seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, collate_fn=collate_deepsets, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_sampler=test_sampler, collate_fn=collate_deepsets, num_workers=0
    )

    feature_names = cache["feature_names"]
    model_config = DeepSetsModelConfig(
        boundary_feature_dim=len(feature_names["boundary"]),
        customer_static_dim=len(feature_names["customer_static"]),
        assignment_feature_dim=len(feature_names["assignment"]),
        group_feature_dim=len(feature_names["group"]),
        global_feature_dim=len(selected_feature_names),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )
    model = HierarchicalDeepSets(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history = []
    best_validation = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    for epoch in range(config.max_epochs):
        train_sampler.set_epoch(epoch)
        train_losses = _run_epoch(model, train_loader, device, losses, optimizer)
        validation_losses = _run_epoch(model, test_loader, device, losses, None)
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
                f"[DeepSets] epoch={epoch + 1} train={train_losses['total']:.4f} "
                f"validation={validation_total:.4f}",
                flush=True,
            )
        if stale_epochs >= config.patience:
            break

    model.load_state_dict(best_state)
    evaluation = evaluate_deepsets(
        model, test_loader, device, cost_limit=float(cache["cost_limit"])
    )
    report = {
        "train_instance_count": len(train_ids),
        "test_instance_count": len(test_ids),
        "train_candidate_count": len(train_dataset),
        "test_candidate_count": len(test_dataset),
        "selected_global_feature_count": len(selected_feature_names),
        "selected_global_feature_names": list(selected_feature_names),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "training_config": asdict(config),
        "loss_config": asdict(losses),
        "model_config": asdict(model_config),
        "train_instance_ids": list(train_ids),
        "test_instance_ids": list(test_ids),
        "evaluation": evaluation,
        "history": history,
    }

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_config": asdict(model_config),
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "global_feature_names": selected_feature_names,
        "global_feature_indices": feature_indices,
        "global_feature_mean": feature_mean,
        "global_feature_scale": feature_scale,
        "cost_limit": float(cache["cost_limit"]),
    }
    torch.save(checkpoint, directory / "deepsets_model.pt")
    (directory / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model.cpu(), report


def load_deepsets_model(path: str | Path) -> tuple[HierarchicalDeepSets, dict[str, Any]]:
    """输入可信的本地检查点，输出评估模式的CPU模型及归一化元数据。"""
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location="cpu")
    model = HierarchicalDeepSets(DeepSetsModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint

