"""在统一实例切分上运行人工特征、Deep Sets、全局MLP及融合模型消融。"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .deep_sets import DeepSetsLossConfig, POSITIVE_TARGET_INDICES
from .deep_sets_data import (
    InstanceSplit,
    select_global_feature_indices,
    split_instance_ids_three_way,
)
from .deep_sets_training import (
    DeepSetsTrainingConfig,
    _classification_metrics,
    evaluate_candidate_predictions,
    train_deepsets,
)
from .models import REGRESSION_TARGETS


def _records_to_arrays(
    cache: dict[str, Any],
    instance_ids: tuple[str, ...],
    feature_indices: np.ndarray,
) -> dict[str, Any]:
    """
    将指定实例的缓存记录转换为人工特征基线所需数组。

    输入缓存、实例编号和特征列；输出特征、标签、掩码、状态及候选身份信息。
    """
    allowed = set(instance_ids)
    records = [record for record in cache["records"] if record["instance_id"] in allowed]
    return {
        "features": np.stack([
            np.asarray(record["global_features"])[feature_indices] for record in records
        ]).astype(np.float64),
        "targets": np.stack([
            np.asarray(record["targets_raw"], dtype=np.float64) for record in records
        ]),
        "target_mask": np.stack([
            np.asarray(record["target_mask"], dtype=bool) for record in records
        ]),
        "right_censored": np.asarray(
            [record["right_censored"] for record in records], dtype=np.int32
        ),
        "cost_feasible": np.asarray(
            [record["cost_feasible"] for record in records], dtype=np.int32
        ),
        "cost_feasible_mask": np.asarray(
            [record["cost_feasible_mask"] for record in records], dtype=bool
        ),
        "instance_ids": np.asarray([record["instance_id"] for record in records]),
        "candidate_names": np.asarray([record["candidate_name"] for record in records]),
    }


def _new_classifier(labels: np.ndarray, random_seed: int):
    """输入训练标签与随机种子，输出可处理单类别标签的分类器。"""
    classes = np.unique(labels)
    if len(classes) == 1:
        return DummyClassifier(strategy="constant", constant=int(classes[0]))
    return HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=random_seed,
    )


def _positive_probability(model, features: np.ndarray) -> np.ndarray:
    """输入已训练分类器和特征，输出阳性类别概率。"""
    probabilities = model.predict_proba(features)
    classes = list(map(int, model.classes_))
    if 1 not in classes:
        return np.zeros(len(features), dtype=np.float64)
    return probabilities[:, classes.index(1)]


def _fit_handcrafted_models(
    train: dict[str, Any], random_seed: int
) -> dict[str, Any]:
    """在训练实例的89维无算法名称人工特征上拟合梯度提升模型。"""
    regression_models: list[HistGradientBoostingRegressor] = []
    for target_index in range(len(REGRESSION_TARGETS)):
        mask = train["target_mask"][:, target_index]
        labels = train["targets"][mask, target_index]
        if target_index in POSITIVE_TARGET_INDICES:
            labels = np.log1p(np.clip(labels, 0.0, None))
        model = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_seed,
        )
        model.fit(train["features"][mask], labels)
        regression_models.append(model)

    right_censored = _new_classifier(train["right_censored"], random_seed)
    right_censored.fit(train["features"], train["right_censored"])
    feasible_mask = train["cost_feasible_mask"]
    cost_feasible = _new_classifier(train["cost_feasible"][feasible_mask], random_seed)
    cost_feasible.fit(
        train["features"][feasible_mask], train["cost_feasible"][feasible_mask]
    )
    return {
        "regression": regression_models,
        "right_censored": right_censored,
        "cost_feasible": cost_feasible,
    }


def _predict_handcrafted(models: dict[str, Any], dataset: dict[str, Any]) -> np.ndarray:
    """输入人工特征模型和数据数组，输出七个原始量纲的回归预测。"""
    columns = []
    for target_index, model in enumerate(models["regression"]):
        values = model.predict(dataset["features"])
        if target_index in POSITIVE_TARGET_INDICES:
            values = np.expm1(values)
            values = np.clip(values, 0.0, None)
        columns.append(values)
    return np.column_stack(columns)


def _evaluate_handcrafted(
    models: dict[str, Any], dataset: dict[str, Any], cost_limit: float
) -> dict[str, Any]:
    """输入人工特征模型和一个数据子集，输出与神经网络一致的评价报告。"""
    predictions = _predict_handcrafted(models, dataset)
    report = evaluate_candidate_predictions(
        dataset["targets"],
        predictions,
        dataset["target_mask"],
        dataset["instance_ids"],
        dataset["candidate_names"],
        cost_limit=cost_limit,
    )
    censor_probability = _positive_probability(models["right_censored"], dataset["features"])
    report["right_censored"] = _classification_metrics(
        dataset["right_censored"], censor_probability
    )
    feasible_mask = dataset["cost_feasible_mask"]
    feasible_probability = _positive_probability(
        models["cost_feasible"], dataset["features"]
    )
    report["cost_feasible"] = _classification_metrics(
        dataset["cost_feasible"][feasible_mask], feasible_probability[feasible_mask]
    )
    return report


def train_handcrafted_baseline(
    cache: dict[str, Any],
    instance_split: InstanceSplit,
    output_dir: str | Path,
    *,
    random_seed: int,
) -> dict[str, Any]:
    """
    训练不含算法名称的人工特征梯度提升基线并保存模型。

    输入缓存、固定实例切分和输出目录；输出验证集与独立测试集报告。
    """
    feature_indices, feature_names = select_global_feature_indices(
        cache, include_method_features=False
    )
    train = _records_to_arrays(cache, instance_split.train_ids, feature_indices)
    validation = _records_to_arrays(cache, instance_split.validation_ids, feature_indices)
    test = _records_to_arrays(cache, instance_split.test_ids, feature_indices)
    models = _fit_handcrafted_models(train, random_seed)
    cost_limit = float(cache["cost_limit"])
    report = {
        "model_name": "handcrafted_gradient_boosting",
        "selected_global_feature_count": len(feature_names),
        "selected_global_feature_names": list(feature_names),
        "train_instance_ids": list(instance_split.train_ids),
        "validation_instance_ids": list(instance_split.validation_ids),
        "test_instance_ids": list(instance_split.test_ids),
        "validation_evaluation": _evaluate_handcrafted(models, validation, cost_limit),
        "evaluation": _evaluate_handcrafted(models, test, cost_limit),
    }

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "format_version": 1,
        "feature_indices": feature_indices,
        "feature_names": feature_names,
        "models": models,
        "cost_limit": cost_limit,
    }, directory / "handcrafted_model.joblib")
    (directory / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _summary_row(model_name: str, report: dict[str, Any]) -> dict[str, Any]:
    """输入单个模型报告，输出消融表所需的核心测试指标。"""
    evaluation = report["evaluation"]
    phase2 = evaluation["regression"]["phase2_serial_seconds"]
    cost = evaluation["regression"]["final_cost"]
    policy = evaluation["joint_policy"]
    return {
        "model": model_name,
        "phase2_r2": phase2["r2"],
        "phase2_within_instance_spearman": phase2["mean_within_instance_spearman"],
        "phase2_top3_hit_fraction": phase2["true_fastest_top3_hit_fraction"],
        "phase2_selection_regret_ratio": phase2["predicted_fastest_mean_regret_ratio"],
        "final_cost_r2": cost["r2"],
        "final_cost_within_instance_spearman": cost["mean_within_instance_spearman"],
        "final_cost_top3_hit_fraction": cost["true_fastest_top3_hit_fraction"],
        "cost_violation_fraction": policy["true_cost_violation_fraction"],
        "selected_time_saving_vs_mst": policy["mean_selected_time_saving_vs_mst"],
    }


def run_ablation_suite(
    cache: dict[str, Any],
    output_dir: str | Path,
    *,
    training_config: DeepSetsTrainingConfig | None = None,
    loss_config: DeepSetsLossConfig | None = None,
) -> dict[str, Any]:
    """
    在同一三路实例切分上运行四种模型并生成统一消融报告。

    四组依次为人工特征梯度提升、纯Deep Sets、纯全局MLP及二者融合模型。
    """
    config = training_config or DeepSetsTrainingConfig()
    losses = loss_config or DeepSetsLossConfig()
    config = replace(config, include_method_features=False)
    split = split_instance_ids_three_way(
        cache,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        random_seed=config.random_seed,
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict[str, Any]] = {}
    print("[ablation] 训练人工特征梯度提升基线", flush=True)
    reports["handcrafted_gradient_boosting"] = train_handcrafted_baseline(
        cache,
        split,
        directory / "handcrafted_gradient_boosting",
        random_seed=config.random_seed,
    )
    for variant in ("deepsets_only", "global_mlp", "fused"):
        print(f"[ablation] 训练 {variant}", flush=True)
        _, report = train_deepsets(
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
    (directory / "ablation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
