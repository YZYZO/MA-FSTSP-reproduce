"""训练划分性能预测模型，并按实例隔离训练集与测试集。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

# 明确限制 sklearn/joblib 的底层并行度，保证服务器上记录的求解时间不受模型训练抢核干扰。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .reporting import deduplicate_candidate_records


MODEL_NAMES = ("ridge", "gradient_boosting", "shallow_mlp")
REGRESSION_TARGETS = (
    "time_saving_ratio",
    "phase2_serial_seconds",
    "phase3_seconds",
    "downstream_total_seconds",
    "solver_work_sum",
    "cost_change_ratio",
    "final_cost",
)
TARGET_INDEX = {name: index for index, name in enumerate(REGRESSION_TARGETS)}
POSITIVE_TARGETS = {
    "phase2_serial_seconds",
    "phase3_seconds",
    "downstream_total_seconds",
    "solver_work_sum",
    "final_cost",
}


@dataclass
class ModelEnsemble:
    """保存多目标回归器、状态分类器以及训练分布的 OOD 参考。"""

    feature_names: tuple[str, ...]
    models: dict[str, dict[str, object]]
    training_mean: np.ndarray
    training_scale: np.ndarray
    standardized_training: np.ndarray

    def vectorize(self, features: dict[str, float]) -> np.ndarray:
        """输入特征字典，输出与训练列顺序一致的一行二维数组。"""
        return np.asarray([[float(features.get(name, 0.0)) for name in self.feature_names]])

    def predict(self, features: dict[str, float]) -> dict[str, object]:
        """
        预测一个候选的二三阶段时间、最终目标值、相对变化和标签可靠性。

        输入为求解前可获得的划分特征；输出逐模型预测、集成均值、不确定性和 OOD 距离。
        正值目标在输出端截断到零，避免数值外推产生没有物理意义的负时间或负成本。
        """
        vector = self.vectorize(features)
        predictions: dict[str, dict[str, float]] = {}
        for name, model_group in self.models.items():
            item = {
                target: float(model_group[target].predict(vector)[0])
                for target in REGRESSION_TARGETS
            }
            for target in POSITIVE_TARGETS:
                item[target] = max(item[target], 0.0)
            item["right_censored_probability"] = float(
                _positive_probability(model_group["right_censored"], vector)[0]
            )
            item["guard_probability"] = float(
                _positive_probability(model_group["guard"], vector)[0]
            )
            # 保留旧字段名，避免已有主动采样代码和外部分析脚本立即失效。
            item["censored_probability"] = item["right_censored_probability"]
            predictions[name] = item

        def values(target: str) -> np.ndarray:
            """提取同一目标的三个模型预测，供集成统计使用。"""
            return np.asarray([item[target] for item in predictions.values()], dtype=np.float64)

        time_saving = values("time_saving_ratio")
        cost_change = values("cost_change_ratio")
        downstream = values("downstream_total_seconds")
        solver_work = values("solver_work_sum")
        final_cost = values("final_cost")
        standardized = (vector[0] - self.training_mean) / self.training_scale
        distances = np.sqrt(np.mean((self.standardized_training - standardized) ** 2, axis=1))
        return {
            "models": predictions,
            "mean_time_saving_ratio": float(np.mean(time_saving)),
            "mean_cost_change_ratio": float(np.mean(cost_change)),
            "mean_downstream_total_seconds": float(np.mean(downstream)),
            "mean_solver_work_sum": float(np.mean(solver_work)),
            "mean_final_cost": float(np.mean(final_cost)),
            "time_uncertainty": float(np.std(time_saving)),
            "cost_uncertainty": float(np.std(cost_change)),
            "downstream_uncertainty": float(np.std(downstream)),
            "solver_work_uncertainty": float(np.std(solver_work)),
            "final_cost_uncertainty": float(np.std(final_cost)),
            "mean_right_censored_probability": float(np.mean(values("right_censored_probability"))),
            "mean_guard_probability": float(np.mean(values("guard_probability"))),
            "ood_distance": float(np.min(distances)),
        }


def _records_to_dataset(records: Iterable[dict]) -> dict[str, object]:
    """
    将候选记录转换为特征、七个回归目标、两个状态目标和实例分组。

    新记录显式提供二三阶段总时间和精确性标志；旧记录缺少这些字段时，按
    `总时间=Phase 2`、`无删失且无回退=精确`兼容读取，但不会把保护回退用于精确回归。
    """
    rows = deduplicate_candidate_records(records)
    feature_names = tuple(sorted({name for row in rows for name in row["features"]}))
    matrix = np.asarray(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    target_rows = []
    right_censored, guard, exact, relative_exact = [], [], [], []
    for row in rows:
        labels = row["labels"]
        evaluation = row.get("evaluation", {})
        phase2 = float(labels.get(
            "phase2_serial_effective_seconds",
            evaluation.get("phase2_serial_effective_seconds", evaluation.get("phase2_effective_seconds", 0.0)),
        ))
        phase3 = float(labels.get("phase3_seconds", evaluation.get("phase3_seconds", 0.0)))
        downstream = float(labels.get(
            "downstream_total_seconds",
            evaluation.get("downstream_total_seconds", phase2 + phase3),
        ))
        solver_work = float(labels.get(
            "solver_work_sum", evaluation.get("solver_work_sum", 0.0)
        ))
        final_cost = float(labels.get("final_cost", evaluation.get("final_cost", 0.0)))
        target_rows.append([
            float(labels["time_saving_ratio"]),
            phase2,
            phase3,
            downstream,
            solver_work,
            float(labels["cost_change_ratio"]),
            final_cost,
        ])
        is_censored = bool(labels.get(
            "candidate_right_censored",
            labels.get("candidate_censored", evaluation.get("right_censored_groups", 0)),
        ))
        has_guard = bool(labels.get("guard_groups", evaluation.get("guard_groups", 0)))
        has_fallback = bool(labels.get("fallback_groups", evaluation.get("fallback_groups", 0)))
        legacy_exact = not is_censored and not has_guard and not has_fallback
        right_censored.append(is_censored)
        guard.append(has_guard)
        is_exact = bool(labels.get("candidate_exact", evaluation.get("candidate_exact", legacy_exact)))
        exact.append(is_exact)
        baseline_exact = bool(labels.get("baseline_exact", True))
        relative_exact.append(bool(labels.get(
            "relative_labels_exact", is_exact and baseline_exact
        )))
    return {
        "rows": rows,
        "matrix": matrix,
        "targets": np.asarray(target_rows, dtype=np.float64),
        "right_censored": np.asarray(right_censored, dtype=np.int32),
        "guard": np.asarray(guard, dtype=np.int32),
        "exact": np.asarray(exact, dtype=bool),
        "relative_exact": np.asarray(relative_exact, dtype=bool),
        "groups": np.asarray([str(row["instance_id"]) for row in rows]),
        "feature_names": feature_names,
    }


def _new_regressor(name: str, random_seed: int):
    """按模型族创建尚未拟合的标量回归器。"""
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=3.0))
    if name == "gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_seed,
        )
    if name == "shallow_mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=800,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
                random_state=random_seed,
            ),
        )
    raise ValueError(f"未知模型：{name}")


def _new_positive_regressor(name: str, random_seed: int):
    """为非负的时间或成本目标创建 log1p/expm1 变换回归器。"""
    return TransformedTargetRegressor(
        regressor=_new_regressor(name, random_seed),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def _new_classifier(name: str, random_seed: int, labels: np.ndarray):
    """创建状态分类器；训练标签只有一个类别时返回固定概率模型。"""
    classes, class_counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return DummyClassifier(strategy="constant", constant=int(labels[0]))
    if name == "ridge":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_seed,
        )
    if name == "shallow_mlp":
        # early_stopping 会在内部做分层验证切分；少数类只有一条时 sklearn 会直接报错。
        # 此处仍保留全部稀有状态样本训练，只关闭无法成立的内部验证切分。
        use_early_stopping = bool(np.min(class_counts) >= 2 and len(labels) >= 20)
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=1e-3,
                max_iter=600,
                early_stopping=use_early_stopping,
                random_state=random_seed,
            ),
        )
    raise ValueError(f"未知模型：{name}")


def _positive_probability(model, matrix: np.ndarray) -> np.ndarray:
    """返回分类器对类别 1 的概率，并兼容单类别 DummyClassifier。"""
    probabilities = model.predict_proba(matrix)
    classes = list(map(int, model.classes_))
    if 1 not in classes:
        return np.zeros(len(matrix), dtype=np.float64)
    return probabilities[:, classes.index(1)]


def _metric_dict(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """计算 MAE、R² 和 Spearman；样本过少或常量向量时返回可序列化的零相关值。"""
    if float(np.ptp(actual)) < 1e-12 or float(np.ptp(predicted)) < 1e-12:
        correlation = 0.0
    else:
        correlation = spearmanr(actual, predicted).statistic
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else 0.0,
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
    }


def _group_ranking_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """计算实例内排序相关、真实最优 Top-3 命中率和模型选择后悔率。"""
    correlations, top3_hits, regrets = [], [], []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            continue
        group_actual = actual[indices]
        group_predicted = predicted[indices]
        if float(np.ptp(group_actual)) < 1e-12 or float(np.ptp(group_predicted)) < 1e-12:
            correlations.append(0.0)
        else:
            correlation = spearmanr(group_actual, group_predicted).statistic
            correlations.append(float(correlation) if np.isfinite(correlation) else 0.0)
        true_best_local = int(np.argmin(group_actual))
        predicted_order = np.argsort(group_predicted)
        top3_hits.append(true_best_local in set(map(int, predicted_order[: min(3, len(indices))])))
        selected_actual = float(group_actual[int(predicted_order[0])])
        best_actual = max(float(np.min(group_actual)), 1e-9)
        regrets.append((selected_actual - best_actual) / best_actual)
    return {
        "mean_within_instance_spearman": float(np.mean(correlations)) if correlations else 0.0,
        "true_fastest_top3_hit_fraction": float(np.mean(top3_hits)) if top3_hits else 0.0,
        "predicted_fastest_mean_regret_ratio": float(np.mean(regrets)) if regrets else 0.0,
    }


def _joint_policy_metrics(
    actual_time: np.ndarray,
    actual_cost_change: np.ndarray,
    predicted_time: np.ndarray,
    predicted_cost_change: np.ndarray,
    groups: np.ndarray,
    candidate_names: np.ndarray,
    *,
    cost_limit: float,
) -> dict[str, float | int]:
    """
    评价“预测成本可行时选择预测最快划分”的完整决策策略。

    输入为测试候选的真实/预测总时间、成本变化、实例分组与候选名；输出约束违反率、
    可行选择的时间后悔率，以及相对 MST 保持划分的实际节时。若预测没有可行候选，
    策略退回预测成本最低者，并单独记录该情况，避免把阈值失配悄悄隐藏。
    """
    violations, feasible_regrets, selected_savings, oracle_savings = [], [], [], []
    predicted_empty = []
    evaluated_instances = 0
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        true_feasible = indices[actual_cost_change[indices] <= cost_limit]
        if len(true_feasible) == 0:
            continue
        predicted_feasible = indices[predicted_cost_change[indices] <= cost_limit]
        predicted_empty.append(len(predicted_feasible) == 0)
        if len(predicted_feasible):
            selected = int(predicted_feasible[np.argmin(predicted_time[predicted_feasible])])
        else:
            selected = int(indices[np.argmin(predicted_cost_change[indices])])
        oracle = int(true_feasible[np.argmin(actual_time[true_feasible])])
        stay_candidates = indices[candidate_names[indices] == "stay"]
        if len(stay_candidates) == 0:
            continue
        stay = int(stay_candidates[0])
        baseline_time = max(float(actual_time[stay]), 1e-9)
        is_violation = bool(actual_cost_change[selected] > cost_limit)
        violations.append(is_violation)
        if not is_violation:
            oracle_time = max(float(actual_time[oracle]), 1e-9)
            feasible_regrets.append(max(
                0.0, (float(actual_time[selected]) - oracle_time) / oracle_time
            ))
        selected_savings.append((baseline_time - float(actual_time[selected])) / baseline_time)
        oracle_savings.append((baseline_time - float(actual_time[oracle])) / baseline_time)
        evaluated_instances += 1
    return {
        "cost_limit": float(cost_limit),
        "instance_count": int(evaluated_instances),
        "true_cost_violation_fraction": float(np.mean(violations)) if violations else 0.0,
        "predicted_empty_feasible_fraction": float(np.mean(predicted_empty)) if predicted_empty else 0.0,
        "feasible_selection_count": int(len(feasible_regrets)),
        "mean_feasible_time_regret_ratio": (
            float(np.mean(feasible_regrets)) if feasible_regrets else 0.0
        ),
        "median_feasible_time_regret_ratio": (
            float(np.median(feasible_regrets)) if feasible_regrets else 0.0
        ),
        "mean_selected_time_saving_vs_mst": (
            float(np.mean(selected_savings)) if selected_savings else 0.0
        ),
        "mean_oracle_time_saving_vs_mst": (
            float(np.mean(oracle_savings)) if oracle_savings else 0.0
        ),
    }


def _classifier_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """计算状态概率的 Brier 分数、阳性比例，并在双类别测试集上计算 AUC。"""
    result = {
        "brier": float(brier_score_loss(actual, probability)),
        "positive_fraction": float(np.mean(actual)),
    }
    if len(np.unique(actual)) == 2:
        result["roc_auc"] = float(roc_auc_score(actual, probability))
    return result


def train_model_ensemble(
    records: Iterable[dict],
    output_dir: str | Path,
    *,
    random_seed: int = 260915,
    test_fraction: float = 0.2,
    cost_limit: float = 0.10,
) -> tuple[ModelEnsemble, dict[str, object]]:
    """
    以实例为单位划分数据，训练三类多目标性能预测器。

    回归仅使用 Phase 2 得到精确顺序的候选，防止超时下界和启发式回退污染真值；
    右删失与复杂度保护使用全部候选训练分类器。最终返回用全部可用数据重训的模型和留出评估。
    """
    dataset = _records_to_dataset(records)
    rows = dataset["rows"]
    matrix = dataset["matrix"]
    targets = dataset["targets"]
    right_censored = dataset["right_censored"]
    guard = dataset["guard"]
    exact = dataset["exact"]
    relative_exact = dataset["relative_exact"]
    groups = dataset["groups"]
    feature_names = dataset["feature_names"]
    exact_indices = np.flatnonzero(exact)
    relative_exact_indices = np.flatnonzero(relative_exact)
    exact_groups = groups[exact_indices]
    relative_exact_groups = groups[relative_exact_indices]
    if len(np.unique(relative_exact_groups)) < 2:
        raise ValueError("基线与候选均精确的样本不足两个独立实例，无法进行实例级训练/测试划分。")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_seed)
    train_local, test_local = next(splitter.split(
        matrix[relative_exact_indices], groups=relative_exact_groups
    ))
    train_group_names = set(relative_exact_groups[train_local])
    test_group_names = set(relative_exact_groups[test_local])
    train_exact = np.asarray([
        index for index in exact_indices if groups[index] in train_group_names
    ], dtype=int)
    test_exact = np.asarray([
        index for index in exact_indices if groups[index] in test_group_names
    ], dtype=int)
    train_relative_exact = np.asarray([
        index for index in relative_exact_indices if groups[index] in train_group_names
    ], dtype=int)
    test_relative_exact = np.asarray([
        index for index in relative_exact_indices if groups[index] in test_group_names
    ], dtype=int)
    # 没有精确候选的实例只用于最终状态分类器，不进入留出集指标。
    train_indices = np.asarray([i for i, group in enumerate(groups) if group in train_group_names], dtype=int)
    test_indices = np.asarray([i for i, group in enumerate(groups) if group in test_group_names], dtype=int)

    report: dict[str, object] = {
        "row_count": len(rows),
        "exact_row_count": int(np.sum(exact)),
        "relative_exact_row_count": int(np.sum(relative_exact)),
        "excluded_regression_row_count": int(len(rows) - np.sum(exact)),
        "instance_count": int(len(np.unique(groups))),
        "exact_instance_count": int(len(np.unique(exact_groups))),
        "relative_exact_instance_count": int(len(np.unique(relative_exact_groups))),
        "train_instance_count": int(len(train_group_names)),
        "test_instance_count": int(len(test_group_names)),
        "train_exact_row_count": int(len(train_exact)),
        "test_exact_row_count": int(len(test_exact)),
        "train_relative_exact_row_count": int(len(train_relative_exact)),
        "test_relative_exact_row_count": int(len(test_relative_exact)),
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "joint_policy_cost_limit": float(cost_limit),
        "regression_policy": {
            "absolute_targets": "candidate_exact_only",
            "relative_targets": "candidate_and_baseline_exact_only",
        },
        "models": {},
    }
    final_models: dict[str, dict[str, object]] = {}
    for model_offset, name in enumerate(MODEL_NAMES):
        fitted: dict[str, object] = {}
        model_report: dict[str, object] = {}
        validation_models: dict[str, object] = {}
        for target_offset, target_name in enumerate(REGRESSION_TARGETS):
            target_values = targets[:, target_offset]
            is_relative_target = target_name in {"time_saving_ratio", "cost_change_ratio"}
            train_target_indices = train_relative_exact if is_relative_target else train_exact
            test_target_indices = test_relative_exact if is_relative_target else test_exact
            final_target_indices = relative_exact_indices if is_relative_target else exact_indices
            factory = _new_positive_regressor if target_name in POSITIVE_TARGETS else _new_regressor
            model = factory(name, random_seed + 50 * target_offset + model_offset)
            model.fit(matrix[train_target_indices], target_values[train_target_indices])
            prediction = model.predict(matrix[test_target_indices])
            if target_name in POSITIVE_TARGETS:
                prediction = np.maximum(prediction, 0.0)
            metrics = _metric_dict(target_values[test_target_indices], prediction)
            if target_name in {
                "phase2_serial_seconds", "phase3_seconds", "downstream_total_seconds",
                "solver_work_sum", "final_cost"
            }:
                metrics.update(_group_ranking_metrics(
                    target_values[test_target_indices], prediction, groups[test_target_indices]
                ))
            model_report[target_name] = metrics
            validation_models[target_name] = model

            final_model = factory(name, random_seed + 50 * target_offset + model_offset)
            final_model.fit(matrix[final_target_indices], target_values[final_target_indices])
            fitted[target_name] = final_model

        # 用同一留出实例同时预测成本和时间，评价模型最终会执行的联合选择，而非孤立 R²。
        joint_time_prediction = np.maximum(
            validation_models["downstream_total_seconds"].predict(matrix[test_relative_exact]),
            0.0,
        )
        joint_cost_prediction = validation_models["cost_change_ratio"].predict(
            matrix[test_relative_exact]
        )
        model_report["joint_policy"] = _joint_policy_metrics(
            targets[test_relative_exact, TARGET_INDEX["downstream_total_seconds"]],
            targets[test_relative_exact, TARGET_INDEX["cost_change_ratio"]],
            joint_time_prediction,
            joint_cost_prediction,
            groups[test_relative_exact],
            np.asarray([
                str(rows[index].get("candidate_name", "")) for index in test_relative_exact
            ]),
            cost_limit=cost_limit,
        )

        for label_offset, (label_name, label_values) in enumerate((
            ("right_censored", right_censored),
            ("guard", guard),
        )):
            classifier = _new_classifier(
                name,
                random_seed + 500 + 50 * label_offset + model_offset,
                label_values[train_indices],
            )
            classifier.fit(matrix[train_indices], label_values[train_indices])
            probability = _positive_probability(classifier, matrix[test_indices])
            model_report[label_name] = _classifier_metrics(label_values[test_indices], probability)

            final_classifier = _new_classifier(
                name,
                random_seed + 500 + 50 * label_offset + model_offset,
                label_values,
            )
            final_classifier.fit(matrix, label_values)
            fitted[label_name] = final_classifier
        # 旧报告读取器仍可通过 censored 访问真实右删失分类指标。
        model_report["censored"] = model_report["right_censored"]
        report["models"][name] = model_report
        final_models[name] = fitted

    training_mean = np.mean(matrix, axis=0)
    training_scale = np.std(matrix, axis=0)
    training_scale[training_scale < 1e-9] = 1.0
    ensemble = ModelEnsemble(
        feature_names=feature_names,
        models=final_models,
        training_mean=training_mean,
        training_scale=training_scale,
        standardized_training=(matrix - training_mean) / training_scale,
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, directory / "model_ensemble.joblib")
    (directory / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ensemble, report


def load_model_ensemble(path: str | Path) -> ModelEnsemble:
    """输入 joblib 路径，输出已训练模型集成。"""
    return joblib.load(path)
