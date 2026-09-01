"""客户组成本、时间、超时与不确定性的 PyTorch 多任务代理模型。"""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SurrogateModelConfig:
    """
    保存多任务代理模型的网络结构与训练超参数。

    输入：输入维度、隐藏层、Dropout、优化器、批量、轮数和随机种子。
    输出：不可变配置对象。
    逻辑：默认结构面向当前小型表格特征数据，不使用大型图神经网络。
    """

    input_dim: int
    hidden_dims: tuple = (128, 64)
    dropout: float = 0.15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 400
    early_stopping_patience: int = 50
    cost_loss_weight: float = 1.0
    time_loss_weight: float = 1.0
    timeout_loss_weight: float = 0.5
    approximate_cost_weight: float = 0.25
    seed: int = 0


class MultiTaskSurrogate(nn.Module):
    """
    使用共享编码器同时预测成本、时间、超时和回归不确定性。

    输入：形状为 `[batch, input_dim]` 的标准化客户组特征。
    输出：成本/时间均值与对数方差，以及超时 logit。
    逻辑：两个异方差回归头学习数据噪声，Dropout 重复前向用于估计模型不确定性。
    """

    def __init__(self, config):
        """
        根据配置建立共享 MLP 编码器和五个预测头。

        输入：`SurrogateModelConfig`。
        输出：初始化后的 PyTorch 模型。
        逻辑：每层使用线性变换、LayerNorm、SiLU 和 Dropout，适合小批量表格数据。
        """
        super().__init__()
        layers = []
        previous_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend((
                nn.Linear(previous_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            ))
            previous_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.cost_mean_head = nn.Linear(previous_dim, 1)
        self.cost_log_variance_head = nn.Linear(previous_dim, 1)
        self.time_mean_head = nn.Linear(previous_dim, 1)
        self.time_log_variance_head = nn.Linear(previous_dim, 1)
        self.timeout_head = nn.Linear(previous_dim, 1)

    def forward(self, features):
        """
        对一批标准化客户组特征执行多任务预测。

        输入：二维浮点张量。
        输出：包含五个一维张量的字典。
        逻辑：对数方差限制在合理区间，避免小数据训练时数值爆炸。
        """
        encoded = self.encoder(features)
        return {
            "cost_mean": self.cost_mean_head(encoded).squeeze(-1),
            "cost_log_variance": self.cost_log_variance_head(encoded).squeeze(-1).clamp(-7.0, 5.0),
            "time_mean": self.time_mean_head(encoded).squeeze(-1),
            "time_log_variance": self.time_log_variance_head(encoded).squeeze(-1).clamp(-7.0, 5.0),
            "timeout_logit": self.timeout_head(encoded).squeeze(-1),
        }


class SurrogateTensorDataset(Dataset):
    """
    将标准化客户组特征和多任务标签包装成 PyTorch 数据集。

    输入：特征、成本、对数时间、超时、时间删失和近似成本标记。
    输出：按行返回张量字典。
    逻辑：训练代码不依赖 pandas，便于单元测试和后续策略环境复用。
    """

    def __init__(self, arrays):
        """
        把 NumPy 数组转换为 CPU 浮点张量。

        输入：包含固定键的数组字典。
        输出：初始化后的 Dataset。
        逻辑：所有标签使用 float32，布尔标记在损失中以 0/1 处理。
        """
        self.tensors = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in arrays.items()
        }

    def __len__(self):
        """返回客户组样本行数。"""
        return len(self.tensors["features"])

    def __getitem__(self, index):
        """按索引返回一个客户组的全部训练张量。"""
        return {key: value[index] for key, value in self.tensors.items()}


def fit_normalization(train_arrays):
    """
    仅使用训练集拟合特征、成本和对数时间的均值与标准差。

    输入：未标准化训练数组字典。
    输出：可序列化的归一化参数字典。
    逻辑：验证和测试集不参与统计量计算，避免拆分泄漏。
    """
    feature_mean = train_arrays["features"].mean(axis=0)
    feature_std = train_arrays["features"].std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    cost_mean = float(np.mean(train_arrays["cost"]))
    cost_std = float(np.std(train_arrays["cost"]))
    time_mean = float(np.mean(train_arrays["log_time"]))
    time_std = float(np.std(train_arrays["log_time"]))
    return {
        "feature_mean": feature_mean.tolist(),
        "feature_std": np.where(feature_std < 1e-8, 1.0, feature_std).tolist(),
        "cost_mean": cost_mean,
        "cost_std": cost_std if cost_std >= 1e-8 else 1.0,
        "time_mean": time_mean,
        "time_std": time_std if time_std >= 1e-8 else 1.0,
    }


def normalize_arrays(arrays, normalization):
    """
    使用训练集统计量标准化任意数据拆分。

    输入：原始数组字典和归一化参数。
    输出：保留分类标记、替换连续量的数组副本。
    逻辑：成本和时间进入异方差损失前缩放到相近量级。
    """
    normalized = {
        key: np.asarray(value).copy()
        for key, value in arrays.items()
    }
    normalized["features"] = (
        normalized["features"] - np.asarray(normalization["feature_mean"])
    ) / np.asarray(normalization["feature_std"])
    normalized["cost"] = (
        normalized["cost"] - normalization["cost_mean"]
    ) / normalization["cost_std"]
    normalized["log_time"] = (
        normalized["log_time"] - normalization["time_mean"]
    ) / normalization["time_std"]
    return normalized


def _heteroscedastic_loss(mean, log_variance, target, censored=None):
    """
    计算普通或右删失高斯异方差负对数似然。

    输入：预测均值、对数方差、目标和可选删失标记。
    输出：每个样本一个损失值。
    逻辑：超时样本的真实时间只知道更大，因此使用高斯生存概率而不是当作精确值。
    """
    inverse_variance = torch.exp(-log_variance)
    exact_loss = 0.5 * (
        log_variance + (target - mean) ** 2 * inverse_variance
    )
    if censored is None:
        return exact_loss
    standard_deviation = torch.exp(0.5 * log_variance).clamp_min(1e-6)
    z_value = (target - mean) / standard_deviation
    survival_probability = 0.5 * torch.erfc(z_value / math.sqrt(2.0))
    censored_loss = -torch.log(survival_probability.clamp_min(1e-8))
    return torch.where(censored > 0.5, censored_loss, exact_loss)


def compute_multitask_loss(outputs, batch, config, timeout_positive_weight):
    """
    组合成本、时间和超时三个任务的训练损失。

    输入：模型输出、批量标签、训练配置和超时正类权重。
    输出：总损失及各子损失字典。
    逻辑：超时成本标签降权，时间标签按右删失处理，分类使用带正类权重的 BCE。
    """
    cost_losses = _heteroscedastic_loss(
        outputs["cost_mean"],
        outputs["cost_log_variance"],
        batch["cost"],
    )
    cost_weights = torch.where(
        batch["approximate_cost"] > 0.5,
        torch.full_like(cost_losses, config.approximate_cost_weight),
        torch.ones_like(cost_losses),
    )
    cost_loss = (cost_losses * cost_weights).sum() / cost_weights.sum().clamp_min(1.0)
    time_loss = _heteroscedastic_loss(
        outputs["time_mean"],
        outputs["time_log_variance"],
        batch["log_time"],
        censored=batch["time_censored"],
    ).mean()
    timeout_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["timeout_logit"],
        batch["timeout"],
        pos_weight=timeout_positive_weight,
    )
    total_loss = (
        config.cost_loss_weight * cost_loss
        + config.time_loss_weight * time_loss
        + config.timeout_loss_weight * timeout_loss
    )
    return {
        "total": total_loss,
        "cost": cost_loss,
        "time": time_loss,
        "timeout": timeout_loss,
    }


def _epoch_loss(model, data_loader, config, timeout_positive_weight, optimizer=None):
    """
    执行一个训练或验证轮次并汇总平均损失。

    输入：模型、DataLoader、配置、正类权重和可选优化器。
    输出：四类样本加权平均损失。
    逻辑：提供优化器时反向传播，否则在 `no_grad` 下只评估。
    """
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "cost": 0.0, "time": 0.0, "timeout": 0.0}
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in data_loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["features"])
            losses = compute_multitask_loss(
                outputs,
                batch,
                config,
                timeout_positive_weight,
            )
            if training:
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            batch_size = len(batch["features"])
            sample_count += batch_size
            for name in totals:
                totals[name] += float(losses[name].detach()) * batch_size
    return {
        name: value / max(sample_count, 1)
        for name, value in totals.items()
    }


def train_surrogate_model(
    train_arrays,
    validation_arrays,
    config,
    progress_callback=None,
):
    """
    训练多任务代理模型并按验证损失执行早停。

    输入：训练/验证数组、模型配置和可选进度回调。
    输出：最佳模型、归一化参数和逐轮历史。
    逻辑：所有随机源固定；只用训练集拟合归一化和超时类别权重。
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    normalization = fit_normalization(train_arrays)
    normalized_train = normalize_arrays(train_arrays, normalization)
    normalized_validation = normalize_arrays(validation_arrays, normalization)

    train_dataset = SurrogateTensorDataset(normalized_train)
    validation_dataset = SurrogateTensorDataset(normalized_validation)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(config.batch_size, len(validation_dataset)),
        shuffle=False,
    )

    model = MultiTaskSurrogate(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    positive_count = float(np.sum(train_arrays["timeout"]))
    negative_count = float(len(train_arrays["timeout"]) - positive_count)
    positive_weight_value = negative_count / positive_count if positive_count > 0 else 1.0
    timeout_positive_weight = torch.tensor(positive_weight_value, dtype=torch.float32)

    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _epoch_loss(
            model,
            train_loader,
            config,
            timeout_positive_weight,
            optimizer=optimizer,
        )
        validation_loss = _epoch_loss(
            model,
            validation_loader,
            config,
            timeout_positive_weight,
        )
        history_row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_loss.items()},
            **{f"validation_{key}": value for key, value in validation_loss.items()},
        }
        history.append(history_row)
        if progress_callback is not None:
            progress_callback(history_row)

        if validation_loss["total"] < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss["total"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    return model, {
        "normalization": normalization,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "timeout_positive_weight": positive_weight_value,
    }


def predict_with_uncertainty(model, features, normalization, mc_samples=30):
    """
    使用 MC Dropout 输出原量纲预测及总不确定性。

    输入：模型、原始特征、归一化参数和重复前向次数。
    输出：成本、对数时间、墙钟时间、超时概率及标准差字典。
    逻辑：总方差等于多次均值的模型方差与预测异方差均值之和。
    """
    normalized_features = (
        np.asarray(features, dtype=np.float32)
        - np.asarray(normalization["feature_mean"], dtype=np.float32)
    ) / np.asarray(normalization["feature_std"], dtype=np.float32)
    feature_tensor = torch.as_tensor(normalized_features, dtype=torch.float32)

    model.train(mc_samples > 1)
    cost_means = []
    cost_variances = []
    time_means = []
    time_variances = []
    timeout_probabilities = []
    with torch.no_grad():
        for _ in range(max(mc_samples, 1)):
            outputs = model(feature_tensor)
            cost_means.append(outputs["cost_mean"].cpu().numpy())
            cost_variances.append(torch.exp(outputs["cost_log_variance"]).cpu().numpy())
            time_means.append(outputs["time_mean"].cpu().numpy())
            time_variances.append(torch.exp(outputs["time_log_variance"]).cpu().numpy())
            timeout_probabilities.append(torch.sigmoid(outputs["timeout_logit"]).cpu().numpy())

    cost_means = np.asarray(cost_means)
    cost_variances = np.asarray(cost_variances)
    time_means = np.asarray(time_means)
    time_variances = np.asarray(time_variances)
    timeout_probabilities = np.asarray(timeout_probabilities)

    normalized_cost_mean = cost_means.mean(axis=0)
    normalized_cost_variance = cost_means.var(axis=0) + cost_variances.mean(axis=0)
    normalized_time_mean = time_means.mean(axis=0)
    normalized_time_variance = time_means.var(axis=0) + time_variances.mean(axis=0)

    cost_mean = (
        normalized_cost_mean * normalization["cost_std"]
        + normalization["cost_mean"]
    )
    cost_std = np.sqrt(normalized_cost_variance) * normalization["cost_std"]
    log_time_mean = (
        normalized_time_mean * normalization["time_std"]
        + normalization["time_mean"]
    )
    log_time_std = np.sqrt(normalized_time_variance) * normalization["time_std"]
    return {
        "cost_mean": cost_mean,
        "cost_std": cost_std,
        "log_time_mean": log_time_mean,
        "log_time_std": log_time_std,
        "wall_time_mean": np.expm1(np.clip(log_time_mean, -20.0, 20.0)),
        "timeout_probability": timeout_probabilities.mean(axis=0),
        "timeout_probability_std": timeout_probabilities.std(axis=0),
    }


def _safe_spearman(expected, predicted):
    """
    计算有限的 Spearman 排序相关系数。

    输入：真实值和预测值。
    输出：有限浮点数；样本过少或常量序列返回 `None`。
    逻辑：排序指标是代理模型能否支持后续局部动作选择的核心验收项。
    """
    if len(expected) < 3:
        return None
    coefficient = spearmanr(expected, predicted).statistic
    return float(coefficient) if np.isfinite(coefficient) else None


def evaluate_surrogate_model(model, arrays, normalization, mc_samples=30):
    """
    在一个数据拆分上计算质量、时间、超时和不确定性指标。

    输入：模型、原始数组、归一化参数和 MC Dropout 次数。
    输出：JSON 可序列化指标与原量纲预测。
    逻辑：超时 AUC 仅在同时存在正负样本时计算，成本另报告精确标签子集排序。
    """
    predictions = predict_with_uncertainty(
        model,
        arrays["features"],
        normalization,
        mc_samples=mc_samples,
    )
    expected_cost = np.asarray(arrays["cost"], dtype=float)
    expected_log_time = np.asarray(arrays["log_time"], dtype=float)
    expected_wall_time = np.expm1(expected_log_time)
    expected_timeout = np.asarray(arrays["timeout"], dtype=int)
    predicted_timeout = (predictions["timeout_probability"] >= 0.5).astype(int)
    exact_mask = np.asarray(arrays["approximate_cost"]) < 0.5
    approximate_mask = ~exact_mask

    timeout_auc = None
    timeout_average_precision = None
    if len(np.unique(expected_timeout)) == 2:
        timeout_auc = float(
            roc_auc_score(expected_timeout, predictions["timeout_probability"])
        )
        timeout_average_precision = float(
            average_precision_score(
                expected_timeout,
                predictions["timeout_probability"],
            )
        )

    metrics = {
        "sample_count": len(expected_cost),
        "timeout_positive_count": int(expected_timeout.sum()),
        "cost_mae": float(mean_absolute_error(expected_cost, predictions["cost_mean"])),
        "cost_rmse": float(
            math.sqrt(mean_squared_error(expected_cost, predictions["cost_mean"]))
        ),
        "cost_spearman": _safe_spearman(expected_cost, predictions["cost_mean"]),
        "exact_cost_spearman": _safe_spearman(
            expected_cost[exact_mask],
            predictions["cost_mean"][exact_mask],
        ),
        "exact_cost_mae": (
            float(
                mean_absolute_error(
                    expected_cost[exact_mask],
                    predictions["cost_mean"][exact_mask],
                )
            )
            if np.any(exact_mask)
            else None
        ),
        "exact_cost_rmse": (
            float(
                math.sqrt(
                    mean_squared_error(
                        expected_cost[exact_mask],
                        predictions["cost_mean"][exact_mask],
                    )
                )
            )
            if np.any(exact_mask)
            else None
        ),
        "approximate_cost_mae": (
            float(
                mean_absolute_error(
                    expected_cost[approximate_mask],
                    predictions["cost_mean"][approximate_mask],
                )
            )
            if np.any(approximate_mask)
            else None
        ),
        "log_time_mae": float(mean_absolute_error(expected_log_time, predictions["log_time_mean"])),
        "log_time_rmse": float(
            math.sqrt(
                mean_squared_error(
                    expected_log_time,
                    predictions["log_time_mean"],
                )
            )
        ),
        "time_spearman": _safe_spearman(expected_log_time, predictions["log_time_mean"]),
        "wall_time_mae_seconds": float(mean_absolute_error(expected_wall_time, predictions["wall_time_mean"])),
        "timeout_roc_auc": timeout_auc,
        "timeout_average_precision": timeout_average_precision,
        "timeout_precision": float(precision_score(expected_timeout, predicted_timeout, zero_division=0)),
        "timeout_recall": float(recall_score(expected_timeout, predicted_timeout, zero_division=0)),
        "timeout_f1": float(f1_score(expected_timeout, predicted_timeout, zero_division=0)),
        "cost_one_sigma_coverage": float(
            np.mean(
                np.abs(expected_cost - predictions["cost_mean"])
                <= predictions["cost_std"]
            )
        ),
        "cost_two_sigma_coverage": float(
            np.mean(
                np.abs(expected_cost - predictions["cost_mean"])
                <= 2.0 * predictions["cost_std"]
            )
        ),
        "time_one_sigma_coverage": float(
            np.mean(
                np.abs(expected_log_time - predictions["log_time_mean"])
                <= predictions["log_time_std"]
            )
        ),
        "time_two_sigma_coverage": float(
            np.mean(
                np.abs(expected_log_time - predictions["log_time_mean"])
                <= 2.0 * predictions["log_time_std"]
            )
        ),
    }
    return metrics, predictions


def checkpoint_payload(model, config, feature_names, training_artifacts, metrics):
    """
    构造可直接由后续策略环境加载的模型检查点字典。

    输入：模型、配置、特征名、训练产物和评估指标。
    输出：仅包含状态字典与可序列化元数据的字典。
    逻辑：显式保存特征顺序和归一化参数，避免推理阶段列错位。
    """
    return {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "feature_names": list(feature_names),
        "normalization": training_artifacts["normalization"],
        "best_epoch": training_artifacts["best_epoch"],
        "metrics": metrics,
    }
