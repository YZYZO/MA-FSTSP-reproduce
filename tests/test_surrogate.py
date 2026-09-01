"""多任务代理模型、删失损失和不确定性输出测试。"""

import unittest

import numpy as np
import torch

from src.learning.surrogate import (
    MultiTaskSurrogate,
    SurrogateModelConfig,
    checkpoint_payload,
    compute_multitask_loss,
    evaluate_surrogate_model,
    predict_with_uncertainty,
    train_surrogate_model,
)


def _synthetic_arrays(sample_count, seed):
    """
    生成成本、时间和超时具有明确共同结构的合成数据。

    输入：样本数和随机种子。
    输出：代理模型标准数组字典。
    逻辑：成本与前两维线性相关，时间由第三维驱动，超时取高时间区域。
    """
    random_state = np.random.RandomState(seed)
    features = random_state.normal(size=(sample_count, 4)).astype(np.float32)
    cost = (5.0 + 1.8 * features[:, 0] - 0.7 * features[:, 1]).astype(np.float32)
    log_time = (1.0 + 1.2 * features[:, 2] + 0.2 * features[:, 0]).astype(np.float32)
    timeout = (log_time > 1.45).astype(np.float32)
    return {
        "features": features,
        "cost": cost,
        "log_time": log_time,
        "timeout": timeout,
        "time_censored": timeout.copy(),
        "approximate_cost": timeout.copy(),
    }


class SurrogateModelTest(unittest.TestCase):
    """验证多头形状、损失、短训练、指标和检查点。"""

    def test_forward_and_multitask_loss_are_finite(self):
        """
        验证五个预测头和删失多任务损失可正常反向传播。

        输入：八行随机特征与混合超时标签。
        输出：每个预测头形状为八，总损失有限且梯度存在。
        逻辑：覆盖模型最基础的训练数值路径。
        """
        config = SurrogateModelConfig(input_dim=4, hidden_dims=(16, 8))
        model = MultiTaskSurrogate(config)
        arrays = _synthetic_arrays(8, seed=1)
        batch = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in arrays.items()
        }
        outputs = model(batch["features"])
        self.assertEqual(set(outputs), {
            "cost_mean",
            "cost_log_variance",
            "time_mean",
            "time_log_variance",
            "timeout_logit",
        })
        self.assertTrue(all(value.shape == (8,) for value in outputs.values()))
        losses = compute_multitask_loss(
            outputs,
            batch,
            config,
            timeout_positive_weight=torch.tensor(1.0),
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_short_training_returns_metrics_and_uncertainty(self):
        """
        验证短训练可以生成排序指标、正不确定性和可保存检查点。

        输入：合成训练集与验证集。
        输出：最佳轮次、有限预测、非负标准差和完整检查点字段。
        逻辑：不要求小测试达到正式质量门槛，只验证训练评估接口闭环。
        """
        train_arrays = _synthetic_arrays(72, seed=2)
        validation_arrays = _synthetic_arrays(32, seed=3)
        config = SurrogateModelConfig(
            input_dim=4,
            hidden_dims=(24, 12),
            dropout=0.1,
            batch_size=16,
            max_epochs=40,
            early_stopping_patience=10,
            seed=4,
        )
        model, artifacts = train_surrogate_model(
            train_arrays,
            validation_arrays,
            config,
        )
        metrics, predictions = evaluate_surrogate_model(
            model,
            validation_arrays,
            artifacts["normalization"],
            mc_samples=5,
        )
        self.assertGreaterEqual(artifacts["best_epoch"], 1)
        self.assertTrue(np.isfinite(predictions["cost_mean"]).all())
        self.assertTrue((predictions["cost_std"] >= 0.0).all())
        self.assertTrue((predictions["log_time_std"] >= 0.0).all())
        self.assertIsNotNone(metrics["cost_spearman"])
        self.assertIsNotNone(metrics["exact_cost_mae"])
        self.assertIsNotNone(metrics["time_spearman"])
        payload = checkpoint_payload(
            model,
            config,
            ["f0", "f1", "f2", "f3"],
            artifacts,
            {"validation": metrics},
        )
        self.assertIn("model_state_dict", payload)
        self.assertEqual(payload["feature_names"], ["f0", "f1", "f2", "f3"])

    def test_direct_uncertainty_prediction_has_expected_length(self):
        """
        验证 MC Dropout 推理输出与输入行数一致。

        输入：未训练模型、五行特征和单位归一化参数。
        输出：所有预测数组长度为五。
        逻辑：后续强化学习环境将逐批调用同一推理接口。
        """
        config = SurrogateModelConfig(input_dim=4, hidden_dims=(8,))
        model = MultiTaskSurrogate(config)
        normalization = {
            "feature_mean": [0.0] * 4,
            "feature_std": [1.0] * 4,
            "cost_mean": 0.0,
            "cost_std": 1.0,
            "time_mean": 0.0,
            "time_std": 1.0,
        }
        predictions = predict_with_uncertainty(
            model,
            np.zeros((5, 4), dtype=np.float32),
            normalization,
            mc_samples=3,
        )
        self.assertTrue(all(len(values) == 5 for values in predictions.values()))


if __name__ == "__main__":
    unittest.main()
