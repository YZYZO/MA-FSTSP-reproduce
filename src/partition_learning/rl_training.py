"""强化学习轨迹采集、代理贪心基线和结果压缩工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .candidates import moved_customer_count
from .rl_environment import PartitionRLAction, SurrogatePartitionEnvironment
from .rl_policy import PartitionActorCritic, TrajectoryStep


@dataclass(frozen=True)
class RolloutResult:
    """保存一条完整轨迹、总奖励、动作记录和最终代理结果。"""

    steps: list[TrajectoryStep]
    total_reward: float
    initial_score: float
    final_score: float
    trace: list[dict[str, Any]]
    final_partition: dict[int, tuple[int, ...]]
    final_prediction: dict[str, float]
    final_uncertainty: dict[str, float]
    moved_customers: int


def _compact_output(output: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """从统一代理输出提取报告需要的预测值和主模型不确定性。"""
    prediction_names = (
        "phase2_serial_seconds",
        "phase3_seconds",
        "downstream_total_seconds",
        "final_cost",
        "cost_change_ratio",
        "time_rank_score",
        "cost_rank_score",
        "cost_feasible_probability",
        "downstream_time_p50",
        "downstream_time_p90",
        "right_censored_probability",
    )
    prediction = {
        name: float(output["prediction"][name])
        for name in prediction_names
        if name in output["prediction"]
    }
    uncertainty = {
        name: float(value)
        for name, value in output["uncertainty"]["numeric"].items()
    }
    return prediction, uncertainty


def _result(
    environment: SurrogatePartitionEnvironment,
    *,
    steps: list[TrajectoryStep],
    total_reward: float,
    initial_score: float,
    trace: list[dict[str, Any]],
) -> RolloutResult:
    """把环境终态转换成可序列化且不包含逐成员大数组的轨迹结果。"""
    assert environment.current_prediction is not None
    prediction, uncertainty = _compact_output(
        environment.current_prediction
    )
    return RolloutResult(
        steps=steps,
        total_reward=float(total_reward),
        initial_score=float(initial_score),
        final_score=float(environment.current_score),
        trace=trace,
        final_partition=environment.current,
        final_prediction=prediction,
        final_uncertainty=uncertainty,
        moved_customers=moved_customer_count(
            environment.baseline,
            environment.current,
        ),
    )


def policy_rollout(
    environment: SurrogatePartitionEnvironment,
    policy: PartitionActorCritic,
    *,
    deterministic: bool,
    device: str = "cpu",
) -> RolloutResult:
    """让 PPO 策略完成一条轨迹，并保留训练所需的旧策略统计量。"""
    state, actions = environment.reset()
    initial_score = environment.current_score
    trajectory: list[TrajectoryStep] = []
    trace: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False
    while not done:
        action_matrix = np.stack(
            [action.features for action in actions],
            axis=0,
        )
        action_index, log_probability, value, entropy = policy.select_action(
            state,
            action_matrix,
            deterministic=deterministic,
            device=device,
        )
        action = actions[action_index]
        next_state, reward, done, info = environment.step(action)
        trajectory.append(TrajectoryStep(
            state_features=state.copy(),
            action_features=action_matrix.copy(),
            action_index=action_index,
            old_log_probability=log_probability,
            old_value=value,
            reward=reward,
            done=done,
        ))
        trace.append({
            "step": len(trace) + 1,
            "action": action.name,
            "kind": action.kind,
            "description": action.description,
            "reward": float(reward),
            "score": float(info["score"]),
            "entropy": float(entropy),
        })
        total_reward += reward
        state = next_state
        if not done:
            actions = environment.available_actions()
    return _result(
        environment,
        steps=trajectory,
        total_reward=total_reward,
        initial_score=initial_score,
        trace=trace,
    )


def greedy_rollout(
    environment: SurrogatePartitionEnvironment,
) -> RolloutResult:
    """每步选择风险调整分数最低的动作，形成监督代理贪心上界基线。"""
    environment.reset()
    initial_score = environment.current_score
    trace: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False
    while not done:
        actions = environment.available_actions()
        evaluations = environment.evaluate_actions(actions)
        scores = np.asarray(
            [components["total"] for _, components in evaluations],
            dtype=np.float64,
        )
        action_index = int(np.argmin(scores))
        # 如果没有动作能覆盖步长惩罚，则显式选择停止。
        if (
            action_index != 0
            and scores[action_index] + environment.config.step_penalty
            >= environment.current_score
        ):
            action_index = 0
        action = actions[action_index]
        _, reward, done, info = environment.step(action)
        trace.append({
            "step": len(trace) + 1,
            "action": action.name,
            "kind": action.kind,
            "description": action.description,
            "reward": float(reward),
            "score": float(info["score"]),
        })
        total_reward += reward
    return _result(
        environment,
        steps=[],
        total_reward=total_reward,
        initial_score=initial_score,
        trace=trace,
    )


def random_rollout(
    environment: SurrogatePartitionEnvironment,
    random_state: np.random.Generator,
) -> RolloutResult:
    """从合法动作集合均匀采样，形成与 PPO 相同信赖域的随机基线。"""
    environment.reset()
    initial_score = environment.current_score
    trace: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False
    while not done:
        actions = environment.available_actions()
        action_index = int(random_state.integers(0, len(actions)))
        action = actions[action_index]
        _, reward, done, info = environment.step(action)
        trace.append({
            "step": len(trace) + 1,
            "action": action.name,
            "kind": action.kind,
            "description": action.description,
            "reward": float(reward),
            "score": float(info["score"]),
        })
        total_reward += reward
    return _result(
        environment,
        steps=[],
        total_reward=total_reward,
        initial_score=initial_score,
        trace=trace,
    )


def rollout_to_dict(result: RolloutResult) -> dict[str, Any]:
    """把轨迹结果转换成 JSON 可写的摘要，不导出训练张量。"""
    return {
        "total_reward": result.total_reward,
        "initial_score": result.initial_score,
        "final_score": result.final_score,
        "score_improvement": result.initial_score - result.final_score,
        "moved_customers": result.moved_customers,
        "step_count": len(result.trace),
        "trace": result.trace,
        "final_prediction": result.final_prediction,
        "final_uncertainty": result.final_uncertainty,
        "final_partition": {
            str(depot): list(customers)
            for depot, customers in result.final_partition.items()
        },
    }
