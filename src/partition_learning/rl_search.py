"""在统一代理查询预算下比较客户划分搜索算法。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from .candidates import Partition, partition_key
from .rl_environment import PartitionRLAction, SurrogatePartitionEnvironment
from .rl_policy import PartitionActorCritic
from .rl_training import _compact_output


@dataclass(frozen=True)
class SearchNode:
    """保存束搜索节点的划分、深度、路径访问集合和动作轨迹。"""

    partition: Partition
    depth: int
    visited: frozenset[tuple[tuple[int, ...], ...]]
    trace: tuple[str, ...]


def _conservative_metrics(
    environment: SurrogatePartitionEnvironment,
    output: dict[str, Any],
    *,
    uncertainty_kappa: float,
    baseline: bool = False,
) -> dict[str, float | bool]:
    """输入代理输出，计算时间、成本均值、成本上界和硬约束可行性。"""
    prediction = output["prediction"]
    final_cost = float(prediction["final_cost"])
    uncertainty = float(
        output["uncertainty"]["numeric"].get("final_cost", 0.0)
    )
    denominator = max(abs(environment.baseline_cost), 1e-6)
    mean_change = (final_cost - environment.baseline_cost) / denominator
    ucb_change = (
        final_cost + uncertainty_kappa * max(uncertainty, 0.0)
        - environment.baseline_cost
    ) / denominator
    # 新版风险模型直接预测成本变化P90；与集成UCB取更保守者。
    if "cost_change_p90" in prediction:
        ucb_change = max(ucb_change, float(prediction["cost_change_p90"]))
    feasible = baseline or ucb_change <= environment.config.cost_limit
    return {
        "predicted_time": max(float(prediction["downstream_total_seconds"]), 0.0),
        "mean_cost_change": mean_change,
        "cost_ucb_change": ucb_change,
        "constraint_violation": max(0.0, ucb_change - environment.config.cost_limit),
        "feasible": bool(feasible),
    }


def _candidate(
    environment: SurrogatePartitionEnvironment,
    partition: Partition,
    output: dict[str, Any],
    *,
    uncertainty_kappa: float,
    name: str,
    trace: Iterable[str],
    baseline: bool = False,
) -> dict[str, Any]:
    """把搜索到的划分和代理输出压缩为统一候选记录。"""
    prediction, uncertainty = _compact_output(output)
    return {
        "name": str(name),
        "partition": partition,
        "trace": list(trace),
        "prediction": prediction,
        "uncertainty": uncertainty,
        **_conservative_metrics(
            environment,
            output,
            uncertainty_kappa=uncertainty_kappa,
            baseline=baseline,
        ),
    }


def _begin_search(
    environment: SurrogatePartitionEnvironment,
    *,
    uncertainty_kappa: float,
) -> tuple[dict[str, Any], dict[tuple[tuple[int, ...], ...], dict[str, Any]]]:
    """清空方法间共享缓存并查询MST，返回基线候选和候选索引。"""
    environment.reset_query_budget(clear_cache=True)
    environment.reset()
    assert environment.current_prediction is not None
    baseline = _candidate(
        environment,
        environment.baseline,
        environment.current_prediction,
        uncertainty_kappa=uncertainty_kappa,
        name="mst",
        trace=(),
        baseline=True,
    )
    return baseline, {partition_key(environment.baseline, environment.depots): baseline}


def _budgeted_actions(
    environment: SurrogatePartitionEnvironment,
    actions: Iterable[PartitionRLAction],
    *,
    query_budget: int,
) -> list[PartitionRLAction]:
    """按动作顺序保留不会使唯一代理查询数超过预算的动作。"""
    selected: list[PartitionRLAction] = []
    reserved: set[tuple[tuple[int, ...], ...]] = set()
    for action in actions:
        if action.kind == "stop":
            continue
        key = partition_key(action.partition, environment.depots)
        cached = key in environment.prediction_cache or key in reserved
        if not cached and environment.surrogate_query_count + len(reserved) >= query_budget:
            continue
        selected.append(action)
        if not cached:
            reserved.add(key)
    return selected


def _remember(
    candidates: dict[tuple[tuple[int, ...], ...], dict[str, Any]],
    environment: SurrogatePartitionEnvironment,
    action: PartitionRLAction,
    output: dict[str, Any],
    *,
    uncertainty_kappa: float,
    trace: Iterable[str],
) -> dict[str, Any]:
    """记录唯一划分并返回该候选；缓存命中不会重复增加候选数量。"""
    key = partition_key(action.partition, environment.depots)
    if key not in candidates:
        candidates[key] = _candidate(
            environment,
            action.partition,
            output,
            uncertainty_kappa=uncertainty_kappa,
            name=action.name,
            trace=trace,
        )
    return candidates[key]


def _finish(
    method: str,
    environment: SurrogatePartitionEnvironment,
    baseline: dict[str, Any],
    candidates: dict[tuple[tuple[int, ...], ...], dict[str, Any]],
    *,
    query_budget: int,
) -> dict[str, Any]:
    """在所有保守可行候选中选择最快者；无改进时严格返回MST。"""
    feasible = [candidate for candidate in candidates.values() if candidate["feasible"]]
    best = min(feasible, key=lambda candidate: candidate["predicted_time"])
    if best["predicted_time"] >= baseline["predicted_time"]:
        best = baseline
    return {
        "method": method,
        "query_budget": int(query_budget),
        "surrogate_query_count": int(environment.surrogate_query_count),
        "unique_candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible),
        "returned_mst": best is baseline,
        "baseline_predicted_time": float(baseline["predicted_time"]),
        "predicted_time_saving_ratio": (
            float(baseline["predicted_time"]) - float(best["predicted_time"])
        ) / max(float(baseline["predicted_time"]), 1e-6),
        "best": {
            **best,
            "partition": {
                str(depot): list(customers)
                for depot, customers in best["partition"].items()
            },
        },
    }


def greedy_budget_search(
    environment: SurrogatePartitionEnvironment,
    *,
    query_budget: int = 200,
    uncertainty_kappa: float = 1.0,
) -> dict[str, Any]:
    """在每一步查询全部预算允许邻居，并沿保守可行的最快改进继续。"""
    baseline, candidates = _begin_search(
        environment, uncertainty_kappa=uncertainty_kappa
    )
    current = baseline
    visited = {partition_key(environment.baseline, environment.depots)}
    for depth in range(environment.config.max_steps):
        actions = _budgeted_actions(
            environment,
            environment.available_actions(),
            query_budget=query_budget,
        )
        if not actions:
            break
        evaluations = environment.evaluate_actions(actions)
        evaluated = [
            (
                action,
                _remember(
                    candidates,
                    environment,
                    action,
                    output,
                    uncertainty_kappa=uncertainty_kappa,
                    trace=tuple(current["trace"]) + (action.name,),
                ),
            )
            for action, (output, _) in zip(actions, evaluations)
        ]
        feasible = [item for item in evaluated if item[1]["feasible"]]
        if not feasible:
            break
        action, selected = min(feasible, key=lambda item: item[1]["predicted_time"])
        if selected["predicted_time"] >= current["predicted_time"]:
            break
        current = selected
        visited.add(partition_key(action.partition, environment.depots))
        environment.set_search_state(
            action.partition,
            step_index=depth + 1,
            visited_partition_keys=visited,
        )
    return _finish(
        "greedy", environment, baseline, candidates, query_budget=query_budget
    )


def random_budget_search(
    environment: SurrogatePartitionEnvironment,
    random_state: np.random.Generator,
    *,
    query_budget: int = 200,
    uncertainty_kappa: float = 1.0,
    stop_probability: float = 0.15,
) -> dict[str, Any]:
    """反复从MST随机采样局部轨迹，直到耗尽唯一代理查询预算。"""
    baseline, candidates = _begin_search(
        environment, uncertainty_kappa=uncertainty_kappa
    )
    stagnant_restarts = 0
    while environment.surrogate_query_count < query_budget and stagnant_restarts < 10:
        before = environment.surrogate_query_count
        visited = {partition_key(environment.baseline, environment.depots)}
        environment.set_search_state(
            environment.baseline,
            step_index=0,
            visited_partition_keys=visited,
        )
        trace: list[str] = []
        for depth in range(environment.config.max_steps):
            if depth > 0 and random_state.random() < stop_probability:
                break
            actions = _budgeted_actions(
                environment,
                environment.available_actions(),
                query_budget=query_budget,
            )
            uncached = [
                action for action in actions
                if partition_key(action.partition, environment.depots)
                not in environment.prediction_cache
            ]
            pool = uncached or actions
            if not pool:
                break
            action = pool[int(random_state.integers(0, len(pool)))]
            output, _ = environment.evaluate_actions([action])[0]
            trace.append(action.name)
            _remember(
                candidates,
                environment,
                action,
                output,
                uncertainty_kappa=uncertainty_kappa,
                trace=trace,
            )
            visited.add(partition_key(action.partition, environment.depots))
            environment.set_search_state(
                action.partition,
                step_index=depth + 1,
                visited_partition_keys=visited,
            )
        stagnant_restarts = stagnant_restarts + 1 if before == environment.surrogate_query_count else 0
    return _finish(
        "random", environment, baseline, candidates, query_budget=query_budget
    )


def beam_budget_search(
    environment: SurrogatePartitionEnvironment,
    *,
    query_budget: int = 200,
    beam_width: int = 5,
    uncertainty_kappa: float = 1.0,
) -> dict[str, Any]:
    """按保守可行性优先、预测时间次优的规则执行固定宽度束搜索。"""
    baseline, candidates = _begin_search(
        environment, uncertainty_kappa=uncertainty_kappa
    )
    baseline_key = partition_key(environment.baseline, environment.depots)
    frontier = [SearchNode(
        environment.baseline,
        0,
        frozenset({baseline_key}),
        (),
    )]
    for depth in range(environment.config.max_steps):
        children: list[tuple[tuple[float, float, float], SearchNode]] = []
        for node in frontier:
            environment.set_search_state(
                node.partition,
                step_index=node.depth,
                visited_partition_keys=node.visited,
            )
            actions = _budgeted_actions(
                environment,
                environment.available_actions(),
                query_budget=query_budget,
            )
            if not actions:
                continue
            evaluations = environment.evaluate_actions(actions)
            for action, (output, _) in zip(actions, evaluations):
                trace = node.trace + (action.name,)
                candidate = _remember(
                    candidates,
                    environment,
                    action,
                    output,
                    uncertainty_kappa=uncertainty_kappa,
                    trace=trace,
                )
                key = partition_key(action.partition, environment.depots)
                child = SearchNode(
                    action.partition,
                    depth + 1,
                    frozenset(set(node.visited) | {key}),
                    trace,
                )
                rank = (
                    0.0 if candidate["feasible"] else 1.0,
                    float(candidate["constraint_violation"]),
                    float(candidate["predicted_time"]),
                )
                children.append((rank, child))
        if not children:
            break
        children.sort(key=lambda item: item[0])
        unique: dict[tuple[tuple[int, ...], ...], SearchNode] = {}
        for _, child in children:
            unique.setdefault(
                partition_key(child.partition, environment.depots), child
            )
            if len(unique) >= beam_width:
                break
        frontier = list(unique.values())
    return _finish(
        "beam", environment, baseline, candidates, query_budget=query_budget
    )


def annealing_budget_search(
    environment: SurrogatePartitionEnvironment,
    random_state: np.random.Generator,
    *,
    query_budget: int = 200,
    uncertainty_kappa: float = 1.0,
    initial_temperature: float = 0.20,
) -> dict[str, Any]:
    """以时间和约束违反组成的能量执行多次重启模拟退火。"""
    baseline, candidates = _begin_search(
        environment, uncertainty_kappa=uncertainty_kappa
    )

    def energy(candidate: dict[str, Any]) -> float:
        """输入候选，输出以MST时间归一化并强罚成本越界的退火能量。"""
        return (
            float(candidate["predicted_time"]) / max(float(baseline["predicted_time"]), 1e-6)
            + 10.0 * float(candidate["constraint_violation"])
        )

    stagnant_restarts = 0
    while environment.surrogate_query_count < query_budget and stagnant_restarts < 10:
        before = environment.surrogate_query_count
        current = baseline
        visited = {partition_key(environment.baseline, environment.depots)}
        environment.set_search_state(
            environment.baseline,
            step_index=0,
            visited_partition_keys=visited,
        )
        trace: list[str] = []
        for depth in range(environment.config.max_steps):
            actions = _budgeted_actions(
                environment,
                environment.available_actions(),
                query_budget=query_budget,
            )
            uncached = [
                action for action in actions
                if partition_key(action.partition, environment.depots)
                not in environment.prediction_cache
            ]
            pool = uncached or actions
            if not pool:
                break
            action = pool[int(random_state.integers(0, len(pool)))]
            output, _ = environment.evaluate_actions([action])[0]
            proposed_trace = trace + [action.name]
            proposed = _remember(
                candidates,
                environment,
                action,
                output,
                uncertainty_kappa=uncertainty_kappa,
                trace=proposed_trace,
            )
            temperature = initial_temperature * max(
                0.05,
                1.0 - environment.surrogate_query_count / max(query_budget, 1),
            )
            delta = energy(proposed) - energy(current)
            accepted = delta <= 0.0 or random_state.random() < math.exp(
                -delta / max(temperature, 1e-6)
            )
            visited.add(partition_key(action.partition, environment.depots))
            if accepted:
                current = proposed
                trace = proposed_trace
                environment.set_search_state(
                    action.partition,
                    step_index=depth + 1,
                    visited_partition_keys=visited,
                )
        stagnant_restarts = stagnant_restarts + 1 if before == environment.surrogate_query_count else 0
    return _finish(
        "simulated_annealing",
        environment,
        baseline,
        candidates,
        query_budget=query_budget,
    )


def ppo_budget_search(
    environment: SurrogatePartitionEnvironment,
    policy: PartitionActorCritic,
    random_state: np.random.Generator,
    *,
    query_budget: int = 200,
    uncertainty_kappa: float = 1.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """在相同唯一查询预算内反复采样PPO轨迹，并硬过滤最终候选。"""
    baseline, candidates = _begin_search(
        environment, uncertainty_kappa=uncertainty_kappa
    )
    attempts_without_query = 0
    deterministic = True
    while environment.surrogate_query_count < query_budget and attempts_without_query < 10:
        before = environment.surrogate_query_count
        visited = {partition_key(environment.baseline, environment.depots)}
        environment.set_search_state(
            environment.baseline,
            step_index=0,
            visited_partition_keys=visited,
        )
        trace: list[str] = []
        for depth in range(environment.config.max_steps):
            actions = environment.available_actions()
            allowed = [actions[0]] + _budgeted_actions(
                environment, actions[1:], query_budget=query_budget
            )
            action_matrix = np.stack([action.features for action in allowed])
            state = environment.state_features()
            action_index, _, _, _ = policy.select_action(
                state,
                action_matrix,
                deterministic=deterministic,
                device=device,
            )
            # 首条确定性轨迹之后均使用策略采样；随机数状态由策略自身维护。
            action = allowed[action_index]
            if action.kind == "stop":
                break
            output, _ = environment.evaluate_actions([action])[0]
            trace.append(action.name)
            _remember(
                candidates,
                environment,
                action,
                output,
                uncertainty_kappa=uncertainty_kappa,
                trace=trace,
            )
            visited.add(partition_key(action.partition, environment.depots))
            environment.set_search_state(
                action.partition,
                step_index=depth + 1,
                visited_partition_keys=visited,
            )
        deterministic = False
        # 消耗一次生成器状态，使外部重复运行拥有明确且可复现的采样序列边界。
        random_state.random()
        attempts_without_query = (
            attempts_without_query + 1
            if before == environment.surrogate_query_count else 0
        )
    return _finish(
        "ppo", environment, baseline, candidates, query_budget=query_budget
    )
