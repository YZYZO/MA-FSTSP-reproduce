"""里程碑 D 使用的客户划分选项和对比方法统一适配层。"""

from dataclasses import dataclass, replace
from pathlib import Path
import time

import torch

from src.fstsp import MultiAgentFlyingSidekickTSP
from src.learning.hierarchical_policy import (
    HierarchicalPartitionPolicy,
    HierarchicalPolicyConfig,
    rollout_policy,
)
from src.learning.partition_env import (
    PartitionEnvironmentConfig,
    PartitionRepairEnvironment,
    SurrogateGroupScorer,
)
from src.learning.single_level_policy import (
    SingleLevelActorCritic,
    SingleLevelPolicyConfig,
    rollout_single_level,
)
from src.partitioning import partition_customers


PRODUCTION_PARTITION_OPTIONS = (
    "legacy_mst",
    "symmetric_mst",
    "solver_aware_hrl",
)

EVALUATION_PARTITION_METHODS = (
    "legacy_mst",
    "symmetric_mst",
    "balanced_mst",
    "solver_aware_greedy",
    "single_level_rl",
    "solver_aware_hrl",
)


def _stable_node_key(node):
    """
    为仓库和客户节点生成稳定排序键。

    输入：任意节点编号。
    输出：类型名与文本表示组成的元组。
    逻辑：所有方法输出统一的仓库及组内客户顺序。
    """
    return type(node).__name__, repr(node)


def _copy_groups(groups):
    """
    复制并稳定排序客户分区。

    输入：`{depot: customers}` 字典。
    输出：新的列表字典。
    逻辑：避免学习环境或实验评估修改其他方法的基准分区。
    """
    return {
        depot: sorted(customers, key=_stable_node_key)
        for depot, customers in groups.items()
    }


@dataclass(frozen=True)
class PartitionMethodContext:
    """
    保存同一实例上全部划分方法共享的模型和候选集合。

    输入：原 MA-FSTSP 模型、边界候选集合及其构造时间。
    输出：不可变上下文对象。
    逻辑：各对比方法复用完全相同的第一阶段候选节点，避免重复计算和口径差异。
    """

    model: object
    convex_sets: dict
    boundary_construction_seconds: float


@dataclass(frozen=True)
class PartitionMethodResult:
    """
    保存一种划分方法的客户组、耗时和策略遥测。

    输入：方法、分组、模型加载/决策时间、移动数、代理前后分数和真实复核次数。
    输出：不可变结果对象。
    逻辑：传统方法没有代理分数时使用 `None`，仍可由统一评估器处理。
    """

    method: str
    groups: dict
    model_loading_seconds: float
    partition_strategy_seconds: float
    move_count: int = 0
    predicted_initial_score: float | None = None
    predicted_final_score: float | None = None
    strategy_real_solver_calls: int = 0
    stop_reason: str | None = None


def build_partition_method_context(
    graph,
    depots,
    cities,
    distance,
    drone_count,
    drone_limit=1.5,
    drone_speed=1.6,
    theta=(0.5, 0.5),
):
    """
    为一个道路实例构建各划分方法共享的上下文。

    输入：路网、仓库、客户、距离和原算法车队参数。
    输出：`PartitionMethodContext`。
    逻辑：只构造一次模型和边界候选集合，原模型三阶段函数保持不变。
    """
    model = MultiAgentFlyingSidekickTSP(
        graph,
        depots,
        cities,
        distance,
        drone_count,
        limit=drone_limit,
        speed=drone_speed,
        theta=theta,
    )
    start = time.perf_counter()
    convex_sets = model.get_boundary_convex_sets(model.theta[0])
    boundary_seconds = time.perf_counter() - start
    return PartitionMethodContext(
        model=model,
        convex_sets=convex_sets,
        boundary_construction_seconds=float(boundary_seconds),
    )


def _mst_groups(context, edge_mode):
    """
    使用指定边权模式生成 MST 客户分区。

    输入：共享上下文及 `legacy/mean` 等边权模式。
    输出：稳定排序后的分组字典。
    逻辑：复用新增独立分区模块，不修改原 `set_mst`。
    """
    model = context.model
    groups = partition_customers(
        model.depots,
        model.cities,
        context.convex_sets,
        model.distance["truck"],
        model.distance["drone"],
        model.speed,
        edge_mode=edge_mode,
        coefficient=model.const,
    )
    return _copy_groups(groups)


def _depot_affinity(customer, depot, truck_distance):
    """
    计算均衡基线使用的客户—仓库双向平均道路距离。

    输入：客户、仓库和有向道路距离表。
    输出：浮点亲和距离。
    逻辑：移动客户时最小化相对原仓库的亲和力增量。
    """
    return float(
        (truck_distance[depot][customer] + truck_distance[customer][depot])
        / 2.0
    )


def balance_mst_groups(groups, depots, truck_distance):
    """
    将 symmetric MST 分区修复为客户数量相差至多一的容量均衡基线。

    输入：初始组、仓库顺序和有向道路距离。
    输出：客户守恒且组规模位于 `floor/ceil(N/K)` 的新分区。
    逻辑：每次从过大组向不足组移动亲和力代价增量最小的客户。
    """
    balanced = _copy_groups(groups)
    ordered_depots = tuple(sorted(depots, key=_stable_node_key))
    customer_count = sum(len(balanced[depot]) for depot in ordered_depots)
    lower_capacity = customer_count // len(ordered_depots)
    upper_capacity = (customer_count + len(ordered_depots) - 1) // len(ordered_depots)

    while True:
        below_lower = [
            depot for depot in ordered_depots
            if len(balanced[depot]) < lower_capacity
        ]
        above_upper = [
            depot for depot in ordered_depots
            if len(balanced[depot]) > upper_capacity
        ]
        if below_lower:
            targets = below_lower
            sources = [
                depot for depot in ordered_depots
                if len(balanced[depot]) > lower_capacity
            ]
        elif above_upper:
            targets = [
                depot for depot in ordered_depots
                if len(balanced[depot]) < upper_capacity
            ]
            sources = above_upper
        else:
            break

        candidates = []
        for source in sources:
            for target in targets:
                if source == target:
                    continue
                for customer in balanced[source]:
                    delta = (
                        _depot_affinity(customer, target, truck_distance)
                        - _depot_affinity(customer, source, truck_distance)
                    )
                    candidates.append(
                        (
                            delta,
                            _stable_node_key(source),
                            _stable_node_key(target),
                            _stable_node_key(customer),
                            source,
                            target,
                            customer,
                        )
                    )
        _, _, _, _, source, target, customer = min(candidates)
        balanced[source].remove(customer)
        balanced[target].append(customer)
        balanced[source].sort(key=_stable_node_key)
        balanced[target].sort(key=_stable_node_key)
    return balanced


def _load_torch_checkpoint(path):
    """
    在 CPU 上读取策略检查点。

    输入：检查点路径。
    输出：PyTorch 字典。
    逻辑：策略和代理均为受信任的本项目产物，需要恢复配置元数据。
    """
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _resolve_surrogate_checkpoint(explicit_path, policy_checkpoint=None):
    """
    确定学习型方法应使用的代理检查点。

    输入：显式路径和可选策略检查点字典。
    输出：代理模型路径。
    逻辑：显式参数优先，否则使用训练策略记录的代理版本。
    """
    if explicit_path is not None:
        return Path(explicit_path)
    if policy_checkpoint is not None:
        return Path(policy_checkpoint["surrogate_checkpoint"])
    raise ValueError("学习型划分方法需要 surrogate checkpoint。")


def _load_surrogate_scorer(context, surrogate_path, mc_samples):
    """
    从检查点加载当前实例使用的代理评分器。

    输入：实例上下文、代理路径和 MC Dropout 次数。
    输出：尚未执行客户组预测的 `SurrogateGroupScorer`。
    逻辑：将纯模型加载与环境初始化推理解耦，保证策略耗时统计口径准确。
    """
    model = context.model
    return SurrogateGroupScorer.from_checkpoint(
        model.graph,
        model.distance,
        context.convex_sets,
        surrogate_path,
        mc_samples=mc_samples,
    )


def _build_learning_environment(context, initial_groups, scorer, environment_config):
    """
    为贪心、单层 RL 或 HRL 建立统一代理修复环境。

    输入：实例上下文、初始组、已加载评分器和环境配置。
    输出：不调用真实求解器的 `PartitionRepairEnvironment`。
    逻辑：正式下游评估稍后统一执行，因此策略阶段的真实复核预算强制为零。
    """
    model = context.model
    config = replace(environment_config, real_verification_budget=0)
    return PartitionRepairEnvironment(
        model.graph,
        model.distance,
        model.depots,
        context.convex_sets,
        initial_groups,
        scorer,
        config=config,
    )


def run_partition_method(
    context,
    method,
    surrogate_checkpoint=None,
    hrl_checkpoint=None,
    single_level_checkpoint=None,
):
    """
    在共享实例上下文上运行一种传统或学习型客户划分方法。

    输入：上下文、方法名以及可能需要的三个模型路径。
    输出：`PartitionMethodResult`。
    逻辑：所有学习方法都从 symmetric MST 开始，传统方法不加载 PyTorch 模型。
    """
    if method not in EVALUATION_PARTITION_METHODS:
        raise ValueError(f"不支持的划分方法：{method}")

    loading_seconds = 0.0
    if method in ("legacy_mst", "symmetric_mst", "balanced_mst"):
        start = time.perf_counter()
        edge_mode = "legacy" if method == "legacy_mst" else "mean"
        groups = _mst_groups(context, edge_mode)
        if method == "balanced_mst":
            groups = balance_mst_groups(
                groups,
                context.model.depots,
                context.model.distance["truck"],
            )
        partition_seconds = time.perf_counter() - start
        return PartitionMethodResult(
            method=method,
            groups=groups,
            model_loading_seconds=0.0,
            partition_strategy_seconds=float(partition_seconds),
        )

    initial_partition_start = time.perf_counter()
    initial_groups = _mst_groups(context, "mean")
    initial_partition_seconds = time.perf_counter() - initial_partition_start
    if method == "solver_aware_greedy":
        load_start = time.perf_counter()
        environment_config = PartitionEnvironmentConfig()
        surrogate_path = _resolve_surrogate_checkpoint(surrogate_checkpoint)
        scorer = _load_surrogate_scorer(
            context,
            surrogate_path,
            environment_config.mc_samples,
        )
        loading_seconds = time.perf_counter() - load_start
        strategy_start = time.perf_counter()
        environment = _build_learning_environment(
            context,
            initial_groups,
            scorer,
            environment_config,
        )
        environment.reset()
        while not environment.done:
            upper_index, lower_index, _ = environment.best_predicted_action()
            environment.step(upper_index, lower_index)
        partition_seconds = (
            initial_partition_seconds + time.perf_counter() - strategy_start
        )
        return PartitionMethodResult(
            method=method,
            groups=_copy_groups(environment.groups),
            model_loading_seconds=float(loading_seconds),
            partition_strategy_seconds=float(partition_seconds),
            move_count=environment.steps,
            predicted_initial_score=environment.initial_snapshot.score,
            predicted_final_score=environment.current_snapshot.score,
            strategy_real_solver_calls=0,
            stop_reason=environment.stop_reason,
        )

    if method == "solver_aware_hrl":
        if hrl_checkpoint is None:
            raise ValueError("solver_aware_hrl 需要 HRL checkpoint。")
        load_start = time.perf_counter()
        checkpoint = _load_torch_checkpoint(hrl_checkpoint)
        policy = HierarchicalPartitionPolicy(
            HierarchicalPolicyConfig(**checkpoint["policy_config"])
        )
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()
        surrogate_path = _resolve_surrogate_checkpoint(
            surrogate_checkpoint,
            checkpoint,
        )
        environment_config = PartitionEnvironmentConfig(
            **checkpoint["environment_config"]
        )
        scorer = _load_surrogate_scorer(
            context,
            surrogate_path,
            environment_config.mc_samples,
        )
        loading_seconds = time.perf_counter() - load_start
        strategy_start = time.perf_counter()
        environment = _build_learning_environment(
            context,
            initial_groups,
            scorer,
            environment_config,
        )
        rollout = rollout_policy(policy, environment, deterministic=True)
        partition_seconds = (
            initial_partition_seconds + time.perf_counter() - strategy_start
        )
        return PartitionMethodResult(
            method=method,
            groups=_copy_groups(rollout["groups"]),
            model_loading_seconds=float(loading_seconds),
            partition_strategy_seconds=float(partition_seconds),
            move_count=rollout["move_count"],
            predicted_initial_score=rollout["initial_snapshot"].score,
            predicted_final_score=rollout["final_snapshot"].score,
            strategy_real_solver_calls=rollout["real_solver_calls"],
            stop_reason=rollout["stop_reason"],
        )

    if single_level_checkpoint is None:
        raise ValueError("single_level_rl 需要 single-level checkpoint。")
    load_start = time.perf_counter()
    checkpoint = _load_torch_checkpoint(single_level_checkpoint)
    policy = SingleLevelActorCritic(
        SingleLevelPolicyConfig(**checkpoint["policy_config"])
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    surrogate_path = _resolve_surrogate_checkpoint(
        surrogate_checkpoint,
        checkpoint,
    )
    environment_config = PartitionEnvironmentConfig(
        **checkpoint["environment_config"]
    )
    scorer = _load_surrogate_scorer(
        context,
        surrogate_path,
        environment_config.mc_samples,
    )
    loading_seconds = time.perf_counter() - load_start
    strategy_start = time.perf_counter()
    environment = _build_learning_environment(
        context,
        initial_groups,
        scorer,
        environment_config,
    )
    rollout = rollout_single_level(policy, environment, deterministic=True)
    partition_seconds = (
        initial_partition_seconds + time.perf_counter() - strategy_start
    )
    return PartitionMethodResult(
        method=method,
        groups=_copy_groups(rollout["groups"]),
        model_loading_seconds=float(loading_seconds),
        partition_strategy_seconds=float(partition_seconds),
        move_count=rollout["move_count"],
        predicted_initial_score=rollout["initial_snapshot"].score,
        predicted_final_score=rollout["final_snapshot"].score,
        strategy_real_solver_calls=rollout["real_solver_calls"],
        stop_reason=rollout["stop_reason"],
    )


def partition_instance(
    graph,
    depots,
    cities,
    distance,
    drone_count,
    partition_option,
    surrogate_checkpoint=None,
    hrl_checkpoint=None,
):
    """
    使用三个正式划分选项之一返回可交给原第二、三阶段的客户组。

    输入：完整实例、车队规模、`legacy_mst/symmetric_mst/solver_aware_hrl` 和模型路径。
    输出：`PartitionMethodResult`。
    逻辑：这是里程碑 D 的稳定接入入口，不改变原 `MultiAgentFlyingSidekickTSP.solve`。
    """
    if partition_option not in PRODUCTION_PARTITION_OPTIONS:
        raise ValueError(
            f"partition_option 必须属于 {PRODUCTION_PARTITION_OPTIONS}。"
        )
    context = build_partition_method_context(
        graph,
        depots,
        cities,
        distance,
        drone_count,
    )
    return run_partition_method(
        context,
        partition_option,
        surrogate_checkpoint=surrogate_checkpoint,
        hrl_checkpoint=hrl_checkpoint,
    )
