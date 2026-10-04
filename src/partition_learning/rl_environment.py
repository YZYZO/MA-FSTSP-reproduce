"""受信赖域约束的客户划分强化学习环境。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from .candidates import (
    Partition,
    canonical_partition,
    moved_customer_count,
    partition_key,
)
from .surrogate_bundle import PartitionSurrogateBundle


STATE_FEATURE_NAMES = (
    "step_fraction",
    "moved_fraction",
    "size_std_ratio",
    "size_max_ratio",
    "size_min_ratio",
    "size_gini",
    "complexity_mean_log_ratio",
    "complexity_max_log_ratio",
    "complexity_gini",
    "empty_group_fraction",
)

ACTION_FEATURE_NAMES = (
    "is_stop",
    "is_relocate",
    "is_swap",
    "source_size_ratio",
    "target_size_ratio",
    "source_after_ratio",
    "target_after_ratio",
    "boundary_size_ratio",
    "road_delta_signed_log",
    "local_complexity_delta_ratio",
    "after_max_size_ratio",
    "after_size_gini",
    "after_complexity_gini",
    "after_moved_fraction",
    "after_max_complexity_ratio",
)


@dataclass(frozen=True)
class SurrogateRewardConfig:
    """保存代理奖励、动作数量和信赖域约束。"""

    max_steps: int = 8
    max_actions: int = 24
    max_moved_fraction: float = 0.20
    max_group_ratio: float = 1.75
    complexity_growth_limit: float = 1.15
    cost_limit: float = 0.10
    cost_penalty: float = 2.0
    uncertainty_weight: float = 0.10
    p90_weight: float = 0.05
    timeout_weight: float = 0.25
    step_penalty: float = 0.002
    source_group_count: int = 3
    customers_per_source: int = 6
    targets_per_customer: int = 2
    max_swap_actions: int = 6


@dataclass(frozen=True)
class PartitionRLAction:
    """保存一个局部划分动作、动作后完整分区和固定长度策略特征。"""

    name: str
    kind: str
    partition: Partition
    features: np.ndarray
    description: str


def _gini(values: Iterable[float]) -> float:
    """输入非负序列，输出零到一之间的基尼系数。"""
    array = np.sort(np.asarray(tuple(values), dtype=np.float64))
    if array.size == 0 or float(array.sum()) <= 0.0:
        return 0.0
    ranks = np.arange(1, array.size + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(ranks * array) / (array.size * array.sum()))
        - (array.size + 1.0) / array.size
    )


def _group_complexity(customers: Iterable[int], boundary_sizes: dict[int, int]) -> float:
    """按 Set-TSP 二元变量主导项估计一个仓库组的组合复杂度。"""
    set_sizes = [1] + [int(boundary_sizes[customer]) for customer in customers]
    set_count = len(set_sizes)
    return float(
        set_count * set_count
        + sum(size * size for size in set_sizes)
        + sum(set_sizes) ** 2
    )


def _relocate(
    partition: Partition,
    source: int,
    target: int,
    customer: int,
) -> Partition:
    """输入单客户迁移动作，输出不修改原对象的新完整分区。"""
    result = {depot: list(customers) for depot, customers in partition.items()}
    result[source].remove(customer)
    result[target].append(customer)
    return canonical_partition(result, result)


def _swap(
    partition: Partition,
    left_depot: int,
    right_depot: int,
    left_customer: int,
    right_customer: int,
) -> Partition:
    """输入两个仓库组及客户，输出交换后的新完整分区。"""
    result = {depot: list(customers) for depot, customers in partition.items()}
    result[left_depot].remove(left_customer)
    result[right_depot].remove(right_customer)
    result[left_depot].append(right_customer)
    result[right_depot].append(left_customer)
    return canonical_partition(result, result)


class SurrogatePartitionEnvironment:
    """
    以冻结监督代理模型为奖励函数的局部客户划分环境。

    状态从 MST 划分开始；动作由单客户迁移、道路相邻组交换和停止组成。环境在生成
    动作时限制移动客户比例、最大组规模和 Set-TSP 复杂度增长，并缓存已查询划分。
    """

    def __init__(
        self,
        bundle: PartitionSurrogateBundle,
        reward_config: SurrogateRewardConfig | None = None,
    ):
        self.bundle = bundle
        self.config = reward_config or SurrogateRewardConfig()
        self.instance = bundle.instance
        self.baseline = canonical_partition(
            self.instance.partition,
            self.instance.partition,
        )
        self.depots = tuple(self.baseline)
        self.customer_count = sum(map(len, self.baseline.values()))
        self.ideal_group_size = self.customer_count / max(len(self.depots), 1)
        self.boundary_sizes = {
            int(customer): int(size)
            for customer, size in self.instance.boundary_sizes.items()
        }
        self.maximum_boundary_size = max(self.boundary_sizes.values(), default=1)
        self.truck_query = bundle.predictors["numeric"].context["truck"].query

        # 信赖域允许改善原始不平衡划分，但不允许比 MST 的极端组规模明显更差。
        baseline_sizes = [len(self.baseline[depot]) for depot in self.depots]
        self.group_size_cap = max(
            max(baseline_sizes),
            math.ceil(self.ideal_group_size * self.config.max_group_ratio),
        )
        baseline_complexities = self._complexities(self.baseline)
        self.baseline_max_complexity = max(baseline_complexities.values())
        self.complexity_cap = (
            self.baseline_max_complexity * self.config.complexity_growth_limit
        )
        self.max_moved_customers = max(
            1,
            math.ceil(self.customer_count * self.config.max_moved_fraction),
        )
        baseline_affinities = [
            self._affinity(depot, customer)
            for depot, customers in self.baseline.items()
            for customer in customers
        ]
        self.road_log_scale = max(
            float(np.mean(np.log1p(np.asarray(baseline_affinities)))),
            1.0,
        )

        # 预测缓存是训练可行性的关键：同一策略轨迹反复访问的划分只推理一次。
        self.prediction_cache: dict[
            tuple[tuple[int, ...], ...],
            dict[str, Any],
        ] = {}
        self.current = self.baseline
        self.current_prediction: dict[str, Any] | None = None
        self.current_score = 0.0
        self.baseline_time = 1.0
        self.baseline_cost = 1.0
        self.step_index = 0
        self.visited_partition_keys: set[tuple[tuple[int, ...], ...]] = set()

    def _affinity(self, depot: int, customer: int) -> float:
        """返回仓库与客户之间的有向往返道路距离。"""
        return float(
            self.truck_query(int(depot), int(customer))
            + self.truck_query(int(customer), int(depot))
        )

    def _complexities(self, partition: Partition) -> dict[int, float]:
        """输入完整分区，输出每个仓库组的 Set-TSP 复杂度代理。"""
        return {
            depot: _group_complexity(partition[depot], self.boundary_sizes)
            for depot in self.depots
        }

    def _state_features(self, partition: Partition) -> np.ndarray:
        """把当前划分压缩成与客户和仓库编号无关的固定长度状态特征。"""
        sizes = np.asarray(
            [len(partition[depot]) for depot in self.depots],
            dtype=np.float64,
        )
        complexities = np.asarray(
            list(self._complexities(partition).values()),
            dtype=np.float64,
        )
        baseline_log = math.log1p(max(self.baseline_max_complexity, 1.0))
        return np.asarray(
            [
                self.step_index / max(self.config.max_steps, 1),
                moved_customer_count(self.baseline, partition)
                / max(self.customer_count, 1),
                float(np.std(sizes)) / max(self.ideal_group_size, 1.0),
                float(np.max(sizes)) / max(self.ideal_group_size, 1.0),
                float(np.min(sizes)) / max(self.ideal_group_size, 1.0),
                _gini(sizes),
                float(np.mean(np.log1p(complexities))) / baseline_log,
                float(np.max(np.log1p(complexities))) / baseline_log,
                _gini(complexities),
                float(np.mean(sizes == 0)),
            ],
            dtype=np.float32,
        )

    def state_features(self) -> np.ndarray:
        """返回当前环境状态的固定长度策略特征。"""
        return self._state_features(self.current)

    def _within_trust_region(self, partition: Partition) -> bool:
        """判断候选是否满足移动数量、组规模和复杂度增长限制。"""
        if moved_customer_count(self.baseline, partition) > self.max_moved_customers:
            return False
        if max(map(len, partition.values())) > self.group_size_cap:
            return False
        return max(self._complexities(partition).values()) <= self.complexity_cap

    def _action_features(
        self,
        kind: str,
        partition: Partition,
        *,
        source: int | None = None,
        target: int | None = None,
        moved_customers: tuple[int, ...] = (),
        road_delta: float = 0.0,
    ) -> np.ndarray:
        """编码动作类型、局部组变化、道路增量和动作后全局平衡程度。"""
        before_complexities = self._complexities(self.current)
        after_complexities = self._complexities(partition)
        source_size = len(self.current[source]) if source is not None else 0
        target_size = len(self.current[target]) if target is not None else 0
        source_after = len(partition[source]) if source is not None else 0
        target_after = len(partition[target]) if target is not None else 0
        before_local = (
            before_complexities[source] + before_complexities[target]
            if source is not None and target is not None
            else 1.0
        )
        after_local = (
            after_complexities[source] + after_complexities[target]
            if source is not None and target is not None
            else before_local
        )
        boundary_ratio = (
            float(np.mean([
                self.boundary_sizes[customer] for customer in moved_customers
            ]))
            / max(self.maximum_boundary_size, 1)
            if moved_customers
            else 0.0
        )
        sizes_after = np.asarray(
            [len(partition[depot]) for depot in self.depots],
            dtype=np.float64,
        )
        complexities_after = np.asarray(
            list(after_complexities.values()),
            dtype=np.float64,
        )
        signed_road = (
            math.copysign(math.log1p(abs(road_delta)), road_delta)
            / self.road_log_scale
            if road_delta
            else 0.0
        )
        return np.asarray(
            [
                float(kind == "stop"),
                float(kind == "relocate"),
                float(kind == "swap"),
                source_size / max(self.ideal_group_size, 1.0),
                target_size / max(self.ideal_group_size, 1.0),
                source_after / max(self.ideal_group_size, 1.0),
                target_after / max(self.ideal_group_size, 1.0),
                boundary_ratio,
                signed_road,
                (after_local - before_local) / max(before_local, 1.0),
                float(np.max(sizes_after)) / max(self.ideal_group_size, 1.0),
                _gini(sizes_after),
                _gini(complexities_after),
                moved_customer_count(self.baseline, partition)
                / max(self.customer_count, 1),
                float(np.max(complexities_after))
                / max(self.baseline_max_complexity, 1.0),
            ],
            dtype=np.float32,
        )

    def _relocation_actions(self) -> list[tuple[float, PartitionRLAction]]:
        """生成高复杂度组削减、边界客户迁移和道路友好迁移动作。"""
        complexities = self._complexities(self.current)
        sources = sorted(
            [depot for depot in self.depots if self.current[depot]],
            key=lambda depot: (
                -len(self.current[depot]) / max(self.ideal_group_size, 1.0),
                -complexities[depot],
                depot,
            ),
        )[: self.config.source_group_count]
        generated: list[tuple[float, PartitionRLAction]] = []
        for source in sources:
            customers = tuple(self.current[source])
            by_boundary = sorted(
                customers,
                key=lambda customer: (-self.boundary_sizes[customer], customer),
            )
            by_alternative = sorted(
                customers,
                key=lambda customer: min(
                    self._affinity(target, customer)
                    - self._affinity(source, customer)
                    for target in self.depots
                    if target != source
                ),
            )
            # 交替保留“集合边界大”和“道路迁移友好”的客户，避免简单截断后
            # 第二类客户完全进不了候选集合。
            selected_customers_list: list[int] = []
            for rank in range(self.config.customers_per_source):
                for ranked_customers in (by_boundary, by_alternative):
                    if rank >= len(ranked_customers):
                        continue
                    customer = ranked_customers[rank]
                    if customer not in selected_customers_list:
                        selected_customers_list.append(customer)
                    if (
                        len(selected_customers_list)
                        >= self.config.customers_per_source
                    ):
                        break
                if (
                    len(selected_customers_list)
                    >= self.config.customers_per_source
                ):
                    break
            selected_customers = tuple(selected_customers_list)
            for customer in selected_customers:
                targets = sorted(
                    [depot for depot in self.depots if depot != source],
                    key=lambda target: (
                        self._affinity(target, customer)
                        - self._affinity(source, customer)
                        + 0.05 * len(self.current[target]),
                        target,
                    ),
                )[: self.config.targets_per_customer]
                for target in targets:
                    partition = _relocate(self.current, source, target, customer)
                    if not self._within_trust_region(partition):
                        continue
                    road_delta = (
                        self._affinity(target, customer)
                        - self._affinity(source, customer)
                    )
                    before = complexities[source] + complexities[target]
                    after_complexities = self._complexities(partition)
                    after = after_complexities[source] + after_complexities[target]
                    complexity_gain = (before - after) / max(before, 1.0)
                    balance_gain = (
                        abs(len(self.current[source]) - self.ideal_group_size)
                        + abs(len(self.current[target]) - self.ideal_group_size)
                        - abs(len(partition[source]) - self.ideal_group_size)
                        - abs(len(partition[target]) - self.ideal_group_size)
                    ) / max(self.ideal_group_size, 1.0)
                    heuristic = complexity_gain + 0.05 * balance_gain - 0.01 * max(
                        0.0,
                        math.log1p(max(road_delta, 0.0)),
                    )
                    generated.append((
                        heuristic,
                        PartitionRLAction(
                            name=f"relocate_{customer}_{source}_{target}",
                            kind="relocate",
                            partition=partition,
                            features=self._action_features(
                                "relocate",
                                partition,
                                source=source,
                                target=target,
                                moved_customers=(customer,),
                                road_delta=road_delta,
                            ),
                            description=(
                                f"客户 {customer} 从仓库 {source} 迁移到 {target}"
                            ),
                        ),
                    ))
        return generated

    def _swap_actions(self) -> list[tuple[float, PartitionRLAction]]:
        """生成道路代价友好的跨仓库客户交换动作。"""
        pairs = [
            (left, right)
            for left_index, left in enumerate(self.depots)
            for right in self.depots[left_index + 1 :]
            if self.current[left] and self.current[right]
        ]
        pairs.sort(
            key=lambda pair: (
                -abs(len(self.current[pair[0]]) - len(self.current[pair[1]])),
                pair,
            )
        )
        generated: list[tuple[float, PartitionRLAction]] = []
        for left, right in pairs[: self.config.max_swap_actions]:
            left_customer = min(
                self.current[left],
                key=lambda customer: (
                    self._affinity(right, customer)
                    - self._affinity(left, customer),
                    customer,
                ),
            )
            right_customer = min(
                self.current[right],
                key=lambda customer: (
                    self._affinity(left, customer)
                    - self._affinity(right, customer),
                    customer,
                ),
            )
            partition = _swap(
                self.current,
                left,
                right,
                left_customer,
                right_customer,
            )
            if not self._within_trust_region(partition):
                continue
            road_delta = (
                self._affinity(right, left_customer)
                - self._affinity(left, left_customer)
                + self._affinity(left, right_customer)
                - self._affinity(right, right_customer)
            )
            current_complexities = self._complexities(self.current)
            after_complexities = self._complexities(partition)
            before = sum(current_complexities[depot] for depot in (left, right))
            after = sum(after_complexities[depot] for depot in (left, right))
            heuristic = (before - after) / max(before, 1.0) - 0.01 * max(
                0.0,
                math.log1p(max(road_delta, 0.0)),
            )
            generated.append((
                heuristic,
                PartitionRLAction(
                    name=(
                        f"swap_{left_customer}_{right_customer}_{left}_{right}"
                    ),
                    kind="swap",
                    partition=partition,
                    features=self._action_features(
                        "swap",
                        partition,
                        source=left,
                        target=right,
                        moved_customers=(left_customer, right_customer),
                        road_delta=road_delta,
                    ),
                    description=(
                        f"交换客户 {left_customer} 与 {right_customer}"
                    ),
                ),
            ))
        return generated

    def available_actions(self) -> list[PartitionRLAction]:
        """返回停止动作及按启发式多样性筛选的有限局部动作集合。"""
        stop = PartitionRLAction(
            name="stop",
            kind="stop",
            partition=self.current,
            features=self._action_features("stop", self.current),
            description="结束当前划分轨迹",
        )
        relocation_actions = self._relocation_actions()
        swap_actions = self._swap_actions()
        relocation_actions.sort(key=lambda item: (-item[0], item[1].name))
        swap_actions.sort(key=lambda item: (-item[0], item[1].name))

        # 给交换动作预留少量名额，其余仍优先用于能直接削减最大组的迁移动作。
        action_budget = max(self.config.max_actions - 1, 0)
        swap_budget = min(
            len(swap_actions),
            self.config.max_swap_actions,
            max(1, action_budget // 4) if action_budget else 0,
        )
        relocation_budget = max(action_budget - swap_budget, 0)
        generated = (
            relocation_actions[:relocation_budget]
            + swap_actions[:swap_budget]
        )
        generated.sort(key=lambda item: (-item[0], item[1].name))
        # 一个回合内禁止重访历史划分，避免策略通过“迁出—迁回”空耗步数。
        seen = set(self.visited_partition_keys)
        seen.add(partition_key(self.current, self.depots))
        actions = [stop]
        for _, action in generated:
            key = partition_key(action.partition, self.depots)
            if key in seen:
                continue
            seen.add(key)
            actions.append(action)
            if len(actions) >= self.config.max_actions:
                break
        return actions

    def _predict_many(
        self,
        actions: list[PartitionRLAction],
    ) -> list[dict[str, Any]]:
        """批量查询未缓存动作并按动作顺序返回统一代理预测。"""
        missing_actions = [
            action
            for action in actions
            if partition_key(action.partition, self.depots)
            not in self.prediction_cache
        ]
        if missing_actions:
            outputs = self.bundle.predict_many(
                [action.partition for action in missing_actions],
                kinds=[action.kind for action in missing_actions],
                strengths=[
                    moved_customer_count(self.baseline, action.partition)
                    / max(self.customer_count, 1)
                    for action in missing_actions
                ],
            )
            for action, output in zip(missing_actions, outputs):
                self.prediction_cache[
                    partition_key(action.partition, self.depots)
                ] = output
        return [
            self.prediction_cache[partition_key(action.partition, self.depots)]
            for action in actions
        ]

    def score_components(self, output: dict[str, Any]) -> dict[str, float]:
        """把统一代理输出转换成时间、成本、不确定性和超时风险分量。"""
        prediction = output["prediction"]
        numeric_uncertainty = output["uncertainty"]["numeric"]
        time_value = max(float(prediction["downstream_total_seconds"]), 0.0)
        p90 = max(float(prediction["downstream_time_p90"]), time_value)
        uncertainty = max(
            float(numeric_uncertainty["downstream_total_seconds"]),
            0.0,
        )
        # 成本约束以本实例的 MST 预测成本为参照，不依赖代理模型另一个可能
        # 存在校准误差的 cost_change_ratio 输出。
        relative_cost_change = (
            float(prediction["final_cost"]) - self.baseline_cost
        ) / max(abs(self.baseline_cost), 1e-6)
        cost_violation = max(
            0.0,
            relative_cost_change - self.config.cost_limit,
        )
        timeout_probability = float(prediction["right_censored_probability"])
        denominator = max(self.baseline_time, 1e-6)
        components = {
            "time": time_value / denominator,
            "cost": self.config.cost_penalty * cost_violation,
            "uncertainty": self.config.uncertainty_weight
            * uncertainty
            / denominator,
            "p90": self.config.p90_weight
            * max(0.0, p90 - time_value)
            / denominator,
            "timeout": self.config.timeout_weight * timeout_probability,
            "relative_cost_change": relative_cost_change,
        }
        components["total"] = sum(
            components[name]
            for name in ("time", "cost", "uncertainty", "p90", "timeout")
        )
        return components

    def reset(self) -> tuple[np.ndarray, list[PartitionRLAction]]:
        """重置到 MST 基线并返回初始状态和合法动作。"""
        self.current = self.baseline
        self.step_index = 0
        self.visited_partition_keys = {
            partition_key(self.baseline, self.depots)
        }
        baseline_action = PartitionRLAction(
            "baseline",
            "stay",
            self.baseline,
            np.zeros(len(ACTION_FEATURE_NAMES), dtype=np.float32),
            "MST基线",
        )
        self.current_prediction = self._predict_many([baseline_action])[0]
        self.baseline_time = max(
            float(
                self.current_prediction["prediction"][
                    "downstream_total_seconds"
                ]
            ),
            1e-6,
        )
        self.baseline_cost = float(
            self.current_prediction["prediction"]["final_cost"]
        )
        self.current_score = self.score_components(
            self.current_prediction
        )["total"]
        return self.state_features(), self.available_actions()

    def evaluate_actions(
        self,
        actions: list[PartitionRLAction],
    ) -> list[tuple[dict[str, Any], dict[str, float]]]:
        """批量返回动作后的代理预测与风险调整分数，供贪心基线和诊断使用。"""
        outputs = self._predict_many(actions)
        return [
            (output, self.score_components(output))
            for output in outputs
        ]

    def step(
        self,
        action: PartitionRLAction,
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """执行一个动作，返回新状态、代理奖励、结束标志和诊断信息。"""
        if action.kind == "stop":
            assert self.current_prediction is not None
            return self.state_features(), 0.0, True, {
                "action": action.name,
                "score": self.current_score,
                "prediction": self.current_prediction,
                "score_components": self.score_components(
                    self.current_prediction
                ),
            }
        output = self._predict_many([action])[0]
        components = self.score_components(output)
        reward = (
            self.current_score
            - components["total"]
            - self.config.step_penalty
        )
        self.current = action.partition
        self.visited_partition_keys.add(
            partition_key(self.current, self.depots)
        )
        self.current_prediction = output
        self.current_score = components["total"]
        self.step_index += 1
        done = self.step_index >= self.config.max_steps
        return self.state_features(), float(reward), done, {
            "action": action.name,
            "description": action.description,
            "score": self.current_score,
            "prediction": output,
            "score_components": components,
            "moved_customers": moved_customer_count(
                self.baseline,
                self.current,
            ),
        }
