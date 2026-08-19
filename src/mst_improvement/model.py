"""在不修改原论文模型的前提下组织 MST/MSF 分区改进流程。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from src.fstsp import MultiAgentFlyingSidekickTSP

from .directed_costs import build_directed_set_costs
from .local_search import PartitionSearchConfig, improve_partition
from .partition import partition_corrected_mst, partition_rooted_msf


@dataclass(frozen=True)
class MSTImprovementConfig:
    """配置初始分区方法、对称化方式和解码器引导搜索预算。"""

    partition_method: str = 'rooted_msf'
    symmetrization: str = 'mean'
    enable_relocate: bool = True
    enable_swap: bool = True
    exact_candidate_count: int = 5
    max_iterations: int = 20
    time_limit_seconds: float = 300.0
    improvement_tolerance: float = 1e-9

    def __post_init__(self):
        """在模型运行前尽早拒绝拼写错误或无效搜索预算。"""
        if self.partition_method not in {'corrected_mst', 'rooted_msf'}:
            raise ValueError(f'未知初始分区方法：{self.partition_method!r}。')
        if self.symmetrization not in {'mean', 'sum', 'max', 'min'}:
            raise ValueError(f'未知对称化策略：{self.symmetrization!r}。')
        if self.exact_candidate_count <= 0:
            raise ValueError('exact_candidate_count 必须大于 0。')
        if self.max_iterations < 0:
            raise ValueError('max_iterations 不能为负数。')
        if self.time_limit_seconds <= 0:
            raise ValueError('time_limit_seconds 必须大于 0。')


@dataclass
class GroupEvaluation:
    """缓存一个仓库客户集合经过 Phase 2 和 Phase 3 后的完整结果。"""

    cost: float
    raw_solution: dict[str, list]
    set_tsp_sequence: list
    visit_route: list
    phase2_seconds: float
    phase3_seconds: float


class ImprovedMSTMultiAgentFlyingSidekickTSP(MultiAgentFlyingSidekickTSP):
    """
    使用显式有向代价、修正 MST/MSF 和可选局部搜索的实验模型。

    输入与原 `MultiAgentFlyingSidekickTSP` 相同，额外接受 `improvement_config`。
    输出仍为 `(solution, cost)`，因此实验脚本可以把它与原算法并列运行。
    """

    def __init__(
        self,
        graph,
        depots,
        cities,
        distance,
        drone,
        limit=1.5,
        speed=1.6,
        theta=(0.5, 0.5),
        improvement_config: MSTImprovementConfig | None = None,
    ):
        """初始化原论文模型状态以及本实验独立的配置、缓存和遥测。"""
        super().__init__(graph, depots, cities, distance, drone, limit, speed, theta)
        self.improvement_config = improvement_config or MSTImprovementConfig()
        # 组缓存键由仓库与规范客户元组组成，避免搜索反复求解相同 Set-TSP。
        self.group_cache: dict[tuple[Any, tuple], GroupEvaluation] = {}
        self.telemetry: dict[str, Any] = {}

    def _canonical_group(self, cities) -> list:
        """
        按全局客户顺序规范化一个仓库组。

        输入：任意顺序的客户序列。
        输出：无重复且顺序稳定的客户列表。
        """
        city_set = set(cities)
        return [city for city in self.cities if city in city_set]

    def _evaluate_group(self, depot, cities, convex_sets) -> GroupEvaluation:
        """
        在不永久修改 `self.groups/self.cost/self.solution` 的情况下评价单个仓库组。

        输入：仓库、候选客户集合和全局边界候选点映射。
        输出：Set-TSP 顺序、Phase 3 路线、成本和分阶段耗时。

        实现逻辑：
        1. 使用规范化客户集合查询缓存。
        2. 临时替换当前仓库组，以复用未修改的原 `get_seq()`。
        3. 在 `finally` 中恢复原状态，再调用原 Phase 3 动态规划。
        """
        canonical_cities = self._canonical_group(cities)
        cache_key = (depot, tuple(canonical_cities))
        if cache_key in self.group_cache:
            self.telemetry['group_cache_hits'] += 1
            return self.group_cache[cache_key]

        if not canonical_cities:
            evaluation = GroupEvaluation(
                cost=0.0,
                raw_solution={'truck': [depot, depot], 'drone': []},
                set_tsp_sequence=[],
                visit_route=[depot, depot],
                phase2_seconds=0.0,
                phase3_seconds=0.0,
            )
            self.group_cache[cache_key] = evaluation
            self.telemetry['group_evaluations'] += 1
            return evaluation

        local_convex_sets = [[depot]] + [convex_sets[city] for city in canonical_cities]
        previous_group = list(self.groups[depot])
        phase2_started = perf_counter()
        self.groups[depot] = list(canonical_cities)
        try:
            sequence = self.get_seq(depot, local_convex_sets)
        finally:
            self.groups[depot] = previous_group
        phase2_seconds = perf_counter() - phase2_started

        visit_route = [depot] + [
            canonical_cities[index - 1]
            for index in sequence[1:-1]
        ] + [depot]
        phase3_started = perf_counter()
        raw_solution, cost = self.local_search_multi_drone_appr(visit_route, depot)
        phase3_seconds = perf_counter() - phase3_started

        evaluation = GroupEvaluation(
            cost=float(cost),
            raw_solution=raw_solution,
            set_tsp_sequence=list(sequence),
            visit_route=visit_route,
            phase2_seconds=phase2_seconds,
            phase3_seconds=phase3_seconds,
        )
        self.group_cache[cache_key] = evaluation
        self.telemetry['group_evaluations'] += 1
        self.telemetry['phase2_evaluation_seconds'] += phase2_seconds
        self.telemetry['phase3_evaluation_seconds'] += phase3_seconds
        return evaluation

    def _initial_partition(self, directed_costs) -> dict[Any, list[Any]]:
        """
        根据配置运行显式对称化的原 MST+DP 或超级根 MSF。

        输入：完整有向集合代价矩阵。
        输出：初始客户分组。
        """
        config = self.improvement_config
        if config.partition_method == 'corrected_mst':
            return partition_corrected_mst(
                directed_costs,
                self.depots,
                self.cities,
                config.symmetrization,
            )
        return partition_rooted_msf(
            directed_costs,
            self.depots,
            self.cities,
            config.symmetrization,
        )

    def solve(self):
        """
        执行有向代价构造、初始分区、可选局部搜索和原 Phase 2/3 解码。

        输入：无显式输入，使用初始化时保存的问题实例与改进配置。
        输出：与原算法一致的 `(self.solution, self.cost)`。

        原算法类和原 MST 分区函数不会被修改；所有新增时间与搜索统计写入 `self.telemetry`。
        """
        total_started = perf_counter()
        self.groups = {depot: [] for depot in self.depots}
        self.solution = []
        self.cost = 0.0
        self.group_cache = {}
        self.telemetry = {
            'config': asdict(self.improvement_config),
            'group_evaluations': 0,
            'group_cache_hits': 0,
            'phase2_evaluation_seconds': 0.0,
            'phase3_evaluation_seconds': 0.0,
        }

        boundary_started = perf_counter()
        convex_sets = self.get_boundary_convex_sets(self.theta[0])
        self.telemetry['boundary_seconds'] = perf_counter() - boundary_started

        directed_started = perf_counter()
        directed_costs = build_directed_set_costs(self, convex_sets)
        self.telemetry['directed_cost_seconds'] = perf_counter() - directed_started

        partition_started = perf_counter()
        self.groups = self._initial_partition(directed_costs)
        self.telemetry['initial_partition_seconds'] = perf_counter() - partition_started
        self.telemetry['initial_groups'] = {
            depot: list(cities)
            for depot, cities in self.groups.items()
        }

        config = self.improvement_config
        if config.enable_relocate or config.enable_swap:
            search_config = PartitionSearchConfig(
                enable_relocate=config.enable_relocate,
                enable_swap=config.enable_swap,
                exact_candidate_count=config.exact_candidate_count,
                max_iterations=config.max_iterations,
                time_limit_seconds=config.time_limit_seconds,
                improvement_tolerance=config.improvement_tolerance,
            )
            search_result = improve_partition(
                self.groups,
                directed_costs,
                lambda depot, cities: self._evaluate_group(
                    depot, cities, convex_sets
                ).cost,
                search_config,
            )
            self.groups = search_result.groups
            self.telemetry['search'] = {
                'iterations': search_result.iterations,
                'evaluated_candidates': search_result.evaluated_candidates,
                'accepted_relocates': search_result.accepted_relocates,
                'accepted_swaps': search_result.accepted_swaps,
                'elapsed_seconds': search_result.elapsed_seconds,
                'stop_reason': search_result.stop_reason,
            }
            # 使用同一 Phase 2/3 评价器记录局部搜索前后的分区成本，便于归因。
            self.telemetry['partition_cost_before_search'] = (
                search_result.initial_total_cost
            )
            self.telemetry['partition_cost_after_search'] = (
                search_result.final_total_cost
            )
        else:
            self.telemetry['search'] = {
                'iterations': 0,
                'evaluated_candidates': 0,
                'accepted_relocates': 0,
                'accepted_swaps': 0,
                'elapsed_seconds': 0.0,
                'stop_reason': 'disabled',
            }

        final_evaluations = {}
        for depot in self.depots:
            evaluation = self._evaluate_group(depot, self.groups[depot], convex_sets)
            final_evaluations[depot] = evaluation
            self.solution.append(self.convert(evaluation.raw_solution))
            self.cost += evaluation.cost

        group_sizes = [len(self.groups[depot]) for depot in self.depots]
        self.telemetry['final_groups'] = {
            depot: list(self.groups[depot])
            for depot in self.depots
        }
        self.telemetry['group_sizes'] = group_sizes
        self.telemetry['max_group_size'] = max(group_sizes, default=0)
        self.telemetry['group_size_std'] = float(np.std(group_sizes)) if group_sizes else 0.0
        self.telemetry['final_visit_routes'] = {
            depot: list(final_evaluations[depot].visit_route)
            for depot in self.depots
        }
        self.telemetry['final_group_costs'] = {
            depot: float(final_evaluations[depot].cost)
            for depot in self.depots
        }
        # 未启用局部搜索时，初始分区就是最终分区，两项成本应保持一致。
        if 'partition_cost_before_search' not in self.telemetry:
            partition_cost = float(sum(
                evaluation.cost for evaluation in final_evaluations.values()
            ))
            self.telemetry['partition_cost_before_search'] = partition_cost
            self.telemetry['partition_cost_after_search'] = partition_cost
        self.telemetry['total_seconds'] = perf_counter() - total_started
        self.telemetry['final_cost'] = float(self.cost)
        return self.solution, float(self.cost)
