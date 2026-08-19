"""基于有向路线代理和下游评价器的客户分区局部搜索。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class PartitionMove:
    """记录一次 relocate 或 swap 候选操作及其代理成本变化。"""

    kind: str
    source: Any
    target: Any
    source_city: Any
    target_city: Any | None
    approximate_delta: float


@dataclass(frozen=True)
class PartitionSearchConfig:
    """控制局部搜索候选规模、迭代次数和停止条件。"""

    enable_relocate: bool = True
    enable_swap: bool = True
    exact_candidate_count: int = 5
    max_iterations: int = 20
    time_limit_seconds: float = 300.0
    improvement_tolerance: float = 1e-9


@dataclass
class PartitionSearchResult:
    """保存局部搜索最终分区及可复核统计信息。"""

    groups: dict[Any, list[Any]]
    group_costs: dict[Any, float]
    initial_total_cost: float
    final_total_cost: float
    iterations: int
    evaluated_candidates: int
    accepted_relocates: int
    accepted_swaps: int
    elapsed_seconds: float
    stop_reason: str


def route_proxy_cost(route: Sequence, directed_costs: Mapping) -> float:
    """
    计算一条仓库闭合访问顺序的有向集合转移代理成本。

    输入：首尾为同一仓库的节点顺序和有向代价矩阵。
    输出：相邻有向转移代价之和。
    """
    return sum(
        float(directed_costs[start][end])
        for start, end in zip(route, route[1:])
    )


def cheapest_insertion_route(depot, cities: Sequence, directed_costs: Mapping) -> list:
    """
    用有向 cheapest insertion 为一个仓库组生成快速代理路线。

    输入：仓库、客户序列和有向代价矩阵。
    输出：首尾均为仓库的访问顺序。
    """
    route = [depot, depot]
    for city in cities:
        best_position = 1
        best_delta = float('inf')
        for position in range(1, len(route)):
            previous = route[position - 1]
            following = route[position]
            delta = (
                directed_costs[previous][city]
                + directed_costs[city][following]
                - directed_costs[previous][following]
            )
            if delta < best_delta:
                best_delta = float(delta)
                best_position = position
        route.insert(best_position, city)
    return route


def _removal_delta(route: Sequence, city, directed_costs: Mapping) -> float:
    """计算从当前代理路线删除一个客户的有向成本变化。"""
    position = route.index(city)
    previous = route[position - 1]
    following = route[position + 1]
    return float(
        directed_costs[previous][following]
        - directed_costs[previous][city]
        - directed_costs[city][following]
    )


def _best_insertion_delta(route: Sequence, city, directed_costs: Mapping) -> float:
    """计算把一个客户插入目标代理路线的最小有向成本增量。"""
    best = float('inf')
    for position in range(1, len(route)):
        previous = route[position - 1]
        following = route[position]
        delta = (
            directed_costs[previous][city]
            + directed_costs[city][following]
            - directed_costs[previous][following]
        )
        best = min(best, float(delta))
    return best


def _replacement_delta(route: Sequence, old_city, new_city, directed_costs: Mapping) -> float:
    """计算在原位置用另一个客户替换当前客户的代理成本变化。"""
    position = route.index(old_city)
    previous = route[position - 1]
    following = route[position + 1]
    return float(
        directed_costs[previous][new_city]
        + directed_costs[new_city][following]
        - directed_costs[previous][old_city]
        - directed_costs[old_city][following]
    )


def generate_partition_moves(
    groups: Mapping,
    directed_costs: Mapping,
    enable_relocate: bool = True,
    enable_swap: bool = True,
) -> list[PartitionMove]:
    """
    枚举全部单客户 relocate 和跨仓库 swap，并按代理增量排序。

    输入：当前分组、有向代价以及两个邻域开关。
    输出：代理增量从小到大排列的候选操作列表。
    """
    depots = list(groups)
    routes = {
        depot: cheapest_insertion_route(depot, groups[depot], directed_costs)
        for depot in depots
    }
    moves: list[PartitionMove] = []

    if enable_relocate:
        for source in depots:
            for city in groups[source]:
                removal = _removal_delta(routes[source], city, directed_costs)
                for target in depots:
                    if target == source:
                        continue
                    insertion = _best_insertion_delta(routes[target], city, directed_costs)
                    moves.append(
                        PartitionMove(
                            kind='relocate',
                            source=source,
                            target=target,
                            source_city=city,
                            target_city=None,
                            approximate_delta=removal + insertion,
                        )
                    )

    if enable_swap:
        for source_index, source in enumerate(depots):
            for target in depots[source_index + 1:]:
                for source_city in groups[source]:
                    for target_city in groups[target]:
                        delta = _replacement_delta(
                            routes[source], source_city, target_city, directed_costs
                        ) + _replacement_delta(
                            routes[target], target_city, source_city, directed_costs
                        )
                        moves.append(
                            PartitionMove(
                                kind='swap',
                                source=source,
                                target=target,
                                source_city=source_city,
                                target_city=target_city,
                                approximate_delta=delta,
                            )
                        )

    return sorted(
        moves,
        key=lambda move: (
            move.approximate_delta,
            move.kind,
            repr(move.source),
            repr(move.target),
            repr(move.source_city),
            repr(move.target_city),
        ),
    )


def apply_partition_move(groups: Mapping, move: PartitionMove) -> dict[Any, list[Any]]:
    """
    在分组副本上应用一次候选操作，不修改输入对象。

    输入：当前分组和 relocate/swap 候选。
    输出：应用候选后的全新分组字典。
    """
    updated = {depot: list(cities) for depot, cities in groups.items()}
    updated[move.source].remove(move.source_city)
    if move.kind == 'relocate':
        updated[move.target].append(move.source_city)
        return updated
    if move.kind == 'swap' and move.target_city is not None:
        updated[move.target].remove(move.target_city)
        updated[move.source].append(move.target_city)
        updated[move.target].append(move.source_city)
        return updated
    raise ValueError(f'不支持或不完整的客户分区操作：{move!r}。')


def improve_partition(
    initial_groups: Mapping,
    directed_costs: Mapping,
    evaluate_group: Callable[[Any, Sequence], float],
    config: PartitionSearchConfig,
) -> PartitionSearchResult:
    """
    使用代理筛选和下游组评价器执行解码器引导的分区局部搜索。

    输入：
    - initial_groups: MST/MSF 初始客户分组。
    - directed_costs: 有向集合转移代价。
    - evaluate_group: 返回单仓库组高保真成本的回调；调用方可在内部缓存。
    - config: 搜索预算与邻域配置。

    输出：
    - 最终分区、仓库组成本及搜索统计。

    每轮只精确评价代理排名前 `exact_candidate_count` 的候选，并且只重算受影响的两个仓库。
    """
    if config.exact_candidate_count <= 0:
        raise ValueError('exact_candidate_count 必须大于 0。')
    if config.max_iterations < 0:
        raise ValueError('max_iterations 不能为负数。')

    started = perf_counter()
    groups = {depot: list(cities) for depot, cities in initial_groups.items()}
    group_costs = {
        depot: float(evaluate_group(depot, cities))
        for depot, cities in groups.items()
    }
    # 初始分区下游成本用于量化 relocate/swap 带来的纯分区收益。
    initial_total_cost = float(sum(group_costs.values()))
    accepted_relocates = 0
    accepted_swaps = 0
    evaluated_candidates = 0
    completed_iterations = 0
    stop_reason = 'max_iterations'

    for iteration in range(config.max_iterations):
        if perf_counter() - started >= config.time_limit_seconds:
            stop_reason = 'time_limit'
            break

        moves = generate_partition_moves(
            groups,
            directed_costs,
            enable_relocate=config.enable_relocate,
            enable_swap=config.enable_swap,
        )
        if not moves:
            stop_reason = 'no_candidates'
            break

        best_move = None
        best_groups = None
        best_source_cost = None
        best_target_cost = None
        best_delta = float('inf')

        for move in moves[:config.exact_candidate_count]:
            if perf_counter() - started >= config.time_limit_seconds:
                stop_reason = 'time_limit'
                break
            proposal = apply_partition_move(groups, move)
            source_cost = float(evaluate_group(move.source, proposal[move.source]))
            target_cost = float(evaluate_group(move.target, proposal[move.target]))
            delta = (
                source_cost
                + target_cost
                - group_costs[move.source]
                - group_costs[move.target]
            )
            evaluated_candidates += 1
            if delta < best_delta:
                best_delta = delta
                best_move = move
                best_groups = proposal
                best_source_cost = source_cost
                best_target_cost = target_cost

        if best_move is None:
            break
        if best_delta >= -config.improvement_tolerance:
            stop_reason = 'local_optimum'
            break

        groups = best_groups
        group_costs[best_move.source] = best_source_cost
        group_costs[best_move.target] = best_target_cost
        if best_move.kind == 'relocate':
            accepted_relocates += 1
        else:
            accepted_swaps += 1
        completed_iterations = iteration + 1

    return PartitionSearchResult(
        groups=groups,
        group_costs=group_costs,
        initial_total_cost=initial_total_cost,
        final_total_cost=float(sum(group_costs.values())),
        iterations=completed_iterations,
        evaluated_candidates=evaluated_candidates,
        accepted_relocates=accepted_relocates,
        accepted_swaps=accepted_swaps,
        elapsed_seconds=perf_counter() - started,
        stop_reason=stop_reason,
    )
