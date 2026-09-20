"""生成兼顾负载、边界复杂度和有向道路代价的候选客户分区。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


Partition = dict[int, tuple[int, ...]]


@dataclass(frozen=True)
class PartitionCandidate:
    """保存一个候选完整分区及其生成算法、参数和动作轨迹。"""

    name: str
    kind: str
    strength: float
    partition: Partition
    moved_customers: int
    generator_family: str = ""
    generator_parameters: tuple[tuple[str, object], ...] = ()
    action_trace: tuple[str, ...] = ()
    generation_seconds: float = 0.0
    parent_name: str = "mst"


def canonical_partition(partition: dict[int, Iterable[int]], depots: Iterable[int]) -> Partition:
    """输入任意客户可迭代分区，输出仓库顺序稳定、组内排序稳定的不可变值结构。"""
    return {
        int(depot): tuple(sorted(map(int, partition.get(int(depot), ()))))
        for depot in map(int, depots)
    }


def partition_key(partition: Partition, depots: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """把分区转换成可哈希键，用于候选去重和缓存。"""
    return tuple(tuple(partition[int(depot)]) for depot in map(int, depots))


def moved_customer_count(baseline: Partition, candidate: Partition) -> int:
    """统计相对基线改变所属仓库的客户数量。"""
    before = {city: depot for depot, group in baseline.items() for city in group}
    after = {city: depot for depot, group in candidate.items() for city in group}
    return sum(before[city] != after[city] for city in before)


def _relocate(partition: Partition, source: int, target: int, city: int) -> Partition:
    """迁移一个客户并返回独立新分区，不修改输入对象。"""
    result = {depot: list(group) for depot, group in partition.items()}
    result[source].remove(city)
    result[target].append(city)
    return canonical_partition(result, result)


def _target_capacities(partition: Partition) -> dict[int, int]:
    """按当前组规模排名分配整数均衡容量，尽量减少达到均衡所需迁移数。"""
    depots = list(partition)
    customer_count = sum(map(len, partition.values()))
    quotient, remainder = divmod(customer_count, len(depots))
    ranked = sorted(depots, key=lambda depot: (-len(partition[depot]), depot))
    return {depot: quotient + int(depot in ranked[:remainder]) for depot in depots}


def _balance_path(
    baseline: Partition,
    affinity: Callable[[int, int], float],
) -> list[Partition]:
    """
    沿最小道路归属增量逐个迁移到整数均衡容量。

    输入为基线及 depot-city 有向往返亲和代价；输出每一步完整分区。
    每一步只在超载组与欠载组之间选择道路代价增加最小的客户迁移。
    """
    capacities = _target_capacities(baseline)
    current = baseline
    path: list[Partition] = []
    while any(len(current[depot]) > capacities[depot] for depot in current):
        sources = [depot for depot in current if len(current[depot]) > capacities[depot]]
        targets = [depot for depot in current if len(current[depot]) < capacities[depot]]
        _, source, target, city = min(
            (
                affinity(target, city) - affinity(source, city),
                source,
                target,
                city,
            )
            for source in sources
            for target in targets
            for city in current[source]
        )
        current = _relocate(current, source, target, city)
        path.append(current)
    return path


def _balanced_road_assignment(
    baseline: Partition,
    cities: tuple[int, ...],
    affinity: Callable[[int, int], float],
) -> Partition:
    """用带整数容量的线性指派求全局道路代价最小均衡分区。"""
    capacities = _target_capacities(baseline)
    slots = [depot for depot, capacity in capacities.items() for _ in range(capacity)]
    costs = np.asarray([[affinity(depot, city) for depot in slots] for city in cities])
    city_indices, slot_indices = linear_sum_assignment(costs)
    groups = {depot: [] for depot in baseline}
    for city_index, slot_index in zip(city_indices, slot_indices):
        groups[slots[int(slot_index)]].append(cities[int(city_index)])
    return canonical_partition(groups, baseline)


def _road_voronoi(
    baseline: Partition,
    cities: tuple[int, ...],
    affinity: Callable[[int, int], float],
) -> Partition:
    """把每位客户分给有向往返道路亲和代价最小的仓库。"""
    groups = {depot: [] for depot in baseline}
    for city in cities:
        depot = min(groups, key=lambda candidate: (affinity(candidate, city), candidate))
        groups[depot].append(city)
    return canonical_partition(groups, baseline)


def _group_complexity(group: Iterable[int], boundary_sizes: dict[int, int]) -> float:
    """按 Set-TSP 二元变量主导项估算一个仓库组的建模负担。"""
    customers = list(group)
    set_sizes = [1] + [boundary_sizes[city] for city in customers]
    set_count = len(set_sizes)
    return float(set_count * set_count + sum(size * size for size in set_sizes) + sum(set_sizes) ** 2)


def _burden_path(
    baseline: Partition,
    boundary_sizes: dict[int, int],
    affinity: Callable[[int, int], float],
) -> list[Partition]:
    """逐步选择“复杂度下降/道路代价增量”最优的迁移，返回最多四分之一客户的路径。"""
    current = baseline
    path: list[Partition] = []
    max_steps = max(1, math.ceil(sum(map(len, baseline.values())) * 0.25))
    for _ in range(max_steps):
        sources = sorted(current, key=lambda depot: (-_group_complexity(current[depot], boundary_sizes), depot))[:2]
        choices = []
        for source in sources:
            for target in current:
                if target == source:
                    continue
                for city in current[source]:
                    after = _relocate(current, source, target, city)
                    before_load = _group_complexity(current[source], boundary_sizes) + _group_complexity(
                        current[target], boundary_sizes
                    )
                    after_load = _group_complexity(after[source], boundary_sizes) + _group_complexity(
                        after[target], boundary_sizes
                    )
                    saving = (before_load - after_load) / max(before_load, 1.0)
                    road_penalty = max(0.0, affinity(target, city) - affinity(source, city))
                    score = saving - 0.05 * road_penalty
                    if saving > 0:
                        choices.append((score, -road_penalty, -city, after))
        if not choices:
            break
        current = max(choices, key=lambda item: item[:3])[-1]
        path.append(current)
    return path


def _cluster_candidate(
    baseline: Partition,
    cluster_size: int,
    boundary_sizes: dict[int, int],
    affinity: Callable[[int, int], float],
    symmetric_road: Callable[[int, int], float],
) -> Partition | None:
    """从高负担组迁移道路上相互接近的小簇，返回最佳完整分区。"""
    sources = sorted(baseline, key=lambda depot: (-_group_complexity(baseline[depot], boundary_sizes), depot))[:2]
    best: tuple[float, Partition] | None = None
    for source in sources:
        if len(baseline[source]) < cluster_size:
            continue
        for target in baseline:
            if target == source:
                continue
            seeds = sorted(
                baseline[source],
                key=lambda city: (affinity(target, city) - affinity(source, city), city),
            )[:4]
            for seed in seeds:
                cluster = sorted(
                    baseline[source],
                    key=lambda city: (symmetric_road(seed, city), city),
                )[:cluster_size]
                after = baseline
                for city in cluster:
                    after = _relocate(after, source, target, city)
                before_load = sum(_group_complexity(baseline[depot], boundary_sizes) for depot in baseline)
                after_load = sum(_group_complexity(after[depot], boundary_sizes) for depot in after)
                road_delta = sum(affinity(target, city) - affinity(source, city) for city in cluster)
                score = (before_load - after_load) / max(before_load, 1.0) - 0.03 * max(0.0, road_delta)
                if best is None or score > best[0]:
                    best = (score, after)
    return None if best is None else best[1]


def generate_candidates(
    baseline: Partition,
    cities: Iterable[int],
    boundary_sizes: dict[int, int],
    affinity: Callable[[int, int], float],
    symmetric_road: Callable[[int, int], float],
    *,
    max_candidates: int = 12,
    random_seed: int = 0,
) -> list[PartitionCandidate]:
    """
    生成约十二个由弱到强、显式使用道路结构的候选分区。

    输入为基线、客户、边界规模及道路查询函数；输出去重后的候选列表。
    候选覆盖渐进均衡、全局道路指派、复杂度修复、道路簇迁移和确定性探索动作。
    """
    baseline = canonical_partition(baseline, baseline)
    city_tuple = tuple(map(int, cities))
    candidates: list[PartitionCandidate] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()

    def add(name: str, kind: str, strength: float, partition: Partition | None) -> None:
        """向有序候选池加入一个非重复分区，并计算相对基线移动数。"""
        if partition is None or len(candidates) >= max_candidates:
            return
        canonical = canonical_partition(partition, baseline)
        key = partition_key(canonical, baseline)
        if key in seen:
            return
        seen.add(key)
        candidates.append(PartitionCandidate(
            name=name,
            kind=kind,
            strength=float(strength),
            partition=canonical,
            moved_customers=moved_customer_count(baseline, canonical),
        ))

    add("stay", "stay", 0.0, baseline)
    balance_path = _balance_path(baseline, affinity)
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        if balance_path:
            index = max(0, math.ceil(fraction * len(balance_path)) - 1)
            add(f"balance_{fraction:g}", "balance", fraction, balance_path[index])

    add("road_voronoi", "road", 1.0, _road_voronoi(baseline, city_tuple, affinity))
    add("balanced_road", "road", 1.0, _balanced_road_assignment(baseline, city_tuple, affinity))
    burden_path = _burden_path(baseline, boundary_sizes, affinity)
    if burden_path:
        add("burden_0.25", "burden", 0.25, burden_path[-1])
    for cluster_size in (2, 4, 8):
        add(
            f"road_cluster_{cluster_size}",
            "cluster",
            float(cluster_size),
            _cluster_candidate(baseline, cluster_size, boundary_sizes, affinity, symmetric_road),
        )

    # 当规则候选因分区本来接近平衡而重复时，用确定性随机局部动作补足样本数量。
    random_state = np.random.default_rng(random_seed)
    attempts = 0
    depots = list(baseline)
    while len(candidates) < max_candidates and attempts < max_candidates * 30:
        attempts += 1
        source_choices = [depot for depot in depots if baseline[depot]]
        source = int(random_state.choice(source_choices))
        target = int(random_state.choice([depot for depot in depots if depot != source]))
        city = int(random_state.choice(baseline[source]))
        add(f"explore_{attempts:02d}", "mutation", 1.0, _relocate(baseline, source, target, city))
    return candidates


def generate_active_pool(
    baseline: Partition,
    boundary_sizes: dict[int, int],
    affinity: Callable[[int, int], float],
    *,
    pool_size: int = 64,
    max_depth: int = 8,
    random_seed: int = 0,
) -> list[PartitionCandidate]:
    """
    为模型引导的主动扩充生成随机迁移轨迹池。

    输入为基线、边界规模和道路亲和代价；输出不同深度的未求解候选。
    源组按 Set-TSP 负担加权，目标组兼顾道路增量与负载，使轨迹既探索 OOD 又保留可行改进方向。
    """
    random_state = np.random.default_rng(random_seed)
    depots = list(baseline)
    seen = {partition_key(baseline, depots)}
    pool: list[PartitionCandidate] = []
    attempts = 0
    while len(pool) < pool_size and attempts < pool_size * 40:
        attempts += 1
        current = baseline
        depth = int(random_state.integers(1, max_depth + 1))
        for _ in range(depth):
            sources = [depot for depot in depots if current[depot]]
            source_weights = np.asarray(
                [_group_complexity(current[depot], boundary_sizes) for depot in sources],
                dtype=np.float64,
            )
            source_weights /= source_weights.sum()
            source = int(random_state.choice(sources, p=source_weights))
            # 一半选择高边界客户，一半均匀探索，防止主动池只包含同一种修复动作。
            if random_state.random() < 0.5:
                city = max(current[source], key=lambda item: (boundary_sizes[item], -item))
            else:
                city = int(random_state.choice(current[source]))
            targets = [depot for depot in depots if depot != source]
            target_scores = np.asarray([
                affinity(target, city) - affinity(source, city)
                + 0.05 * len(current[target])
                for target in targets
            ])
            order = np.argsort(target_scores)
            shortlist = [targets[int(index)] for index in order[: min(3, len(order))]]
            target = int(random_state.choice(shortlist))
            current = _relocate(current, source, target, city)
        key = partition_key(current, depots)
        if key in seen:
            continue
        seen.add(key)
        pool.append(PartitionCandidate(
            name=f"active_pool_{len(pool):03d}",
            kind="active",
            strength=float(depth),
            partition=current,
            moved_customers=moved_customer_count(baseline, current),
        ))
    return pool
