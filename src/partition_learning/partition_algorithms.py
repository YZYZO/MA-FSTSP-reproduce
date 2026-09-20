"""生成直接客户分区算法候选与可解释的局部划分动作。"""

from __future__ import annotations

import math
import time
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp

from .candidates import (
    Partition,
    PartitionCandidate,
    canonical_partition,
    moved_customer_count,
    partition_key,
)


def _customer_burden(city: int, boundary_sizes: dict[int, int]) -> float:
    """输入客户和边界规模，输出可加的 Set-TSP 复杂度负担代理。"""
    size = float(boundary_sizes[city])
    return 1.0 + size + size * size


def _group_complexity(group: Iterable[int], boundary_sizes: dict[int, int]) -> float:
    """输入客户组，输出与实际 Set-TSP 二进制变量数同形的规模代理。"""
    members = list(group)
    sizes = [1] + [boundary_sizes[city] for city in members]
    return float(len(sizes) ** 2 + sum(size * size for size in sizes) + sum(sizes) ** 2)


def _relocate(partition: Partition, source: int, target: int, city: int) -> Partition:
    """把一个客户迁移到目标仓库，返回不修改输入的新分区。"""
    groups = {depot: list(group) for depot, group in partition.items()}
    groups[source].remove(city)
    groups[target].append(city)
    return canonical_partition(groups, partition)


def _swap(partition: Partition, first_depot: int, second_depot: int, first: int, second: int) -> Partition:
    """交换两个仓库组中的客户，返回组规模不变的新分区。"""
    groups = {depot: list(group) for depot, group in partition.items()}
    groups[first_depot].remove(first)
    groups[second_depot].remove(second)
    groups[first_depot].append(second)
    groups[second_depot].append(first)
    return canonical_partition(groups, partition)


def _make_candidate(
    *,
    name: str,
    kind: str,
    strength: float,
    partition: Partition,
    baseline: Partition,
    family: str,
    parameters: dict[str, object],
    action_trace: Iterable[str],
    generation_seconds: float,
    parent_name: str = "mst",
) -> PartitionCandidate:
    """将分区结果与可复现的生成元数据组装成候选对象。"""
    canonical = canonical_partition(partition, baseline)
    return PartitionCandidate(
        name=name,
        kind=kind,
        strength=float(strength),
        partition=canonical,
        moved_customers=moved_customer_count(baseline, canonical),
        generator_family=family,
        generator_parameters=tuple(sorted(parameters.items())),
        action_trace=tuple(action_trace),
        generation_seconds=float(generation_seconds),
        parent_name=parent_name,
    )


def _constrained_multiroot_forest(
    baseline: Partition,
    cities: tuple[int, ...],
    boundary_sizes: dict[int, int],
    affinity: Callable[[int, int], float],
    symmetric_road: Callable[[int, int], float],
    *,
    imbalance_tolerance: float,
    complexity_weight: float,
) -> Partition:
    """
    用多仓库同时 Prim 生长构造带容量约束的生成森林分区。

    输入为基线分区、道路距离及平衡参数；输出每个仓库为一个根的完整分区。
    首先为每个仓库选不同种子客户，再按连通边代价和负担压力全局扩展。
    """
    depots = tuple(baseline)
    customer_count = len(cities)
    target_size = customer_count / len(depots)
    upper_size = max(1, math.ceil(target_size * (1.0 + imbalance_tolerance)))
    total_burden = sum(_customer_burden(city, boundary_sizes) for city in cities)
    target_burden = total_burden / len(depots)

    seed_costs = np.asarray([[affinity(depot, city) for city in cities] for depot in depots])
    depot_indices, city_indices = linear_sum_assignment(seed_costs)
    groups = {depot: [] for depot in depots}
    unassigned = set(cities)
    for depot_index, city_index in zip(depot_indices, city_indices):
        depot = depots[int(depot_index)]
        city = cities[int(city_index)]
        groups[depot].append(city)
        unassigned.remove(city)

    finite_affinity = seed_costs[np.isfinite(seed_costs)]
    distance_scale = float(np.median(finite_affinity)) if finite_affinity.size else 1.0
    distance_scale = max(distance_scale, 1e-9)
    burdens = {
        depot: sum(_customer_burden(city, boundary_sizes) for city in group)
        for depot, group in groups.items()
    }
    while unassigned:
        choices = []
        for depot in depots:
            if len(groups[depot]) >= upper_size:
                continue
            tree_nodes = [depot] + groups[depot]
            for city in unassigned:
                link_cost = min(symmetric_road(node, city) for node in tree_nodes)
                size_pressure = ((len(groups[depot]) + 1.0) / max(target_size, 1.0)) ** 2
                after_burden = burdens[depot] + _customer_burden(city, boundary_sizes)
                burden_pressure = (after_burden / max(target_burden, 1.0)) ** 2
                score = (
                    link_cost / distance_scale
                    + 0.15 * affinity(depot, city) / distance_scale
                    + 0.10 * size_pressure
                    + complexity_weight * burden_pressure
                )
                choices.append((score, depot, city))
        if not choices:
            # 整数上限若因极端参数不足，最后客户按当前最小组修复。
            city = min(unassigned)
            depot = min(depots, key=lambda item: (len(groups[item]), affinity(item, city), item))
        else:
            _, depot, city = min(choices)
        groups[depot].append(city)
        burdens[depot] += _customer_burden(city, boundary_sizes)
        unassigned.remove(city)
    return canonical_partition(groups, depots)


def _gap_partition(
    baseline: Partition,
    cities: tuple[int, ...],
    boundary_sizes: dict[int, int],
    affinity: Callable[[int, int], float],
    *,
    imbalance_tolerance: float,
    complexity_weight: float,
    time_limit: float | None = None,
) -> tuple[Partition | None, dict[str, object]]:
    """
    求解带组规模约束和最大负担惩罚的广义指派模型。

    输入为客户、仓库和有向道路亲和代价；输出完整分区及 MILP 状态。
    每个客户恰好分配一次，并以连续变量压低最大可加复杂度负担。
    """
    depots = tuple(baseline)
    city_count, depot_count = len(cities), len(depots)
    assignment_count = city_count * depot_count
    max_load_index = assignment_count
    affinity_matrix = np.asarray([[affinity(depot, city) for depot in depots] for city in cities])
    finite = affinity_matrix[np.isfinite(affinity_matrix)]
    scale = max(float(np.median(finite)) if finite.size else 1.0, 1e-9)
    burdens = np.asarray([_customer_burden(city, boundary_sizes) for city in cities])
    target_burden = max(float(np.sum(burdens) / depot_count), 1.0)

    objective = np.zeros(assignment_count + 1, dtype=np.float64)
    objective[:assignment_count] = (affinity_matrix / (scale * city_count)).reshape(-1)
    objective[max_load_index] = float(complexity_weight / target_burden)
    integrality = np.zeros(assignment_count + 1, dtype=np.int32)
    integrality[:assignment_count] = 1
    lower_bounds = np.zeros(assignment_count + 1, dtype=np.float64)
    upper_bounds = np.ones(assignment_count + 1, dtype=np.float64)
    upper_bounds[max_load_index] = float(np.sum(burdens))

    rows, lower, upper = [], [], []
    for city_index in range(city_count):
        row = np.zeros(assignment_count + 1)
        row[city_index * depot_count:(city_index + 1) * depot_count] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
    target_size = city_count / depot_count
    minimum_size = max(1, math.floor(target_size * (1.0 - imbalance_tolerance)))
    maximum_size = max(minimum_size, math.ceil(target_size * (1.0 + imbalance_tolerance)))
    for depot_index in range(depot_count):
        size_row = np.zeros(assignment_count + 1)
        size_row[depot_index:assignment_count:depot_count] = 1.0
        rows.append(size_row)
        lower.append(float(minimum_size))
        upper.append(float(maximum_size))

        burden_row = np.zeros(assignment_count + 1)
        burden_row[depot_index:assignment_count:depot_count] = burdens
        burden_row[max_load_index] = -1.0
        rows.append(burden_row)
        lower.append(-np.inf)
        upper.append(0.0)

    # GAP 也是候选划分生成器的一部分；默认不截断，避免 150 客户下生成半成品候选。
    options = {"mip_rel_gap": 1e-3}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.asarray(rows), np.asarray(lower), np.asarray(upper)),
        options=options,
    )
    metadata = {
        "milp_status": int(result.status),
        "milp_success": bool(result.success),
        "milp_message": str(result.message),
    }
    if result.x is None:
        return None, metadata
    assignments = result.x[:assignment_count].reshape(city_count, depot_count)
    groups = {depot: [] for depot in depots}
    for city_index, city in enumerate(cities):
        groups[depots[int(np.argmax(assignments[city_index]))]].append(city)
    return canonical_partition(groups, depots), metadata


def _road_distance_matrix(cities: tuple[int, ...], symmetric_road: Callable[[int, int], float]) -> np.ndarray:
    """输入客户序列和对称道路查询，输出有限、对称的客户距离矩阵。"""
    matrix = np.asarray([[symmetric_road(first, second) for second in cities] for first in cities])
    finite = matrix[np.isfinite(matrix)]
    replacement = 10.0 * float(np.max(finite)) if finite.size else 1e6
    matrix[~np.isfinite(matrix)] = replacement
    return 0.5 * (matrix + matrix.T)


def _anchored_graph_partition(
    baseline: Partition,
    cities: tuple[int, ...],
    boundary_sizes: dict[int, int],
    affinity: Callable[[int, int], float],
    symmetric_road: Callable[[int, int], float],
    *,
    neighbors: int,
    cut_weight: float,
    imbalance_tolerance: float = 0.25,
) -> Partition:
    """
    用仓库一元代价与客户 kNN 割边代价构造平衡多终端图分区。

    输入为道路距离、邻居数和割边权重；输出仓库标签下的完整客户分区。
    算法先按仓库代价初始化，再修复容量并以坐标下降减少跨组道路边。
    """
    depots = tuple(baseline)
    city_count, depot_count = len(cities), len(depots)
    distance_matrix = _road_distance_matrix(cities, symmetric_road)
    neighbor_count = min(max(1, int(neighbors)), max(1, city_count - 1))
    neighbor_indices = np.argsort(distance_matrix, axis=1)[:, 1:neighbor_count + 1]
    neighbor_distances = np.take_along_axis(distance_matrix, neighbor_indices, axis=1)
    positive = neighbor_distances[neighbor_distances > 0]
    scale = max(float(np.median(positive)) if positive.size else 1.0, 1e-9)
    adjacency: list[dict[int, float]] = [dict() for _ in cities]
    for city_index in range(city_count):
        for neighbor_index, distance in zip(neighbor_indices[city_index], neighbor_distances[city_index]):
            weight = float(math.exp(-float(distance) / scale))
            adjacency[city_index][int(neighbor_index)] = max(
                weight, adjacency[city_index].get(int(neighbor_index), 0.0)
            )
            adjacency[int(neighbor_index)][city_index] = max(
                weight, adjacency[int(neighbor_index)].get(city_index, 0.0)
            )

    unary = np.asarray([[affinity(depot, city) for depot in depots] for city in cities])
    unary_scale = max(float(np.median(unary[np.isfinite(unary)])), 1e-9)
    unary = unary / unary_scale
    labels = np.argmin(unary, axis=1).astype(np.int32)
    target_size = city_count / depot_count
    minimum_size = max(1, math.floor(target_size * (1.0 - imbalance_tolerance)))
    maximum_size = max(minimum_size, math.ceil(target_size * (1.0 + imbalance_tolerance)))
    burdens = np.asarray([_customer_burden(city, boundary_sizes) for city in cities])
    target_burden = max(float(np.sum(burdens) / depot_count), 1.0)

    def counts_and_loads() -> tuple[np.ndarray, np.ndarray]:
        """根据当前标签返回每个仓库的客户数和可加负担。"""
        counts = np.bincount(labels, minlength=depot_count).astype(np.int32)
        loads = np.asarray([np.sum(burdens[labels == index]) for index in range(depot_count)])
        return counts, loads

    def move_delta(city_index: int, target: int, counts: np.ndarray, loads: np.ndarray) -> float:
        """计算单客户换标签对一元、割边和平衡目标的增量。"""
        source = int(labels[city_index])
        if source == target:
            return 0.0
        delta = float(unary[city_index, target] - unary[city_index, source])
        for neighbor, weight in adjacency[city_index].items():
            before = int(labels[neighbor]) != source
            after = int(labels[neighbor]) != target
            delta += cut_weight * weight * (float(after) - float(before))
        before_balance = (
            ((counts[source] - target_size) / max(target_size, 1.0)) ** 2
            + ((counts[target] - target_size) / max(target_size, 1.0)) ** 2
            + ((loads[source] - target_burden) / target_burden) ** 2
            + ((loads[target] - target_burden) / target_burden) ** 2
        )
        after_balance = (
            ((counts[source] - 1 - target_size) / max(target_size, 1.0)) ** 2
            + ((counts[target] + 1 - target_size) / max(target_size, 1.0)) ** 2
            + ((loads[source] - burdens[city_index] - target_burden) / target_burden) ** 2
            + ((loads[target] + burdens[city_index] - target_burden) / target_burden) ** 2
        )
        return delta + 0.20 * (after_balance - before_balance)

    # 先修复超过上限的标签，再补足低于下限的标签。
    for _ in range(city_count * depot_count):
        counts, loads = counts_and_loads()
        overloaded = [index for index in range(depot_count) if counts[index] > maximum_size]
        if not overloaded:
            break
        choices = [
            (move_delta(city_index, target, counts, loads), city_index, target)
            for source in overloaded
            for city_index in np.flatnonzero(labels == source)
            for target in range(depot_count)
            if target != source and counts[target] < maximum_size
        ]
        _, city_index, target = min(choices)
        labels[city_index] = target
    for _ in range(city_count * depot_count):
        counts, loads = counts_and_loads()
        underloaded = [index for index in range(depot_count) if counts[index] < minimum_size]
        if not underloaded:
            break
        choices = [
            (move_delta(city_index, target, counts, loads), city_index, target)
            for target in underloaded
            for source in range(depot_count)
            if source != target and counts[source] > minimum_size
            for city_index in np.flatnonzero(labels == source)
        ]
        _, city_index, target = min(choices)
        labels[city_index] = target

    for _ in range(12):
        counts, loads = counts_and_loads()
        choices = []
        for city_index in range(city_count):
            source = int(labels[city_index])
            if counts[source] <= minimum_size:
                continue
            for target in range(depot_count):
                if target == source or counts[target] >= maximum_size:
                    continue
                delta = move_delta(city_index, target, counts, loads)
                if delta < -1e-10:
                    choices.append((delta, city_index, target))
        if not choices:
            break
        _, city_index, target = min(choices)
        labels[city_index] = target

    groups = {depot: [] for depot in depots}
    for city_index, city in enumerate(cities):
        groups[depots[int(labels[city_index])]].append(city)
    return canonical_partition(groups, depots)


def _capacitated_road_kmedoids(
    baseline: Partition,
    cities: tuple[int, ...],
    affinity: Callable[[int, int], float],
    symmetric_road: Callable[[int, int], float],
    *,
    initialization: str,
) -> Partition:
    """
    用道路距离的等容量 k-medoids 生成客户簇，再将簇与仓库最优匹配。

    输入为客户距离和初始化方式；输出每个仓库一组的完整分区。
    距离中心始终是真实客户节点，不使用欧氏均值。
    """
    depots = tuple(baseline)
    cluster_count = len(depots)
    distances = _road_distance_matrix(cities, symmetric_road)
    if initialization == "depot_seeded":
        costs = np.asarray([[affinity(depot, city) for city in cities] for depot in depots])
        _, medoid_indices = linear_sum_assignment(costs)
        medoids = np.asarray(medoid_indices, dtype=np.int32)
    else:
        average_depot_cost = np.asarray([
            np.mean([affinity(depot, city) for depot in depots]) for city in cities
        ])
        medoids = [int(np.argmin(average_depot_cost))]
        while len(medoids) < cluster_count:
            nearest = np.min(distances[:, medoids], axis=1)
            nearest[medoids] = -1.0
            medoids.append(int(np.argmax(nearest)))
        medoids = np.asarray(medoids, dtype=np.int32)

    quotient, remainder = divmod(len(cities), cluster_count)
    capacities = [quotient + int(index < remainder) for index in range(cluster_count)]
    slots = [index for index, capacity in enumerate(capacities) for _ in range(capacity)]
    assignments = np.zeros(len(cities), dtype=np.int32)
    for _ in range(12):
        assignment_cost = np.asarray([[distances[city_index, medoids[slot]] for slot in slots]
                                      for city_index in range(len(cities))])
        city_indices, slot_indices = linear_sum_assignment(assignment_cost)
        for city_index, slot_index in zip(city_indices, slot_indices):
            assignments[int(city_index)] = int(slots[int(slot_index)])
        updated = medoids.copy()
        for cluster_index in range(cluster_count):
            members = np.flatnonzero(assignments == cluster_index)
            within = distances[np.ix_(members, members)]
            updated[cluster_index] = int(members[int(np.argmin(np.sum(within, axis=1)))])
        if np.array_equal(updated, medoids):
            break
        medoids = updated

    matching_cost = np.asarray([
        [affinity(depot, cities[int(medoid)]) for medoid in medoids]
        for depot in depots
    ])
    depot_indices, cluster_indices = linear_sum_assignment(matching_cost)
    cluster_to_depot = {
        int(cluster_index): depots[int(depot_index)]
        for depot_index, cluster_index in zip(depot_indices, cluster_indices)
    }
    groups = {depot: [] for depot in depots}
    for city_index, city in enumerate(cities):
        groups[cluster_to_depot[int(assignments[city_index])]].append(city)
    return canonical_partition(groups, depots)


def generate_direct_partition_candidates(
    baseline: Partition,
    cities: Iterable[int],
    boundary_sizes: dict[int, int],
    affinity: Callable[[int, int], float],
    symmetric_road: Callable[[int, int], float],
    *,
    max_candidates: int = 16,
) -> list[PartitionCandidate]:
    """
    生成多根森林、GAP、锚定图分区和道路 k-medoids 四类直接分区候选。

    输入为 MST 基线、道路结构和边界负担；输出去重后、带生成耗时与参数的候选。
    """
    baseline = canonical_partition(baseline, baseline)
    city_tuple = tuple(map(int, cities))
    candidates: list[PartitionCandidate] = []
    # 同一算法族内去重；不同算法得到同一分区时仍保留来源，供算法对比。
    seen: set[tuple[str, tuple[tuple[int, ...], ...]]] = set()

    def add(
        name: str,
        kind: str,
        family: str,
        parameters: dict[str, object],
        builder: Callable[[], Partition | tuple[Partition | None, dict[str, object]]],
    ) -> None:
        """计时执行一个直接分区器，并将非重复结果加入候选池。"""
        if len(candidates) >= max_candidates:
            return
        started_at = time.perf_counter()
        built = builder()
        elapsed = time.perf_counter() - started_at
        extra: dict[str, object] = {}
        if isinstance(built, tuple):
            partition, extra = built
        else:
            partition = built
        if partition is None:
            return
        canonical = canonical_partition(partition, baseline)
        key = (family, partition_key(canonical, baseline))
        if key in seen:
            return
        seen.add(key)
        combined_parameters = {**parameters, **extra}
        candidates.append(_make_candidate(
            name=name,
            kind=kind,
            strength=float(parameters.get("complexity_weight", parameters.get("cut_weight", 1.0))),
            partition=canonical,
            baseline=baseline,
            family=family,
            parameters=combined_parameters,
            action_trace=(f"{family}:{parameters}",),
            generation_seconds=elapsed,
        ))

    for tolerance, complexity_weight in ((0.10, 0.0), (0.20, 0.20), (0.30, 0.50), (0.40, 1.0)):
        parameters = {"imbalance_tolerance": tolerance, "complexity_weight": complexity_weight}
        add(
            f"forest_t{tolerance:g}_c{complexity_weight:g}",
            "forest",
            "constrained_multiroot_forest",
            parameters,
            lambda t=tolerance, c=complexity_weight: _constrained_multiroot_forest(
                baseline, city_tuple, boundary_sizes, affinity, symmetric_road,
                imbalance_tolerance=t, complexity_weight=c,
            ),
        )
    for tolerance, complexity_weight in ((0.10, 0.05), (0.20, 0.20), (0.30, 0.50), (0.40, 1.0)):
        parameters = {"imbalance_tolerance": tolerance, "complexity_weight": complexity_weight}
        add(
            f"gap_t{tolerance:g}_c{complexity_weight:g}",
            "gap",
            "generalized_assignment",
            parameters,
            lambda t=tolerance, c=complexity_weight: _gap_partition(
                baseline, city_tuple, boundary_sizes, affinity,
                imbalance_tolerance=t, complexity_weight=c,
            ),
        )
    for neighbors, cut_weight in ((5, 0.25), (5, 0.75), (10, 0.50), (10, 1.00)):
        parameters = {"neighbors": neighbors, "cut_weight": cut_weight}
        add(
            f"graph_k{neighbors}_w{cut_weight:g}",
            "graph",
            "anchored_balanced_graph",
            parameters,
            lambda k=neighbors, w=cut_weight: _anchored_graph_partition(
                baseline, city_tuple, boundary_sizes, affinity, symmetric_road,
                neighbors=k, cut_weight=w,
            ),
        )
    for initialization in ("depot_seeded", "farthest"):
        parameters = {"initialization": initialization}
        add(
            f"road_kmedoids_{initialization}",
            "kmedoids",
            "capacitated_road_kmedoids",
            parameters,
            lambda mode=initialization: _capacitated_road_kmedoids(
                baseline, city_tuple, affinity, symmetric_road, initialization=mode,
            ),
        )
    return candidates


def generate_partition_action_candidates(
    baseline: Partition,
    boundary_sizes: dict[int, int],
    affinity: Callable[[int, int], float],
    *,
    max_per_action: int = 4,
) -> list[PartitionCandidate]:
    """
    生成单客户迁移、跨组交换、瓶颈削减和双组重分配候选。

    输入为 MST 基线、边界负担和道路亲和代价；输出每条都保留父分区和动作轨迹的分区。
    """
    baseline = canonical_partition(baseline, baseline)
    depots = tuple(baseline)
    candidates: list[PartitionCandidate] = []
    seen = {partition_key(baseline, depots)}
    affinity_values = [affinity(depot, city) for depot, group in baseline.items() for city in group]
    affinity_scale = max(float(np.median(affinity_values)), 1e-9)

    def add(name: str, kind: str, partition: Partition, trace: Iterable[str], started_at: float, strength: float) -> None:
        """把一个局部动作结果去重后加入输出列表。"""
        canonical = canonical_partition(partition, depots)
        key = partition_key(canonical, depots)
        if key in seen:
            return
        seen.add(key)
        candidates.append(_make_candidate(
            name=name,
            kind=kind,
            strength=strength,
            partition=canonical,
            baseline=baseline,
            family=f"action_{kind}",
            parameters={"max_per_action": max_per_action},
            action_trace=trace,
            generation_seconds=time.perf_counter() - started_at,
        ))

    relocation_started = time.perf_counter()
    relocations = []
    for source, group in baseline.items():
        if len(group) <= 1:
            continue
        for city in group:
            for target in depots:
                if target == source:
                    continue
                after = _relocate(baseline, source, target, city)
                before_complexity = _group_complexity(baseline[source], boundary_sizes) + _group_complexity(
                    baseline[target], boundary_sizes
                )
                after_complexity = _group_complexity(after[source], boundary_sizes) + _group_complexity(
                    after[target], boundary_sizes
                )
                complexity_gain = (before_complexity - after_complexity) / max(before_complexity, 1.0)
                road_delta = (affinity(target, city) - affinity(source, city)) / affinity_scale
                relocations.append((-(complexity_gain - 0.10 * road_delta), source, target, city, after))
    for rank, (_, source, target, city, after) in enumerate(sorted(relocations)[:max_per_action], start=1):
        add(
            f"relocate_{rank:02d}", "relocate", after,
            (f"move city={city} source={source} target={target}",),
            relocation_started, 1.0,
        )

    swap_started = time.perf_counter()
    swaps = []
    for first_index, first_depot in enumerate(depots):
        for second_depot in depots[first_index + 1:]:
            for first in baseline[first_depot]:
                for second in baseline[second_depot]:
                    road_delta = (
                        affinity(first_depot, second) + affinity(second_depot, first)
                        - affinity(first_depot, first) - affinity(second_depot, second)
                    ) / affinity_scale
                    boundary_match = abs(boundary_sizes[first] - boundary_sizes[second])
                    swaps.append((road_delta + 0.001 * boundary_match, first_depot, second_depot, first, second))
    for rank, (_, first_depot, second_depot, first, second) in enumerate(sorted(swaps)[:max_per_action], start=1):
        after = _swap(baseline, first_depot, second_depot, first, second)
        add(
            f"swap_{rank:02d}", "swap", after,
            (f"swap city={first}@{first_depot} with city={second}@{second_depot}",),
            swap_started, 2.0,
        )

    bottleneck_started = time.perf_counter()
    current = baseline
    trace: list[str] = []
    snapshots = {1, 2, 4, 8}
    for step in range(1, max(snapshots) + 1):
        source = max(current, key=lambda depot: (_group_complexity(current[depot], boundary_sizes), depot))
        if len(current[source]) <= 1:
            break
        choices = []
        for city in current[source]:
            for target in depots:
                if target == source:
                    continue
                after = _relocate(current, source, target, city)
                max_after = max(_group_complexity(group, boundary_sizes) for group in after.values())
                road_delta = max(0.0, affinity(target, city) - affinity(source, city)) / affinity_scale
                choices.append((max_after + 0.05 * road_delta, city, target, after))
        _, city, target, current = min(choices)
        trace.append(f"shed city={city} source={source} target={target}")
        if step in snapshots:
            add(
                f"bottleneck_depth_{step}", "bottleneck", current, tuple(trace),
                bottleneck_started, float(step),
            )

    pair_started = time.perf_counter()
    pair_scores = sorted(
        (
            -abs(_group_complexity(baseline[first], boundary_sizes) - _group_complexity(baseline[second], boundary_sizes)),
            first,
            second,
        )
        for index, first in enumerate(depots)
        for second in depots[index + 1:]
    )
    for rank, (_, first, second) in enumerate(pair_scores[:max_per_action], start=1):
        joint = tuple(baseline[first] + baseline[second])
        first_capacity = len(joint) // 2 + len(joint) % 2
        slots = [first] * first_capacity + [second] * (len(joint) - first_capacity)
        costs = np.asarray([[affinity(depot, city) for depot in slots] for city in joint])
        city_indices, slot_indices = linear_sum_assignment(costs)
        groups = {depot: list(group) for depot, group in baseline.items()}
        groups[first], groups[second] = [], []
        for city_index, slot_index in zip(city_indices, slot_indices):
            groups[slots[int(slot_index)]].append(joint[int(city_index)])
        after = canonical_partition(groups, depots)
        add(
            f"pair_reassign_{rank:02d}", "pair_reassign", after,
            (f"jointly reassign depots={first},{second} customers={len(joint)}",),
            pair_started, float(len(joint)),
        )
    return candidates
