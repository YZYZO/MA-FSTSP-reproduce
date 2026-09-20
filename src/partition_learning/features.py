"""把完整客户分区编码成适合表格回归器的道路感知动作差值特征。"""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Iterable

import numpy as np

from .candidates import Partition, PartitionCandidate


CANDIDATE_KINDS = (
    "stay", "balance", "road", "burden", "cluster", "mutation", "active",
    "forest", "gap", "graph", "kmedoids", "relocate", "swap", "bottleneck", "pair_reassign",
)


def _gini(values: Iterable[float]) -> float:
    """输入非负负载序列，输出 0 到 1 附近的 Gini 不均衡度。"""
    array = np.sort(np.asarray(list(values), dtype=np.float64))
    if len(array) == 0 or float(array.sum()) == 0.0:
        return 0.0
    indices = np.arange(1, len(array) + 1, dtype=np.float64)
    return float((2.0 * np.sum(indices * array) / np.sum(array) - len(array) - 1.0) / len(array))


def group_variable_proxy(group: Iterable[int], boundary_sizes: dict[int, int]) -> float:
    """输入一个客户组，输出与 Set-TSP 二元变量规模同阶的复杂度代理。"""
    members = list(group)
    set_sizes = np.asarray([1] + [boundary_sizes[city] for city in members], dtype=np.float64)
    set_count = len(set_sizes)
    return float(set_count * set_count + np.sum(set_sizes * set_sizes) + np.sum(set_sizes) ** 2)


def _partition_statistics(
    partition: Partition,
    boundary_sizes: dict[int, int],
    road_distance: Callable[[int, int], float],
    air_distance: Callable[[int, int], float],
) -> dict[str, float]:
    """
    计算一个分区的负载、集合模型规模和有向道路结构摘要。

    输入为分区、边界规模与道路/空中距离；输出固定键的纯数值字典。
    道路特征同时保留去程、回程、非对称性和道路绕行比，使模型不把路网当欧氏平面。
    """
    sizes = np.asarray([len(group) for group in partition.values()], dtype=np.float64)
    boundary_loads = np.asarray(
        [sum(boundary_sizes[city] for city in group) for group in partition.values()],
        dtype=np.float64,
    )
    variable_loads = np.asarray(
        [group_variable_proxy(group, boundary_sizes) for group in partition.values()],
        dtype=np.float64,
    )
    outbound, inbound, asymmetry, stretch, intra_group = [], [], [], [], []
    for depot, group in partition.items():
        for city in group:
            forward = road_distance(depot, city)
            backward = road_distance(city, depot)
            air = max(air_distance(depot, city), 1e-9)
            outbound.append(forward)
            inbound.append(backward)
            asymmetry.append(abs(forward - backward) / max(0.5 * (forward + backward), 1e-9))
            stretch.append(0.5 * (forward + backward) / air)
        # 组内成对道路距离近似描述后续 TSP 的局部紧凑度。
        for first_index, first in enumerate(group):
            for second in group[first_index + 1:]:
                intra_group.append(0.5 * (road_distance(first, second) + road_distance(second, first)))
    outbound_array = np.asarray(outbound, dtype=np.float64)
    inbound_array = np.asarray(inbound, dtype=np.float64)
    asymmetry_array = np.asarray(asymmetry, dtype=np.float64)
    stretch_array = np.asarray(stretch, dtype=np.float64)
    intra_array = np.asarray(intra_group or [0.0], dtype=np.float64)
    return {
        "size_mean": float(np.mean(sizes)),
        "size_std": float(np.std(sizes)),
        "size_max": float(np.max(sizes)),
        "size_min": float(np.min(sizes)),
        "size_gini": _gini(sizes),
        "empty_groups": float(np.sum(sizes == 0)),
        "max_group_share": float(np.max(sizes) / max(np.sum(sizes), 1.0)),
        "boundary_sum": float(np.sum(boundary_loads)),
        "boundary_max": float(np.max(boundary_loads)),
        "boundary_std": float(np.std(boundary_loads)),
        "boundary_gini": _gini(boundary_loads),
        "variable_sum": float(np.sum(variable_loads)),
        "variable_max": float(np.max(variable_loads)),
        "variable_std": float(np.std(variable_loads)),
        "variable_gini": _gini(variable_loads),
        "road_out_mean": float(np.mean(outbound_array)),
        "road_out_p90": float(np.quantile(outbound_array, 0.9)),
        "road_in_mean": float(np.mean(inbound_array)),
        "road_in_p90": float(np.quantile(inbound_array, 0.9)),
        "road_asymmetry_mean": float(np.mean(asymmetry_array)),
        "road_asymmetry_p90": float(np.quantile(asymmetry_array, 0.9)),
        "road_stretch_mean": float(np.mean(stretch_array)),
        "road_stretch_p90": float(np.quantile(stretch_array, 0.9)),
        "intra_group_road_mean": float(np.mean(intra_array)),
        "intra_group_road_p90": float(np.quantile(intra_array, 0.9)),
    }


def _movement_statistics(
    baseline: Partition,
    candidate: Partition,
    boundary_sizes: dict[int, int],
    road_distance: Callable[[int, int], float],
) -> dict[str, float]:
    """
    统计候选相对 MST 移动的客户及其道路归属增量。

    输入为基线、候选、边界规模与有向道路距离；输出只依赖求解前信息的动作摘要。
    """
    before = {city: depot for depot, group in baseline.items() for city in group}
    after = {city: depot for depot, group in candidate.items() for city in group}
    moved = [city for city in before if before[city] != after[city]]
    if not moved:
        return {
            "moved_boundary_sum": 0.0,
            "moved_boundary_mean": 0.0,
            "moved_road_delta_sum": 0.0,
            "moved_road_delta_mean": 0.0,
            "moved_road_delta_max": 0.0,
            "moved_source_size_mean": 0.0,
            "moved_target_size_mean": 0.0,
        }
    road_deltas = np.asarray([
        0.5 * (
            road_distance(after[city], city) + road_distance(city, after[city])
            - road_distance(before[city], city) - road_distance(city, before[city])
        )
        for city in moved
    ])
    moved_boundaries = np.asarray([boundary_sizes[city] for city in moved], dtype=np.float64)
    source_sizes = np.asarray([len(baseline[before[city]]) for city in moved], dtype=np.float64)
    target_sizes = np.asarray([len(baseline[after[city]]) for city in moved], dtype=np.float64)
    return {
        "moved_boundary_sum": float(np.sum(moved_boundaries)),
        "moved_boundary_mean": float(np.mean(moved_boundaries)),
        "moved_road_delta_sum": float(np.sum(road_deltas)),
        "moved_road_delta_mean": float(np.mean(road_deltas)),
        "moved_road_delta_max": float(np.max(road_deltas)),
        "moved_source_size_mean": float(np.mean(source_sizes)),
        "moved_target_size_mean": float(np.mean(target_sizes)),
    }


def action_delta_features(
    baseline: Partition,
    candidate: PartitionCandidate,
    boundary_sizes: dict[int, int],
    road_distance: Callable[[int, int], float],
    air_distance: Callable[[int, int], float],
    *,
    graph_nodes: int,
    graph_edges: int,
) -> dict[str, float]:
    """
    构造监督学习的一条动作差值特征。

    输入为基线、候选和路网查询；输出只依赖求解前信息的特征字典。
    同时写入基线值、候选值和候选减基线的差值，禁止使用真实时间或成本标签。
    """
    baseline_stats = _partition_statistics(baseline, boundary_sizes, road_distance, air_distance)
    candidate_stats = _partition_statistics(candidate.partition, boundary_sizes, road_distance, air_distance)
    customer_count = sum(map(len, baseline.values()))
    features: dict[str, float] = {
        "log_graph_nodes": math.log1p(graph_nodes),
        "log_graph_edges": math.log1p(graph_edges),
        "customer_count": float(customer_count),
        "depot_count": float(len(baseline)),
        "moved_customers": float(candidate.moved_customers),
        "moved_fraction": float(candidate.moved_customers / max(customer_count, 1)),
        "action_strength": float(candidate.strength),
    }
    for kind in CANDIDATE_KINDS:
        features[f"kind_{kind}"] = float(candidate.kind == kind)
    features.update(_movement_statistics(
        baseline,
        candidate.partition,
        boundary_sizes,
        road_distance,
    ))
    for name, value in baseline_stats.items():
        features[f"baseline_{name}"] = value
        features[f"candidate_{name}"] = candidate_stats[name]
        features[f"delta_{name}"] = candidate_stats[name] - value
    return features
