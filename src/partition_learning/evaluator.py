"""在固定客户分区上执行 Set-TSP 与原第三阶段动态规划。"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np

from src.baseline import Baseline
from src.fstsp import MultiAgentFlyingSidekickTSP

from .cache import GroupEvaluationCache
from .candidates import Partition, canonical_partition
from .road import DroneDistanceMatrix, TerminalRoadMatrix


EVALUATOR_VERSION = "set-tsp-v5-capped-censoring-downstream-labels"
UNLIMITED_TIME_SENTINEL = 9999.0


def is_unlimited_solver_time(time_limit: float) -> bool:
    """输入命令行时间参数；当其达到约定值 9999 时返回“不设置求解器时限”。"""
    return float(time_limit) >= UNLIMITED_TIME_SENTINEL


def estimate_set_tsp_binary_variables(convex_sets: list[list[int]]) -> int:
    """输入一个仓库组的集合列表，输出与实际模型一致的二元变量数量估计。"""
    set_count = len(convex_sets)
    sizes = [len(nodes) for nodes in convex_sets]
    return int(set_count * set_count + sum(size * size for size in sizes) + sum(sizes) ** 2)


def _nearest_neighbor_sequence(depot: int, cities: list[int], truck) -> list[int]:
    """输入仓库和客户，按有向道路最短路产生确定性的闭环索引序列。"""
    nodes = [depot] + cities
    remaining = set(range(1, len(nodes)))
    sequence = [0]
    while remaining:
        selected = min(remaining, key=lambda index: (truck[nodes[sequence[-1]]][nodes[index]], index))
        remaining.remove(selected)
        sequence.append(selected)
    return sequence + [0]


def _solve_set_tsp(
    convex_sets: list[list[int]],
    set_distance,
    external_distance,
    *,
    time_limit: float,
    threads: int,
    seed: int,
    mip_gap: float,
) -> tuple[list[int] | None, dict[str, object]]:
    """
    构建并求解一个可选时间预算的 Set-TSP。

    输入为固定集合及内部/外部代价；输出可行环索引或 None，以及建模、优化和状态记录。
    模型约束与原论文仓库中的实现保持一致，只增加资源预算和无解时的安全返回。
    """
    build_started_at = time.perf_counter()
    set_count = len(convex_sets)
    with gp.Model("Partition-Learning-Set-TSP") as model:
        model.Params.OutputFlag = 0
        model.Params.Threads = int(threads)
        model.Params.Seed = int(seed)
        model.Params.MIPGap = float(mip_gap)
        # 9999 是本项目约定的“无限时”标记；此时不向 Gurobi 写入 TimeLimit。
        if not is_unlimited_solver_time(time_limit):
            model.Params.TimeLimit = float(time_limit)

        # select/flow 决定集合访问环，并用单商品流消除不含仓库的子环。
        select = model.addMVar((set_count, set_count), vtype=GRB.BINARY)
        model.addConstrs(select[u, u] == 0 for u in range(set_count))
        model.addConstrs(np.ones(set_count) @ select[:, v] == 1 for v in range(set_count))
        model.addConstrs(np.ones(set_count) @ select[u, :] == 1 for u in range(set_count))
        flow = model.addMVar((set_count, set_count), vtype=GRB.CONTINUOUS)
        model.addConstrs(
            flow[u, v] <= set_count * select[u, v]
            for u in range(set_count)
            for v in range(set_count)
        )
        model.addConstr(np.ones(set_count) @ flow[0, :] == set_count - 1)
        model.addConstr(np.ones(set_count) @ flow[:, 0] == 0)
        model.addConstrs(flow[u, u] == 0 for u in range(set_count))
        model.addConstrs(
            np.ones(set_count) @ flow[:, v] - np.ones(set_count) @ flow[v, :] == 1
            for v in range(1, set_count)
        )

        # internal 选择每个集合的进入/离开节点，external 连接相邻集合的节点。
        internal = [
            [[model.addVar(vtype=GRB.BINARY) for _ in nodes] for _ in nodes]
            for nodes in convex_sets
        ]
        external = [
            [
                [[model.addVar(vtype=GRB.BINARY) for _ in target] for _ in source]
                for target in convex_sets
            ]
            for source in convex_sets
        ]
        model.addConstrs(
            gp.quicksum(internal[index][left][right]
                        for left in range(len(convex_sets[index]))
                        for right in range(len(convex_sets[index]))) == 1
            for index in range(set_count)
        )
        model.addConstrs(
            gp.quicksum(external[u][v][left][right]
                        for left in range(len(convex_sets[u]))
                        for right in range(len(convex_sets[v]))) == select[u, v]
            for u in range(set_count)
            for v in range(set_count)
        )
        model.addConstrs(
            gp.quicksum(external[u][v][left][right]
                        for u in range(set_count)
                        for left in range(len(convex_sets[u])))
            == gp.quicksum(internal[v][right][exit_node]
                           for exit_node in range(len(convex_sets[v])))
            for v in range(set_count)
            for right in range(len(convex_sets[v]))
        )
        model.addConstrs(
            gp.quicksum(external[u][v][left][right]
                        for v in range(set_count)
                        for right in range(len(convex_sets[v])))
            == gp.quicksum(internal[u][entry_node][left]
                           for entry_node in range(len(convex_sets[u])))
            for u in range(set_count)
            for left in range(len(convex_sets[u]))
        )
        model.setObjective(
            gp.quicksum(
                set_distance[index][left][right] * internal[index][left][right]
                for index in range(set_count)
                for left in range(len(convex_sets[index]))
                for right in range(len(convex_sets[index]))
            )
            + gp.quicksum(
                external_distance[u][v][left][right] * external[u][v][left][right]
                for u in range(set_count)
                for v in range(set_count)
                for left in range(len(convex_sets[u]))
                for right in range(len(convex_sets[v]))
            ),
            GRB.MINIMIZE,
        )
        model.update()
        build_seconds = time.perf_counter() - build_started_at
        optimize_started_at = time.perf_counter()
        model.optimize()
        optimize_seconds = time.perf_counter() - optimize_started_at
        has_incumbent = model.SolCount > 0
        sequence = None
        if has_incumbent:
            successors = np.argmax(select.X, axis=1)
            sequence = [0]
            for _ in range(set_count):
                sequence.append(int(successors[sequence[-1]]))
            if sequence[-1] != 0 or sorted(sequence[:-1]) != list(range(set_count)):
                raise RuntimeError("Set-TSP 可行解没有恢复成完整集合环。")
        gap = float(model.MIPGap) if has_incumbent else None
        return sequence, {
            "status": int(model.Status),
            "has_incumbent": bool(has_incumbent),
            "timeout": bool(model.Status == GRB.TIME_LIMIT),
            "mip_gap": gap if gap is not None and math.isfinite(gap) else None,
            "objective": float(model.ObjVal) if has_incumbent else None,
            "num_variables": int(model.NumVars),
            "num_binary_variables": int(model.NumBinVars),
            "num_constraints": int(model.NumConstrs),
            "build_seconds": build_seconds,
            "optimize_seconds": optimize_seconds,
            "solver_work": float(model.Work),
            "requested_time_limit": float(time_limit),
            "applied_time_limit": None if is_unlimited_solver_time(time_limit) else float(time_limit),
            "unlimited_time": is_unlimited_solver_time(time_limit),
        }


def _lean_model(
    depot: int,
    cities: list[int],
    regions: dict[int, list[int]],
    distance: dict,
    drones_per_truck: int,
    drone_limit: float,
    drone_speed: float,
) -> MultiAgentFlyingSidekickTSP:
    """
    创建跳过全图区域扫描的原算法对象。

    输入为单仓库组、预重建区域和距离；输出只供第三阶段调用的轻量模型。
    图节点仅保留本组动态规划会访问的区域节点，避免为每层创建 55K 大字典。
    """
    relevant_nodes = {depot}
    for city in cities:
        relevant_nodes.add(city)
        relevant_nodes.update(regions[city])
    local_graph = nx.DiGraph()
    local_graph.add_nodes_from(sorted(relevant_nodes))
    model = object.__new__(MultiAgentFlyingSidekickTSP)
    Baseline.__init__(
        model,
        local_graph,
        [depot],
        cities,
        distance,
        drones_per_truck,
        drone_limit,
        drone_speed,
    )
    model.groups = {depot: cities}
    model.solution = []
    model.cost = 0.0
    model.theta = (0.5, 0.5)
    model.const = math.sqrt(2.0)
    model.regions = {**{city: regions[city] for city in cities}, depot: [depot]}
    return model


def phase3_cost_vectorized(
    visit_route: list[int],
    depot: int,
    regions: dict[int, list[int]],
    truck: TerminalRoadMatrix,
    drone: DroneDistanceMatrix,
    drones_per_truck: int,
    drone_limit: float,
    drone_speed: float,
) -> float:
    """
    用向量化最小加递推计算原第三阶段的最终成本。

    输入为阶段 2 顺序、固定服务区域和有向道路/空中距离；输出与原动态规划相同的成本。
    逻辑逐客户保存有限个 `appr` 矩阵和奇偶层值，不构造路线回溯字典，也不为无关图节点分配状态。
    """
    customers = list(map(int, visit_route[1:-1]))
    if not customers:
        return 0.0
    region_nodes = [np.asarray([depot], dtype=np.int32)] + [
        np.asarray(regions[city], dtype=np.int32) for city in customers
    ]
    group_counts = [1 for _ in customers]
    # appr[(i, j)] 的行是当前客户区域终点，列是一批无人机共同出发的区域节点。
    appr: dict[tuple[int, int], np.ndarray] = {}
    odd_values: dict[int, np.ndarray] = {}
    previous_even = np.asarray([0.0], dtype=np.float64)

    for customer_index in range(1, len(customers) + 1):
        city = customers[customer_index - 1]
        current_region = region_nodes[customer_index]
        previous_region = region_nodes[customer_index - 1]
        drone_to_region = drone.pairwise([city], current_region)[0]
        internal_truck = truck.pairwise(current_region, current_region)
        drone_time = drone_to_region[:, None] + drone_to_region[None, :]
        direct_drone_or_truck = np.maximum(drone_time / drone_speed, internal_truck.T)
        truck_via_customer = (
            truck.pairwise([city], current_region)[0][:, None]
            + truck.pairwise(current_region, [city])[:, 0][None, :]
        )
        appr[(customer_index, 0)] = np.where(
            drone_time <= drone_limit,
            direct_drone_or_truck,
            truck_via_customer,
        )

        while (
            customer_index - 1 - group_counts[customer_index - 1] >= 0
            and group_counts[customer_index - 1] <= group_counts[customer_index - 2]
            and group_counts[customer_index - 1] < drones_per_truck
        ):
            earlier_city = customers[customer_index - group_counts[customer_index - 1] - 1]
            if drone.query(earlier_city, city) < 2.0 * drone_limit:
                group_counts[customer_index - 1] += 1
            else:
                break

        for joint_index in range(1, group_counts[customer_index - 1]):
            launch_region = region_nodes[customer_index - joint_index]
            previous_appr = appr[(customer_index - 1, joint_index - 1)]
            transition = truck.pairwise(previous_region, current_region)
            # previous_appr.T: launch × middle；transition: middle × current。
            min_plus = np.min(
                previous_appr.T[:, :, None] + transition[None, :, :],
                axis=1,
            ).T
            consumption = (
                drone.pairwise([city], current_region)[0][:, None]
                + drone.pairwise([city], launch_region)[0][None, :]
            )
            joint_cost = np.maximum(min_plus, consumption / drone_speed)
            joint_cost[consumption > drone_limit] = np.inf
            appr[(customer_index, joint_index)] = joint_cost

        transition = truck.pairwise(previous_region, current_region)
        current_odd = np.min(previous_even[:, None] + transition, axis=0)
        odd_values[customer_index] = current_odd
        current_even = np.full(len(current_region), np.inf, dtype=np.float64)
        for joint_index in range(min(drones_per_truck, customer_index)):
            matrix = appr.get((customer_index, joint_index))
            if matrix is None:
                continue
            launch_values = odd_values[customer_index - joint_index]
            current_even = np.minimum(
                current_even,
                np.min(matrix + launch_values[None, :], axis=1),
            )
        previous_even = current_even

        # 后续不会再访问超过无人机数量窗口的 appr 与 odd 层，及时释放大矩阵。
        expired_index = customer_index - drones_per_truck
        if expired_index > 0:
            odd_values.pop(expired_index, None)
            for key in [key for key in appr if key[0] <= expired_index]:
                appr.pop(key, None)

    return float(np.min(previous_even + truck.pairwise(region_nodes[-1], [depot])[:, 0]))


def _group_cache_key(
    instance_id: str,
    depot: int,
    cities: Iterable[int],
    *,
    time_limit: float,
    threads: int,
    seed: int,
    mip_gap: float,
    max_binary_variables: int,
) -> str:
    """输入组身份和求解配置，输出带评价器版本的 SHA-256 缓存键。"""
    payload = {
        "version": EVALUATOR_VERSION,
        "instance_id": instance_id,
        "depot": int(depot),
        "cities": sorted(map(int, cities)),
        "time_limit": float(time_limit),
        "threads": int(threads),
        "seed": int(seed),
        "mip_gap": float(mip_gap),
        "max_binary_variables": int(max_binary_variables),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def evaluate_group(
    *,
    instance_id: str,
    depot: int,
    cities: Iterable[int],
    boundary: dict[int, list[int]],
    regions: dict[int, list[int]],
    distance: dict,
    drones_per_truck: int,
    drone_limit: float,
    drone_speed: float,
    cache: GroupEvaluationCache,
    time_limit: float = 600.0,
    threads: int = 1,
    seed: int = 0,
    mip_gap: float = 1e-4,
    max_binary_variables: int = 0,
) -> dict[str, object]:
    """
    真实评价一个仓库组并缓存结果。

    输入为固定边界、服务区域、道路距离和求解预算；输出阶段 2/3 时间、成本、状态和顺序。
    超过模型规模保护或无可行解时使用有向道路最近邻，并明确记录标签来源。
    有限预算下只有求解器真实达到时间上限才属于右删失；9999 表示完全不设置时间上限。
    """
    group = sorted(map(int, cities))
    cache_key = _group_cache_key(
        instance_id,
        depot,
        group,
        time_limit=time_limit,
        threads=threads,
        seed=seed,
        mip_gap=mip_gap,
        max_binary_variables=max_binary_variables,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return dict(cached, cache_hit=True)
    if not group:
        record = {
            "depot": int(depot),
            "cities": [],
            "complete": True,
            "phase2_wall_seconds": 0.0,
            "phase2_effective_seconds": 0.0,
            "phase3_seconds": 0.0,
            "final_cost": 0.0,
            "sequence": [0, 0],
            "visit_route": [int(depot), int(depot)],
            "timeout": False,
            "censored": False,
            "right_censored": False,
            "runtime_label_valid": True,
            "objective_label_exact": True,
            "solve_outcome": "empty",
            "sequence_source": "empty",
            "fallback_used": False,
            "guard_fallback": False,
            "estimated_binary_variables": 0,
            "truck_only_visit_route_cost": 0.0,
            "phase3_coordination_gain": 0.0,
            "solver": {
                "status": "EMPTY_GROUP",
                "has_incumbent": True,
                "timeout": False,
                "mip_gap": 0.0,
                "objective": 0.0,
                "num_variables": 0,
                "num_binary_variables": 0,
                "num_constraints": 0,
                "build_seconds": 0.0,
                "optimize_seconds": 0.0,
                "distance_seconds": 0.0,
                "solver_work": 0.0,
                "requested_time_limit": float(time_limit),
                "applied_time_limit": None if is_unlimited_solver_time(time_limit) else float(time_limit),
                "unlimited_time": is_unlimited_solver_time(time_limit),
            },
        }
        cache.put(cache_key, record)
        return dict(record, cache_hit=False)

    phase2_started_at = time.perf_counter()
    convex_sets = [[int(depot)]] + [boundary[city] for city in group]
    estimated_binaries = estimate_set_tsp_binary_variables(convex_sets)
    fallback_used = False
    # 非正阈值表示关闭变量规模保护，让服务器尝试把每个模型求解完成。
    over_variable_limit = max_binary_variables > 0 and estimated_binaries > max_binary_variables
    has_empty_set = any(not nodes for nodes in convex_sets)
    guard_fallback = over_variable_limit or has_empty_set
    if over_variable_limit and has_empty_set:
        guard_reason = "variable_limit_and_empty_set"
    elif over_variable_limit:
        guard_reason = "variable_limit"
    elif has_empty_set:
        guard_reason = "empty_set"
    else:
        guard_reason = None
    solver_info: dict[str, object] = {
        "status": "COMPLEXITY_GUARD" if guard_fallback else "NOT_STARTED",
        "has_incumbent": False,
        "timeout": False,
        "build_seconds": 0.0,
        "optimize_seconds": 0.0,
        "requested_time_limit": float(time_limit),
        "applied_time_limit": None if is_unlimited_solver_time(time_limit) else float(time_limit),
        "unlimited_time": is_unlimited_solver_time(time_limit),
    }
    if guard_fallback:
        sequence = _nearest_neighbor_sequence(depot, group, distance["truck"])
        fallback_used = True
    else:
        distance_started_at = time.perf_counter()
        set_distance = [
            [
                [
                    max(
                        distance["truck"][exit_node][entry_node],
                        MultiAgentFlyingSidekickTSP.cut_off(
                            distance["drone"][entry_node][city]
                            + distance["drone"][city][exit_node],
                            drone_limit,
                        ),
                    ) / drone_speed
                    for entry_node in nodes
                ]
                for exit_node in nodes
            ]
            for nodes, city in zip(convex_sets, [depot] + group)
        ]
        external_distance = [
            [
                [
                    [distance["truck"][source_node][target_node] for target_node in target]
                    for source_node in source
                ]
                for target in convex_sets
            ]
            for source in convex_sets
        ]
        distance_seconds = time.perf_counter() - distance_started_at
        sequence, solver_info = _solve_set_tsp(
            convex_sets,
            set_distance,
            external_distance,
            time_limit=time_limit,
            threads=threads,
            seed=seed,
            mip_gap=mip_gap,
        )
        solver_info["distance_seconds"] = distance_seconds
        if sequence is None:
            sequence = _nearest_neighbor_sequence(depot, group, distance["truck"])
            fallback_used = True
    phase2_wall_seconds = time.perf_counter() - phase2_started_at
    timeout = bool(solver_info.get("timeout", False))
    has_incumbent = bool(solver_info.get("has_incumbent", False))
    if guard_fallback:
        solve_outcome = "guard_skipped"
        sequence_source = "nearest_neighbor_guard_fallback"
    elif timeout and has_incumbent:
        solve_outcome = "time_limit_with_incumbent"
        sequence_source = "set_tsp_incumbent"
    elif timeout:
        solve_outcome = "time_limit_without_incumbent"
        sequence_source = "nearest_neighbor_timeout_fallback"
    elif has_incumbent and int(solver_info.get("status", -1)) == int(GRB.OPTIMAL):
        solve_outcome = "optimal"
        sequence_source = "set_tsp_optimal"
    elif has_incumbent:
        solve_outcome = "incomplete_with_incumbent"
        sequence_source = "set_tsp_incumbent"
    else:
        solve_outcome = "solver_without_incumbent"
        sequence_source = "nearest_neighbor_solver_fallback"

    # 复杂度保护没有真正启动求解器，因此不能把 9999 秒当成其运行时间下界。
    right_censored = timeout
    runtime_label_valid = not guard_fallback
    objective_label_exact = solve_outcome == "optimal"
    phase2_effective_seconds = (
        max(phase2_wall_seconds, time_limit)
        if right_censored and not is_unlimited_solver_time(time_limit)
        else phase2_wall_seconds
    )

    visit_route = [depot] + [group[index - 1] for index in sequence[1:-1]] + [depot]
    phase3_started_at = time.perf_counter()
    cost = phase3_cost_vectorized(
        visit_route,
        depot,
        regions,
        distance["truck"],
        distance["drone"],
        drones_per_truck,
        drone_limit,
        drone_speed,
    )
    phase3_seconds = time.perf_counter() - phase3_started_at
    complete = math.isfinite(cost)
    # 该值是固定客户顺序上“全部由卡车访问客户节点”的可解释对照，
    # 不冒充第三阶段最优联合路线的精确卡车分解。
    truck_only_cost = float(sum(
        distance["truck"][visit_route[index]][visit_route[index + 1]]
        for index in range(len(visit_route) - 1)
    ))
    record = {
        "depot": int(depot),
        "cities": group,
        "complete": bool(complete),
        "phase2_wall_seconds": float(phase2_wall_seconds),
        "phase2_effective_seconds": float(phase2_effective_seconds),
        "phase3_seconds": float(phase3_seconds),
        "final_cost": float(cost) if complete else None,
        "sequence": list(map(int, sequence)),
        "visit_route": list(map(int, visit_route)),
        "timeout": timeout,
        "censored": right_censored,
        "right_censored": right_censored,
        "runtime_label_valid": runtime_label_valid,
        "objective_label_exact": objective_label_exact,
        "solve_outcome": solve_outcome,
        "sequence_source": sequence_source,
        "fallback_used": fallback_used,
        "guard_fallback": guard_fallback,
        "guard_reason": guard_reason,
        "estimated_binary_variables": int(estimated_binaries),
        "truck_only_visit_route_cost": truck_only_cost,
        "phase3_coordination_gain": truck_only_cost - float(cost) if complete else None,
        "phase3_method": "vectorized_cost_only_exact_recurrence_v1",
        "solver": solver_info,
    }
    cache.put(cache_key, record)
    return dict(record, cache_hit=False)


def evaluate_partition(
    *,
    instance_id: str,
    partition: Partition,
    depots: Iterable[int],
    boundary: dict[int, list[int]],
    regions: dict[int, list[int]],
    distance: dict,
    drones_per_truck: int,
    drone_limit: float,
    drone_speed: float,
    cache: GroupEvaluationCache,
    time_limit: float = 600.0,
    threads: int = 1,
    seed: int = 0,
    mip_gap: float = 1e-4,
    max_binary_variables: int = 0,
    evaluation_workers: int = 1,
) -> dict[str, object]:
    """
    评价完整客户分区。

    输入为所有仓库组和同一实例的空间/距离数据；输出累加成本、阶段时间与逐组记录。
    每个客户必须恰好出现一次；可选的组间并行只缩短采样等待，标签仍按各组时间求和。
    """
    groups = canonical_partition(partition, depots)
    def solve_one(depot: int) -> dict[str, object]:
        """输入仓库节点，评价该仓库组并返回可缓存记录。"""
        return evaluate_group(
            instance_id=instance_id,
            depot=depot,
            cities=groups[depot],
            boundary=boundary,
            regions=regions,
            distance=distance,
            drones_per_truck=drones_per_truck,
            drone_limit=drone_limit,
            drone_speed=drone_speed,
            cache=cache,
            time_limit=time_limit,
            threads=threads,
            seed=seed,
            mip_gap=mip_gap,
            max_binary_variables=max_binary_variables,
        )

    evaluation_started_at = time.perf_counter()
    if evaluation_workers <= 1:
        group_records = [solve_one(depot) for depot in groups]
    else:
        # 同一分区的仓库组互不依赖；并行仅减少等待时间，汇总标签仍为各组耗时之和。
        with ThreadPoolExecutor(max_workers=evaluation_workers) as executor:
            group_records = list(executor.map(solve_one, groups))
    evaluation_elapsed_seconds = time.perf_counter() - evaluation_started_at
    complete = all(bool(record["complete"]) for record in group_records)
    group_labels = []
    for record in group_records:
        solver = record.get("solver", {})
        group_labels.append({
            "depot": int(record["depot"]),
            "customer_count": len(record["cities"]),
            "phase2_wall_seconds": float(record["phase2_wall_seconds"]),
            "phase2_effective_seconds": float(record["phase2_effective_seconds"]),
            "phase3_seconds": float(record["phase3_seconds"]),
            "final_cost": record["final_cost"],
            "estimated_binary_variables": int(record["estimated_binary_variables"]),
            "timeout": bool(record["timeout"]),
            "censored": bool(record["censored"]),
            "right_censored": bool(record.get("right_censored", record["censored"])),
            "runtime_label_valid": bool(record.get("runtime_label_valid", True)),
            "objective_label_exact": bool(record.get("objective_label_exact", False)),
            "solve_outcome": str(record.get("solve_outcome", "legacy_unknown")),
            "sequence_source": str(record.get("sequence_source", "legacy_unknown")),
            "fallback_used": bool(record["fallback_used"]),
            "guard_fallback": bool(record["guard_fallback"]),
            "guard_reason": record.get("guard_reason"),
            "solver_status": solver.get("status"),
            "has_incumbent": bool(solver.get("has_incumbent", False)),
            "mip_gap": solver.get("mip_gap"),
            "phase2_objective": solver.get("objective"),
            "solver_work": float(solver.get("solver_work", 0.0)),
            "num_variables": int(solver.get("num_variables", 0)),
            "num_binary_variables": int(solver.get("num_binary_variables", 0)),
            "num_constraints": int(solver.get("num_constraints", 0)),
            "build_seconds": float(solver.get("build_seconds", 0.0)),
            "optimize_seconds": float(solver.get("optimize_seconds", 0.0)),
            "distance_seconds": float(solver.get("distance_seconds", 0.0)),
            "requested_time_limit": float(solver.get("requested_time_limit", time_limit)),
            "applied_time_limit": solver.get("applied_time_limit"),
            "unlimited_time": bool(solver.get("unlimited_time", False)),
            "truck_only_visit_route_cost": float(record.get("truck_only_visit_route_cost", 0.0)),
            "phase3_coordination_gain": record.get("phase3_coordination_gain"),
        })
    status_counts = Counter(str(item["solver_status"]) for item in group_labels)
    finite_gaps = [float(item["mip_gap"]) for item in group_labels if item["mip_gap"] is not None]
    wall_times = [float(item["phase2_wall_seconds"]) for item in group_labels]
    effective_times = [float(item["phase2_effective_seconds"]) for item in group_labels]
    runtime_label_valid = all(bool(item["runtime_label_valid"]) for item in group_labels)
    objective_label_exact = all(bool(item["objective_label_exact"]) for item in group_labels)
    phase2_objectives = [item["phase2_objective"] for item in group_labels]
    phase2_objective_complete = all(value is not None for value in phase2_objectives)
    right_censored_groups = sum(bool(item["right_censored"]) for item in group_labels)
    phase2_serial_effective_seconds = float(sum(effective_times))
    phase3_total_seconds = float(sum(record["phase3_seconds"] for record in group_records))
    return {
        "complete": complete,
        "final_cost": float(sum(record["final_cost"] for record in group_records)) if complete else None,
        # 主时间标签为仓库组串行求和；最大组时间只作为瓶颈辅助标签。
        "phase2_wall_seconds": float(sum(wall_times)),
        "phase2_effective_seconds": phase2_serial_effective_seconds,
        "phase2_serial_wall_seconds": float(sum(wall_times)),
        "phase2_serial_effective_seconds": phase2_serial_effective_seconds,
        "phase2_max_group_wall_seconds": float(max(wall_times, default=0.0)),
        "phase2_max_group_effective_seconds": float(max(effective_times, default=0.0)),
        "evaluation_elapsed_seconds": float(evaluation_elapsed_seconds),
        "phase3_seconds": phase3_total_seconds,
        "downstream_total_seconds": phase2_serial_effective_seconds + phase3_total_seconds,
        "runtime_label_valid": runtime_label_valid,
        "objective_label_exact": objective_label_exact,
        # Set-TSP 目标值是第二阶段的辅助标签；最终优化目标仍使用 Phase 3 的 final_cost。
        "phase2_objective_sum": (
            float(sum(float(value) for value in phase2_objectives))
            if phase2_objective_complete else None
        ),
        "phase2_objective_complete": phase2_objective_complete,
        "candidate_exact": runtime_label_valid and objective_label_exact and right_censored_groups == 0,
        "right_censored_groups": right_censored_groups,
        "timeout_groups": sum(bool(record["timeout"]) for record in group_records),
        "censored_groups": sum(bool(record["censored"]) for record in group_records),
        "guard_groups": sum(bool(record["guard_fallback"]) for record in group_records),
        "fallback_groups": sum(bool(record["fallback_used"]) for record in group_records),
        "incumbent_groups": sum(bool(item["has_incumbent"]) for item in group_labels),
        "cache_hit_groups": sum(bool(record["cache_hit"]) for record in group_records),
        "phase2_build_seconds": float(sum(item["build_seconds"] for item in group_labels)),
        "phase2_optimize_seconds": float(sum(item["optimize_seconds"] for item in group_labels)),
        "phase2_distance_seconds": float(sum(item["distance_seconds"] for item in group_labels)),
        "solver_work_sum": float(sum(item["solver_work"] for item in group_labels)),
        "estimated_binary_variables_sum": int(sum(item["estimated_binary_variables"] for item in group_labels)),
        "estimated_binary_variables_max": int(max(
            (item["estimated_binary_variables"] for item in group_labels), default=0
        )),
        "num_variables_sum": int(sum(item["num_variables"] for item in group_labels)),
        "num_variables_max": int(max((item["num_variables"] for item in group_labels), default=0)),
        "num_binary_variables_sum": int(sum(item["num_binary_variables"] for item in group_labels)),
        "num_binary_variables_max": int(max(
            (item["num_binary_variables"] for item in group_labels), default=0
        )),
        "num_constraints_sum": int(sum(item["num_constraints"] for item in group_labels)),
        "num_constraints_max": int(max((item["num_constraints"] for item in group_labels), default=0)),
        "mip_gap_max": max(finite_gaps) if finite_gaps else None,
        "solver_status_counts": dict(sorted(status_counts.items())),
        "solve_outcome_counts": dict(sorted(Counter(
            item["solve_outcome"] for item in group_labels
        ).items())),
        "sequence_source_counts": dict(sorted(Counter(
            item["sequence_source"] for item in group_labels
        ).items())),
        "guard_reason_counts": dict(sorted(Counter(
            item["guard_reason"] for item in group_labels if item["guard_reason"] is not None
        ).items())),
        "truck_only_visit_route_cost": float(sum(
            item["truck_only_visit_route_cost"] for item in group_labels
        )),
        "phase3_coordination_gain": float(sum(
            float(item["phase3_coordination_gain"] or 0.0) for item in group_labels
        )),
        "group_labels": group_labels,
        "group_records": group_records,
    }
