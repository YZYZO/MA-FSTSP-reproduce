"""
Directed Set-GTDS 客户划分算法。

本模块只负责 MA-FSTSP 第一阶段：构造有向集合代理代价、生成仓库感知巨路径，
并在给定代理成本容忍度下控制后续 Set-TSP 子问题的最大模型规模。
"""

from collections import deque
from dataclasses import dataclass
import math
import time

import elkai
import numpy as np


@dataclass(frozen=True)
class DirectedSetCost:
    """
    保存有向集合代价矩阵及节点到矩阵下标的映射。

    输入：固定节点顺序、二维代价矩阵和节点下标字典。
    输出：可通过 ``cost[u, v]`` 读取任意有向代价的只读结果对象。
    逻辑：节点编号与矩阵位置分离，避免调用方依赖客户输入排列。
    """

    nodes: tuple
    values: np.ndarray
    node_to_index: dict

    def __getitem__(self, edge):
        """
        读取一条有向边的集合代理代价。

        输入：二元组 ``(start, end)``。
        输出：对应的浮点代价。
        逻辑：先把节点映射为矩阵下标，再读取矩阵元素。
        """

        start, end = edge
        return float(self.values[self.node_to_index[start], self.node_to_index[end]])


@dataclass(frozen=True)
class SegmentStatistics:
    """
    保存巨路径连续片段的 Set-TSP 规模和路径前缀统计。

    输入：巨路径、候选集合规模前缀和、有向路径代价前缀和与 Q_bin 矩阵。
    输出：供预算搜索和仓库子集 DP 复用的紧凑统计对象。
    逻辑：所有片段统计预先计算，DP 中不再重复遍历候选集合。
    """

    giant_path: tuple
    size_prefix: np.ndarray
    squared_size_prefix: np.ndarray
    path_cost_prefix: np.ndarray
    q_bin: np.ndarray
    q_var: np.ndarray
    q_con: np.ndarray
    candidate_budgets: tuple


@dataclass
class SplitDPResult:
    """
    保存一次固定预算仓库子集 DP 的结果。

    输入：最优代理成本、最终仓库掩码、可选父指针和运行统计。
    输出：供预算搜索判断可行性，或供最终划分回溯。
    逻辑：预算搜索不保存父指针，最终一次 DP 才保留回溯信息。
    """

    cost: float
    final_mask: int | None
    parents: dict | None
    reachable_states: int
    evaluated_windows: int


@dataclass(frozen=True)
class PreparedSetGTDS:
    """
    保存可被多个 GTDS 消融变体共享的只读准备结果。

    输入来自候选集规范化、有向代理代价、巨路径和片段统计四个公共步骤；输出由
    ``solve_prepared_set_gtds`` 消费。这样同一个实例的 epsilon/活跃仓库变体不会
    重复运行 Elkai，也不会因随机启发式调用次数不同而失去严格配对条件。
    """

    depots: tuple
    cities: tuple
    candidate_sets: dict
    directed_cost: DirectedSetCost | None
    giant_path: tuple
    segment_stats: SegmentStatistics | None
    preparation_seconds: float
    directed_cost_seconds: float
    giant_path_seconds: float
    segment_statistics_seconds: float
    scale: int
    lkh_runs: int
    drone_cost_mode: str
    drone_cost_multiplier: float
    drone_cost_factor: float


@dataclass
class SetGTDSResult:
    """
    保存 Directed Set-GTDS 的最终分组与第一阶段遥测。

    输入：分组、巨路径、代理成本、模型预算以及各子阶段耗时。
    输出：主求解器可直接使用的分组和可序列化诊断信息。
    逻辑：候选集合随结果返回，确保 Phase 1 与 Phase 2 使用同一份正规化数据。
    """

    groups: dict
    candidate_sets: dict
    giant_path: list
    surrogate_cost_unbounded: float
    surrogate_cost_final: float
    q_budget: int | None
    max_q_bin: int
    phase1_time: float
    directed_cost_seconds: float
    giant_path_seconds: float
    segment_statistics_seconds: float
    split_seconds: float
    budget_evaluations: int
    dp_runs: int
    reachable_states: int
    evaluated_windows: int
    epsilon: float
    scale: int
    lkh_runs: int
    drone_cost_mode: str
    drone_cost_multiplier: float
    drone_cost_factor: float
    model_budget_enabled: bool
    active_depot_policy: str
    min_active_depots: int
    active_depots: int

    def diagnostics(self):
        """
        生成不含大规模候选集合和分组副本的实验诊断字典。

        输入：无显式输入。
        输出：可直接写入实验过程记录的基础类型字典。
        逻辑：仅保留算法参数结果、规模统计和阶段耗时，控制 NPZ 体积。
        """

        return {
            'giant_path': list(self.giant_path),
            'surrogate_cost_unbounded': float(self.surrogate_cost_unbounded),
            'surrogate_cost_final': float(self.surrogate_cost_final),
            'q_budget': self.q_budget,
            'max_q_bin': int(self.max_q_bin),
            'phase1_time': float(self.phase1_time),
            'directed_cost_seconds': float(self.directed_cost_seconds),
            'giant_path_seconds': float(self.giant_path_seconds),
            'segment_statistics_seconds': float(self.segment_statistics_seconds),
            'split_seconds': float(self.split_seconds),
            'budget_evaluations': int(self.budget_evaluations),
            'dp_runs': int(self.dp_runs),
            'reachable_states': int(self.reachable_states),
            'evaluated_windows': int(self.evaluated_windows),
            'epsilon': float(self.epsilon),
            'scale': int(self.scale),
            'lkh_runs': int(self.lkh_runs),
            'drone_cost_mode': self.drone_cost_mode,
            'drone_cost_multiplier': float(self.drone_cost_multiplier),
            'drone_cost_factor': float(self.drone_cost_factor),
            'model_budget_enabled': bool(self.model_budget_enabled),
            'active_depot_policy': self.active_depot_policy,
            'min_active_depots': int(self.min_active_depots),
            'active_depots': int(self.active_depots),
        }


def _canonical_order(nodes):
    """
    为节点集合建立与输入排列无关的固定顺序。

    输入：可迭代节点集合。
    输出：去重后的稳定节点列表。
    逻辑：优先使用节点的自然顺序；混合类型不可比较时退化为类型名和字符串排序。
    """

    unique_nodes = list(dict.fromkeys(nodes))
    try:
        return sorted(unique_nodes)
    except TypeError:
        return sorted(unique_nodes, key=lambda node: (type(node).__name__, repr(node)))


def resolve_drone_cost_multiplier(drone_cost_mode):
    """
    将具名无人机代价模式转换为公式中的无量纲乘数。

    输入为 ``paper`` 或 ``smst_compatible``；输出分别为 1 与 ``sqrt(2)``。
    逻辑上把论文公式与公开 SMST 代码兼容公式显式分离，避免实验名称隐含代价语义。
    """

    multipliers = {
        'paper': 1.0,
        'smst_compatible': math.sqrt(2.0),
    }
    try:
        return multipliers[str(drone_cost_mode)]
    except KeyError as error:
        raise ValueError(
            f'未知的无人机代价模式：{drone_cost_mode!r}'
        ) from error


def resolve_min_active_depots(policy, depot_count, customer_count):
    """
    将活跃仓库策略解析为当前实例的最少活跃仓库数量。

    输入为 ``all``、``free`` 或非负整数，以及仓库/客户数量；输出整数下界。
    ``all`` 会尽可能启用全部仓库，客户少于仓库时最多启用客户数量个仓库。
    """

    if policy == 'all':
        return min(int(depot_count), int(customer_count))
    if policy == 'free':
        return 1 if int(customer_count) > 0 else 0
    minimum = int(policy)
    if not 0 <= minimum <= int(depot_count):
        raise ValueError('活跃仓库数量必须位于 0 与仓库数之间。')
    if minimum > int(customer_count):
        raise ValueError('客户数少于要求的活跃仓库数，分区不可行。')
    return minimum


def normalize_candidate_sets(cities, raw_sets):
    """
    正规化客户候选集合，保证每个客户至少有一个候选点。

    输入：客户序列和 ``{customer: candidate_nodes}`` 原始映射。
    输出：新的候选集合字典；空集合退化为客户节点自身。
    逻辑：复制已有列表以隔离调用方修改，同时保持原有候选点遍历顺序。
    """

    normalized = {}
    for city in cities:
        candidates = list(raw_sets.get(city, []))
        normalized[city] = candidates if candidates else [city]
    return normalized


def build_directed_set_cost(
    depots,
    cities,
    candidate_sets,
    truck_distance,
    drone_distance,
    speed,
    drone_cost_mode='paper',
):
    """
    构造仓库与客户之间的有向集合代理代价矩阵。

    输入：仓库、客户、正规化候选集合、卡车/无人机距离和无人机相对速度。
    输出：``DirectedSetCost``，同时保存 ``A[u,v]`` 与 ``A[v,u]``。
    逻辑：主模式使用论文式（3）对应的 ``1 / speed``，兼容模式使用公开 SMST
    代码的 ``sqrt(2) / speed``，并与客户节点之间的直接卡车距离取较小值。
    """

    if speed <= 0:
        raise ValueError('speed 必须大于 0。')

    depot_order = _canonical_order(depots)
    city_order = _canonical_order(cities)
    nodes = tuple(depot_order + city_order)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    values = np.zeros((len(nodes), len(nodes)), dtype=float)
    depot_set = set(depot_order)
    drone_multiplier = resolve_drone_cost_multiplier(drone_cost_mode)
    drone_factor = drone_multiplier / float(speed)

    # 仓库候选集合固定为自身；客户集合沿用 Phase 2 使用的正规化结果。
    node_candidate_sets = {
        **{depot: [depot] for depot in depot_order},
        **{city: list(candidate_sets[city]) for city in city_order},
    }

    for start in nodes:
        start_index = node_to_index[start]
        for end in nodes:
            end_index = node_to_index[end]
            if start == end:
                continue

            best_cost = float(truck_distance[start][end])
            for start_node in node_candidate_sets[start]:
                start_drone_cost = (
                    0.0
                    if start in depot_set
                    else drone_factor * float(drone_distance[start][start_node])
                )
                for end_node in node_candidate_sets[end]:
                    end_drone_cost = (
                        0.0
                        if end in depot_set
                        else drone_factor * float(drone_distance[end][end_node])
                    )
                    candidate_cost = (
                        start_drone_cost
                        + float(truck_distance[start_node][end_node])
                        + end_drone_cost
                    )
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
            values[start_index, end_index] = best_cost

    return DirectedSetCost(nodes, values, node_to_index)


def _scaled_giant_path_matrix(depots, cities, directed_cost, scale):
    """
    构造包含虚拟仓库节点的整数 ATSP 矩阵。

    输入：仓库、固定顺序客户、有向集合代价和整数缩放比例。
    输出：``(matrix, virtual_index)``。
    逻辑：虚拟节点的出入代价分别取所有仓库的最优出发和返回代价。
    """

    if scale <= 0:
        raise ValueError('scale 必须大于 0。')

    customer_count = len(cities)
    virtual_index = customer_count
    matrix = [[0 for _ in range(customer_count + 1)] for _ in range(customer_count + 1)]

    for start_index, start in enumerate(cities):
        for end_index, end in enumerate(cities):
            if start_index != end_index:
                matrix[start_index][end_index] = max(
                    1,
                    int(round(scale * directed_cost[start, end])),
                )
        matrix[start_index][virtual_index] = max(
            1,
            int(round(scale * min(directed_cost[start, depot] for depot in depots))),
        )
        matrix[virtual_index][start_index] = max(
            1,
            int(round(scale * min(directed_cost[depot, start] for depot in depots))),
        )
    return matrix, virtual_index


def solve_depot_aware_giant_path(
    depots,
    cities,
    directed_cost,
    scale=1000,
    lkh_runs=10,
):
    """
    使用虚拟仓库节点和 Elkai ATSP 生成仓库感知开放巨路径。

    输入：仓库、客户、有向代价、整数缩放比例和 LKH 运行次数。
    输出：每个客户恰好出现一次的开放路径列表。
    逻辑：客户采用固定顺序建矩阵，求得闭环后从虚拟节点处切开并删除虚拟节点。
    """

    depot_order = _canonical_order(depots)
    city_order = _canonical_order(cities)
    if len(city_order) == 0:
        return []
    if len(city_order) == 1:
        return city_order

    matrix, virtual_index = _scaled_giant_path_matrix(
        depot_order,
        city_order,
        directed_cost,
        scale,
    )
    route = list(elkai.DistanceMatrix(matrix).solve_tsp(runs=lkh_runs))
    if route and route[0] == route[-1]:
        route.pop()
    if len(route) != len(city_order) + 1 or set(route) != set(range(len(city_order) + 1)):
        raise RuntimeError('Elkai 返回的巨路径环包含重复或遗漏节点。')

    virtual_position = route.index(virtual_index)
    rotated = route[virtual_position:] + route[:virtual_position]
    giant_path_indices = rotated[1:]
    return [city_order[index] for index in giant_path_indices]


def calculate_set_tsp_model_size(candidate_sizes):
    """
    根据候选集合大小计算当前 Set-TSP 模型的预求解规模。

    输入：包含仓库集合在内的各集合大小序列。
    输出：``(Q_bin, Q_var, Q_con)``。
    逻辑：严格对应 ``fstsp.py::set_tsp`` 创建的 select、flow、internal 和 external。
    """

    sizes = [int(size) for size in candidate_sizes]
    set_count = len(sizes)
    total_nodes = sum(sizes)
    squared_nodes = sum(size * size for size in sizes)
    q_bin = set_count * set_count + squared_nodes + total_nodes * total_nodes
    q_var = q_bin + set_count * set_count
    q_con = 2 * set_count * set_count + 6 * set_count + 1 + 2 * total_nodes
    return q_bin, q_var, q_con


def build_segment_statistics(giant_path, candidate_sets, directed_cost):
    """
    预计算所有连续客户片段的模型规模与巨路径内部代价。

    输入：巨路径、正规化候选集合和有向集合代价。
    输出：``SegmentStatistics``。
    逻辑：使用候选规模前缀和计算每个片段的 Q 指标，并用路径前缀和支持 O(1)
    片段代理成本查询。
    """

    path = tuple(giant_path)
    customer_count = len(path)
    sizes = np.asarray([len(candidate_sets[city]) for city in path], dtype=np.int64)
    size_prefix = np.zeros(customer_count + 1, dtype=np.int64)
    squared_size_prefix = np.zeros(customer_count + 1, dtype=np.int64)
    size_prefix[1:] = np.cumsum(sizes)
    squared_size_prefix[1:] = np.cumsum(sizes * sizes)

    path_cost_prefix = np.zeros(customer_count, dtype=float)
    for index in range(1, customer_count):
        path_cost_prefix[index] = (
            path_cost_prefix[index - 1]
            + directed_cost[path[index - 1], path[index]]
        )

    q_bin = np.full((customer_count, customer_count), -1, dtype=np.int64)
    q_var = np.full((customer_count, customer_count), -1, dtype=np.int64)
    q_con = np.full((customer_count, customer_count), -1, dtype=np.int64)
    budgets = set()
    for start in range(customer_count):
        for end in range(start, customer_count):
            group_size = end - start + 1
            set_count = group_size + 1
            total_nodes = 1 + int(size_prefix[end + 1] - size_prefix[start])
            squared_nodes = 1 + int(
                squared_size_prefix[end + 1] - squared_size_prefix[start]
            )
            binary_count = (
                set_count * set_count
                + squared_nodes
                + total_nodes * total_nodes
            )
            variable_count = binary_count + set_count * set_count
            constraint_count = (
                2 * set_count * set_count
                + 6 * set_count
                + 1
                + 2 * total_nodes
            )
            q_bin[start, end] = binary_count
            q_var[start, end] = variable_count
            q_con[start, end] = constraint_count
            budgets.add(binary_count)

    return SegmentStatistics(
        giant_path=path,
        size_prefix=size_prefix,
        squared_size_prefix=squared_size_prefix,
        path_cost_prefix=path_cost_prefix,
        q_bin=q_bin,
        q_var=q_var,
        q_con=q_con,
        candidate_budgets=tuple(sorted(budgets)),
    )


def _minimum_segment_starts(segment_stats, q_budget):
    """
    为每个片段终点计算当前预算下最早可行的起点。

    输入：片段统计和可选 Q_bin 预算。
    输出：长度为客户数的整数数组。
    逻辑：Q_bin 随片段向左扩展单调增加，因此用单调双指针在线性时间完成。
    """

    customer_count = len(segment_stats.giant_path)
    if q_budget is None:
        return np.zeros(customer_count, dtype=np.int64)

    starts = np.zeros(customer_count, dtype=np.int64)
    start = 0
    for end in range(customer_count):
        while start <= end and segment_stats.q_bin[start, end] > q_budget:
            start += 1
        starts[end] = start
    return starts


def run_subset_split_dp(
    depots,
    giant_path,
    directed_cost,
    segment_stats,
    q_budget=None,
    keep_parent=False,
    min_active_depots=1,
):
    """
    在固定巨路径和模型预算下精确求解仓库子集 Split。

    输入：仓库、巨路径、有向代价、片段统计、可选预算和父指针开关。
    输出：``SplitDPResult``。
    逻辑：状态 ``F[mask,j]`` 表示已用仓库服务前 j 个客户的最小代理成本；利用
    片段成本的前后缀可分结构，对每个 ``(mask,depot)`` 用滑动窗口最小值扫描，
    将一次预算的复杂度降为 ``O(2^m m n)``。
    """

    depot_order = _canonical_order(depots)
    path = list(giant_path)
    customer_count = len(path)
    if not 0 <= int(min_active_depots) <= len(depot_order):
        raise ValueError('min_active_depots 必须位于 0 与仓库数之间。')
    if int(min_active_depots) > customer_count:
        return SplitDPResult(
            float('inf'),
            None,
            {} if keep_parent else None,
            1,
            0,
        )
    if customer_count == 0:
        final_mask = 0 if int(min_active_depots) == 0 else None
        return SplitDPResult(
            0.0 if final_mask is not None else float('inf'),
            final_mask,
            {} if keep_parent else None,
            1,
            0,
        )

    mask_count = 1 << len(depot_order)
    values = np.full((mask_count, customer_count + 1), np.inf, dtype=float)
    values[0, 0] = 0.0
    parents = {} if keep_parent else None
    minimum_starts = _minimum_segment_starts(segment_stats, q_budget)
    reachable_states = 1
    evaluated_windows = 0

    for mask in range(mask_count):
        if not np.isfinite(values[mask]).any():
            continue
        for depot_index, depot in enumerate(depot_order):
            depot_bit = 1 << depot_index
            if mask & depot_bit:
                continue

            # 窗口元素为 (已覆盖客户数, 与片段终点无关的转移前缀代价)。
            candidates = deque()
            new_mask = mask | depot_bit
            for end in range(customer_count):
                split = end
                if np.isfinite(values[mask, split]):
                    prefix_cost = (
                        values[mask, split]
                        + directed_cost[depot, path[split]]
                        - segment_stats.path_cost_prefix[split]
                    )
                    while candidates and candidates[-1][1] > prefix_cost:
                        candidates.pop()
                    candidates.append((split, prefix_cost))

                minimum_start = int(minimum_starts[end])
                while candidates and candidates[0][0] < minimum_start:
                    candidates.popleft()
                if not candidates:
                    continue

                evaluated_windows += 1
                best_split, best_prefix = candidates[0]
                candidate_cost = (
                    best_prefix
                    + segment_stats.path_cost_prefix[end]
                    + directed_cost[path[end], depot]
                )
                target = end + 1
                if candidate_cost < values[new_mask, target]:
                    if not np.isfinite(values[new_mask, target]):
                        reachable_states += 1
                    values[new_mask, target] = candidate_cost
                    if keep_parent:
                        parents[new_mask, target] = (mask, best_split, depot_index)

    final_candidates = [
        (float(values[mask, customer_count]), mask.bit_count(), mask)
        for mask in range(mask_count)
        if (
            np.isfinite(values[mask, customer_count])
            and mask.bit_count() >= int(min_active_depots)
        )
    ]
    if not final_candidates:
        return SplitDPResult(
            float('inf'),
            None,
            parents,
            reachable_states,
            evaluated_windows,
        )
    best_cost, _, final_mask = min(final_candidates)
    return SplitDPResult(
        best_cost,
        final_mask,
        parents,
        reachable_states,
        evaluated_windows,
    )


def _restore_groups(depots, giant_path, split_result):
    """
    根据最终 DP 父指针恢复每个仓库的连续客户片段。

    输入：仓库、巨路径和带父指针的 DP 结果。
    输出：包含所有仓库键的客户分组字典。
    逻辑：从 ``(final_mask,n)`` 逆向回溯，每次恢复一个仓库及其非空片段。
    """

    depot_order = _canonical_order(depots)
    groups = {depot: [] for depot in depots}
    mask = split_result.final_mask
    end = len(giant_path)
    while end > 0:
        old_mask, start, depot_index = split_result.parents[mask, end]
        groups[depot_order[depot_index]] = list(giant_path[start:end])
        mask = old_mask
        end = start
    return groups


def find_minimum_model_budget(
    depots,
    giant_path,
    directed_cost,
    segment_stats,
    unbounded_cost,
    epsilon=0.01,
    min_active_depots=1,
):
    """
    二分搜索满足代理成本容忍度的最小 Q_bin 预算。

    输入：DP 公共数据、无预算最优成本和容忍比例。
    输出：``(minimum_budget, budget_cost, evaluation_count)``。
    逻辑：预算增大时最优代理成本单调不增，因此可在全部片段预算候选上二分。
    """

    budgets = segment_stats.candidate_budgets
    target_cost = (1.0 + epsilon) * unbounded_cost
    tolerance = 1e-9 * max(1.0, abs(unbounded_cost))
    left, right = 0, len(budgets) - 1
    best_budget = budgets[-1]
    best_cost = float('inf')
    evaluation_count = 0

    while left <= right:
        middle = (left + right) // 2
        budget = budgets[middle]
        result = run_subset_split_dp(
            depots,
            giant_path,
            directed_cost,
            segment_stats,
            q_budget=budget,
            keep_parent=False,
            min_active_depots=min_active_depots,
        )
        evaluation_count += 1
        if result.cost <= target_cost + tolerance:
            best_budget = budget
            best_cost = result.cost
            right = middle - 1
        else:
            left = middle + 1
    return int(best_budget), float(best_cost), evaluation_count


def prepare_set_gtds(
    depots,
    cities,
    candidate_sets,
    truck_distance,
    drone_distance,
    speed,
    scale=1000,
    lkh_runs=10,
    drone_cost_mode='paper',
):
    """
    构造所有 GTDS 变体共享的第一阶段公共数据。

    输入为仓库、客户、候选集、两类距离、巨路径参数和无人机代价模式；函数依次
    规范化候选集、构造有向集合代价、求巨路径并预计算片段统计，输出 ``PreparedSetGTDS``。
    空客户实例仍返回合法准备对象，其中代价和片段统计为 ``None``。
    """

    start_total = time.perf_counter()
    depot_order = tuple(_canonical_order(depots))
    city_order = tuple(_canonical_order(cities))
    if not depot_order:
        raise ValueError('Directed Set-GTDS 至少需要一个仓库。')
    if speed <= 0:
        raise ValueError('speed 必须大于 0。')
    drone_multiplier = resolve_drone_cost_multiplier(drone_cost_mode)

    normalized_sets = normalize_candidate_sets(city_order, candidate_sets)
    if not city_order:
        return PreparedSetGTDS(
            depots=depot_order,
            cities=city_order,
            candidate_sets=normalized_sets,
            directed_cost=None,
            giant_path=(),
            segment_stats=None,
            preparation_seconds=time.perf_counter() - start_total,
            directed_cost_seconds=0.0,
            giant_path_seconds=0.0,
            segment_statistics_seconds=0.0,
            scale=int(scale),
            lkh_runs=int(lkh_runs),
            drone_cost_mode=str(drone_cost_mode),
            drone_cost_multiplier=drone_multiplier,
            drone_cost_factor=drone_multiplier / float(speed),
        )

    start = time.perf_counter()
    directed_cost = build_directed_set_cost(
        depot_order,
        city_order,
        normalized_sets,
        truck_distance,
        drone_distance,
        speed,
        drone_cost_mode=drone_cost_mode,
    )
    directed_cost_seconds = time.perf_counter() - start

    start = time.perf_counter()
    giant_path = tuple(solve_depot_aware_giant_path(
        depot_order,
        city_order,
        directed_cost,
        scale=scale,
        lkh_runs=lkh_runs,
    ))
    giant_path_seconds = time.perf_counter() - start

    start = time.perf_counter()
    segment_stats = build_segment_statistics(
        giant_path,
        normalized_sets,
        directed_cost,
    )
    segment_statistics_seconds = time.perf_counter() - start

    return PreparedSetGTDS(
        depots=depot_order,
        cities=city_order,
        candidate_sets=normalized_sets,
        directed_cost=directed_cost,
        giant_path=giant_path,
        segment_stats=segment_stats,
        preparation_seconds=time.perf_counter() - start_total,
        directed_cost_seconds=directed_cost_seconds,
        giant_path_seconds=giant_path_seconds,
        segment_statistics_seconds=segment_statistics_seconds,
        scale=int(scale),
        lkh_runs=int(lkh_runs),
        drone_cost_mode=str(drone_cost_mode),
        drone_cost_multiplier=drone_multiplier,
        drone_cost_factor=drone_multiplier / float(speed),
    )


def solve_prepared_set_gtds(
    prepared,
    epsilon=0.01,
    apply_model_budget=True,
    min_active_depots=None,
    active_depot_policy='all',
):
    """
    在共享准备结果上求解一个指定的 GTDS 变体。

    输入为 ``PreparedSetGTDS``、代理成本容忍度、模型预算开关和活跃仓库策略；
    先求相同可行域下的无预算最优值，再搜索最小 Q_bin 预算并回溯分组，输出带
    完整诊断的 ``SetGTDSResult``。所有仓库非空可传 ``len(prepared.depots)``。
    """

    split_start = time.perf_counter()
    effective_policy = (
        min_active_depots
        if min_active_depots is not None
        else active_depot_policy
    )
    minimum_active = resolve_min_active_depots(
        effective_policy,
        len(prepared.depots),
        len(prepared.cities),
    )
    policy_name = str(effective_policy)

    if not prepared.cities:
        groups = {depot: [] for depot in prepared.depots}
        return SetGTDSResult(
            groups=groups,
            candidate_sets=prepared.candidate_sets,
            giant_path=[],
            surrogate_cost_unbounded=0.0,
            surrogate_cost_final=0.0,
            q_budget=0 if apply_model_budget else None,
            max_q_bin=0,
            phase1_time=prepared.preparation_seconds,
            directed_cost_seconds=prepared.directed_cost_seconds,
            giant_path_seconds=prepared.giant_path_seconds,
            segment_statistics_seconds=prepared.segment_statistics_seconds,
            split_seconds=0.0,
            budget_evaluations=0,
            dp_runs=0,
            reachable_states=1,
            evaluated_windows=0,
            epsilon=float(epsilon),
            scale=prepared.scale,
            lkh_runs=prepared.lkh_runs,
            drone_cost_mode=prepared.drone_cost_mode,
            drone_cost_multiplier=prepared.drone_cost_multiplier,
            drone_cost_factor=prepared.drone_cost_factor,
            model_budget_enabled=bool(apply_model_budget),
            active_depot_policy=policy_name,
            min_active_depots=minimum_active,
            active_depots=0,
        )

    # 无预算基准必须与当前变体使用同一个活跃仓库可行域，否则 epsilon 保证失真。
    unbounded_result = run_subset_split_dp(
        prepared.depots,
        prepared.giant_path,
        prepared.directed_cost,
        prepared.segment_stats,
        q_budget=None,
        keep_parent=not apply_model_budget,
        min_active_depots=minimum_active,
    )
    if unbounded_result.final_mask is None:
        raise RuntimeError('Directed Set-GTDS 未找到无预算可行分区。')

    if apply_model_budget:
        q_budget, _, budget_evaluations = find_minimum_model_budget(
            prepared.depots,
            prepared.giant_path,
            prepared.directed_cost,
            prepared.segment_stats,
            unbounded_result.cost,
            epsilon=epsilon,
            min_active_depots=minimum_active,
        )
        final_result = run_subset_split_dp(
            prepared.depots,
            prepared.giant_path,
            prepared.directed_cost,
            prepared.segment_stats,
            q_budget=q_budget,
            keep_parent=True,
            min_active_depots=minimum_active,
        )
        dp_runs = budget_evaluations + 2
    else:
        final_result = unbounded_result
        q_budget = None
        budget_evaluations = 0
        dp_runs = 1

    if final_result.final_mask is None:
        raise RuntimeError('Directed Set-GTDS 未找到满足模型预算的可行分区。')
    groups = _restore_groups(
        prepared.depots,
        prepared.giant_path,
        final_result,
    )

    # 从最终连续片段读取每个非空子模型的 Q_bin，作为 Phase 2 难度代理。
    path_index = {
        city: index
        for index, city in enumerate(prepared.giant_path)
    }
    selected_q_bin = []
    for group in groups.values():
        if group:
            start_index = path_index[group[0]]
            end_index = path_index[group[-1]]
            selected_q_bin.append(int(
                prepared.segment_stats.q_bin[start_index, end_index]
            ))
    split_seconds = time.perf_counter() - split_start

    return SetGTDSResult(
        groups=groups,
        candidate_sets=prepared.candidate_sets,
        giant_path=list(prepared.giant_path),
        surrogate_cost_unbounded=float(unbounded_result.cost),
        surrogate_cost_final=float(final_result.cost),
        q_budget=q_budget,
        max_q_bin=max(selected_q_bin, default=0),
        phase1_time=prepared.preparation_seconds + split_seconds,
        directed_cost_seconds=prepared.directed_cost_seconds,
        giant_path_seconds=prepared.giant_path_seconds,
        segment_statistics_seconds=prepared.segment_statistics_seconds,
        split_seconds=split_seconds,
        budget_evaluations=budget_evaluations,
        dp_runs=dp_runs,
        reachable_states=final_result.reachable_states,
        evaluated_windows=final_result.evaluated_windows,
        epsilon=float(epsilon),
        scale=prepared.scale,
        lkh_runs=prepared.lkh_runs,
        drone_cost_mode=prepared.drone_cost_mode,
        drone_cost_multiplier=prepared.drone_cost_multiplier,
        drone_cost_factor=prepared.drone_cost_factor,
        model_budget_enabled=bool(apply_model_budget),
        active_depot_policy=policy_name,
        min_active_depots=minimum_active,
        active_depots=sum(bool(group) for group in groups.values()),
    )


def set_gtds_partition(
    depots,
    cities,
    candidate_sets,
    truck_distance,
    drone_distance,
    speed,
    epsilon=0.01,
    scale=1000,
    apply_model_budget=True,
    lkh_runs=10,
    min_active_depots=None,
    active_depot_policy='all',
    drone_cost_mode='paper',
):
    """
    执行单个 Directed Set-GTDS 变体的兼容入口。

    输入与旧公开函数一致，并新增最少活跃仓库数；内部先准备公共数据再求解指定
    变体，输出 ``SetGTDSResult``。配对实验应直接复用 ``prepare_set_gtds`` 结果。
    """

    prepared = prepare_set_gtds(
        depots=depots,
        cities=cities,
        candidate_sets=candidate_sets,
        truck_distance=truck_distance,
        drone_distance=drone_distance,
        speed=speed,
        scale=scale,
        lkh_runs=lkh_runs,
        drone_cost_mode=drone_cost_mode,
    )
    return solve_prepared_set_gtds(
        prepared,
        epsilon=epsilon,
        apply_model_budget=apply_model_budget,
        min_active_depots=min_active_depots,
        active_depot_policy=active_depot_policy,
    )
