"""P3：Set-TSP 可行上界和指派下界剪枝。"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import (
    CandidateSpace,
    ExternalKey,
    InternalKey,
    PruningEvidence,
    PruningOptions,
    PruningPreconditionError,
    PruningStageReport,
    SetTSPIncumbent,
)
from .validation import validate_cost_tensors, validate_incumbent, validate_pruning_options


def _assignment_cost(cost_matrix: np.ndarray) -> float:
    """求一入一出指派松弛的最小成本，不可行时返回正无穷。"""

    if cost_matrix.size == 0:
        return 0.0
    try:
        rows, columns = linear_sum_assignment(cost_matrix)
    except ValueError:
        return float('inf')
    selected = cost_matrix[rows, columns]
    if len(rows) != cost_matrix.shape[0] or not np.all(np.isfinite(selected)):
        return float('inf')
    return float(np.sum(selected))


def compute_assignment_lower_bound(min_external_cost: np.ndarray) -> float:
    """计算禁止集合自环但允许多个子环的集合级指派下界。"""

    matrix = np.array(min_external_cost, dtype=float, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('指派成本必须是方阵。')
    np.fill_diagonal(matrix, np.inf)
    return _assignment_cost(matrix)


def compute_forced_arc_remaining_bound(
    min_external_cost: np.ndarray,
    source: int,
    target: int,
) -> float:
    """固定集合弧 ``source→target`` 后计算其余行列的指派下界。

    返回值不包含固定弧本身的成本，便于用具体端点距离替换集合最小距离。
    """

    matrix = np.array(min_external_cost, dtype=float, copy=True)
    np.fill_diagonal(matrix, np.inf)
    if source == target:
        return float('inf')
    remaining = np.delete(np.delete(matrix, source, axis=0), target, axis=1)
    return _assignment_cost(remaining)


def _internal_options(
    space: CandidateSpace,
    set_index: int,
    internal_cost: Sequence,
) -> List[Tuple[InternalKey, float]]:
    """按稳定顺序返回一个集合当前可用的内部端点选择。"""

    keys = sorted(
        key
        for key in space.internal | space.fixed_internal_one
        if key[0] == set_index
    )
    return [(key, float(internal_cost[key[0]][key[1]][key[2]])) for key in keys]


def build_fixed_order_incumbent(
    space: CandidateSpace,
    distance: Sequence,
    internal_cost: Sequence,
    order: Optional[Sequence[int]] = None,
    tolerance: float = 1e-12,
) -> SetTSPIncumbent:
    """固定集合顺序后，用分层动态规划求最优端点组合。

    输入：候选空间、两类成本、可选集合排列和比较容差。
    输出：可行 Set-TSP 上界，包含成本以及所选内部/外部变量。

    逻辑：DP 状态是当前集合采用的内部端点对；相邻状态之间仅在存在
    对应外部端点弧时连接，最后补上末集合返回首集合的闭环弧。
    """

    validate_cost_tensors(space, distance, internal_cost)
    n = len(space.set_sizes)
    if n < 2:
        raise PruningPreconditionError('当前 Set-TSP 实现至少需要仓库和一个客户集合。')

    route_order = tuple(range(n)) if order is None else tuple(order)
    if len(route_order) != n or set(route_order) != set(range(n)) or route_order[0] != 0:
        raise PruningPreconditionError('上界顺序必须是从集合 0 开始的完整排列。')

    options = {i: _internal_options(space, i, internal_cost) for i in range(n)}
    if any(not choices for choices in options.values()):
        raise PruningPreconditionError('至少一个集合没有内部端点候选，无法构造上界。')

    best_cost = float('inf')
    best_path: Optional[Tuple[InternalKey, ...]] = None
    for first_key, first_cost in options[route_order[0]]:
        # 后续转移只依赖当前离开端点，因此相同离开端点仅保留一个最佳标签。
        layer: Dict[int, Tuple[float, Tuple[InternalKey, ...]]] = {
            first_key[2]: (first_cost, (first_key,))
        }
        for position in range(1, n):
            previous_set = route_order[position - 1]
            current_set = route_order[position]
            next_layer: Dict[int, Tuple[float, Tuple[InternalKey, ...]]] = {}
            for previous_exit, (prefix_cost, path) in layer.items():
                for current_key, current_cost in options[current_set]:
                    current_entry = current_key[1]
                    current_exit = current_key[2]
                    external_key = (previous_set, current_set, previous_exit, current_entry)
                    if external_key not in space.external:
                        continue
                    candidate_cost = (
                        prefix_cost
                        + float(distance[previous_set][current_set][previous_exit][current_entry])
                        + current_cost
                    )
                    candidate_path = path + (current_key,)
                    incumbent = next_layer.get(current_exit)
                    if (
                        incumbent is None
                        or candidate_cost < incumbent[0] - tolerance
                        or (
                            abs(candidate_cost - incumbent[0]) <= tolerance
                            and candidate_path < incumbent[1]
                        )
                    ):
                        next_layer[current_exit] = (candidate_cost, candidate_path)
            layer = next_layer
            if not layer:
                break

        if not layer:
            continue
        first_entry = first_key[1]
        last_set = route_order[-1]
        first_set = route_order[0]
        for last_exit, (prefix_cost, path) in layer.items():
            closing_key = (last_set, first_set, last_exit, first_entry)
            if closing_key not in space.external:
                continue
            total = prefix_cost + float(distance[last_set][first_set][last_exit][first_entry])
            if (
                total < best_cost - tolerance
                or (
                    abs(total - best_cost) <= tolerance
                    and (best_path is None or path < best_path)
                )
            ):
                best_cost = total
                best_path = path

    if best_path is None or not math.isfinite(best_cost):
        raise PruningPreconditionError('固定集合顺序下不存在可行端点闭环。')

    external_keys: List[ExternalKey] = []
    for position, current_key in enumerate(best_path):
        next_position = (position + 1) % n
        next_key = best_path[next_position]
        external_keys.append(
            (
                route_order[position],
                route_order[next_position],
                current_key[2],
                next_key[1],
            )
        )

    # 固定为 1 的内部变量不再是模型变量，因此只把真实变量写入 MIP Start。
    variable_internal = tuple(key for key in best_path if key in space.internal)
    incumbent = SetTSPIncumbent(
        cost=float(best_cost),
        order=route_order,
        internal=variable_internal,
        external=tuple(external_keys),
    )
    validate_incumbent(space, incumbent, distance, internal_cost, max(tolerance, 1e-9))
    return incumbent


def _minimum_cost_tables(
    space: CandidateSpace,
    distance: Sequence,
    internal_cost: Sequence,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算每个集合的内部最小值和每条集合弧的外部最小值。"""

    n = len(space.set_sizes)
    alpha = np.full(n, np.inf, dtype=float)
    for i, a, b in space.internal | space.fixed_internal_one:
        alpha[i] = min(alpha[i], float(internal_cost[i][a][b]))

    min_external = np.full((n, n), np.inf, dtype=float)
    for i, j, a, b in space.external:
        if i != j:
            min_external[i, j] = min(
                min_external[i, j],
                float(distance[i][j][a][b]),
            )
    return alpha, min_external


def _bound_prunes(
    lower_bound: float,
    upper_bound: float,
    options: PruningOptions,
) -> bool:
    """按“保留一个”或“保留全部最优解”的安全容差判断界剪枝。

    输入：包含指定候选的解的下界、当前可行上界和统一剪枝配置。
    输出：仅当下界位于上界容差带的劣侧时返回 ``True``。

    容差必须加在上界的安全侧；若使用 ``upper_bound - tolerance``，则
    可能删除仍有机会产生轻微更优解的候选。这里宁可保留容差带内的少量
    候选，也不允许浮点比较把精确更优解剪掉。
    """

    # 安全删除阈值：容差带内的候选一律保留，避免数值误差造成误剪。
    pruning_threshold = upper_bound + options.tolerance
    if options.preserve_all_optima:
        return lower_bound > pruning_threshold
    return lower_bound >= pruning_threshold


def apply_p3_assignment_bound_pruning(
    space: CandidateSpace,
    distance: Sequence,
    internal_cost: Sequence,
    options: PruningOptions,
) -> tuple[CandidateSpace, PruningStageReport]:
    """应用可行上界、内部最小值和集合指派松弛剪枝。

    输入：P1 或稠密基线候选空间、成本张量和统一配置。
    输出：P3 后候选空间与逐类指标。

    为保证采用 ``>=`` 时仍保留至少一个最优解，本函数永远保护用于产生
    上界的内部变量和外部端点弧。
    """

    started = time.perf_counter()
    validate_pruning_options(options, 'P3')
    validate_cost_tensors(space, distance, internal_cost)
    before = space.counts()
    incumbent = build_fixed_order_incumbent(
        space,
        distance,
        internal_cost,
        tolerance=options.tolerance,
    )
    validate_incumbent(space, incumbent, distance, internal_cost, options.tolerance)

    alpha, min_external = _minimum_cost_tables(space, distance, internal_cost)
    if not np.all(np.isfinite(alpha)):
        raise PruningPreconditionError('至少一个集合没有有限内部成本。')
    assignment_bound = compute_assignment_lower_bound(min_external)
    if not math.isfinite(assignment_bound):
        raise PruningPreconditionError('集合级指派松弛不可行，停止 P3。')

    result = space.copy()
    result.incumbent = incumbent
    evidence = []
    protected_internal = set(incumbent.internal)
    protected_external = set(incumbent.external)
    alpha_sum = float(np.sum(alpha))
    base_lower_bound = alpha_sum + assignment_bound
    if base_lower_bound > incumbent.cost + options.tolerance:
        raise PruningPreconditionError(
            '基础下界高于已验证可行上界，拒绝使用可能不安全的 P3 结果。'
        )

    removed_internal = set()
    for key in sorted(space.internal):
        i, a, b = key
        lower_bound = (
            float(internal_cost[i][a][b])
            + alpha_sum
            - float(alpha[i])
            + assignment_bound
        )
        if key in protected_internal and lower_bound > incumbent.cost + options.tolerance:
            raise PruningPreconditionError(
                f'内部候选 {key} 的下界与已知上界路线矛盾。'
            )
        if key not in protected_internal and _bound_prunes(lower_bound, incumbent.cost, options):
            removed_internal.add(key)
            if options.record_evidence:
                evidence.append(
                    PruningEvidence(
                        'p3',
                        'internal',
                        key,
                        'assignment_lower_bound',
                        bound=lower_bound,
                        threshold=incumbent.cost,
                    )
                )

    # 同一集合弧的剩余指派下界与具体端点无关，只计算一次。
    forced_remaining: Dict[Tuple[int, int], float] = {}
    external_lower_bounds: Dict[ExternalKey, float] = {}
    removed_external = set()
    for key in sorted(space.external):
        i, j, a, b = key
        if i == j:
            continue
        arc = (i, j)
        if arc not in forced_remaining:
            forced_remaining[arc] = compute_forced_arc_remaining_bound(min_external, i, j)
        lower_bound = (
            alpha_sum
            + float(distance[i][j][a][b])
            + forced_remaining[arc]
        )
        external_lower_bounds[key] = lower_bound
        if key in protected_external and lower_bound > incumbent.cost + options.tolerance:
            raise PruningPreconditionError(
                f'外部候选 {key} 的下界与已知上界路线矛盾。'
            )
        if key not in protected_external and _bound_prunes(lower_bound, incumbent.cost, options):
            removed_external.add(key)
            if options.record_evidence:
                evidence.append(
                    PruningEvidence(
                        'p3',
                        'external',
                        key,
                        'forced_arc_assignment_lower_bound',
                        bound=lower_bound,
                        threshold=incumbent.cost,
                        details={'remaining_assignment_bound': forced_remaining[arc]},
                    )
                )

    result.internal.difference_update(removed_internal)
    result.external.difference_update(removed_external)

    # 某条非自环集合弧已无端点实现时，它的 select/flow 也固定为 0。
    arcs_with_external = {(i, j) for i, j, _, _ in result.external}
    emptied_arcs = {
        arc
        for arc in result.select_arcs
        if arc[0] != arc[1] and arc not in arcs_with_external
    }
    if incumbent.select_arcs & emptied_arcs:
        raise PruningPreconditionError('P3 意外删除了上界使用的集合弧。')
    if options.record_evidence:
        for arc in sorted(emptied_arcs):
            arc_lower_bound = min(
                (
                    bound
                    for key, bound in external_lower_bounds.items()
                    if key[:2] == arc
                ),
                default=float('inf'),
            )
            evidence.append(
                PruningEvidence(
                    'p3',
                    'select',
                    arc,
                    'bound_emptied_set_arc',
                    bound=arc_lower_bound,
                    threshold=incumbent.cost,
                )
            )
            if arc in result.flow_arcs:
                evidence.append(
                    PruningEvidence(
                        'p3',
                        'flow',
                        arc,
                        'bound_emptied_set_arc',
                        bound=arc_lower_bound,
                        threshold=incumbent.cost,
                    )
                )
    result.select_arcs.difference_update(emptied_arcs)
    result.flow_arcs.difference_update(emptied_arcs)

    result.validate()
    validate_incumbent(result, incumbent, distance, internal_cost, options.tolerance)
    after = result.counts()
    removed_by_reason = {
        'select': {'bound_emptied_set_arc': len(emptied_arcs)},
        'flow': {
            'bound_emptied_set_arc': before['flow'] - after['flow'],
        },
        'internal': {'assignment_lower_bound': len(removed_internal)},
        'external': {'forced_arc_assignment_lower_bound': len(removed_external)},
    }
    report = PruningStageReport(
        stage='p3',
        before=before,
        after=after,
        removed_by_reason=removed_by_reason,
        metrics={
            'phase2_incumbent_cost': incumbent.cost,
            'phase2_assignment_lower_bound': base_lower_bound,
            'phase2_internal_bound_pruned': len(removed_internal),
            'phase2_external_bound_pruned': len(removed_external),
            'phase2_set_arcs_bound_pruned': len(emptied_arcs),
            'phase2_bound_preprocessing_seconds': time.perf_counter() - started,
        },
        evidence=evidence,
    )
    report.validate_conservation()
    return result, report
