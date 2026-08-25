"""
实现 P7：Set-TSP 外部端点对支配剪枝。

该实现只依赖已经构造好的集合内部成本和集合间成本，不修改原算法对象，也不依赖 Gurobi。
"""

import math
import time
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .types import (
    DominanceEvidence,
    EndpointDominanceResult,
    EndpointPair,
    PruningMetrics,
    PruningOptions,
    SetArc,
)


def _all_endpoint_pairs(external_costs):
    """
    根据外部成本矩阵构造每条有向集合弧的完整端点候选。

    输入：键为 `(i,j)`、值为二维外部成本矩阵的映射。
    输出：键相同、值为稳定局部索引端点对元组的字典。
    逻辑：按源端点优先、目标端点次优先的顺序生成，作为等价候选的稳定顺序。
    """

    result = {}
    for set_arc in sorted(external_costs):
        rows, columns = np.asarray(external_costs[set_arc]).shape
        result[set_arc] = tuple(
            (source_endpoint, target_endpoint)
            for source_endpoint in range(rows)
            for target_endpoint in range(columns)
        )
    return result


def _validate_costs(internal_costs, external_costs):
    """
    校验 P7 使用的成本矩阵维度、集合弧覆盖和数值有限性。

    输入：每个集合的内部成本矩阵及每条有向集合弧的外部成本矩阵。
    输出：合法时返回空字符串，否则返回可记录的安全回退原因。
    逻辑：任一输入异常都让整个仓库组退回完整候选，避免部分错误索引造成误剪。
    """

    set_count = len(internal_costs)
    if set_count < 2:
        return 'Set-TSP 至少需要仓库集合和一个客户集合。'

    set_sizes = []
    for set_index, matrix in enumerate(internal_costs):
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[0] != values.shape[1]:
            return f'集合 {set_index} 的内部成本矩阵必须为非空方阵。'
        if not np.all(np.isfinite(values)):
            return f'集合 {set_index} 的内部成本矩阵包含非有限值。'
        set_sizes.append(values.shape[0])

    expected_arcs = {
        (source_set, target_set)
        for source_set in range(set_count)
        for target_set in range(set_count)
        if source_set != target_set
    }
    if set(external_costs) != expected_arcs:
        return '外部成本矩阵没有完整覆盖全部 i != j 的有向集合弧。'

    for (source_set, target_set), matrix in external_costs.items():
        values = np.asarray(matrix, dtype=float)
        expected_shape = (set_sizes[source_set], set_sizes[target_set])
        if values.shape != expected_shape:
            return (
                f'集合弧 {(source_set, target_set)} 的外部成本维度 '
                f'{values.shape} 与预期 {expected_shape} 不一致。'
            )
        if not np.all(np.isfinite(values)):
            return f'集合弧 {(source_set, target_set)} 的外部成本包含非有限值。'
    return ''


def _build_fallback_result(external_costs, options, start, reason):
    """
    在输入异常时构造不删除任何候选的安全回退结果。

    输入：外部成本、配置、计时起点和回退原因。
    输出：保留全部候选且指标守恒的 `EndpointDominanceResult`。
    逻辑：回退只影响剪枝收益，不改变 Set-TSP 可行域。
    """

    kept_pairs = _all_endpoint_pairs(external_costs)
    before = sum(len(pairs) for pairs in kept_pairs.values())
    metrics = PruningMetrics(
        endpoint_pair_dominance_enabled=options.endpoint_pair_dominance,
        endpoint_pairs_before=before,
        endpoint_pairs_kept=before,
        endpoint_dominance_seconds=time.perf_counter() - start,
        set_arcs_before=len(kept_pairs),
        set_arcs_after=len(kept_pairs),
        safe_fallback=True,
        fallback_reason=reason,
    )
    metrics.assert_conservation()
    return EndpointDominanceResult(kept_pairs, metrics, [])


def _comparison_budget_remaining(options, checks, start):
    """
    判断支配比较预算是否仍允许继续展开一个比较块。

    输入：剪枝配置、已完成比较数和预处理起始时刻。
    输出：预算仍可用时为 `True`，否则为 `False`。
    逻辑：比较次数和墙钟时间任一耗尽即停止，未检查候选保持不变。
    """

    if options.max_pair_checks is not None and checks >= options.max_pair_checks:
        return False
    if options.max_preprocessing_seconds is not None:
        if time.perf_counter() - start >= options.max_preprocessing_seconds:
            return False
    return True


def _strict_tolerance(options, source_delta, external_delta, target_delta):
    """
    计算严格支配判断使用的尺度相关保守容差。

    输入：配置及支配不等式的三项差分。
    输出：非负绝对容差。
    逻辑：成本尺度越大，要求负裕量越明显；灰区候选一律保留。
    """

    scale = max(
        1.0,
        abs(float(source_delta)),
        abs(float(external_delta)),
        abs(float(target_delta)),
    )
    return options.dominance_epsilon * scale


def _is_exactly_equivalent(
    source_internal,
    target_internal,
    external_matrix,
    dominator,
    victim,
):
    """
    检查两个候选在全部前驱进入点和后继离开点上是否逐项完全等价。

    输入：两个集合的内部成本、外部成本矩阵、候选支配者和被比较候选。
    输出：全部局部总成本按浮点值精确相等时为 `True`。
    逻辑：只对接近零的候选调用；不使用近似相等，防止把微小真实差异误判为等价。
    """

    dominator_source, dominator_target = dominator
    victim_source, victim_target = victim
    source_differences = (
        source_internal[:, dominator_source]
        - source_internal[:, victim_source]
    )
    target_differences = (
        target_internal[dominator_target, :]
        - target_internal[victim_target, :]
    )
    external_difference = (
        external_matrix[dominator_source, dominator_target]
        - external_matrix[victim_source, victim_target]
    )
    full_difference = (
        source_differences[:, None]
        + external_difference
        + target_differences[None, :]
    )
    return bool(np.all(full_difference == 0.0))


def _resolve_terminal_dominator(parent_by_candidate, candidate):
    """
    沿支配父指针找到最终保留代表，并检测意外循环。

    输入：候选到直接支配者的映射及起始候选。
    输出：支配链末端候选。
    逻辑：稳定等价规则和严格支配理论上不会成环；检测用于阻止数值或实现错误进入实验。
    """

    visited = set()
    current = candidate
    while current in parent_by_candidate:
        if current in visited:
            raise RuntimeError(f'端点对支配证据出现循环：{candidate}。')
        visited.add(current)
        current = parent_by_candidate[current]
    return current


def prune_endpoint_pairs(
    internal_costs: Sequence[np.ndarray],
    external_costs: Mapping[SetArc, np.ndarray],
    options: PruningOptions,
) -> EndpointDominanceResult:
    """
    对每条 Set-TSP 有向集合弧执行 P7 外部端点对支配剪枝。

    输入：集合内部成本方阵序列、集合间外部成本矩阵映射和剪枝配置。
    输出：每条集合弧的保留端点对、统一指标和可选审计证据。
    逻辑：预计算进入/离开差分上界，再分块枚举潜在支配者；严格负裕量或稳定完全等价才删除。
    """

    options.validate()
    start = time.perf_counter()
    external_arrays = {
        set_arc: np.asarray(matrix, dtype=float)
        for set_arc, matrix in external_costs.items()
    }
    internal_arrays = [np.asarray(matrix, dtype=float) for matrix in internal_costs]

    validation_error = _validate_costs(internal_arrays, external_arrays)
    if validation_error:
        return _build_fallback_result(external_arrays, options, start, validation_error)

    all_pairs_by_arc = _all_endpoint_pairs(external_arrays)
    before = sum(len(pairs) for pairs in all_pairs_by_arc.values())
    metrics = PruningMetrics(
        endpoint_pair_dominance_enabled=options.endpoint_pair_dominance,
        endpoint_pairs_before=before,
        endpoint_pairs_kept=before,
        set_arcs_before=len(all_pairs_by_arc),
        set_arcs_after=len(all_pairs_by_arc),
    )
    if not options.endpoint_pair_dominance:
        metrics.endpoint_dominance_seconds = time.perf_counter() - start
        metrics.assert_conservation()
        return EndpointDominanceResult(all_pairs_by_arc, metrics, [])

    # exit_delta[a_prime, a] 对应 max_x(g[x,a_prime]-g[x,a])。
    exit_deltas = [
        np.max(matrix[:, :, None] - matrix[:, None, :], axis=0)
        for matrix in internal_arrays
    ]
    # entry_delta[b_prime, b] 对应 max_z(g[b_prime,z]-g[b,z])。
    entry_deltas = [
        np.max(matrix[:, None, :] - matrix[None, :, :], axis=2)
        for matrix in internal_arrays
    ]

    kept_pairs_by_arc: Dict[SetArc, Tuple[EndpointPair, ...]] = {}
    evidence = []
    total_checks = 0
    budget_exhausted = False

    for set_arc in sorted(all_pairs_by_arc):
        source_set, target_set = set_arc
        external_matrix = external_arrays[set_arc]
        candidates = all_pairs_by_arc[set_arc]
        parent_by_candidate: Dict[EndpointPair, EndpointPair] = {}
        evidence_by_victim = {}
        dominance_kind_by_victim = {}

        # 已经获得直接支配证据的 victim 无需再次寻找支配者，但仍可作为其他候选的中间支配者。
        for victim_index, victim in enumerate(candidates):
            if victim in parent_by_candidate:
                continue
            if not _comparison_budget_remaining(options, total_checks, start):
                budget_exhausted = True
                break

            victim_source, victim_target = victim
            found = False
            for block_start in range(0, len(candidates), options.comparison_block_size):
                if not _comparison_budget_remaining(options, total_checks, start):
                    budget_exhausted = True
                    break

                block_end = min(block_start + options.comparison_block_size, len(candidates))
                if options.max_pair_checks is not None:
                    remaining = options.max_pair_checks - total_checks
                    block_end = min(block_end, block_start + remaining)
                if block_end <= block_start:
                    budget_exhausted = True
                    break

                block = candidates[block_start:block_end]
                source_indices = np.fromiter((pair[0] for pair in block), dtype=int)
                target_indices = np.fromiter((pair[1] for pair in block), dtype=int)
                source_terms = exit_deltas[source_set][source_indices, victim_source]
                external_terms = (
                    external_matrix[source_indices, target_indices]
                    - external_matrix[victim_source, victim_target]
                )
                target_terms = entry_deltas[target_set][target_indices, victim_target]
                margins = source_terms + external_terms + target_terms
                total_checks += len(block)

                for offset, dominator in enumerate(block):
                    if dominator == victim:
                        continue
                    source_delta = float(source_terms[offset])
                    external_delta = float(external_terms[offset])
                    target_delta = float(target_terms[offset])
                    margin = float(margins[offset])
                    tolerance = _strict_tolerance(
                        options,
                        source_delta,
                        external_delta,
                        target_delta,
                    )

                    dominance_kind = None
                    if margin < -tolerance:
                        dominance_kind = 'strict'
                    elif dominator < victim and math.isfinite(margin) and abs(margin) <= tolerance:
                        if _is_exactly_equivalent(
                            internal_arrays[source_set],
                            internal_arrays[target_set],
                            external_matrix,
                            dominator,
                            victim,
                        ):
                            dominance_kind = 'equivalent'

                    if dominance_kind is None:
                        continue

                    parent_by_candidate[victim] = dominator
                    dominance_kind_by_victim[victim] = dominance_kind
                    if dominance_kind == 'strict':
                        metrics.endpoint_pairs_strictly_dominated += 1
                    else:
                        metrics.endpoint_pairs_equivalent_dominated += 1
                    if options.record_evidence:
                        record = DominanceEvidence(
                            set_arc=set_arc,
                            victim=victim,
                            dominator=dominator,
                            source_internal_delta=source_delta,
                            external_delta=external_delta,
                            target_internal_delta=target_delta,
                            margin=margin,
                            dominance_kind=dominance_kind,
                        )
                        evidence.append(record)
                        evidence_by_victim[victim] = record
                    found = True
                    break
                if found or budget_exhausted:
                    break
            if budget_exhausted:
                break

        kept = tuple(candidate for candidate in candidates if candidate not in parent_by_candidate)
        if not kept:
            # 理论上严格支配与稳定等价规则不会清空集合弧；若发生则整条弧恢复，禁止误删。
            metrics.set_arcs_emptied_by_dominance += 1
            metrics.set_arcs_restored_by_safety += 1
            metrics.endpoint_pairs_strictly_dominated -= sum(
                kind == 'strict' for kind in dominance_kind_by_victim.values()
            )
            metrics.endpoint_pairs_equivalent_dominated -= sum(
                kind == 'equivalent' for kind in dominance_kind_by_victim.values()
            )
            evidence = [record for record in evidence if record.set_arc != set_arc]
            parent_by_candidate.clear()
            kept = candidates

        if options.record_evidence:
            for victim, record in evidence_by_victim.items():
                if victim in parent_by_candidate:
                    record.terminal_dominator = _resolve_terminal_dominator(
                        parent_by_candidate,
                        victim,
                    )
        kept_pairs_by_arc[set_arc] = kept
        if budget_exhausted:
            # 后续集合弧未经检查，全部候选原样保留。
            remaining_arcs = [arc for arc in sorted(all_pairs_by_arc) if arc not in kept_pairs_by_arc]
            for remaining_arc in remaining_arcs:
                kept_pairs_by_arc[remaining_arc] = all_pairs_by_arc[remaining_arc]
            break

    metrics.endpoint_pair_checks = total_checks
    metrics.endpoint_pairs_kept = sum(len(pairs) for pairs in kept_pairs_by_arc.values())
    metrics.set_arcs_after = sum(bool(pairs) for pairs in kept_pairs_by_arc.values())
    metrics.budget_exhausted = budget_exhausted
    metrics.endpoint_dominance_seconds = time.perf_counter() - start
    metrics.assert_conservation()
    return EndpointDominanceResult(kept_pairs_by_arc, metrics, evidence)
