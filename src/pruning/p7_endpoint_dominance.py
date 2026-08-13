"""P7：Set-TSP 外部端点对支配剪枝。"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Sequence, Set

from .types import (
    CandidateSpace,
    ExternalKey,
    InternalKey,
    PruningEvidence,
    PruningOptions,
    PruningPreconditionError,
    PruningStageReport,
)
from .validation import validate_cost_tensors, validate_incumbent, validate_pruning_options


def _available_internal(space: CandidateSpace) -> set[InternalKey]:
    """返回变量候选和固定为 1 候选的并集。"""

    return space.internal | space.fixed_internal_one


def endpoint_dominance_delta(
    space: CandidateSpace,
    candidate: ExternalKey,
    challenger: ExternalKey,
    distance: Sequence,
    internal_cost: Sequence,
) -> float | None:
    """计算挑战者相对候选的最坏局部成本差。

    输入：同一集合弧上的原候选、挑战者和成本张量。
    输出：若所有需要的替代内部端点仍存在，则返回最大成本差；否则返回
    ``None``，表示在当前稀疏候选空间中不能证明支配。

    这一“替代端点存在性”检查使 P7 能安全接在 P3 后执行。
    """

    i, j, a, b = candidate
    ci, cj, alternative_a, alternative_b = challenger
    if (i, j) != (ci, cj) or candidate == challenger:
        return None

    available = _available_internal(space)
    # 只需考虑原候选可能衔接的进入点 x 和离开点 z。
    entries = sorted(x for x in range(space.set_sizes[i]) if (i, x, a) in available)
    exits = sorted(z for z in range(space.set_sizes[j]) if (j, b, z) in available)
    if not entries or not exits:
        return None
    if any((i, x, alternative_a) not in available for x in entries):
        return None
    if any((j, alternative_b, z) not in available for z in exits):
        return None

    left_delta = max(
        float(internal_cost[i][x][alternative_a]) - float(internal_cost[i][x][a])
        for x in entries
    )
    right_delta = max(
        float(internal_cost[j][alternative_b][z]) - float(internal_cost[j][b][z])
        for z in exits
    )
    external_delta = (
        float(distance[i][j][alternative_a][alternative_b])
        - float(distance[i][j][a][b])
    )
    return left_delta + external_delta + right_delta


def _build_delta_tables(
    space: CandidateSpace,
    source: int,
    target: int,
    internal_cost: Sequence,
):
    """预计算一个集合弧两端的最坏内部成本差。

    输入：候选空间、有向集合弧两端和内部成本。
    输出：``left[(a',a)]`` 与 ``right[(b',b)]``；值为 ``None`` 时说明
    替代端点无法覆盖原端点的全部合法衔接状态。

    逻辑：把 P7 对所有进入点 ``x``、离开点 ``z`` 的最大值预先压缩，
    后续每对外部端点候选只需常数次查表。
    """

    available = _available_internal(space)
    source_size = space.set_sizes[source]
    target_size = space.set_sizes[target]
    left = {}
    right = {}

    for original_exit in range(source_size):
        entries = [
            entry
            for entry in range(source_size)
            if (source, entry, original_exit) in available
        ]
        for alternative_exit in range(source_size):
            key = (alternative_exit, original_exit)
            if not entries or any(
                (source, entry, alternative_exit) not in available
                for entry in entries
            ):
                left[key] = None
            else:
                left[key] = max(
                    float(internal_cost[source][entry][alternative_exit])
                    - float(internal_cost[source][entry][original_exit])
                    for entry in entries
                )

    for original_entry in range(target_size):
        exits = [
            exit_
            for exit_ in range(target_size)
            if (target, original_entry, exit_) in available
        ]
        for alternative_entry in range(target_size):
            key = (alternative_entry, original_entry)
            if not exits or any(
                (target, alternative_entry, exit_) not in available
                for exit_ in exits
            ):
                right[key] = None
            else:
                right[key] = max(
                    float(internal_cost[target][alternative_entry][exit_])
                    - float(internal_cost[target][original_entry][exit_])
                    for exit_ in exits
                )
    return left, right


def _table_delta(
    candidate: ExternalKey,
    challenger: ExternalKey,
    distance: Sequence,
    left,
    right,
) -> float | None:
    """使用预计算差分表返回挑战者相对原候选的最坏成本差。

    输入：同一集合弧的两个端点候选、外部成本和两张差分表。
    输出：支配差值；替代端点不覆盖全部衔接状态时返回 ``None``。
    """

    i, j, a, b = candidate
    _, _, alternative_a, alternative_b = challenger
    left_delta = left[(alternative_a, a)]
    right_delta = right[(alternative_b, b)]
    if left_delta is None or right_delta is None:
        return None
    return (
        left_delta
        + float(distance[i][j][alternative_a][alternative_b])
        - float(distance[i][j][a][b])
        + right_delta
    )


def _resolve_final_witness(
    candidate: ExternalKey,
    replacements: Dict[ExternalKey, Set[ExternalKey]],
    protected: Set[ExternalKey],
    memo: Dict[ExternalKey, ExternalKey],
    visiting: Set[ExternalKey],
) -> ExternalKey:
    """沿支配图寻找最终幸存者，并拒绝任何数值导致的支配环。

    输入：当前候选、候选到挑战者的有向边、受保护上界候选和 DFS 状态。
    输出：一个最终保留的稳定见证；若支配关系成环则抛出安全回退异常。
    """

    if candidate in memo:
        return memo[candidate]
    if candidate in protected or not replacements[candidate]:
        memo[candidate] = candidate
        return candidate
    if candidate in visiting:
        raise PruningPreconditionError(f'P7 检测到端点支配环：{candidate}')

    visiting.add(candidate)
    final_witnesses = {
        _resolve_final_witness(challenger, replacements, protected, memo, visiting)
        for challenger in replacements[candidate]
    }
    visiting.remove(candidate)
    if not final_witnesses:
        raise PruningPreconditionError(f'P7 候选 {candidate} 没有最终幸存见证。')
    memo[candidate] = min(final_witnesses)
    return memo[candidate]


def apply_p7_endpoint_dominance_pruning(
    space: CandidateSpace,
    distance: Sequence,
    internal_cost: Sequence,
    options: PruningOptions,
) -> tuple[CandidateSpace, PruningStageReport]:
    """删除同一有向集合弧上被另一个端点对支配的候选。

    P3 上界使用的外部弧会被保护。容差只用于要求严格支配具有足够负裕量，
    不允许成本略高的挑战者借助容差删除候选。
    """

    started = time.perf_counter()
    validate_pruning_options(options, 'P7')
    validate_cost_tensors(space, distance, internal_cost)
    before = space.counts()
    result = space.copy()
    protected = set(space.incumbent.external) if space.incumbent is not None else set()

    by_set_arc = defaultdict(list)
    for key in space.external:
        if key[0] != key[1]:
            by_set_arc[(key[0], key[1])].append(key)

    removed: Set[ExternalKey] = set()
    evidence = []
    for arc in sorted(by_set_arc):
        candidates = sorted(by_set_arc[arc])
        left, right = _build_delta_tables(
            space,
            arc[0],
            arc[1],
            internal_cost,
        )

        # replacement[c] 存放能够无成本恶化地替换候选 c 的挑战者。
        replacements: Dict[ExternalKey, Set[ExternalKey]] = {
            candidate: set() for candidate in candidates
        }
        for candidate in candidates:
            if candidate in protected:
                continue
            for challenger in candidates:
                if candidate == challenger:
                    continue
                delta = _table_delta(
                    candidate,
                    challenger,
                    distance,
                    left,
                    right,
                )
                if delta is None:
                    continue
                if options.preserve_all_optima:
                    # 保留全部最优解时，只有对所有局部上下文都明显严格更优才删除。
                    dominates = delta < -options.tolerance
                else:
                    # “保留一个最优解”也绝不允许正成本差；完全等价时按稳定键定向。
                    if delta > 0.0:
                        dominates = False
                    else:
                        reverse_delta = _table_delta(
                            challenger,
                            candidate,
                            distance,
                            left,
                            right,
                        )
                        mutually_equivalent = (
                            reverse_delta is not None
                            and reverse_delta <= 0.0
                        )
                        dominates = challenger < candidate if mutually_equivalent else True
                if dominates:
                    replacements[candidate].add(challenger)

        # 先建立完整支配图，再统一解析到最终幸存者，避免已删除候选相互删除。
        final_witness: Dict[ExternalKey, ExternalKey] = {}
        for candidate in candidates:
            _resolve_final_witness(
                candidate,
                replacements,
                protected,
                final_witness,
                set(),
            )
        arc_removed = {
            candidate
            for candidate in candidates
            if final_witness[candidate] != candidate
        }
        arc_survivors = set(candidates) - arc_removed
        if not arc_survivors:
            raise PruningPreconditionError(f'P7 意外删空集合弧 {arc}。')

        for candidate in sorted(arc_removed):
            witness = final_witness[candidate]
            if witness not in arc_survivors:
                raise PruningPreconditionError(
                    f'P7 候选 {candidate} 的最终见证 {witness} 未幸存。'
                )
            final_delta = _table_delta(
                candidate,
                witness,
                distance,
                left,
                right,
            )
            if final_delta is None or final_delta > 0.0:
                raise PruningPreconditionError(
                    f'P7 候选 {candidate} 到最终见证 {witness} 的成本证明无效。'
                )
            removed.add(candidate)
            if not options.record_evidence:
                continue
            evidence.append(
                PruningEvidence(
                    'p7',
                    'external',
                    candidate,
                    'endpoint_pair_dominance',
                    witness=witness,
                    bound=final_delta,
                    threshold=0.0,
                )
            )

    result.external.difference_update(removed)
    before_nonempty_arcs = set(by_set_arc)
    after_nonempty_arcs = {(i, j) for i, j, _, _ in result.external if i != j}
    newly_emptied_arcs = before_nonempty_arcs - after_nonempty_arcs
    if newly_emptied_arcs:
        raise PruningPreconditionError(
            f'P7 意外删空集合弧：{sorted(newly_emptied_arcs)}'
        )

    result.validate()
    if result.incumbent is not None:
        validate_incumbent(
            result,
            result.incumbent,
            distance,
            internal_cost,
            options.tolerance,
        )
    after = result.counts()
    removed_by_reason = {
        'select': {},
        'flow': {},
        'internal': {},
        'external': {'endpoint_pair_dominance': len(removed)},
    }
    report = PruningStageReport(
        stage='p7',
        before=before,
        after=after,
        removed_by_reason=removed_by_reason,
        metrics={
            'phase2_endpoint_pairs_before': sum(len(items) for items in by_set_arc.values()),
            'phase2_endpoint_pairs_dominated': len(removed),
            'phase2_endpoint_dominance_seconds': time.perf_counter() - started,
            'phase2_set_arcs_emptied_by_dominance': 0,
        },
        evidence=evidence,
    )
    report.validate_conservation()
    return result, report
