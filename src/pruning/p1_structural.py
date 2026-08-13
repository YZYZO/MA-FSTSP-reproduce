"""P1：删除 Set-TSP 中结构上已固定的变量。"""

from __future__ import annotations

from .types import CandidateSpace, PruningEvidence, PruningStageReport


def build_p1_structural_space(
    set_sizes: tuple[int, ...],
    record_evidence: bool = False,
) -> tuple[CandidateSpace, PruningStageReport]:
    """直接构造 P1 空间及报告，不物化结构上必删的稠密索引。

    输入：集合规模和证据开关。
    输出：结构稀疏候选空间与等价的 P1 计数报告。
    """

    result = CandidateSpace.structural(set_sizes)
    result.validate()
    before = CandidateSpace.dense_counts(set_sizes)
    after = result.counts()
    evidence = []
    if record_evidence:
        n = len(set_sizes)
        evidence.extend(
            PruningEvidence('p1', 'select', (i, i), 'structural_self_arc_zero')
            for i in range(n)
        )
        evidence.extend(
            PruningEvidence('p1', 'flow', (i, i), 'structural_self_arc_zero')
            for i in range(n)
        )
        evidence.extend(
            PruningEvidence('p1', 'external', (i, i, a, b), 'structural_self_arc_zero')
            for i, size in enumerate(set_sizes)
            for a in range(size)
            for b in range(size)
        )
        evidence.extend(
            PruningEvidence('p1', 'internal', key, 'structural_fixed_one')
            for key in sorted(result.fixed_internal_one)
        )

    external_skipped = before['external'] - after['external']
    fixed_internal = len(result.fixed_internal_one)
    removed_by_reason = {
        'select': {'structural_self_arc_zero': before['select'] - after['select']},
        'flow': {'structural_self_arc_zero': before['flow'] - after['flow']},
        'internal': {'structural_fixed_one': fixed_internal},
        'external': {'structural_self_arc_zero': external_skipped},
    }
    report = PruningStageReport(
        stage='p1',
        before=before,
        after=after,
        removed_by_reason=removed_by_reason,
        metrics={
            'phase2_structural_variables_skipped': (
                (before['select'] - after['select'])
                + (before['flow'] - after['flow'])
                + external_skipped
                + fixed_internal
            ),
            'phase2_external_self_arc_variables_skipped': external_skipped,
            'phase2_fixed_internal_variables': fixed_internal,
        },
        evidence=evidence,
    )
    report.validate_conservation()
    return result, report


def apply_p1_structural_pruning(
    space: CandidateSpace,
    record_evidence: bool = False,
) -> tuple[CandidateSpace, PruningStageReport]:
    """删除自环变量，并把单节点集合的内部选择代入常数 1。

    输入：稠密或部分稀疏的候选空间，以及是否记录逐变量证据。
    输出：新的候选空间和 P1 报告；输入对象不会被原地修改。

    逻辑：原模型已通过约束固定 ``select[i,i]``、``flow[i,i]`` 和对应
    ``external`` 为 0；单节点集合的唯一内部变量则被固定为 1。
    """

    space.validate()
    before = space.counts()
    result = space.copy()
    evidence = []

    self_select = {(i, i) for i in range(len(space.set_sizes))} & result.select_arcs
    self_flow = {(i, i) for i in range(len(space.set_sizes))} & result.flow_arcs
    self_external = {key for key in result.external if key[0] == key[1]}

    result.select_arcs.difference_update(self_select)
    result.flow_arcs.difference_update(self_flow)
    result.external.difference_update(self_external)

    # 任意单节点集合的唯一内部选择都由“恰好选择一次”约束固定为 1。
    fixed_internal = {
        (i, 0, 0)
        for i, size in enumerate(result.set_sizes)
        if size == 1 and (i, 0, 0) in result.internal
    }
    result.internal.difference_update(fixed_internal)
    result.fixed_internal_one.update(fixed_internal)

    if record_evidence:
        evidence.extend(
            PruningEvidence('p1', 'select', key, 'structural_self_arc_zero')
            for key in sorted(self_select)
        )
        evidence.extend(
            PruningEvidence('p1', 'flow', key, 'structural_self_arc_zero')
            for key in sorted(self_flow)
        )
        evidence.extend(
            PruningEvidence('p1', 'external', key, 'structural_self_arc_zero')
            for key in sorted(self_external)
        )
        evidence.extend(
            PruningEvidence('p1', 'internal', key, 'structural_fixed_one')
            for key in sorted(fixed_internal)
        )

    result.validate()
    after = result.counts()
    removed_by_reason = {
        'select': {'structural_self_arc_zero': len(self_select)},
        'flow': {'structural_self_arc_zero': len(self_flow)},
        'internal': {'structural_fixed_one': len(fixed_internal)},
        'external': {'structural_self_arc_zero': len(self_external)},
    }
    report = PruningStageReport(
        stage='p1',
        before=before,
        after=after,
        removed_by_reason=removed_by_reason,
        metrics={
            'phase2_structural_variables_skipped': (
                len(self_select) + len(self_flow) + len(self_external) + len(fixed_internal)
            ),
            'phase2_external_self_arc_variables_skipped': len(self_external),
            'phase2_fixed_internal_variables': len(fixed_internal),
        },
        evidence=evidence,
    )
    report.validate_conservation()
    return result, report
