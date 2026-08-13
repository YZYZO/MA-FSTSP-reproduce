"""P1、P3、P7 的可配置顺序流水线。"""

from __future__ import annotations

from typing import Sequence

from .p1_structural import build_p1_structural_space
from .p3_assignment_bound import apply_p3_assignment_bound_pruning
from .p7_endpoint_dominance import apply_p7_endpoint_dominance_pruning
from .types import (
    CandidateSpace,
    PruningOptions,
    PruningPipelineReport,
    PruningPreconditionError,
    PruningStageReport,
)
from .validation import validate_cost_tensors


def _fallback_report(stage: str, space: CandidateSpace, reason: Exception) -> PruningStageReport:
    """构造“不应用当前剪枝”的安全回退报告。"""

    counts = space.counts()
    return PruningStageReport(
        stage=stage,
        before=counts,
        after=dict(counts),
        removed_by_reason={
            'select': {},
            'flow': {},
            'internal': {},
            'external': {},
        },
        metrics={f'{stage}_applied': False},
        applied=False,
        fallback_reason=f'{type(reason).__name__}: {reason}',
    )


def apply_pruning_pipeline(
    set_sizes: Sequence[int],
    distance: Sequence,
    internal_cost: Sequence,
    options: PruningOptions | None = None,
) -> tuple[CandidateSpace, PruningPipelineReport]:
    """按 P1→P3→P7 顺序构造最终候选空间。

    输入：集合规模、成本张量和配置；配置为空时所有剪枝关闭。
    输出：最终候选空间和完整流水线报告。

    每个预处理器都在副本上工作。若前提或数值检查失败，仅回退该阶段，
    不会把部分删除结果传给后续模型。
    """

    selected_options = options or PruningOptions()
    normalized_sizes = tuple(int(size) for size in set_sizes)
    initial_counts = CandidateSpace.dense_counts(normalized_sizes)
    reports = []

    if selected_options.structural_stsp:
        try:
            current, report = build_p1_structural_space(
                normalized_sizes,
                record_evidence=selected_options.record_evidence,
            )
            validate_cost_tensors(current, distance, internal_cost)
        except (PruningPreconditionError, ValueError, ArithmeticError) as exc:
            current = CandidateSpace.dense(normalized_sizes)
            report = _fallback_report('p1', current, exc)
        reports.append(report)
    else:
        current = CandidateSpace.dense(normalized_sizes)
        validate_cost_tensors(current, distance, internal_cost)

    if selected_options.assignment_bound:
        try:
            current, report = apply_p3_assignment_bound_pruning(
                current,
                distance,
                internal_cost,
                selected_options,
            )
        except (PruningPreconditionError, ValueError, ArithmeticError) as exc:
            report = _fallback_report('p3', current, exc)
        reports.append(report)

    if selected_options.endpoint_pair_dominance:
        try:
            current, report = apply_p7_endpoint_dominance_pruning(
                current,
                distance,
                internal_cost,
                selected_options,
            )
        except (PruningPreconditionError, ValueError, ArithmeticError) as exc:
            report = _fallback_report('p7', current, exc)
        reports.append(report)

    current.validate()
    pipeline_report = PruningPipelineReport(
        options=selected_options,
        initial_counts=initial_counts,
        final_counts=current.counts(),
        stages=reports,
    )
    pipeline_report.validate_conservation()
    return current, pipeline_report
