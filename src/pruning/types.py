"""剪枝模块共享的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# 以下键均使用候选集合中的局部下标，避免依赖路网节点标签是否连续。
SetArc = Tuple[int, int]
InternalKey = Tuple[int, int, int]
ExternalKey = Tuple[int, int, int, int]


@dataclass(frozen=True)
class SetTSPIncumbent:
    """保存一个经过验证的 Set-TSP 可行上界及其变量选择。

    输入由上界构造器生成；输出供 P3/P7 保护至少一个可行解，并可作为
    Gurobi MIP Start 使用。
    """

    cost: float
    order: Tuple[int, ...]
    internal: Tuple[InternalKey, ...]
    external: Tuple[ExternalKey, ...]

    @property
    def select_arcs(self) -> Set[SetArc]:
        """返回上界路线使用的集合弧集合。"""

        return {(i, j) for i, j, _, _ in self.external}


@dataclass
class CandidateSpace:
    """表示当前 Set-TSP 中仍允许创建的变量索引。

    ``fixed_internal_one`` 存放被 P1 代入常数 1 的单节点集合内部变量。
    其他未出现在集合中的变量统一解释为固定为 0。
    """

    set_sizes: Tuple[int, ...]
    select_arcs: Set[SetArc]
    flow_arcs: Set[SetArc]
    internal: Set[InternalKey]
    external: Set[ExternalKey]
    fixed_internal_one: Set[InternalKey] = field(default_factory=set)
    incumbent: Optional[SetTSPIncumbent] = None

    @classmethod
    def dense(cls, set_sizes: Tuple[int, ...]) -> 'CandidateSpace':
        """按照原始实现创建完整稠密候选空间。

        输入：每个集合包含的节点数。
        输出：包含自环以及全部端点组合的候选空间。
        """

        n = len(set_sizes)
        select_arcs = {(i, j) for i in range(n) for j in range(n)}
        internal = {
            (i, a, b)
            for i, size in enumerate(set_sizes)
            for a in range(size)
            for b in range(size)
        }
        external = {
            (i, j, a, b)
            for i, left_size in enumerate(set_sizes)
            for j, right_size in enumerate(set_sizes)
            for a in range(left_size)
            for b in range(right_size)
        }
        return cls(
            set_sizes=tuple(set_sizes),
            select_arcs=set(select_arcs),
            flow_arcs=set(select_arcs),
            internal=internal,
            external=external,
        )

    @classmethod
    def structural(cls, set_sizes: Tuple[int, ...]) -> 'CandidateSpace':
        """直接生成 P1 稀疏空间，避免先物化随后删除结构零候选。

        输入：每个集合包含的节点数。
        输出：无集合自环，且单节点内部选择已代入常数的候选空间。
        """

        n = len(set_sizes)
        set_arcs = {(i, j) for i in range(n) for j in range(n) if i != j}
        fixed_internal = {
            (i, 0, 0)
            for i, size in enumerate(set_sizes)
            if size == 1
        }
        internal = {
            (i, a, b)
            for i, size in enumerate(set_sizes)
            if size > 1
            for a in range(size)
            for b in range(size)
        }
        external = {
            (i, j, a, b)
            for i, left_size in enumerate(set_sizes)
            for j, right_size in enumerate(set_sizes)
            if i != j
            for a in range(left_size)
            for b in range(right_size)
        }
        return cls(
            set_sizes=tuple(set_sizes),
            select_arcs=set(set_arcs),
            flow_arcs=set(set_arcs),
            internal=internal,
            external=external,
            fixed_internal_one=fixed_internal,
        )

    @classmethod
    def dense_counts(cls, set_sizes: Tuple[int, ...]) -> Dict[str, int]:
        """不创建索引集合，直接计算原稠密模型的候选数量。"""

        n = len(set_sizes)
        total_nodes = sum(set_sizes)
        return {
            'select': n * n,
            'flow': n * n,
            'internal': sum(size * size for size in set_sizes),
            'external': total_nodes * total_nodes,
            'fixed_internal_one': 0,
        }

    def copy(self) -> 'CandidateSpace':
        """深复制可变索引集合，并安全复用不可变上界记录。"""

        return CandidateSpace(
            set_sizes=self.set_sizes,
            select_arcs=set(self.select_arcs),
            flow_arcs=set(self.flow_arcs),
            internal=set(self.internal),
            external=set(self.external),
            fixed_internal_one=set(self.fixed_internal_one),
            incumbent=self.incumbent,
        )

    def counts(self) -> Dict[str, int]:
        """返回各类候选数量，供守恒检查和实验记录使用。"""

        return {
            'select': len(self.select_arcs),
            'flow': len(self.flow_arcs),
            'internal': len(self.internal),
            'external': len(self.external),
            'fixed_internal_one': len(self.fixed_internal_one),
        }

    def validate(self) -> None:
        """检查索引范围和变量依赖关系，异常时拒绝继续剪枝或建模。"""

        n = len(self.set_sizes)
        if n == 0:
            raise ValueError('Set-TSP 至少需要一个集合。')
        if any(size <= 0 for size in self.set_sizes):
            raise ValueError('每个 Set-TSP 集合都必须至少包含一个节点。')
        if not self.flow_arcs.issubset(self.select_arcs):
            raise ValueError('flow 候选必须对应一个 select 候选。')

        for i, j in self.select_arcs | self.flow_arcs:
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f'集合弧索引越界：{(i, j)}')
        for i, a, b in self.internal | self.fixed_internal_one:
            if not (0 <= i < n and 0 <= a < self.set_sizes[i] and 0 <= b < self.set_sizes[i]):
                raise ValueError(f'内部端点索引越界：{(i, a, b)}')
        if self.internal & self.fixed_internal_one:
            raise ValueError('同一个内部变量不能同时是候选变量和固定常量。')
        for i, j, a, b in self.external:
            if not (
                0 <= i < n
                and 0 <= j < n
                and 0 <= a < self.set_sizes[i]
                and 0 <= b < self.set_sizes[j]
            ):
                raise ValueError(f'外部端点索引越界：{(i, j, a, b)}')
            if (i, j) not in self.select_arcs:
                raise ValueError(f'外部端点候选缺少对应集合弧：{(i, j, a, b)}')


@dataclass(frozen=True)
class PruningOptions:
    """控制 P1、P3、P7 的组合及共同数值口径。"""

    structural_stsp: bool = False
    assignment_bound: bool = False
    endpoint_pair_dominance: bool = False
    preserve_all_optima: bool = False
    tolerance: float = 1e-7
    record_evidence: bool = False

    @classmethod
    def p1(cls) -> 'PruningOptions':
        """返回仅启用 P1 的配置。"""

        return cls(structural_stsp=True)

    @classmethod
    def p1_p3(cls) -> 'PruningOptions':
        """返回累积启用 P1、P3 的配置。"""

        return cls(structural_stsp=True, assignment_bound=True)

    @classmethod
    def p1_p3_p7(cls) -> 'PruningOptions':
        """返回累积启用 P1、P3、P7 的配置。"""

        return cls(
            structural_stsp=True,
            assignment_bound=True,
            endpoint_pair_dominance=True,
        )


@dataclass(frozen=True)
class SetTSPSolverOptions:
    """固定统一 Set-TSP 构造器的随机种子与停止精度。

    ``Threads`` 刻意不属于本配置，求解时沿用 Gurobi 的默认自动线程策略。
    """

    seed: int = 0
    mip_gap: float = 0.0
    mip_gap_abs: float = 1e-9
    optimality_tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-9
    integrality_tolerance: float = 1e-9
    output_flag: int = 0


@dataclass(frozen=True)
class PruningEvidence:
    """记录单个候选被删除时可复核的数学证据。"""

    stage: str
    candidate_type: str
    candidate: Tuple[int, ...]
    reason: str
    witness: Optional[Tuple[int, ...]] = None
    bound: Optional[float] = None
    threshold: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PruningStageReport:
    """保存一个剪枝阶段的计数、耗时、证据和安全回退信息。"""

    stage: str
    before: Dict[str, int]
    after: Dict[str, int]
    removed_by_reason: Dict[str, Dict[str, int]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    evidence: List[PruningEvidence] = field(default_factory=list)
    applied: bool = True
    fallback_reason: Optional[str] = None

    def validate_conservation(self) -> None:
        """严格验证候选变化、分原因计数和可选逐候选证据一致。

        输入：当前阶段的前后计数、分原因删除数和可选证据。
        输出：无；存在漏记、重复记账或非法增加时抛出异常。
        """

        candidate_types = ('select', 'flow', 'internal', 'external')
        unknown_types = set(self.removed_by_reason) - set(candidate_types)
        if unknown_types:
            raise ValueError(f'{self.stage} 含未知候选类型：{sorted(unknown_types)}')
        for candidate_type in candidate_types:
            before_count = int(self.before.get(candidate_type, 0))
            after_count = int(self.after.get(candidate_type, 0))
            reason_counts = self.removed_by_reason.get(candidate_type, {})
            removed_count = 0
            for reason, count in reason_counts.items():
                if not isinstance(reason, str) or not reason:
                    raise ValueError(f'{self.stage} 的删除原因名称非法：{reason!r}')
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(
                        f'{self.stage} 的 {candidate_type}/{reason} 删除计数非法：{count!r}'
                    )
                removed_count += count
            if before_count != after_count + removed_count:
                raise ValueError(
                    f'{self.stage} 的 {candidate_type} 不守恒：'
                    f'before={before_count}, after={after_count}, removed={removed_count}。'
                )

        # P1 把单节点内部变量转换为常量，必须同时体现在固定常量数量中。
        fixed_before = int(self.before.get('fixed_internal_one', 0))
        fixed_after = int(self.after.get('fixed_internal_one', 0))
        fixed_added = self.removed_by_reason.get('internal', {}).get(
            'structural_fixed_one',
            0,
        )
        if fixed_after - fixed_before != fixed_added:
            raise ValueError(
                f'{self.stage} 的单节点内部常量不守恒：'
                f'before={fixed_before}, after={fixed_after}, added={fixed_added}。'
            )

        if self.evidence:
            evidence_counts: Dict[Tuple[str, str], int] = {}
            for item in self.evidence:
                key = (item.candidate_type, item.reason)
                evidence_counts[key] = evidence_counts.get(key, 0) + 1
            expected_evidence_counts = {
                (candidate_type, reason): count
                for candidate_type in candidate_types
                for reason, count in self.removed_by_reason.get(candidate_type, {}).items()
                if count > 0
            }
            if evidence_counts != expected_evidence_counts:
                raise ValueError(f'{self.stage} 的逐候选证据与分原因删除计数不一致。')


@dataclass
class PruningPipelineReport:
    """汇总一次 P1→P3→P7 流水线的全部阶段报告。"""

    options: PruningOptions
    initial_counts: Dict[str, int]
    final_counts: Dict[str, int]
    stages: List[PruningStageReport] = field(default_factory=list)

    def metrics(self) -> Dict[str, Any]:
        """将各阶段指标展平成适合实验落盘的单层字典。"""

        flattened: Dict[str, Any] = {}
        for report in self.stages:
            flattened.update(report.metrics)
            if not report.applied:
                flattened[f'{report.stage}_fallback'] = report.fallback_reason
        return flattened

    def validate_conservation(self) -> None:
        """验证每个阶段内部以及相邻阶段之间的候选数量守恒。"""

        if not self.stages:
            if self.initial_counts != self.final_counts:
                raise ValueError('无剪枝阶段时，流水线初始与最终候选计数必须一致。')
            return

        previous_counts = self.initial_counts
        for report in self.stages:
            if report.before != previous_counts:
                raise ValueError(
                    f'{report.stage} 的 before 与上一阶段 after 不一致。'
                )
            report.validate_conservation()
            previous_counts = report.after
        if previous_counts != self.final_counts:
            raise ValueError('流水线最终计数与最后阶段 after 不一致。')


class PruningPreconditionError(RuntimeError):
    """表示剪枝前提无法验证；流水线捕获后应安全回退。"""
