"""剪枝输入与可行上界的独立验证函数。"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

from .types import (
    CandidateSpace,
    PruningOptions,
    PruningPreconditionError,
    SetTSPIncumbent,
    SetTSPSolverOptions,
)


def validate_pruning_options(options: PruningOptions, stage: str) -> None:
    """验证 P3/P7 共用容差，阻止非法数值进入候选删除逻辑。

    输入：剪枝配置和即将执行的阶段名称。
    输出：无；容差不是有限非负实数时抛出可安全回退的前提异常。
    """

    tolerance = options.tolerance
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0.0
    ):
        raise PruningPreconditionError(
            f'{stage} 要求 tolerance 为有限非负实数，实际为 {tolerance!r}。'
        )


def validate_solver_options(options: SetTSPSolverOptions) -> None:
    """验证统一 Set-TSP 求解器的随机种子和停止精度。

    输入：求解器配置。
    输出：无；配置超出 Gurobi 合法范围时抛出 ``ValueError``。
    """

    if (
        isinstance(options.seed, bool)
        or not isinstance(options.seed, Integral)
        or not 0 <= int(options.seed) <= 2_000_000_000
    ):
        raise ValueError(f'Gurobi Seed 必须位于 [0, 2000000000]：{options.seed!r}')
    for name, value in (
        ('MIPGap', options.mip_gap),
        ('MIPGapAbs', options.mip_gap_abs),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f'{name} 必须是有限非负实数：{value!r}')
    for name, value, upper_bound in (
        ('OptimalityTol', options.optimality_tolerance, 1e-2),
        ('FeasibilityTol', options.feasibility_tolerance, 1e-2),
        ('IntFeasTol', options.integrality_tolerance, 1e-1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not 1e-9 <= float(value) <= upper_bound
        ):
            raise ValueError(
                f'{name} 必须位于 [1e-9, {upper_bound}]：{value!r}'
            )
    if isinstance(options.output_flag, bool) or not isinstance(options.output_flag, Integral):
        raise ValueError(f'OutputFlag 必须是整数：{options.output_flag!r}')


def validate_cost_tensors(
    space: CandidateSpace,
    distance: Sequence,
    internal_cost: Sequence,
) -> None:
    """验证距离张量维度以及候选位置上的数值有效性。

    输入：候选空间、集合间距离和集合内部成本。
    输出：无；发现维度不匹配、NaN 或负成本时抛出异常。
    """

    space.validate()
    n = len(space.set_sizes)
    if len(distance) != n or len(internal_cost) != n:
        raise ValueError('成本张量的集合维度与候选空间不一致。')

    for i, size in enumerate(space.set_sizes):
        if len(internal_cost[i]) != size or any(len(row) != size for row in internal_cost[i]):
            raise ValueError(f'集合 {i} 的内部成本矩阵维度错误。')
    for i, left_size in enumerate(space.set_sizes):
        if len(distance[i]) != n:
            raise ValueError(f'集合 {i} 的外部距离张量维度错误。')
        for j, right_size in enumerate(space.set_sizes):
            matrix = distance[i][j]
            if len(matrix) != left_size or any(len(row) != right_size for row in matrix):
                raise ValueError(f'集合弧 {(i, j)} 的端点距离矩阵维度错误。')

    for i, a, b in space.internal | space.fixed_internal_one:
        value = float(internal_cost[i][a][b])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'内部成本必须有限且非负：{(i, a, b)}={value}')
    for i, j, a, b in space.external:
        value = float(distance[i][j][a][b])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'外部成本必须有限且非负：{(i, j, a, b)}={value}')


def validate_incumbent(
    space: CandidateSpace,
    incumbent: SetTSPIncumbent,
    distance: Sequence,
    internal_cost: Sequence,
    tolerance: float,
) -> None:
    """重新计算上界路线成本并检查每个选择仍在候选空间中。

    输入：候选空间、上界、成本张量和容差。
    输出：无；上界不可行或成本不一致时抛出异常。
    """

    n = len(space.set_sizes)
    if len(incumbent.order) != n or set(incumbent.order) != set(range(n)):
        raise ValueError('上界的集合顺序不是完整排列。')
    if incumbent.order[0] != 0:
        raise ValueError('上界顺序必须从仓库集合 0 开始。')
    if len(incumbent.internal) + len(space.fixed_internal_one) < n:
        raise ValueError('上界没有为每个集合提供内部端点选择。')
    if len(incumbent.external) != n:
        raise ValueError('上界没有形成包含所有集合的闭环。')

    selected_internal = set(incumbent.internal)
    if not selected_internal.issubset(space.internal):
        raise ValueError('上界使用了已不存在的内部候选。')
    if not set(incumbent.external).issubset(space.external):
        raise ValueError('上界使用了已不存在的外部候选。')

    recomputed = 0.0
    internal_by_set = {}
    for key in selected_internal | space.fixed_internal_one:
        i, a, b = key
        if i in internal_by_set:
            raise ValueError(f'上界为集合 {i} 选择了多个内部端点对。')
        internal_by_set[i] = (a, b)
        recomputed += float(internal_cost[i][a][b])
    if set(internal_by_set) != set(range(n)):
        raise ValueError('上界没有恰好覆盖所有集合内部选择。')

    incoming = {}
    outgoing = {}
    for i, j, a, b in incumbent.external:
        if i in outgoing or j in incoming:
            raise ValueError('上界不满足集合一入一出。')
        outgoing[i] = (j, a)
        incoming[j] = (i, b)
        recomputed += float(distance[i][j][a][b])
    if set(incoming) != set(range(n)) or set(outgoing) != set(range(n)):
        raise ValueError('上界外部弧没有覆盖全部集合。')

    for i, (entry, exit_) in internal_by_set.items():
        if incoming[i][1] != entry or outgoing[i][1] != exit_:
            raise ValueError(f'上界在集合 {i} 的端点连续性不成立。')
    if abs(recomputed - incumbent.cost) > tolerance:
        raise ValueError(f'上界成本校验失败：记录 {incumbent.cost}，重算 {recomputed}。')
