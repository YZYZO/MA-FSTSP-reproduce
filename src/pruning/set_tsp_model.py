"""从统一候选空间构造并求解 Set-TSP 模型。"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import time
from typing import Dict, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

from .types import (
    CandidateSpace,
    ExternalKey,
    InternalKey,
    SetArc,
    SetTSPSolverOptions,
)
from .validation import validate_cost_tensors, validate_incumbent, validate_solver_options


@dataclass(frozen=True)
class SetTSPResult:
    """保存 Set-TSP 解、目标值和求解统计。"""

    sequence: Tuple[int, ...]
    objective: float
    status: int
    variable_count: int
    constraint_count: int
    node_count: float
    model_build_seconds: float
    runtime_seconds: float
    total_seconds: float
    mip_gap: float
    objective_bound: float
    solver_seed: int
    solver_mip_gap_target: float
    solver_mip_gap_abs_target: float
    solver_optimality_tolerance: float
    solver_feasibility_tolerance: float
    solver_integrality_tolerance: float
    gurobi_version: Tuple[int, int, int]
    selected_internal: Tuple[InternalKey, ...]
    selected_external: Tuple[ExternalKey, ...]


def _linear_sum(variables, constant: float = 0.0) -> gp.LinExpr:
    """安全构造允许空变量集合和常数项的线性表达式。"""

    expression = gp.LinExpr(float(constant))
    for variable in variables:
        expression.add(variable)
    return expression


def _apply_mip_start(
    space: CandidateSpace,
    select: Dict[SetArc, gp.Var],
    flow: Dict[SetArc, gp.Var],
    internal: Dict[InternalKey, gp.Var],
    external: Dict[ExternalKey, gp.Var],
) -> None:
    """把 P3 可行上界写入当前稀疏模型作为 MIP Start。"""

    incumbent = space.incumbent
    if incumbent is None:
        return
    for arc in incumbent.select_arcs:
        if arc in select:
            select[arc].Start = 1.0
    for key in incumbent.internal:
        if key in internal:
            internal[key].Start = 1.0
    for key in incumbent.external:
        if key in external:
            external[key].Start = 1.0

    # GG 流从仓库带 n-1 单位出发，每经过一个客户集合减少一单位。
    n = len(incumbent.order)
    for position, source in enumerate(incumbent.order):
        target = incumbent.order[(position + 1) % n]
        arc = (source, target)
        if arc in flow:
            flow[arc].Start = float(n - 1 - position) if position < n - 1 else 0.0


def solve_set_tsp(
    space: CandidateSpace,
    distance: Sequence,
    internal_cost: Sequence,
    output_flag: int | None = None,
    solver_options: SetTSPSolverOptions | None = None,
) -> SetTSPResult:
    """用最终候选空间建立与原算法等价的 GG Set-TSP 模型。

    输入：候选空间、集合间距离、集合内部成本、兼容日志开关和求解配置。
    输出：最优集合顺序、目标值、变量选择及 Gurobi 统计。

    缺失变量按固定为 0 处理；P1 的单节点内部变量按常数 1 代入约束
    与目标函数，因此无需依赖 Gurobi Presolve 才能体现规模下降。
    """

    solve_started = time.perf_counter()
    selected_solver_options = solver_options or SetTSPSolverOptions()
    validate_solver_options(selected_solver_options)
    validate_cost_tensors(space, distance, internal_cost)
    if space.incumbent is not None:
        validate_incumbent(space, space.incumbent, distance, internal_cost, 1e-7)

    n = len(space.set_sizes)
    model = gp.Model('Pruned-Set-TSP')
    effective_output_flag = (
        selected_solver_options.output_flag
        if output_flag is None
        else int(output_flag)
    )
    model.setParam('OutputFlag', effective_output_flag)
    model.setParam('Seed', int(selected_solver_options.seed))
    model.setParam('MIPGap', float(selected_solver_options.mip_gap))
    model.setParam('MIPGapAbs', float(selected_solver_options.mip_gap_abs))
    model.setParam(
        'OptimalityTol',
        float(selected_solver_options.optimality_tolerance),
    )
    model.setParam(
        'FeasibilityTol',
        float(selected_solver_options.feasibility_tolerance),
    )
    model.setParam(
        'IntFeasTol',
        float(selected_solver_options.integrality_tolerance),
    )
    # 按实验约定不设置 Threads，沿用 Gurobi 默认自动线程策略。

    select = {
        key: model.addVar(vtype=GRB.BINARY, name=f'select[{key[0]},{key[1]}]')
        for key in sorted(space.select_arcs)
    }
    flow = {
        key: model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f'flow[{key[0]},{key[1]}]')
        for key in sorted(space.flow_arcs)
    }
    internal = {
        key: model.addVar(vtype=GRB.BINARY, name=f'internal[{key[0]},{key[1]},{key[2]}]')
        for key in sorted(space.internal)
    }
    external = {
        key: model.addVar(
            vtype=GRB.BINARY,
            name=f'external[{key[0]},{key[1]},{key[2]},{key[3]}]',
        )
        for key in sorted(space.external)
    }

    # 预建约束邻接索引，避免每个端点约束反复扫描全部外部变量。
    select_in = defaultdict(list)
    select_out = defaultdict(list)
    flow_in = defaultdict(list)
    flow_out = defaultdict(list)
    external_by_arc = defaultdict(list)
    external_by_entry = defaultdict(list)
    external_by_exit = defaultdict(list)
    internal_by_set = defaultdict(list)
    internal_by_entry = defaultdict(list)
    internal_by_exit = defaultdict(list)
    for (i, j), variable in select.items():
        select_out[i].append(variable)
        select_in[j].append(variable)
    for (i, j), variable in flow.items():
        flow_out[i].append(variable)
        flow_in[j].append(variable)
    for (i, a, b), variable in internal.items():
        internal_by_set[i].append(variable)
        internal_by_entry[(i, a)].append(variable)
        internal_by_exit[(i, b)].append(variable)
    for (i, j, a, b), variable in external.items():
        external_by_arc[(i, j)].append(variable)
        external_by_exit[(i, a)].append(variable)
        external_by_entry[(j, b)].append(variable)

    for i in range(n):
        if (i, i) in select:
            model.addConstr(select[i, i] == 0, name=f'no_self_select[{i}]')
        model.addConstr(
            _linear_sum(select_in[i]) == 1,
            name=f'select_in[{i}]',
        )
        model.addConstr(
            _linear_sum(select_out[i]) == 1,
            name=f'select_out[{i}]',
        )

    for arc, variable in flow.items():
        model.addConstr(variable <= n * select[arc], name=f'flow_link[{arc[0]},{arc[1]}]')
    model.addConstr(
        _linear_sum(flow_out[0]) == n - 1,
        name='flow_source_out',
    )
    model.addConstr(
        _linear_sum(flow_in[0]) == 0,
        name='flow_source_in',
    )
    for i in range(n):
        if (i, i) in flow:
            model.addConstr(flow[i, i] == 0, name=f'no_self_flow[{i}]')
    for node in range(1, n):
        model.addConstr(
            _linear_sum(flow_in[node])
            - _linear_sum(flow_out[node])
            == 1,
            name=f'flow_balance[{node}]',
        )

    for i in range(n):
        fixed_count = sum(1 for key in space.fixed_internal_one if key[0] == i)
        model.addConstr(
            _linear_sum(internal_by_set[i], fixed_count) == 1,
            name=f'internal_once[{i}]',
        )

    for i, j in sorted(space.select_arcs):
        model.addConstr(
            _linear_sum(external_by_arc[(i, j)])
            == select[i, j],
            name=f'external_select[{i},{j}]',
        )

    for set_index, size in enumerate(space.set_sizes):
        for node in range(size):
            incoming_external = _linear_sum(
                external_by_entry[(set_index, node)]
            )
            fixed_entry = sum(
                1
                for key in space.fixed_internal_one
                if key[0] == set_index and key[1] == node
            )
            internal_from_entry = _linear_sum(
                internal_by_entry[(set_index, node)],
                fixed_entry,
            )
            model.addConstr(
                incoming_external == internal_from_entry,
                name=f'entry_continuity[{set_index},{node}]',
            )

            outgoing_external = _linear_sum(
                external_by_exit[(set_index, node)]
            )
            fixed_exit = sum(
                1
                for key in space.fixed_internal_one
                if key[0] == set_index and key[2] == node
            )
            internal_to_exit = _linear_sum(
                internal_by_exit[(set_index, node)],
                fixed_exit,
            )
            model.addConstr(
                outgoing_external == internal_to_exit,
                name=f'exit_continuity[{set_index},{node}]',
            )

    objective = gp.LinExpr()
    for key, variable in internal.items():
        i, a, b = key
        objective.add(variable, float(internal_cost[i][a][b]))
    for i, a, b in space.fixed_internal_one:
        objective.addConstant(float(internal_cost[i][a][b]))
    for key, variable in external.items():
        i, j, a, b = key
        objective.add(variable, float(distance[i][j][a][b]))
    model.setObjective(objective, GRB.MINIMIZE)

    _apply_mip_start(space, select, flow, internal, external)
    model_build_seconds = time.perf_counter() - solve_started
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f'Set-TSP 未达到最优状态，Gurobi status={model.Status}。')

    sequence = [0]
    for _ in range(n):
        current = sequence[-1]
        successors = [j for (i, j), variable in select.items() if i == current and variable.X > 0.5]
        if len(successors) != 1:
            raise RuntimeError(f'集合 {current} 的最优后继数量不是 1。')
        sequence.append(successors[0])
    if sequence[-1] != 0 or len(set(sequence[:-1])) != n:
        raise RuntimeError(f'恢复出的集合序列不是单一 Hamilton 闭环：{sequence}')

    selected_internal = tuple(
        sorted(
            space.fixed_internal_one
            | {key for key, variable in internal.items() if variable.X > 0.5}
        )
    )
    selected_external = tuple(sorted(key for key, variable in external.items() if variable.X > 0.5))
    return SetTSPResult(
        sequence=tuple(sequence),
        objective=float(model.ObjVal),
        status=int(model.Status),
        variable_count=int(model.NumVars),
        constraint_count=int(model.NumConstrs),
        node_count=float(model.NodeCount),
        model_build_seconds=model_build_seconds,
        runtime_seconds=float(model.Runtime),
        total_seconds=time.perf_counter() - solve_started,
        mip_gap=float(model.MIPGap),
        objective_bound=float(model.ObjBound),
        solver_seed=int(selected_solver_options.seed),
        solver_mip_gap_target=float(selected_solver_options.mip_gap),
        solver_mip_gap_abs_target=float(selected_solver_options.mip_gap_abs),
        solver_optimality_tolerance=float(selected_solver_options.optimality_tolerance),
        solver_feasibility_tolerance=float(selected_solver_options.feasibility_tolerance),
        solver_integrality_tolerance=float(selected_solver_options.integrality_tolerance),
        gurobi_version=tuple(int(part) for part in gp.gurobi.version()),
        selected_internal=selected_internal,
        selected_external=selected_external,
    )
