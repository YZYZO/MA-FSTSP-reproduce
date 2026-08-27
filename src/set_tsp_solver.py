"""Set-TSP 的独立建模、求解与遥测模块。"""

from dataclasses import asdict, dataclass
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB


@dataclass
class SetTSPSolveResult:
    """
    保存一次单仓库 Set-TSP 求解的完整结果。

    输入由 ``solve_set_tsp`` 填充；输出同时包含可选访问序列、求解状态、目标值、
    建模/优化耗时、真实模型规模与 MIP 遥测。无可用 incumbent 时 ``sequence=None``。
    """

    sequence: list | None
    status: str
    objective: float | None
    build_seconds: float
    optimize_seconds: float
    num_bin_vars: int
    num_vars: int
    num_constraints: int
    node_count: float | None
    mip_gap: float | None
    solution_count: int
    time_limit_reached: bool
    has_incumbent: bool
    error_message: str | None = None

    def diagnostics(self):
        """
        将求解结果转换为可直接写入 JSON/NPZ 的基础类型字典。

        输入为当前结果对象；输出不包含 Gurobi 对象，仅保留序列和标量遥测。
        """

        return asdict(self)


def build_set_tsp_model(convex_sets, distance, convex_set_distance):
    """
    构造 Set-TSP 数学模型但不调用优化器。

    输入为候选集合、集合间卡车距离和集合内部切换代价；输出 ``(model, select)``，
    其中 ``select`` 用于从 incumbent 恢复集合访问顺序。函数在返回前调用
    ``model.update()``，因此 NumVars/NumBinVars/NumConstrs 均为可读取的真实值。
    """

    set_count = len(convex_sets)
    if set_count == 0:
        raise ValueError('Set-TSP 至少需要一个候选集合。')
    if any(len(convex_set) == 0 for convex_set in convex_sets):
        raise ValueError('Set-TSP 的候选集合不能为空。')

    model = gp.Model('Set-TSP')
    model.setParam('OutputFlag', 0)

    # ``select[u,v]`` 和 ``flow[u,v]`` 描述集合层面的 Hamilton 回路与 GG 流。
    select = model.addMVar((set_count, set_count), vtype=GRB.BINARY)
    model.addConstrs(select[u, u] == 0 for u in range(set_count))
    model.addConstrs(
        np.ones((set_count,)) @ select[:, target] == 1
        for target in range(set_count)
    )
    model.addConstrs(
        np.ones((set_count,)) @ select[source, :] == 1
        for source in range(set_count)
    )
    flow = model.addMVar((set_count, set_count), vtype=GRB.CONTINUOUS)
    model.addConstrs(
        flow[source, target] <= set_count * select[source, target]
        for source in range(set_count)
        for target in range(set_count)
    )
    model.addConstr(np.ones((set_count,)) @ flow[0, :] == set_count - 1)
    model.addConstr(np.ones((set_count,)) @ flow[:, 0] == 0)
    model.addConstrs(flow[source, source] == 0 for source in range(set_count))
    model.addConstrs(
        np.ones((set_count,)) @ flow[:, target]
        - np.ones((set_count,)) @ flow[target, :] == 1
        for target in range(1, set_count)
    )

    # ``internal`` 选择同一集合的进入/离开节点，``external`` 连接不同集合节点。
    internal = [
        [
            [model.addVar(vtype=GRB.BINARY) for _ in convex_set]
            for _ in convex_set
        ]
        for convex_set in convex_sets
    ]
    external = [
        [
            [
                [model.addVar(vtype=GRB.BINARY) for _ in target_set]
                for _ in source_set
            ]
            for target_set in convex_sets
        ]
        for source_set in convex_sets
    ]
    model.addConstrs(
        gp.quicksum(
            internal[set_index][entry][leave]
            for entry in range(len(convex_sets[set_index]))
            for leave in range(len(convex_sets[set_index]))
        ) == 1
        for set_index in range(set_count)
    )
    model.addConstrs(
        gp.quicksum(
            external[source][target][source_node][target_node]
            for source_node in range(len(convex_sets[source]))
            for target_node in range(len(convex_sets[target]))
        ) == select[source, target]
        for source in range(set_count)
        for target in range(set_count)
    )
    model.addConstrs(
        gp.quicksum(
            external[source][target][source_node][target_node]
            for source in range(set_count)
            for source_node in range(len(convex_sets[source]))
        ) == gp.quicksum(
            internal[target][target_node][leave]
            for leave in range(len(convex_sets[target]))
        )
        for target in range(set_count)
        for target_node in range(len(convex_sets[target]))
    )
    model.addConstrs(
        gp.quicksum(
            external[source][target][source_node][target_node]
            for target in range(set_count)
            for target_node in range(len(convex_sets[target]))
        ) == gp.quicksum(
            internal[source][entry][source_node]
            for entry in range(len(convex_sets[source]))
        )
        for source in range(set_count)
        for source_node in range(len(convex_sets[source]))
    )
    model.setObjective(
        gp.quicksum(
            convex_set_distance[set_index][entry][leave]
            * internal[set_index][entry][leave]
            for set_index in range(set_count)
            for entry in range(len(convex_sets[set_index]))
            for leave in range(len(convex_sets[set_index]))
        )
        + gp.quicksum(
            distance[source][target][source_node][target_node]
            * external[source][target][source_node][target_node]
            for source in range(set_count)
            for target in range(set_count)
            for source_node in range(len(convex_sets[source]))
            for target_node in range(len(convex_sets[target]))
        ),
        GRB.MINIMIZE,
    )
    model.update()
    return model, select


def _status_name(status_code):
    """
    将 Gurobi 数值状态映射为稳定的实验字符串。

    输入为 ``model.Status``；输出不会随日志语言变化的状态名，未知状态保留编号。
    """

    names = {
        GRB.OPTIMAL: 'optimal',
        GRB.INFEASIBLE: 'infeasible',
        GRB.INF_OR_UNBD: 'inf_or_unbd',
        GRB.UNBOUNDED: 'unbounded',
        GRB.TIME_LIMIT: 'time_limit',
        GRB.INTERRUPTED: 'interrupted',
        GRB.SUBOPTIMAL: 'suboptimal',
        GRB.NUMERIC: 'numeric',
    }
    return names.get(status_code, f'gurobi_status_{status_code}')


def _extract_sequence(select, set_count):
    """
    从有 incumbent 的集合层变量恢复以仓库集合 0 开始并回到 0 的访问序列。

    输入为 ``select`` MVar 和集合数；输出长度 ``set_count + 1`` 的索引序列。
    若变量值不是合法单回路则抛出 ``RuntimeError``，避免原实现发生无限循环。
    """

    sequence = [0]
    visited = {0}
    for _ in range(set_count):
        current = sequence[-1]
        successors = [
            target
            for target in range(set_count)
            if float(select[current, target].X) > 0.5
        ]
        if len(successors) != 1:
            raise RuntimeError('Set-TSP incumbent 的集合后继不唯一。')
        target = successors[0]
        sequence.append(target)
        if target == 0:
            break
        if target in visited:
            raise RuntimeError('Set-TSP incumbent 在返回仓库前出现子回路。')
        visited.add(target)
    if len(sequence) != set_count + 1 or sequence[-1] != 0:
        raise RuntimeError('Set-TSP incumbent 未形成覆盖全部集合的闭合回路。')
    return sequence


def solve_set_tsp(
    convex_sets,
    distance,
    convex_set_distance,
    time_limit=None,
):
    """
    构建并求解 Set-TSP，同时返回可恢复失败的正式遥测。

    输入为三类模型数据和可选秒级时限；``None`` 表示不设置上限。输出
    ``SetTSPSolveResult``。达到时限但已有 incumbent 时仍返回序列；无 incumbent、
    不可行或异常时返回 ``sequence=None``，由上层标记实例未完成而不中断整批实验。
    """

    if time_limit is not None and float(time_limit) <= 0:
        raise ValueError('time_limit 必须为正数或 None。')

    build_start = time.perf_counter()
    try:
        model, select = build_set_tsp_model(
            convex_sets,
            distance,
            convex_set_distance,
        )
    except Exception as error:  # Gurobi 许可和建模错误也必须成为可记录结果。
        return SetTSPSolveResult(
            sequence=None,
            status='build_error',
            objective=None,
            build_seconds=time.perf_counter() - build_start,
            optimize_seconds=0.0,
            num_bin_vars=0,
            num_vars=0,
            num_constraints=0,
            node_count=None,
            mip_gap=None,
            solution_count=0,
            time_limit_reached=False,
            has_incumbent=False,
            error_message=f'{type(error).__name__}: {error}',
        )
    build_seconds = time.perf_counter() - build_start

    if time_limit is not None:
        model.setParam('TimeLimit', float(time_limit))
    optimize_start = time.perf_counter()
    try:
        model.optimize()
        optimize_seconds = time.perf_counter() - optimize_start
        status = _status_name(model.Status)
        solution_count = int(model.SolCount)
        has_incumbent = solution_count > 0
        sequence = (
            _extract_sequence(select, len(convex_sets))
            if has_incumbent
            else None
        )
        objective = float(model.ObjVal) if has_incumbent else None
        mip_gap = float(model.MIPGap) if has_incumbent else None
        return SetTSPSolveResult(
            sequence=sequence,
            status=status,
            objective=objective,
            build_seconds=build_seconds,
            optimize_seconds=optimize_seconds,
            num_bin_vars=int(model.NumBinVars),
            num_vars=int(model.NumVars),
            num_constraints=int(model.NumConstrs),
            node_count=float(model.NodeCount),
            mip_gap=mip_gap,
            solution_count=solution_count,
            time_limit_reached=model.Status == GRB.TIME_LIMIT,
            has_incumbent=has_incumbent,
        )
    except Exception as error:
        return SetTSPSolveResult(
            sequence=None,
            status='solve_error',
            objective=None,
            build_seconds=build_seconds,
            optimize_seconds=time.perf_counter() - optimize_start,
            num_bin_vars=int(model.NumBinVars),
            num_vars=int(model.NumVars),
            num_constraints=int(model.NumConstrs),
            node_count=None,
            mip_gap=None,
            solution_count=0,
            time_limit_reached=False,
            has_incumbent=False,
            error_message=f'{type(error).__name__}: {error}',
        )
