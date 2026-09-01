"""
带计算预算和求解遥测的 Set-TSP 独立求解器。

本模块复制论文原始 Set-TSP 数学模型，但不修改 `src/fstsp.py`。学习数据生成
和后续策略评估可以使用这里的结构化结果，原始论文实验仍沿用旧入口。
"""

from dataclasses import asdict, dataclass
import math

import gurobipy as gp
from gurobipy import GRB
import numpy as np


@dataclass(frozen=True)
class SetTSPSolveResult:
    """
    保存一次 Set-TSP 求解的顺序、状态和计算难度信息。

    输入：由 `solve_set_tsp_with_telemetry` 填充的求解结果字段。
    输出：不可变结果对象，可通过 `to_dict` 写入缓存或数据集。
    逻辑：把标签生成所需信息集中保存，避免调用方直接读取 Gurobi 模型。
    """

    sequence: tuple
    status: int
    status_name: str
    objective: float | None
    runtime_seconds: float
    node_count: float
    mip_gap: float | None
    objective_bound: float | None
    solution_count: int
    timed_out: bool
    has_incumbent: bool
    fallback_used: bool
    sequence_source: str

    def to_dict(self):
        """
        将求解结果转换成可 JSON 序列化的字典。

        输入：当前结果对象。
        输出：其中访问顺序为列表的普通字典。
        逻辑：先使用 dataclass 转换，再把元组显式改成列表。
        """
        payload = asdict(self)
        payload["sequence"] = list(self.sequence)
        return payload

    @classmethod
    def from_dict(cls, payload):
        """
        从缓存字典恢复 Set-TSP 求解结果。

        输入：`to_dict` 生成的字典。
        输出：新的不可变结果对象。
        逻辑：把访问顺序恢复为元组后调用构造函数。
        """
        values = dict(payload)
        values["sequence"] = tuple(values["sequence"])
        return cls(**values)


def _status_name(status):
    """
    把 Gurobi 整数状态转换成稳定的文本标签。

    输入：Gurobi `Status` 整数。
    输出：如 `OPTIMAL`、`TIME_LIMIT` 的字符串。
    逻辑：使用显式映射，未知的新状态保留数值信息。
    """
    status_names = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return status_names.get(status, f"STATUS_{status}")


def _finite_or_none(value):
    """
    将 Gurobi 数值属性转换成有限浮点数或空值。

    输入：可能为无穷或不可转换的数值。
    输出：有限浮点数；否则返回 `None`。
    逻辑：缓存和 JSON 标签不保存无穷值。
    """
    try:
        number = float(value)
    except (TypeError, ValueError, AttributeError, gp.GurobiError):
        return None
    return number if math.isfinite(number) else None


def _extract_sequence(select, number_of_sets):
    """
    从 Gurobi 选边变量中恢复以仓库集合 0 为首尾的访问顺序。

    输入：二维选择变量和集合数量。
    输出：完整合法的顺序元组；无法恢复时返回空元组。
    逻辑：逐步跟随值大于 0.5 的唯一出边，并检查长度、首尾和节点覆盖。
    """
    sequence = [0]
    for _ in range(number_of_sets):
        current = sequence[-1]
        next_nodes = [
            node
            for node in range(number_of_sets)
            if select[current, node].X > 0.5
        ]
        if len(next_nodes) != 1:
            return tuple()
        sequence.append(next_nodes[0])
        if sequence[-1] == 0:
            break

    expected_nodes = set(range(number_of_sets))
    if (
        len(sequence) != number_of_sets + 1
        or sequence[0] != 0
        or sequence[-1] != 0
        or set(sequence[:-1]) != expected_nodes
    ):
        return tuple()
    return tuple(sequence)


def _trivial_result():
    """
    返回只有仓库、没有客户时的显然 Set-TSP 结果。

    输入：无。
    输出：顺序为 `(0, 0)` 的最优结果。
    逻辑：避免为一个集合建立没有研究价值的 MIP。
    """
    return SetTSPSolveResult(
        sequence=(0, 0),
        status=GRB.OPTIMAL,
        status_name="OPTIMAL",
        objective=0.0,
        runtime_seconds=0.0,
        node_count=0.0,
        mip_gap=0.0,
        objective_bound=0.0,
        solution_count=1,
        timed_out=False,
        has_incumbent=True,
        fallback_used=False,
        sequence_source="trivial",
    )


def solve_set_tsp_with_telemetry(
    convex_sets,
    distance,
    convex_set_distance,
    time_limit_seconds=None,
    threads=1,
    seed=0,
    fallback_sequence=None,
):
    """
    求解 Set-TSP，并返回可用于机器学习标签的完整遥测信息。

    输入：
    - convex_sets: 仓库和客户的候选节点集合列表；
    - distance: 集合之间的候选点卡车距离；
    - convex_set_distance: 各集合内部起终点切换代价；
    - time_limit_seconds: 可选 Gurobi 时间上限；
    - threads/seed: 可复现求解配置；
    - fallback_sequence: 无可用 MIP 顺序时采用的 LKH 等后备顺序。

    输出：`SetTSPSolveResult`。
    逻辑：复用论文 GG 流模型；有 incumbent 时恢复其顺序，否则安全返回后备顺序。
    """
    number_of_sets = len(convex_sets)
    if number_of_sets == 1:
        return _trivial_result()

    model = gp.Model("Set-TSP-Telemetry")
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", threads)
    model.setParam("Seed", seed)
    if time_limit_seconds is not None:
        model.setParam("TimeLimit", float(time_limit_seconds))

    # 集合层面的访问顺序与单商品流变量，对应论文原始 Set-TSP 模型。
    select = model.addMVar((number_of_sets, number_of_sets), vtype=GRB.BINARY)
    model.addConstrs(select[u, u] == 0 for u in range(number_of_sets))
    model.addConstrs(
        np.ones((number_of_sets,)) @ select[:, v] == 1
        for v in range(number_of_sets)
    )
    model.addConstrs(
        np.ones((number_of_sets,)) @ select[u, :] == 1
        for u in range(number_of_sets)
    )

    flow = model.addMVar((number_of_sets, number_of_sets), vtype=GRB.CONTINUOUS)
    model.addConstrs(
        flow[u, v] <= number_of_sets * select[u, v]
        for u in range(number_of_sets)
        for v in range(number_of_sets)
    )
    model.addConstr(np.ones((number_of_sets,)) @ flow[0, :] == number_of_sets - 1)
    model.addConstr(np.ones((number_of_sets,)) @ flow[:, 0] == 0)
    model.addConstrs(flow[u, u] == 0 for u in range(number_of_sets))
    model.addConstrs(
        np.ones((number_of_sets,)) @ flow[:, v]
        - np.ones((number_of_sets,)) @ flow[v, :]
        == 1
        for v in range(1, number_of_sets)
    )

    # `internal` 和 `external` 与原代码含义相同，分别选择集合内部节点对和集合间连接节点对。
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
                [model.addVar(vtype=GRB.BINARY) for _ in end_set]
                for _ in start_set
            ]
            for end_set in convex_sets
        ]
        for start_set in convex_sets
    ]

    model.addConstrs(
        gp.quicksum(
            internal[index][start][end]
            for start in range(len(convex_sets[index]))
            for end in range(len(convex_sets[index]))
        )
        == 1
        for index in range(number_of_sets)
    )
    model.addConstrs(
        gp.quicksum(
            external[start_set][end_set][start][end]
            for start in range(len(convex_sets[start_set]))
            for end in range(len(convex_sets[end_set]))
        )
        == select[start_set, end_set]
        for start_set in range(number_of_sets)
        for end_set in range(number_of_sets)
    )
    model.addConstrs(
        gp.quicksum(
            external[start_set][end_set][start][end]
            for start_set in range(number_of_sets)
            for start in range(len(convex_sets[start_set]))
        )
        == gp.quicksum(
            internal[end_set][end][internal_end]
            for internal_end in range(len(convex_sets[end_set]))
        )
        for end_set in range(number_of_sets)
        for end in range(len(convex_sets[end_set]))
    )
    model.addConstrs(
        gp.quicksum(
            external[start_set][end_set][start][end]
            for end_set in range(number_of_sets)
            for end in range(len(convex_sets[end_set]))
        )
        == gp.quicksum(
            internal[start_set][internal_start][start]
            for internal_start in range(len(convex_sets[start_set]))
        )
        for start_set in range(number_of_sets)
        for start in range(len(convex_sets[start_set]))
    )

    internal_cost = gp.quicksum(
        convex_set_distance[index][start][end]
        * internal[index][start][end]
        for index in range(number_of_sets)
        for start in range(len(convex_sets[index]))
        for end in range(len(convex_sets[index]))
    )
    external_cost = gp.quicksum(
        distance[start_set][end_set][start][end]
        * external[start_set][end_set][start][end]
        for start_set in range(number_of_sets)
        for end_set in range(number_of_sets)
        for start in range(len(convex_sets[start_set]))
        for end in range(len(convex_sets[end_set]))
    )
    model.setObjective(internal_cost + external_cost, GRB.MINIMIZE)
    model.optimize()

    status = int(model.Status)
    solution_count = int(model.SolCount)
    has_incumbent = solution_count > 0
    mip_sequence = _extract_sequence(select, number_of_sets) if has_incumbent else tuple()

    fallback = tuple(fallback_sequence) if fallback_sequence is not None else tuple()
    fallback_used = len(mip_sequence) == 0 and len(fallback) > 0
    sequence = mip_sequence if len(mip_sequence) > 0 else fallback
    sequence_source = "gurobi_incumbent" if len(mip_sequence) > 0 else "fallback"
    if len(sequence) == 0:
        sequence_source = "unavailable"

    result = SetTSPSolveResult(
        sequence=sequence,
        status=status,
        status_name=_status_name(status),
        objective=_finite_or_none(model.ObjVal) if has_incumbent else None,
        runtime_seconds=float(model.Runtime),
        node_count=float(model.NodeCount),
        mip_gap=_finite_or_none(model.MIPGap) if has_incumbent else None,
        objective_bound=_finite_or_none(model.ObjBound),
        solution_count=solution_count,
        timed_out=status == GRB.TIME_LIMIT,
        has_incumbent=has_incumbent,
        fallback_used=fallback_used,
        sequence_source=sequence_source,
    )
    model.dispose()
    return result
