"""
根据保留端点索引构造并求解稀疏 Set-TSP Gurobi 模型。

模型语义与 `src.fstsp.MultiAgentFlyingSidekickTSP.set_tsp` 一致，仅省略结构零变量和 P7 已证明冗余的外部变量。
"""

from dataclasses import dataclass
import time
from typing import Mapping, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .types import EndpointDominanceResult, PruningMetrics, PruningOptions, SetArc


@dataclass
class SetTSPSolveResult:
    """
    保存一次稀疏 Set-TSP 求解结果。

    输入：集合访问顺序、最优目标值和已经补充求解器指标的剪枝统计。
    输出：供模型适配器恢复原算法 `get_seq` 返回值并记录实验数据。
    逻辑：访问顺序以仓库集合 0 开头和结尾，长度为集合数加一。
    """

    sequence: Tuple[int, ...]
    objective: float
    metrics: PruningMetrics


def _validate_fixed_sequence(fixed_sequence, set_count):
    """
    校验用于等价顺序审计的固定集合序列。

    输入：可选固定序列和集合数量。
    输出：合法时返回集合弧元组，未固定时返回 `None`。
    逻辑：要求仓库 0 首尾出现且其余集合恰好出现一次，防止审计模型改变问题定义。
    """

    if fixed_sequence is None:
        return None
    sequence = tuple(int(value) for value in fixed_sequence)
    if len(sequence) != set_count + 1 or sequence[0] != 0 or sequence[-1] != 0:
        raise ValueError('固定 Set-TSP 序列必须以 0 开始和结束，且长度为集合数加一。')
    if sorted(sequence[:-1]) != list(range(set_count)):
        raise ValueError('固定 Set-TSP 序列必须恰好访问每个集合一次。')
    return tuple(zip(sequence[:-1], sequence[1:]))


def _extract_sequence(select, set_count):
    """
    从最优 `select` 变量中恢复以仓库集合 0 为根的唯一环。

    输入：Gurobi 集合弧变量字典和集合数量。
    输出：首尾为 0 的集合访问顺序元组。
    逻辑：先构造唯一后继映射，再走恰好 `set_count` 条弧并验证无子环或断链。
    """

    successors = {}
    for (source_set, target_set), variable in select.items():
        if variable.X > 0.5:
            if source_set in successors:
                raise RuntimeError(f'集合 {source_set} 在最优解中出现多个后继。')
            successors[source_set] = target_set

    sequence = [0]
    for _ in range(set_count):
        current = sequence[-1]
        if current not in successors:
            raise RuntimeError(f'无法从集合 {current} 恢复 Set-TSP 后继。')
        sequence.append(successors[current])
    if sequence[-1] != 0 or len(set(sequence[:-1])) != set_count:
        raise RuntimeError(f'恢复的 Set-TSP 顺序不是覆盖全部集合的单一环：{sequence}。')
    return tuple(sequence)


def solve_sparse_set_tsp(
    internal_costs: Sequence[np.ndarray],
    external_costs: Mapping[SetArc, np.ndarray],
    dominance_result: EndpointDominanceResult,
    options: PruningOptions,
    fixed_sequence: Optional[Sequence[int]] = None,
) -> SetTSPSolveResult:
    """
    构造并求解与原模型等价的稀疏 Set-TSP。

    输入：内部成本、外部成本、P7 保留索引、求解配置和可选固定集合顺序。
    输出：集合访问顺序、目标值及完整模型/剪枝指标。
    逻辑：仅为 `i != j` 和保留端点对创建变量，随后复刻 GG 流、内部选择及端点连续约束。
    """

    options.validate()
    total_start = time.perf_counter()
    set_count = len(internal_costs)
    fixed_arcs = _validate_fixed_sequence(fixed_sequence, set_count)
    metrics = dominance_result.metrics

    model_build_start = time.perf_counter()
    model = gp.Model('Sparse-Set-TSP-P7')
    model.setParam('OutputFlag', 1 if options.gurobi_output else 0)
    model.setParam('Seed', options.gurobi_seed)

    # 集合弧只包含 `i != j` 且仍有端点候选的弧，作为 select 与 flow 的公共稀疏索引。
    set_arcs = tuple(
        set_arc
        for set_arc in sorted(dominance_result.kept_pairs_by_arc)
        if dominance_result.kept_pairs_by_arc[set_arc]
    )
    select = model.addVars(set_arcs, vtype=GRB.BINARY, name='select')
    flow = model.addVars(set_arcs, lb=0.0, vtype=GRB.CONTINUOUS, name='flow')

    internal_keys = tuple(
        (set_index, entry_endpoint, exit_endpoint)
        for set_index, matrix in enumerate(internal_costs)
        for entry_endpoint in range(np.asarray(matrix).shape[0])
        for exit_endpoint in range(np.asarray(matrix).shape[1])
    )
    internal = model.addVars(internal_keys, vtype=GRB.BINARY, name='internal')

    external_keys = tuple(
        (source_set, target_set, source_endpoint, target_endpoint)
        for (source_set, target_set) in set_arcs
        for source_endpoint, target_endpoint
        in dominance_result.kept_pairs_by_arc[(source_set, target_set)]
    )
    external = model.addVars(external_keys, vtype=GRB.BINARY, name='external')

    # 每个集合恰好选择一条入弧和一条出弧。
    for set_index in range(set_count):
        model.addConstr(
            gp.quicksum(select[source_set, set_index] for source_set in range(set_count)
                        if (source_set, set_index) in select) == 1,
            name=f'set_in_degree[{set_index}]',
        )
        model.addConstr(
            gp.quicksum(select[set_index, target_set] for target_set in range(set_count)
                        if (set_index, target_set) in select) == 1,
            name=f'set_out_degree[{set_index}]',
        )

    # Gavish-Graves 单商品流约束与原实现保持一致。
    for set_arc in set_arcs:
        model.addConstr(
            flow[set_arc] <= set_count * select[set_arc],
            name=f'flow_select{set_arc}',
        )
    model.addConstr(
        gp.quicksum(flow[0, target_set] for target_set in range(set_count)
                    if (0, target_set) in flow) == set_count - 1,
        name='flow_from_depot',
    )
    model.addConstr(
        gp.quicksum(flow[source_set, 0] for source_set in range(set_count)
                    if (source_set, 0) in flow) == 0,
        name='flow_to_depot',
    )
    for set_index in range(1, set_count):
        model.addConstr(
            gp.quicksum(flow[source_set, set_index] for source_set in range(set_count)
                        if (source_set, set_index) in flow)
            - gp.quicksum(flow[set_index, target_set] for target_set in range(set_count)
                          if (set_index, target_set) in flow) == 1,
            name=f'flow_balance[{set_index}]',
        )

    # 每个集合选择一个进入端点和一个离开端点组成的内部服务动作。
    for set_index, matrix in enumerate(internal_costs):
        endpoint_count = np.asarray(matrix).shape[0]
        model.addConstr(
            gp.quicksum(
                internal[set_index, entry_endpoint, exit_endpoint]
                for entry_endpoint in range(endpoint_count)
                for exit_endpoint in range(endpoint_count)
            ) == 1,
            name=f'internal_selection[{set_index}]',
        )

    # 每条选中的集合弧恰好选择一个仍被保留的外部端点对。
    for source_set, target_set in set_arcs:
        model.addConstr(
            gp.quicksum(
                external[source_set, target_set, source_endpoint, target_endpoint]
                for source_endpoint, target_endpoint
                in dominance_result.kept_pairs_by_arc[(source_set, target_set)]
            ) == select[source_set, target_set],
            name=f'external_selection[{source_set},{target_set}]',
        )

    # 关键索引表将端点连续性约束从全量四重循环变为对实际存在变量求和。
    incoming_external = {
        (set_index, endpoint): []
        for set_index, matrix in enumerate(internal_costs)
        for endpoint in range(np.asarray(matrix).shape[0])
    }
    outgoing_external = {
        (set_index, endpoint): []
        for set_index, matrix in enumerate(internal_costs)
        for endpoint in range(np.asarray(matrix).shape[0])
    }
    for key in external_keys:
        source_set, target_set, source_endpoint, target_endpoint = key
        outgoing_external[source_set, source_endpoint].append(external[key])
        incoming_external[target_set, target_endpoint].append(external[key])

    for set_index, matrix in enumerate(internal_costs):
        endpoint_count = np.asarray(matrix).shape[0]
        for entry_endpoint in range(endpoint_count):
            model.addConstr(
                gp.quicksum(incoming_external[set_index, entry_endpoint])
                == gp.quicksum(
                    internal[set_index, entry_endpoint, exit_endpoint]
                    for exit_endpoint in range(endpoint_count)
                ),
                name=f'entry_continuity[{set_index},{entry_endpoint}]',
            )
        for exit_endpoint in range(endpoint_count):
            model.addConstr(
                gp.quicksum(outgoing_external[set_index, exit_endpoint])
                == gp.quicksum(
                    internal[set_index, entry_endpoint, exit_endpoint]
                    for entry_endpoint in range(endpoint_count)
                ),
                name=f'exit_continuity[{set_index},{exit_endpoint}]',
            )

    objective = gp.quicksum(
        float(np.asarray(internal_costs[set_index])[entry_endpoint, exit_endpoint])
        * internal[set_index, entry_endpoint, exit_endpoint]
        for set_index, entry_endpoint, exit_endpoint in internal_keys
    ) + gp.quicksum(
        float(np.asarray(external_costs[source_set, target_set])[source_endpoint, target_endpoint])
        * external[source_set, target_set, source_endpoint, target_endpoint]
        for source_set, target_set, source_endpoint, target_endpoint in external_keys
    )
    model.setObjective(objective, GRB.MINIMIZE)

    if fixed_arcs is not None:
        fixed_arc_set = set(fixed_arcs)
        missing_arcs = fixed_arc_set - set(set_arcs)
        if missing_arcs:
            raise ValueError(f'固定序列包含没有端点候选的集合弧：{sorted(missing_arcs)}。')
        for set_arc in set_arcs:
            model.addConstr(
                select[set_arc] == (1 if set_arc in fixed_arc_set else 0),
                name=f'fixed_select{set_arc}',
            )

    model.update()
    metrics.model_build_seconds = time.perf_counter() - model_build_start
    metrics.select_variables = len(select)
    metrics.flow_variables = len(flow)
    metrics.internal_variables = len(internal)
    metrics.external_variables = len(external)
    metrics.model_variables = int(model.NumVars)
    metrics.model_constraints = int(model.NumConstrs)

    model.optimize()
    # 使用 Gurobi 自身 Runtime，避免把 Python 调用开销混入正式求解器时间。
    metrics.gurobi_seconds = float(model.Runtime)
    metrics.model_status = int(model.Status)
    metrics.gurobi_node_count = float(model.NodeCount)

    if model.SolCount <= 0:
        raise RuntimeError(
            f'稀疏 Set-TSP 没有得到可用解，Gurobi status={model.Status}。'
        )
    if model.IsMIP:
        metrics.gurobi_mip_gap = float(model.MIPGap)
    objective_value = float(model.ObjVal)
    metrics.set_tsp_objective = objective_value
    metrics.set_tsp_total_seconds = time.perf_counter() - total_start

    sequence = _extract_sequence(select, set_count)
    metrics.assert_conservation()
    return SetTSPSolveResult(sequence, objective_value, metrics)
