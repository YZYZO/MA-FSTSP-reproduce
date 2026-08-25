"""
把 P7 预处理和稀疏 Set-TSP 接入原 MA-FSTSP 类，而不修改 `src/fstsp.py`。
"""

from typing import Dict, Sequence

import numpy as np

from src.fstsp import MultiAgentFlyingSidekickTSP

from .endpoint_pair_dominance import prune_endpoint_pairs
from .set_tsp_solver import solve_sparse_set_tsp
from .types import PruningOptions


def build_current_set_tsp_costs(model, depot, convex_sets):
    """
    按当前原算法的精确语义构造 Set-TSP 内部成本和外部成本。

    输入：MA-FSTSP 模型、当前仓库和按 `[仓库, 客户...]` 排列的候选集合。
    输出：内部成本 NumPy 方阵列表，以及键为 `(i,j)` 的外部成本矩阵字典。
    逻辑：逐字保持原 `get_seq` 的速度归一化、航程罚值和有向卡车距离，不在 P7 中修正原公式。
    """

    group = list(model.groups[depot])
    internal_costs = []
    for convex_set, city in zip(convex_sets, [depot] + group):
        matrix = np.asarray([
            [
                max(
                    model.distance['truck'][entry_node][exit_node],
                    model.cut_off(
                        model.distance['drone'][exit_node][city]
                        + model.distance['drone'][city][entry_node],
                        model.limit,
                    ),
                ) / model.speed
                for exit_node in convex_set
            ]
            for entry_node in convex_set
        ], dtype=float)
        internal_costs.append(matrix)

    external_costs = {}
    for source_set, source_nodes in enumerate(convex_sets):
        for target_set, target_nodes in enumerate(convex_sets):
            if source_set == target_set:
                continue
            external_costs[source_set, target_set] = np.asarray([
                [model.distance['truck'][source_node][target_node] for target_node in target_nodes]
                for source_node in source_nodes
            ], dtype=float)
    return internal_costs, external_costs


class PruningMultiAgentFlyingSidekickTSP(MultiAgentFlyingSidekickTSP):
    """
    通过继承方式为原算法增加稀疏 Set-TSP 与可选 P7，不改变原类文件。

    输入：原算法全部初始化参数，以及可选 `PruningOptions`。
    输出：行为兼容原算法的模型实例；Phase 2 额外保存剪枝和求解指标。
    逻辑：仅覆写 `get_seq`，Phase 1、Phase 3、解格式和总成本累加逻辑全部继承原实现。
    """

    def __init__(
        self,
        graph,
        depots,
        cities,
        distance,
        drone,
        limit=1.5,
        speed=1.6,
        theta=(0.5, 0.5),
        pruning_options=None,
    ):
        """
        初始化剪枝适配模型及每仓库 Phase 2 报告容器。

        输入：原 MA-FSTSP 参数和可选剪枝配置。
        输出：无；完成原模型初始化并准备审计缓存。
        逻辑：P7 默认关闭，因此显式启用前只使用稀疏结构基线。
        """

        super().__init__(graph, depots, cities, distance, drone, limit, speed, theta)
        self.pruning_options = pruning_options or PruningOptions()
        self.pruning_options.validate()
        # 键为仓库节点；每次 `get_seq` 写入一个可 JSON 序列化的 Phase 2 报告。
        self.phase2_pruning_reports: Dict[object, dict] = {}
        # 只在发生等价顺序复核时使用，正常实验不会额外求解。
        self._phase2_problem_cache: Dict[object, dict] = {}

    def get_seq(self, depot, convex_sets):
        """
        使用原算法成本构造、P7 预处理和稀疏模型获得客户集合顺序。

        输入：当前仓库和 `[仓库集合, 客户集合...]`。
        输出：与原实现格式相同、首尾为集合 0 的顺序列表。
        逻辑：关闭 Set-TSP 时完全委托原方法；否则记录剪枝证据、模型指标和可选复核缓存。
        """

        if self.theta[1] == 0:
            return super().get_seq(depot, convex_sets)

        internal_costs, external_costs = build_current_set_tsp_costs(
            self,
            depot,
            convex_sets,
        )
        dominance_result = prune_endpoint_pairs(
            internal_costs,
            external_costs,
            self.pruning_options,
        )
        solve_result = solve_sparse_set_tsp(
            internal_costs,
            external_costs,
            dominance_result,
            self.pruning_options,
        )
        self.phase2_pruning_reports[depot] = {
            'solver': (
                'Sparse-Set-TSP-P7'
                if self.pruning_options.endpoint_pair_dominance
                else 'Sparse-Set-TSP-Baseline'
            ),
            'sequence': list(solve_result.sequence),
            'metrics': solve_result.metrics.to_dict(),
            'evidence': [record.to_dict() for record in dominance_result.evidence],
        }
        self._phase2_problem_cache[depot] = {
            'internal_costs': internal_costs,
            'external_costs': external_costs,
            'dominance_result': dominance_result,
        }
        return list(solve_result.sequence)

    def evaluate_fixed_sequence(self, depot, sequence: Sequence[int]):
        """
        在当前模型的端点候选空间中复核一个固定集合顺序的最优端点成本。

        输入：仓库节点和首尾为 0 的集合访问顺序。
        输出：固定该顺序后的最优 Set-TSP 目标值。
        逻辑：仅用于 B/C 顺序不同时的审计，不写回正式运行指标或最终路线。
        """

        if depot not in self._phase2_problem_cache:
            raise KeyError(f'仓库 {depot!r} 没有可用于固定顺序复核的 Phase 2 缓存。')
        cached = self._phase2_problem_cache[depot]
        # 深复制指标，避免审计模型覆盖正式模型的建模和 Gurobi 时间。
        original_result = cached['dominance_result']
        audit_result = type(original_result)(
            kept_pairs_by_arc=original_result.kept_pairs_by_arc,
            metrics=type(original_result.metrics)(
                **{
                    key: value
                    for key, value in original_result.metrics.__dict__.items()
                    if key in type(original_result.metrics).__dataclass_fields__
                }
            ),
            evidence=[],
        )
        result = solve_sparse_set_tsp(
            cached['internal_costs'],
            cached['external_costs'],
            audit_result,
            self.pruning_options,
            fixed_sequence=sequence,
        )
        return result.objective


def attach_pruning_process_data(model, process_data):
    """
    把适配模型保存的每仓库 P7 报告附加到现有三阶段 `process_data`。

    输入：已求解模型和 `_solve_model_with_process_data` 返回的过程字典。
    输出：原位补充后同时返回的过程字典。
    逻辑：按仓库节点匹配记录；旧保存器会忽略新增字段，因此保持既有 NPZ 格式兼容。
    """

    reports = getattr(model, 'phase2_pruning_reports', {})
    for record in process_data.get('depot_records', []):
        report = reports.get(record['depot_node'])
        if report is None:
            continue
        record['set_tsp_solver'] = report['solver']
        record['pruning'] = report
    process_data['pruning_reports'] = [
        reports[depot]
        for depot in model.depots
        if depot in reports
    ]
    return process_data

