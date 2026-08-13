"""保持原算法不变的 P1/P3/P7 Set-TSP 剪枝入口。"""

from __future__ import annotations

import time
from typing import Optional

from .fstsp import MultiAgentFlyingSidekickTSP
from .pruning import (
    PruningOptions,
    SetTSPSolverOptions,
    apply_pruning_pipeline,
    solve_set_tsp,
)


class PrunedMultiAgentFlyingSidekickTSP(MultiAgentFlyingSidekickTSP):
    """在原三阶段算法上仅替换 Phase 2 Set-TSP 实现。

    输入与原类一致，额外接受 ``pruning_options`` 控制 P1、P3、P7。
    输出仍由继承的 ``solve`` 返回，便于和原算法直接对照。
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
        pruning_options: Optional[PruningOptions] = None,
        solver_options: Optional[SetTSPSolverOptions] = None,
    ):
        """初始化原算法状态以及独立的 Phase 2 剪枝记录。"""

        super().__init__(graph, depots, cities, distance, drone, limit, speed, theta)
        self.pruning_options = pruning_options or PruningOptions()
        self.set_tsp_solver_options = solver_options or SetTSPSolverOptions()
        # 每个仓库会产生一次 Set-TSP 运行，分别保留报告和求解结果。
        self.phase2_pruning_reports = []
        self.phase2_set_tsp_results = []
        self.phase2_pipeline_seconds = []
        self.phase2_total_seconds = []

    def set_tsp(self, convex_sets, distance, convex_set_distance):
        """应用配置的剪枝流水线并调用统一稀疏 Set-TSP 模型。

        输入/输出与原 ``set_tsp`` 保持兼容，因此 Phase 1 和 Phase 3 可直接
        复用原实现，实验差异被限制在 Phase 2。
        """

        total_started = time.perf_counter()
        set_sizes = tuple(len(convex_set) for convex_set in convex_sets)
        pruning_started = time.perf_counter()
        candidate_space, report = apply_pruning_pipeline(
            set_sizes,
            distance,
            convex_set_distance,
            self.pruning_options,
        )
        pipeline_seconds = time.perf_counter() - pruning_started
        result = solve_set_tsp(
            candidate_space,
            distance,
            convex_set_distance,
            solver_options=getattr(self, 'set_tsp_solver_options', None),
        )
        self.phase2_pruning_reports.append(report)
        self.phase2_set_tsp_results.append(result)
        self.phase2_pipeline_seconds.append(pipeline_seconds)
        self.phase2_total_seconds.append(time.perf_counter() - total_started)
        return list(result.sequence)

    def phase2_metrics(self):
        """返回每个仓库独立的候选计数、剪枝指标与求解统计。"""

        records = []
        for report, result, pipeline_seconds, total_seconds in zip(
            self.phase2_pruning_reports,
            self.phase2_set_tsp_results,
            self.phase2_pipeline_seconds,
            self.phase2_total_seconds,
        ):
            record = report.metrics()
            record.update(
                {
                    'phase2_variables': result.variable_count,
                    'phase2_constraints': result.constraint_count,
                    'phase2_gurobi_nodes': result.node_count,
                    'phase2_model_build_seconds': result.model_build_seconds,
                    'phase2_gurobi_seconds': result.runtime_seconds,
                    'phase2_pruning_pipeline_seconds': pipeline_seconds,
                    'phase2_total_seconds': total_seconds,
                    'phase2_mip_gap': result.mip_gap,
                    'phase2_objective_bound': result.objective_bound,
                    'phase2_objective': result.objective,
                    'phase2_solver_seed': result.solver_seed,
                    'phase2_solver_mip_gap_target': result.solver_mip_gap_target,
                    'phase2_solver_mip_gap_abs_target': result.solver_mip_gap_abs_target,
                    'phase2_solver_optimality_tolerance': result.solver_optimality_tolerance,
                    'phase2_solver_feasibility_tolerance': result.solver_feasibility_tolerance,
                    'phase2_solver_integrality_tolerance': result.solver_integrality_tolerance,
                    'phase2_gurobi_version': '.'.join(map(str, result.gurobi_version)),
                }
            )
            records.append(record)
        return records
