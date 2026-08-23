"""剪枝实验结果分析脚本的独立测试。"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "analyze_pruning_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_pruning_results", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(
    group: str,
    *,
    total_seconds: float,
    variables: int,
    sequence: str,
    full_objective: float,
) -> dict:
    """构造一个最小但字段完整的人工逐仓库实验记录。"""

    # 初始和最终候选数量用于验证分析脚本的变量守恒检查。
    initial = {"select": 4, "flow": 4, "internal": 4, "external": 8}
    final = {"select": 4, "flow": 4, "internal": 4, "external": variables - 12}
    return {
        "scenario": "toy",
        "customer_count": 2,
        "instance_index": 0,
        "depot_index": 0,
        "depot_node": "0",
        "assigned_customer_count": 2,
        "group": group,
        "phase2_total_seconds": total_seconds,
        "phase2_gurobi_seconds": total_seconds * 0.7,
        "phase2_model_build_seconds": total_seconds * 0.2,
        "phase2_pruning_pipeline_seconds": total_seconds * 0.1 if group != "C0" else 0.0,
        "phase2_variables": variables,
        "phase2_constraints": variables + 3,
        "phase2_gurobi_nodes": 1.0,
        "phase2_objective": 5.0,
        "phase2_objective_delta_from_c0": 0.0,
        "phase2_objective_consistent": True,
        "full_objective": full_objective,
        "full_objective_delta_from_c0": abs(full_objective - 10.0),
        "full_solve_seconds": total_seconds + 5.0,
        "set_tsp_sequence": sequence,
        **{f"phase2_initial_{name}": count for name, count in initial.items()},
        **{f"phase2_final_{name}": count for name, count in final.items()},
        "phase2_initial_fixed_internal_one": 0,
        "phase2_final_fixed_internal_one": 0,
    }


def test_analysis_separates_phase2_correctness_from_full_objective_change():
    """Phase 2 目标相同但序列变化时，应单独报告端到端目标变化。"""

    frame = pd.DataFrame(
        [
            _record("C0", total_seconds=4.0, variables=20, sequence="[0,1,2,0]", full_objective=10.0),
            _record("P1", total_seconds=2.0, variables=16, sequence="[0,1,2,0]", full_objective=10.0),
            _record("P3", total_seconds=1.0, variables=14, sequence="[0,2,1,0]", full_objective=10.5),
            _record("P7", total_seconds=2.0, variables=15, sequence="[0,1,2,0]", full_objective=10.0),
            _record("P1_P3_P7", total_seconds=1.0, variables=13, sequence="[0,2,1,0]", full_objective=10.5),
        ]
    )

    summary = MODULE.analyze_results(frame)
    p3 = next(item for item in summary["full_objective"] if item["group"] == "P3")
    phase2_p3 = next(item for item in summary["phase2_overall"] if item["group"] == "P3")

    assert summary["integrity"]["all_phase2_objectives_consistent"] is True
    assert p3["objective_changed_runs"] == 1
    assert p3["sequence_changed_runs"] == 1
    assert p3["changed_objective_without_sequence_change"] == 0
    assert math.isclose(phase2_p3["aggregate_phase2_speedup"], 4.0)
    assert math.isclose(phase2_p3["weighted_variable_reduction_rate"], 0.3)


def test_integrity_reports_a_missing_combination_group_without_aborting():
    """服务器中断导致组合组缺失时，应保留分析并明确列出缺失键。"""

    rows = [
        _record("C0", total_seconds=4.0, variables=20, sequence="[0,1,2,0]", full_objective=10.0),
        _record("P1", total_seconds=3.0, variables=18, sequence="[0,1,2,0]", full_objective=10.0),
        _record("P3", total_seconds=3.0, variables=18, sequence="[0,1,2,0]", full_objective=10.0),
        _record("P7", total_seconds=3.0, variables=18, sequence="[0,1,2,0]", full_objective=10.0),
    ]
    integrity = MODULE.validate_integrity(pd.DataFrame(rows))

    assert integrity["duplicate_depot_group_rows"] == 0
    assert integrity["missing_or_unexpected_runs"] == [
        {
            "scenario": "toy",
            "customer_count": 2,
            "instance_index": 0,
            "missing_groups": ["P1_P3_P7"],
            "unexpected_groups": [],
        }
    ]


def test_final_candidate_counts_must_equal_solver_variable_count():
    """最终候选计数和求解器变量数不一致时，完整性检查应标红。"""

    row = _record("C0", total_seconds=4.0, variables=20, sequence="[0,1,2,0]", full_objective=10.0)
    row["phase2_final_external"] += 1
    integrity = MODULE.validate_integrity(pd.DataFrame([row]), expected_groups=("C0",))

    assert integrity["final_counts_equal_solver_variables"] is False
