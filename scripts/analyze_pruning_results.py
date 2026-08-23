"""分析 P1、P3、P7 Set-TSP 剪枝消融实验结果。

脚本把逐仓库的 Phase 2 指标与逐实例的端到端指标分开聚合，避免同一个
``full_objective`` 因活跃仓库数量不同而被重复计权。默认在输入 CSV 同目录
生成一份 Markdown 报告和一份机器可读 JSON 摘要。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


EXPECTED_GROUPS = ("C0", "P1", "P3", "P7", "P1_P3_P7")
DEPOT_KEY = ("scenario", "customer_count", "instance_index", "depot_index")
RUN_KEY = ("scenario", "customer_count", "instance_index")
CANDIDATE_TYPES = ("select", "flow", "internal", "external")
OBJECTIVE_TOLERANCE = 1e-7


def _python_scalar(value: Any) -> Any:
    """把 NumPy/Pandas 标量转换成可直接写入 JSON 的 Python 标量。

    输入：任意汇总值。
    输出：``int``、``float``、``bool``、``None`` 或原始 Python 对象；
    非有限浮点数转换为 ``None``，避免生成非标准 JSON。
    """

    if pd.isna(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """将数据表转换为不含 NumPy 标量和 NaN 的 JSON 记录列表。"""

    return [
        {str(key): _python_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """逐元素计算安全比值，分母非正或缺失时返回 NaN。"""

    valid = denominator.notna() & (denominator > 0)
    result = pd.Series(np.nan, index=denominator.index, dtype=float)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def _quantile(series: pd.Series, probability: float) -> float:
    """返回去除缺失值后的分位数；空序列返回 NaN。"""

    clean = series.dropna()
    return float(clean.quantile(probability)) if not clean.empty else math.nan


def _format_markdown_table(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[tuple[str, str, str]],
) -> str:
    """用固定列定义生成无需额外依赖的 Markdown 表格。

    输入：记录列表，以及 ``(字段名, 表头, 格式)`` 三元组列表。
    输出：可直接写入报告的 Markdown 表格文本。
    """

    def format_value(value: Any, specifier: str) -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "-"
        if specifier == "":
            return str(value)
        return format(value, specifier)

    header = "| " + " | ".join(title for _, title, _ in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(
                format_value(row.get(field), specifier)
                for field, _, specifier in columns
            )
            + " |"
        )
    return "\n".join([header, separator, *body])


def load_results(csv_path: Path) -> pd.DataFrame:
    """读取并规范化剪枝实验 CSV。

    输入：实验 CSV 路径。
    输出：按场景、规模、实例、组、仓库排序的数据表。
    """

    frame = pd.read_csv(csv_path)
    required = {
        *DEPOT_KEY,
        "group",
        "phase2_total_seconds",
        "phase2_variables",
        "phase2_objective",
        "phase2_objective_delta_from_c0",
        "phase2_objective_consistent",
        "full_objective",
        "full_objective_delta_from_c0",
        "full_solve_seconds",
        "set_tsp_sequence",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"实验 CSV 缺少必要字段：{missing}")
    return frame.sort_values([*RUN_KEY, "group", "depot_index"]).reset_index(drop=True)


def validate_integrity(
    frame: pd.DataFrame,
    expected_groups: Sequence[str] = EXPECTED_GROUPS,
    tolerance: float = OBJECTIVE_TOLERANCE,
) -> dict[str, Any]:
    """检查记录唯一性、实验组覆盖、候选计数和基线一致性。

    输入：原始数据、预期实验组和目标容差。
    输出：完整性诊断字典；不因缺失运行直接抛错，以便中断结果仍可分析。
    """

    duplicate_mask = frame.duplicated([*DEPOT_KEY, "group"], keep=False)
    duplicate_count = int(duplicate_mask.sum())

    run_groups = frame.groupby(list(RUN_KEY), sort=True)["group"].agg(set)
    missing_runs = []
    expected_group_set = set(expected_groups)
    for run_key, observed in run_groups.items():
        missing = sorted(expected_group_set - set(observed))
        unexpected = sorted(set(observed) - expected_group_set)
        if missing or unexpected:
            missing_runs.append(
                {
                    "scenario": run_key[0],
                    "customer_count": int(run_key[1]),
                    "instance_index": int(run_key[2]),
                    "missing_groups": missing,
                    "unexpected_groups": unexpected,
                }
            )

    # 同一实例/实验组的端到端指标应在所有活跃仓库行中完全重复。
    run_metric_variation = (
        frame.groupby([*RUN_KEY, "group"])[
            ["full_objective", "full_solve_seconds", "full_objective_delta_from_c0"]
        ]
        .nunique(dropna=False)
        .max(axis=1)
    )
    inconsistent_run_metric_groups = int((run_metric_variation > 1).sum())

    initial_count_consistency = True
    initial_columns = [f"phase2_initial_{name}" for name in CANDIDATE_TYPES]
    if set(initial_columns).issubset(frame.columns):
        initial_count_consistency = bool(
            (
                frame.groupby(list(DEPOT_KEY))[initial_columns]
                .nunique(dropna=False)
                .le(1)
                .all(axis=None)
            )
        )

    final_columns = [f"phase2_final_{name}" for name in CANDIDATE_TYPES]
    variable_count_consistency = None
    if set(final_columns).issubset(frame.columns):
        final_sum = frame[final_columns].sum(axis=1)
        variable_count_consistency = bool(
            np.allclose(final_sum, frame["phase2_variables"], atol=0.0, rtol=0.0)
        )

    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "run_count": int(frame.groupby([*RUN_KEY, "group"]).ngroups),
        "base_instance_count": int(frame.groupby(list(RUN_KEY)).ngroups),
        "duplicate_depot_group_rows": duplicate_count,
        "missing_or_unexpected_runs": missing_runs,
        "inconsistent_run_metric_groups": inconsistent_run_metric_groups,
        "initial_counts_equal_across_groups": initial_count_consistency,
        "final_counts_equal_solver_variables": variable_count_consistency,
        "all_phase2_objectives_consistent": bool(
            frame["phase2_objective_consistent"].fillna(False).all()
        ),
        "max_phase2_objective_delta": float(
            frame["phase2_objective_delta_from_c0"].max()
        ),
        "phase2_delta_exceeds_tolerance": int(
            (frame["phase2_objective_delta_from_c0"] > tolerance).sum()
        ),
        "max_full_objective_absolute_delta": float(
            frame["full_objective_delta_from_c0"].max()
        ),
        "groups": {
            str(group): int(count)
            for group, count in frame["group"].value_counts().sort_index().items()
        },
    }


def build_paired_depot_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """把每个剪枝组的逐仓库记录与同一仓库 C0 基线配对。

    输入：原始逐仓库实验表。
    输出：仅含非 C0 记录的数据表，并附带后缀 ``_c0`` 的基线指标。
    """

    baseline_columns = [
        *DEPOT_KEY,
        "phase2_total_seconds",
        "phase2_gurobi_seconds",
        "phase2_model_build_seconds",
        "phase2_variables",
        "phase2_constraints",
        "phase2_gurobi_nodes",
        "phase2_objective",
        "set_tsp_sequence",
    ]
    baseline = frame.loc[frame["group"] == "C0", baseline_columns]
    baseline = baseline.rename(
        columns={name: f"{name}_c0" for name in baseline_columns if name not in DEPOT_KEY}
    )
    paired = frame.loc[frame["group"] != "C0"].merge(
        baseline,
        on=list(DEPOT_KEY),
        how="left",
        validate="many_to_one",
    )
    if paired["phase2_total_seconds_c0"].isna().any():
        raise ValueError("存在没有 C0 仓库基线的剪枝记录。")

    paired["phase2_speedup_paired"] = _safe_ratio(
        paired["phase2_total_seconds_c0"], paired["phase2_total_seconds"]
    )
    paired["gurobi_speedup_paired"] = _safe_ratio(
        paired["phase2_gurobi_seconds_c0"], paired["phase2_gurobi_seconds"]
    )
    paired["variable_reduction_rate"] = (
        1.0 - paired["phase2_variables"] / paired["phase2_variables_c0"]
    )
    paired["constraint_reduction_rate"] = (
        1.0 - paired["phase2_constraints"] / paired["phase2_constraints_c0"]
    )
    paired["sequence_changed"] = (
        paired["set_tsp_sequence"] != paired["set_tsp_sequence_c0"]
    )
    return paired


def build_run_rows(frame: pd.DataFrame, paired_depots: pd.DataFrame) -> pd.DataFrame:
    """构造逐实例记录并汇总该实例所有活跃仓库的 Phase 2 时间。

    输入：原始数据与已经匹配 C0 的逐仓库数据。
    输出：非 C0 的逐实例表，每个实例/组恰好一行。
    """

    run_metrics = (
        frame.groupby([*RUN_KEY, "group"], as_index=False)
        .agg(
            full_objective=("full_objective", "first"),
            full_solve_seconds=("full_solve_seconds", "first"),
            full_objective_delta_from_c0=("full_objective_delta_from_c0", "first"),
            phase2_total_seconds_sum=("phase2_total_seconds", "sum"),
            active_depots=("depot_index", "size"),
        )
    )
    base = run_metrics.loc[run_metrics["group"] == "C0"].drop(columns="group")
    base = base.rename(
        columns={
            "full_objective": "full_objective_c0",
            "full_solve_seconds": "full_solve_seconds_c0",
            "full_objective_delta_from_c0": "full_objective_delta_from_c0_c0",
            "phase2_total_seconds_sum": "phase2_total_seconds_sum_c0",
            "active_depots": "active_depots_c0",
        }
    )
    runs = run_metrics.loc[run_metrics["group"] != "C0"].merge(
        base,
        on=list(RUN_KEY),
        how="left",
        validate="many_to_one",
    )
    runs["phase2_run_speedup"] = _safe_ratio(
        runs["phase2_total_seconds_sum_c0"], runs["phase2_total_seconds_sum"]
    )
    runs["full_run_speedup"] = _safe_ratio(
        runs["full_solve_seconds_c0"], runs["full_solve_seconds"]
    )
    runs["full_objective_signed_delta"] = (
        runs["full_objective"] - runs["full_objective_c0"]
    )

    sequence_by_run = (
        paired_depots.groupby([*RUN_KEY, "group"], as_index=False)
        .agg(
            changed_depots=("sequence_changed", "sum"),
            any_sequence_changed=("sequence_changed", "any"),
        )
    )
    return runs.merge(
        sequence_by_run,
        on=[*RUN_KEY, "group"],
        how="left",
        validate="one_to_one",
    )


def summarize_phase2(
    paired_depots: pd.DataFrame,
    run_rows: pd.DataFrame,
) -> pd.DataFrame:
    """汇总每个剪枝组的 Phase 2 规模削减与配对耗时表现。"""

    summaries = []
    for group, depots in paired_depots.groupby("group", sort=False):
        runs = run_rows.loc[run_rows["group"] == group]
        baseline_total = float(depots["phase2_total_seconds_c0"].sum())
        pruned_total = float(depots["phase2_total_seconds"].sum())
        baseline_gurobi = float(depots["phase2_gurobi_seconds_c0"].sum())
        pruned_gurobi = float(depots["phase2_gurobi_seconds"].sum())
        pipeline_total = float(depots["phase2_pruning_pipeline_seconds"].sum())
        summaries.append(
            {
                "group": group,
                "runs": int(len(runs)),
                "depots": int(len(depots)),
                "aggregate_phase2_speedup": baseline_total / pruned_total,
                "median_run_speedup": float(runs["phase2_run_speedup"].median()),
                "run_speedup_q1": _quantile(runs["phase2_run_speedup"], 0.25),
                "run_speedup_q3": _quantile(runs["phase2_run_speedup"], 0.75),
                "faster_run_rate": float((runs["phase2_run_speedup"] > 1.0).mean()),
                "aggregate_gurobi_speedup": baseline_gurobi / pruned_gurobi,
                "median_variable_reduction_rate": float(
                    depots["variable_reduction_rate"].median()
                ),
                "weighted_variable_reduction_rate": float(
                    1.0 - depots["phase2_variables"].sum() / depots["phase2_variables_c0"].sum()
                ),
                "weighted_constraint_reduction_rate": float(
                    1.0
                    - depots["phase2_constraints"].sum()
                    / depots["phase2_constraints_c0"].sum()
                ),
                "pipeline_share": pipeline_total / pruned_total,
                "sequence_changed_depot_rate": float(depots["sequence_changed"].mean()),
            }
        )
    return pd.DataFrame(summaries)


def summarize_phase2_by_scale(
    paired_depots: pd.DataFrame,
    run_rows: pd.DataFrame,
) -> pd.DataFrame:
    """按地图、客户规模和实验组汇总 Phase 2 配对结果。"""

    summaries = []
    scale_columns = ["scenario", "customer_count", "group"]
    for scale_key, depots in paired_depots.groupby(scale_columns, sort=True):
        scenario, customer_count, group = scale_key
        run_mask = (
            (run_rows["scenario"] == scenario)
            & (run_rows["customer_count"] == customer_count)
            & (run_rows["group"] == group)
        )
        runs = run_rows.loc[run_mask]
        baseline_total = float(depots["phase2_total_seconds_c0"].sum())
        pruned_total = float(depots["phase2_total_seconds"].sum())
        summaries.append(
            {
                "scenario": scenario,
                "customer_count": int(customer_count),
                "group": group,
                "runs": int(len(runs)),
                "depots": int(len(depots)),
                "aggregate_phase2_speedup": baseline_total / pruned_total,
                "median_run_speedup": float(runs["phase2_run_speedup"].median()),
                "faster_run_rate": float((runs["phase2_run_speedup"] > 1.0).mean()),
                "weighted_variable_reduction_rate": float(
                    1.0 - depots["phase2_variables"].sum() / depots["phase2_variables_c0"].sum()
                ),
                "pipeline_share": float(
                    depots["phase2_pruning_pipeline_seconds"].sum() / pruned_total
                ),
            }
        )
    return pd.DataFrame(summaries)


def summarize_stage_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """汇总 P1/P3/P7 直接记录的候选删除率与上下界质量。"""

    rows = []
    initial_total = frame[[f"phase2_initial_{name}" for name in CANDIDATE_TYPES]].sum(axis=1)
    for group, subset in frame.loc[frame["group"] != "C0"].groupby("group", sort=False):
        indices = subset.index
        row: dict[str, Any] = {
            "group": group,
            "initial_variables": int(initial_total.loc[indices].sum()),
            "final_variables": int(subset["phase2_variables"].sum()),
        }
        row["total_variable_reduction_rate"] = (
            1.0 - row["final_variables"] / row["initial_variables"]
        )

        def summed(column: str) -> int:
            return int(subset[column].fillna(0).sum()) if column in subset else 0

        row.update(
            {
                "p1_structural_removed": summed("phase2_structural_variables_skipped"),
                "p3_internal_removed": summed("phase2_internal_bound_pruned"),
                "p3_external_removed": summed("phase2_external_bound_pruned"),
                "p3_set_arcs_removed": summed("phase2_set_arcs_bound_pruned"),
                "p7_endpoint_pairs_removed": summed("phase2_endpoint_pairs_dominated"),
            }
        )
        p7_before = summed("phase2_endpoint_pairs_before")
        row["p7_dominance_rate"] = (
            row["p7_endpoint_pairs_removed"] / p7_before if p7_before else math.nan
        )

        # 兼容只包含公共指标的旧版或人工最小结果；缺少 P3 字段时留空统计。
        p3_columns = {"phase2_assignment_lower_bound", "phase2_incumbent_cost"}
        p3_rows = (
            subset.dropna(subset=["phase2_assignment_lower_bound"])
            if p3_columns.issubset(subset.columns)
            else subset.iloc[0:0]
        )
        positive_objective = p3_rows["phase2_objective"] > 0
        p3_rows = p3_rows.loc[positive_objective]
        if not p3_rows.empty:
            lower_bound_ratio = (
                p3_rows["phase2_assignment_lower_bound"] / p3_rows["phase2_objective"]
            )
            incumbent_ratio = p3_rows["phase2_incumbent_cost"] / p3_rows["phase2_objective"]
            row["median_assignment_lb_to_optimum"] = float(lower_bound_ratio.median())
            row["median_incumbent_gap"] = float((incumbent_ratio - 1.0).median())
        else:
            row["median_assignment_lb_to_optimum"] = math.nan
            row["median_incumbent_gap"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_full_objective(run_rows: pd.DataFrame) -> pd.DataFrame:
    """按实例统计端到端目标和总运行时间相对 C0 的变化。"""

    rows = []
    for group, subset in run_rows.groupby("group", sort=False):
        signed = subset["full_objective_signed_delta"]
        changed = signed.abs() > OBJECTIVE_TOLERANCE
        same_objective = subset.loc[~changed]
        rows.append(
            {
                "group": group,
                "runs": int(len(subset)),
                "objective_changed_runs": int(changed.sum()),
                "objective_improved_runs": int((signed < -OBJECTIVE_TOLERANCE).sum()),
                "objective_worsened_runs": int((signed > OBJECTIVE_TOLERANCE).sum()),
                "max_absolute_delta": float(signed.abs().max()),
                "mean_signed_delta": float(signed.mean()),
                "sequence_changed_runs": int(subset["any_sequence_changed"].sum()),
                "changed_objective_without_sequence_change": int(
                    (changed & ~subset["any_sequence_changed"]).sum()
                ),
                "aggregate_full_speedup": float(
                    subset["full_solve_seconds_c0"].sum()
                    / subset["full_solve_seconds"].sum()
                ),
                "median_full_speedup": float(subset["full_run_speedup"].median()),
                "median_full_speedup_same_objective": float(
                    same_objective["full_run_speedup"].median()
                )
                if not same_objective.empty
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_combo_by_scale(
    scale_summary: pd.DataFrame,
    run_rows: pd.DataFrame,
) -> pd.DataFrame:
    """提取组合组逐地图/规模性能并附加端到端目标变化计数。"""

    combo = scale_summary.loc[scale_summary["group"] == "P1_P3_P7"].copy()
    changed = (
        run_rows.loc[run_rows["group"] == "P1_P3_P7"]
        .assign(
            objective_changed=lambda value: (
                value["full_objective_signed_delta"].abs() > OBJECTIVE_TOLERANCE
            )
        )
        .groupby(["scenario", "customer_count"], as_index=False)
        .agg(
            objective_changed_runs=("objective_changed", "sum"),
            max_full_objective_delta=("full_objective_signed_delta", lambda value: value.abs().max()),
        )
    )
    return combo.merge(changed, on=["scenario", "customer_count"], how="left")


def identify_outliers(
    paired_depots: pd.DataFrame,
    run_rows: pd.DataFrame,
    limit: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    """找出最明显的 Phase 2 减速与端到端目标变化实例。"""

    phase2_columns = [
        *DEPOT_KEY,
        "group",
        "assigned_customer_count",
        "phase2_speedup_paired",
        "variable_reduction_rate",
        "phase2_total_seconds_c0",
        "phase2_total_seconds",
    ]
    phase2_slowest = paired_depots.nsmallest(limit, "phase2_speedup_paired")[phase2_columns]

    objective_columns = [
        *RUN_KEY,
        "group",
        "full_objective_c0",
        "full_objective",
        "full_objective_signed_delta",
        "changed_depots",
        "phase2_run_speedup",
        "full_run_speedup",
    ]
    largest_objective_changes = (
        run_rows.assign(abs_delta=run_rows["full_objective_signed_delta"].abs())
        .nlargest(limit, "abs_delta")[objective_columns]
    )
    return {
        "slowest_phase2_depots": _records(phase2_slowest),
        "largest_full_objective_changes": _records(largest_objective_changes),
    }


def analyze_results(frame: pd.DataFrame) -> dict[str, Any]:
    """执行完整分析并返回可序列化摘要。

    输入：经过 ``load_results`` 规范化的实验数据。
    输出：完整性、Phase 2、分剪枝阶段、端到端指标与异常点摘要。
    """

    integrity = validate_integrity(frame)
    paired_depots = build_paired_depot_rows(frame)
    run_rows = build_run_rows(frame, paired_depots)
    phase2 = summarize_phase2(paired_depots, run_rows)
    phase2_by_scale = summarize_phase2_by_scale(paired_depots, run_rows)
    stages = summarize_stage_effects(frame)
    full = summarize_full_objective(run_rows)
    combo = summarize_combo_by_scale(phase2_by_scale, run_rows)
    return {
        "integrity": integrity,
        "phase2_overall": _records(phase2),
        "phase2_by_scale": _records(phase2_by_scale),
        "stage_effects": _records(stages),
        "full_objective": _records(full),
        "combo_by_scale": _records(combo),
        "outliers": identify_outliers(paired_depots, run_rows),
    }


def render_markdown(summary: dict[str, Any], source_path: Path) -> str:
    """把分析摘要渲染为中文 Markdown 报告。"""

    integrity = summary["integrity"]
    missing_runs = integrity["missing_or_unexpected_runs"]
    missing_text = (
        "；".join(
            f'{item["scenario"]}/{item["customer_count"]}/实例{item["instance_index"]}'
            f' 缺少 {",".join(item["missing_groups"])}'
            for item in missing_runs
        )
        if missing_runs
        else "无"
    )
    phase2_table = _format_markdown_table(
        summary["phase2_overall"],
        (
            ("group", "组", ""),
            ("runs", "实例运行", "d"),
            ("aggregate_phase2_speedup", "Phase 2 聚合加速", ".3f"),
            ("median_run_speedup", "实例中位加速", ".3f"),
            ("faster_run_rate", "加速实例比例", ".1%"),
            ("weighted_variable_reduction_rate", "加权变量削减", ".1%"),
            ("pipeline_share", "预处理占组内时间", ".1%"),
            ("sequence_changed_depot_rate", "仓库序列变化率", ".1%"),
        ),
    )
    stage_table = _format_markdown_table(
        summary["stage_effects"],
        (
            ("group", "组", ""),
            ("total_variable_reduction_rate", "总变量削减", ".1%"),
            ("p1_structural_removed", "P1 删除", "d"),
            ("p3_internal_removed", "P3 内部删除", "d"),
            ("p3_external_removed", "P3 外部删除", "d"),
            ("p3_set_arcs_removed", "P3 集合弧删除", "d"),
            ("p7_endpoint_pairs_removed", "P7 端点对删除", "d"),
            ("p7_dominance_rate", "P7 支配率", ".1%"),
            ("median_assignment_lb_to_optimum", "P3 下界/最优中位数", ".3f"),
            ("median_incumbent_gap", "P3 上界 Gap 中位数", ".1%"),
        ),
    )
    full_table = _format_markdown_table(
        summary["full_objective"],
        (
            ("group", "组", ""),
            ("runs", "实例运行", "d"),
            ("objective_changed_runs", "端到端目标变化", "d"),
            ("objective_improved_runs", "改善", "d"),
            ("objective_worsened_runs", "变差", "d"),
            ("max_absolute_delta", "最大绝对变化", ".6f"),
            ("sequence_changed_runs", "序列变化实例", "d"),
            ("aggregate_full_speedup", "端到端聚合加速", ".3f"),
            ("median_full_speedup_same_objective", "同目标实例中位加速", ".3f"),
        ),
    )
    combo_table = _format_markdown_table(
        summary["combo_by_scale"],
        (
            ("scenario", "场景", ""),
            ("customer_count", "客户数", "d"),
            ("runs", "实例", "d"),
            ("aggregate_phase2_speedup", "Phase 2 聚合加速", ".3f"),
            ("median_run_speedup", "实例中位加速", ".3f"),
            ("weighted_variable_reduction_rate", "加权变量削减", ".1%"),
            ("pipeline_share", "预处理占比", ".1%"),
            ("objective_changed_runs", "端到端目标变化", "d"),
        ),
    )

    return f"""# P1/P3/P7 剪枝实验分析

数据源：`{source_path.as_posix()}`

## 1. 数据完整性与正确性

- 共 {integrity['row_count']} 条逐仓库记录、{integrity['run_count']} 个实验组运行、{integrity['base_instance_count']} 个基础实例。
- 缺失运行：{missing_text}。
- 重复仓库-实验组键：{integrity['duplicate_depot_group_rows']}。
- Phase 2 目标在 `1e-7` 容差内全部一致：{integrity['all_phase2_objectives_consistent']}；最大差值为 {integrity['max_phase2_objective_delta']:.3e}。
- 端到端目标相对 C0 的最大绝对差值为 {integrity['max_full_objective_absolute_delta']:.6f}。
- 初始候选计数跨组一致：{integrity['initial_counts_equal_across_groups']}；最终候选计数等于求解器变量数：{integrity['final_counts_equal_solver_variables']}。

## 2. Phase 2 总体表现

聚合加速比按“全部配对仓库的 C0 总耗时 / 剪枝组总耗时”计算；实例中位数先对每个实例汇总全部活跃仓库，再做中位数。

{phase2_table}

## 3. 各剪枝的候选削减

{stage_table}

## 4. 端到端目标与总时间

Phase 2 只保证当前 Set-TSP 数学模型的最优目标不变。若多个 Set-TSP 序列等价最优，稀疏模型可能返回不同序列，而 Phase 3 的动态规划依赖该序列，因此端到端目标可能变化。

{full_table}

## 5. 组合组按场景与规模

{combo_table}

## 6. 解读限制

- 每个实例/实验组只有一次计时，且组按固定顺序运行；当前数据可用于工程趋势判断，不足以独立估计机器噪声或给出稳健显著性结论。
- 实验记录没有峰值内存和距离查询数，尚未覆盖方案文档 P0 的全部指标。
- `P1_P3_P7` 缺少一个完整实例，跨组汇总采用现有可配对记录，正式论文表格前应补跑。
- 端到端时间只有在目标与序列口径明确后才宜作为剪枝收益；Phase 2 的 `pipeline + model + optimize` 总时间是当前最直接的性能指标。
"""


def write_outputs(
    summary: dict[str, Any],
    csv_path: Path,
    output_prefix: Path | None = None,
) -> tuple[Path, Path]:
    """原子写出 Markdown 和 JSON 分析产物。

    输入：分析摘要、源 CSV 和可选输出前缀。
    输出：Markdown 与 JSON 文件路径。
    """

    prefix = output_prefix or csv_path.with_name(f"{csv_path.stem}_analysis")
    markdown_path = prefix.with_suffix(".md")
    json_path = prefix.with_suffix(".json")
    markdown_temporary = markdown_path.with_suffix(".md.tmp")
    json_temporary = json_path.with_suffix(".json.tmp")
    markdown_temporary.write_text(
        render_markdown(summary, csv_path), encoding="utf-8"
    )
    json_temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown_temporary.replace(markdown_path)
    json_temporary.replace(json_path)
    return markdown_path, json_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """解析命令行参数并返回输入路径与可选输出前缀。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="剪枝实验 CSV 路径")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="输出前缀；默认与 CSV 同目录并追加 _analysis",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """读取实验、执行分析、写出报告并打印产物路径。"""

    args = parse_args(argv)
    frame = load_results(args.csv_path)
    summary = analyze_results(frame)
    markdown_path, json_path = write_outputs(summary, args.csv_path, args.output_prefix)
    print(f"Markdown analysis: {markdown_path}")
    print(f"JSON analysis: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
