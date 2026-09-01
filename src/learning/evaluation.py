"""里程碑 D 客户划分方法的真实下游评估与统一指标汇总。"""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from src.learning.evaluator import GroupEvaluator
from src.learning.settings import SetTSPSolverSettings


def _jsonable(value):
    """
    将评估记录递归转换为 JSON 基础类型。

    输入：可能包含 NumPy 标量、数组、路径和非有限数的对象。
    输出：可由标准 JSON 编码器处理的值。
    逻辑：详细 CSV 与摘要 JSON 共用同一无状态序列化口径。
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _percentile(values, percentile):
    """
    计算一个非空或空数列的百分位数。

    输入：数值列表和 0—100 百分位。
    输出：浮点百分位；空列表返回零。
    逻辑：统一 P90、最大值等实验统计的 NumPy 实现。
    """
    return float(np.percentile(values, percentile)) if values else 0.0


def evaluate_partition_result(
    context,
    partition_result,
    map_id,
    map_label,
    customer_size,
    instance_index,
    cache,
    time_limit_seconds=30.0,
    solver_threads=1,
    solver_seed=0,
    group_progress_callback=None,
):
    """
    对一种客户划分真实执行第二阶段 Set-TSP 和第三阶段 DP。

    输入：实例上下文、划分结果、实验标识、缓存和求解器配置。
    输出：包含实例总成本、各组时间、超时与调用次数的详细记录。
    逻辑：每个仓库客户组独立评估并共享 SQLite 缓存，超时成本显式标记为近似值。
    """
    model = context.model
    evaluator = GroupEvaluator(
        model,
        context.convex_sets,
        map_id=map_id,
        cache=cache,
        solver_settings=SetTSPSolverSettings(
            time_limit_seconds=time_limit_seconds,
            threads=solver_threads,
            seed=solver_seed,
        ),
    )
    group_records = []
    for depot in model.depots:
        customers = partition_result.groups[depot]
        evaluation = evaluator.evaluate(depot, customers)
        solver_result = evaluation.set_tsp_result
        group_record = {
                "depot": depot,
                "customer_count": len(customers),
                "customers": list(evaluation.customers),
                "final_cost": evaluation.final_cost,
                "set_tsp_wall_seconds": evaluation.set_tsp_wall_seconds,
                "set_tsp_solver_runtime_seconds": solver_result.runtime_seconds,
                "phase3_seconds": evaluation.phase3_seconds,
                "downstream_seconds": (
                    evaluation.set_tsp_wall_seconds + evaluation.phase3_seconds
                ),
                "set_tsp_status": solver_result.status_name,
                "set_tsp_timeout": int(solver_result.timed_out),
                "set_tsp_mip_gap": solver_result.mip_gap,
                "cost_is_approximate": int(solver_result.timed_out),
                "cache_hit": int(evaluation.cache_hit),
            }
        group_records.append(group_record)
        if group_progress_callback is not None:
            group_progress_callback(group_record)

    downstream_seconds = sum(row["downstream_seconds"] for row in group_records)
    final_cost = sum(row["final_cost"] for row in group_records)
    downstream_real_calls = sum(1 - row["cache_hit"] for row in group_records)
    total_real_calls = (
        partition_result.strategy_real_solver_calls + downstream_real_calls
    )
    end_to_end_seconds = (
        context.boundary_construction_seconds
        + partition_result.model_loading_seconds
        + partition_result.partition_strategy_seconds
        + downstream_seconds
    )
    return _jsonable(
        {
            "map_id": map_id,
            "map_label": map_label,
            "customer_size": int(customer_size),
            "instance_index": int(instance_index),
            "method": partition_result.method,
            "groups": partition_result.groups,
            "group_sizes": [row["customer_count"] for row in group_records],
            "boundary_construction_seconds": context.boundary_construction_seconds,
            "model_loading_seconds": partition_result.model_loading_seconds,
            "partition_strategy_seconds": partition_result.partition_strategy_seconds,
            "move_count": partition_result.move_count,
            "stop_reason": partition_result.stop_reason,
            "predicted_initial_score": partition_result.predicted_initial_score,
            "predicted_final_score": partition_result.predicted_final_score,
            "final_cost": float(final_cost),
            "cost_has_approximate_group": int(
                any(row["cost_is_approximate"] for row in group_records)
            ),
            "set_tsp_total_seconds": float(
                sum(row["set_tsp_wall_seconds"] for row in group_records)
            ),
            "phase3_total_seconds": float(
                sum(row["phase3_seconds"] for row in group_records)
            ),
            "downstream_total_seconds": float(downstream_seconds),
            "end_to_end_seconds": float(end_to_end_seconds),
            "timeout_count": int(
                sum(row["set_tsp_timeout"] for row in group_records)
            ),
            "strategy_real_solver_calls": partition_result.strategy_real_solver_calls,
            "downstream_real_solver_calls": downstream_real_calls,
            "total_real_solver_calls": total_real_calls,
            "group_records": group_records,
        }
    )


def summarize_partition_evaluations(records, cost_tolerance=0.01):
    """
    按地图、客户规模和方法汇总成本、P90 时间、超时率及调用次数。

    输入：实例详细记录和相对 symmetric MST 的成本容忍比例。
    输出：包含汇总行、约束结论和配置的 JSON 对象。
    逻辑：P90/最大时间按客户组统计，策略耗时按实例统计，成本按实例总目标聚合。
    """
    records = list(records)
    buckets = {}
    for record in records:
        key = (record["map_label"], record["customer_size"], record["method"])
        buckets.setdefault(key, []).append(record)

    summary_rows = []
    for (map_label, customer_size, method), bucket in sorted(buckets.items()):
        group_rows = [
            group
            for record in bucket
            for group in record["group_records"]
        ]
        set_tsp_times = [row["set_tsp_wall_seconds"] for row in group_rows]
        downstream_times = [row["downstream_seconds"] for row in group_rows]
        strategy_times = [row["partition_strategy_seconds"] for row in bucket]
        end_to_end_times = [row["end_to_end_seconds"] for row in bucket]
        summary_rows.append(
            {
                "map_label": map_label,
                "customer_size": int(customer_size),
                "method": method,
                "instance_count": len(bucket),
                "group_count": len(group_rows),
                "final_cost_total": float(sum(row["final_cost"] for row in bucket)),
                "final_cost_mean": float(np.mean([row["final_cost"] for row in bucket])),
                "approximate_cost_instance_count": int(
                    sum(row["cost_has_approximate_group"] for row in bucket)
                ),
                "set_tsp_p90_seconds": _percentile(set_tsp_times, 90),
                "set_tsp_max_seconds": max(set_tsp_times, default=0.0),
                "downstream_p90_seconds": _percentile(downstream_times, 90),
                "downstream_max_seconds": max(downstream_times, default=0.0),
                "timeout_rate": (
                    sum(row["set_tsp_timeout"] for row in group_rows)
                    / max(len(group_rows), 1)
                ),
                "strategy_mean_seconds": float(np.mean(strategy_times)),
                "strategy_p90_seconds": _percentile(strategy_times, 90),
                "strategy_max_seconds": max(strategy_times, default=0.0),
                "end_to_end_p90_seconds": _percentile(end_to_end_times, 90),
                "end_to_end_max_seconds": max(end_to_end_times, default=0.0),
                "max_group_customer_count": max(
                    (row["customer_count"] for row in group_rows),
                    default=0,
                ),
                "strategy_real_solver_calls": int(
                    sum(row["strategy_real_solver_calls"] for row in bucket)
                ),
                "downstream_group_evaluation_count": len(group_rows),
                "downstream_real_solver_calls": int(
                    sum(row["downstream_real_solver_calls"] for row in bucket)
                ),
                "total_real_solver_calls": int(
                    sum(row["total_real_solver_calls"] for row in bucket)
                ),
            }
        )

    baseline_costs = {
        (row["map_label"], row["customer_size"]): row["final_cost_total"]
        for row in summary_rows
        if row["method"] == "symmetric_mst"
    }
    baseline_runtime = {
        (row["map_label"], row["customer_size"]): row
        for row in summary_rows
        if row["method"] == "symmetric_mst"
    }
    for row in summary_rows:
        key = (row["map_label"], row["customer_size"])
        baseline_cost = baseline_costs.get(key)
        baseline_row = baseline_runtime.get(key)
        if baseline_cost is None or baseline_cost == 0:
            row["cost_change_vs_symmetric"] = None
            row["cost_constraint_pass"] = None
        else:
            cost_change = row["final_cost_total"] / baseline_cost - 1.0
            row["cost_change_vs_symmetric"] = float(cost_change)
            row["cost_constraint_pass"] = bool(cost_change <= cost_tolerance)
        if baseline_row is None:
            row["set_tsp_p90_change_vs_symmetric"] = None
            row["timeout_rate_change_vs_symmetric"] = None
            row["runtime_or_timeout_improved"] = None
        else:
            p90_change = (
                row["set_tsp_p90_seconds"]
                / max(baseline_row["set_tsp_p90_seconds"], 1e-12)
                - 1.0
            )
            timeout_change = row["timeout_rate"] - baseline_row["timeout_rate"]
            row["set_tsp_p90_change_vs_symmetric"] = float(p90_change)
            row["timeout_rate_change_vs_symmetric"] = float(timeout_change)
            row["runtime_or_timeout_improved"] = bool(
                p90_change < 0.0 or timeout_change < 0.0
            )
        row["joint_acceptance_pass"] = (
            bool(row["cost_constraint_pass"])
            and bool(row["runtime_or_timeout_improved"])
            if row["cost_constraint_pass"] is not None
            and row["runtime_or_timeout_improved"] is not None
            else None
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cost_tolerance": cost_tolerance,
        "reference_method": "symmetric_mst",
        "record_count": len(records),
        "summary_rows": summary_rows,
    }


def _markdown_report(summary):
    """
    将评估摘要转换为便于直接阅读的 Markdown 表格。

    输入：`summarize_partition_evaluations` 输出。
    输出：Markdown 文本。
    逻辑：保留用户要求的成本、P90、最大时间、超时率、策略耗时和调用次数。
    """
    lines = [
        "# 客户划分方法评估结果",
        "",
        f"成本约束：相对 symmetric MST 总成本最多上升 {summary['cost_tolerance']:.1%}。",
        "",
        "| 地图 | 客户数 | 方法 | 平均成本 | 成本变化 | Set-TSP P90(s) | 最大(s) | 超时率 | 策略均时(s) | 最大组 | 策略真实调用 | 下游缓存未命中 | 联合通过 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["summary_rows"]:
        cost_change = row["cost_change_vs_symmetric"]
        change_text = "—" if cost_change is None else f"{cost_change:.2%}"
        joint = row["joint_acceptance_pass"]
        joint_text = "—" if joint is None else ("是" if joint else "否")
        lines.append(
            "| {map_label} | {customer_size} | {method} | {final_cost_mean:.4f} | "
            "{change} | {set_tsp_p90_seconds:.3f} | {set_tsp_max_seconds:.3f} | "
            "{timeout_rate:.1%} | {strategy_mean_seconds:.3f} | "
            "{max_group_customer_count} | {strategy_real_solver_calls} | "
            "{downstream_real_solver_calls} | {joint} |".format(
                change=change_text,
                joint=joint_text,
                **row,
            )
        )
    lines.append("")
    lines.append(
        "> 超时实例的最终成本来自 incumbent 或回退顺序，摘要中另有 approximate_cost_instance_count 标记。"
    )
    return "\n".join(lines) + "\n"


def write_partition_evaluation_outputs(
    records,
    output_directory,
    cost_tolerance=0.01,
    run_id=None,
):
    """
    保存详细 CSV、摘要 JSON 和 Markdown 报告。

    输入：评估记录、目录、成本容忍和可选运行标识。
    输出：三个输出路径及摘要对象。
    逻辑：嵌套字段在 CSV 中编码为 JSON，JSON 和 Markdown 使用同一汇总结果。
    """
    records = [_jsonable(record) for record in records]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    detail_path = output_directory / f"partition_evaluation_{run_id}.csv"
    summary_path = output_directory / f"partition_evaluation_{run_id}_summary.json"
    report_path = output_directory / f"partition_evaluation_{run_id}_report.md"

    fieldnames = list(records[0].keys()) if records else []
    with detail_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in record.items()
                }
            )

    summary = summarize_partition_evaluations(records, cost_tolerance=cost_tolerance)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return detail_path, summary_path, report_path, summary
