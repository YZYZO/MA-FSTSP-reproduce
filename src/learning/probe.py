"""小规模 Set-TSP 求解难度探测与统计汇总。"""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr

from src.fstsp import MultiAgentFlyingSidekickTSP
from src.learning.evaluator import GroupEvaluator
from src.learning.features import extract_group_feature_dict
from src.learning.settings import SetTSPSolverSettings
from src.partitioning import partition_customers


@dataclass(frozen=True)
class ProbeSettings:
    """
    保存小规模探测实验的固定算法参数。

    输入：无人机数、候选区域阈值、MST 模式、Set-TSP 上限、线程和种子。
    输出：不可变配置对象。
    逻辑：默认值对应获批的 Manhattan 1K 推荐探测设置。
    """

    drone_count: int = 3
    theta: tuple = (0.5, 0.5)
    edge_mode: str = "mean"
    set_tsp_time_limit_seconds: float = 30.0
    solver_threads: int = 1
    solver_seed: int = 0


def _jsonable(value):
    """
    将探测记录转换为 JSON 和 CSV 可写的基础类型。

    输入：可能含 NumPy 标量、数组、元组和字典的值。
    输出：递归转换后的基础容器。
    逻辑：保证实验结果不依赖 pickle，便于人工检查和后续训练读取。
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


def probe_instance(
    graph,
    depots,
    cities,
    distance,
    map_id,
    customer_size,
    instance_index,
    cache,
    settings=None,
    progress_callback=None,
):
    """
    探测一个多仓库实例中全部客户组的第二、第三阶段真实表现。

    输入：已准备路网、实例节点、距离、标识、缓存、探测配置和可选进度回调。
    输出：每个仓库一条的扁平记录列表。
    逻辑：构造边界集合，用显式双向 MST 划分，再逐组提取特征并调用可缓存评估器。
    """
    settings = settings or ProbeSettings()
    model = MultiAgentFlyingSidekickTSP(
        graph,
        depots,
        cities,
        distance,
        settings.drone_count,
        theta=settings.theta,
    )

    boundary_start = time.perf_counter()
    convex_sets = model.get_boundary_convex_sets(model.theta[0])
    boundary_seconds = time.perf_counter() - boundary_start

    partition_start = time.perf_counter()
    groups = partition_customers(
        model.depots,
        model.cities,
        convex_sets,
        model.distance["truck"],
        model.distance["drone"],
        model.speed,
        edge_mode=settings.edge_mode,
        coefficient=model.const,
    )
    partition_seconds = time.perf_counter() - partition_start

    evaluator = GroupEvaluator(
        model,
        convex_sets,
        map_id=map_id,
        cache=cache,
        solver_settings=SetTSPSolverSettings(
            time_limit_seconds=settings.set_tsp_time_limit_seconds,
            threads=settings.solver_threads,
            seed=settings.solver_seed,
        ),
    )

    records = []
    for depot_index, depot in enumerate(model.depots):
        customers = groups[depot]
        feature_values = extract_group_feature_dict(
            graph,
            distance,
            depot,
            customers,
            convex_sets,
        )
        evaluation = evaluator.evaluate(depot, customers)
        solver_result = evaluation.set_tsp_result

        # CSV 每行对应一个客户组；结构特征直接展开，便于后续建模和人工排序。
        record = {
            "map_id": map_id,
            "customer_size": int(customer_size),
            "instance_index": int(instance_index),
            "depot_index": int(depot_index),
            "depot_node": depot,
            "customers": list(evaluation.customers),
            "boundary_construction_seconds": float(boundary_seconds),
            "mst_partition_seconds": float(partition_seconds),
            "mst_edge_mode": settings.edge_mode,
            "set_tsp_time_limit_seconds": float(
                settings.set_tsp_time_limit_seconds
            ),
            "set_tsp_status": solver_result.status_name,
            "set_tsp_status_code": solver_result.status,
            "set_tsp_runtime_seconds": solver_result.runtime_seconds,
            "set_tsp_wall_seconds": evaluation.set_tsp_wall_seconds,
            "set_tsp_node_count": solver_result.node_count,
            "set_tsp_mip_gap": solver_result.mip_gap,
            "set_tsp_objective": solver_result.objective,
            "set_tsp_objective_bound": solver_result.objective_bound,
            "set_tsp_solution_count": solver_result.solution_count,
            "set_tsp_timed_out": solver_result.timed_out,
            "set_tsp_has_incumbent": solver_result.has_incumbent,
            "set_tsp_fallback_used": solver_result.fallback_used,
            "set_tsp_sequence_source": solver_result.sequence_source,
            "phase3_seconds": evaluation.phase3_seconds,
            "final_cost": evaluation.final_cost,
            "cache_hit": evaluation.cache_hit,
            **feature_values,
        }
        records.append(_jsonable(record))
        if progress_callback is not None:
            progress_callback(records[-1])
    return records


def _percentile(values, percentile):
    """
    计算数值序列分位数，空序列返回空值。

    输入：数值序列和 0 到 100 的分位点。
    输出：浮点分位数或 `None`。
    逻辑：集中处理汇总中的空组情况。
    """
    return float(np.percentile(values, percentile)) if values else None


def _safe_spearman(first_values, second_values):
    """
    计算两个变量的 Spearman 排序相关系数。

    输入：等长数值序列。
    输出：有限相关系数；样本过少或常量输入时返回 `None`。
    逻辑：探测样本中可能存在全部零时间的空客户组，需要安全处理。
    """
    if len(first_values) < 3:
        return None
    coefficient = spearmanr(first_values, second_values).statistic
    return float(coefficient) if np.isfinite(coefficient) else None


def _summarize_group_records(records):
    """
    汇总一组客户组记录的时间、状态、规模和成本指标。

    输入：同一范围内的客户组记录。
    输出：适合 JSON 展示的统计字典。
    逻辑：同时保留 P50、P90、最大值与长尾超时数量。
    """
    runtimes = [float(record["set_tsp_runtime_seconds"]) for record in records]
    wall_times = [float(record["set_tsp_wall_seconds"]) for record in records]
    phase3_times = [float(record["phase3_seconds"]) for record in records]
    downstream_times = [
        float(record["set_tsp_wall_seconds"]) + float(record["phase3_seconds"])
        for record in records
    ]
    group_sizes = [int(record["customer_count"]) for record in records]
    complexity_values = [float(record["set_tsp_complexity_proxy"]) for record in records]
    log_runtimes = [math.log1p(value) for value in runtimes]
    log_wall_times = [math.log1p(value) for value in wall_times]
    log_downstream_times = [math.log1p(value) for value in downstream_times]

    statuses = {}
    for record in records:
        status = record["set_tsp_status"]
        statuses[status] = statuses.get(status, 0) + 1

    return {
        "group_count": len(records),
        "empty_group_count": sum(size == 0 for size in group_sizes),
        "timeout_count": sum(bool(record["set_tsp_timed_out"]) for record in records),
        "timeout_rate": (
            sum(bool(record["set_tsp_timed_out"]) for record in records)
            / len(records)
            if records
            else 0.0
        ),
        "fallback_count": sum(bool(record["set_tsp_fallback_used"]) for record in records),
        "cache_hit_count": sum(bool(record["cache_hit"]) for record in records),
        "status_counts": statuses,
        "group_size": {
            "p50": _percentile(group_sizes, 50),
            "p90": _percentile(group_sizes, 90),
            "max": max(group_sizes) if group_sizes else None,
        },
        "set_tsp_runtime_seconds": {
            "p50": _percentile(runtimes, 50),
            "p90": _percentile(runtimes, 90),
            "max": max(runtimes) if runtimes else None,
            "sum": float(sum(runtimes)),
        },
        "set_tsp_wall_seconds": {
            "p50": _percentile(wall_times, 50),
            "p90": _percentile(wall_times, 90),
            "max": max(wall_times) if wall_times else None,
            "sum": float(sum(wall_times)),
        },
        "phase3_seconds": {
            "p50": _percentile(phase3_times, 50),
            "p90": _percentile(phase3_times, 90),
            "max": max(phase3_times) if phase3_times else None,
            "sum": float(sum(phase3_times)),
        },
        "downstream_total_seconds": {
            "p50": _percentile(downstream_times, 50),
            "p90": _percentile(downstream_times, 90),
            "max": max(downstream_times) if downstream_times else None,
            "sum": float(sum(downstream_times)),
        },
        "group_size_vs_log_runtime_spearman": _safe_spearman(
            group_sizes,
            log_runtimes,
        ),
        "complexity_proxy_vs_log_runtime_spearman": _safe_spearman(
            complexity_values,
            log_runtimes,
        ),
        "group_size_vs_log_wall_time_spearman": _safe_spearman(
            group_sizes,
            log_wall_times,
        ),
        "complexity_proxy_vs_log_wall_time_spearman": _safe_spearman(
            complexity_values,
            log_wall_times,
        ),
        "group_size_vs_log_downstream_time_spearman": _safe_spearman(
            group_sizes,
            log_downstream_times,
        ),
        "complexity_proxy_vs_log_downstream_time_spearman": _safe_spearman(
            complexity_values,
            log_downstream_times,
        ),
    }


def summarize_probe_records(records):
    """
    生成整个探测批次及各客户规模的统计摘要。

    输入：全部客户组明细记录。
    输出：包含总体、分规模和分实例指标的字典。
    逻辑：分实例聚合各组最终成本与串行时间，保持论文当前顺序执行语义。
    """
    records = list(records)
    customer_sizes = sorted({int(record["customer_size"]) for record in records})
    by_customer_size = {
        str(customer_size): _summarize_group_records(
            [
                record
                for record in records
                if int(record["customer_size"]) == customer_size
            ]
        )
        for customer_size in customer_sizes
    }

    instance_groups = {}
    for record in records:
        key = (int(record["customer_size"]), int(record["instance_index"]))
        instance_groups.setdefault(key, []).append(record)
    instance_metrics = []
    for (customer_size, instance_index), group_records in sorted(instance_groups.items()):
        instance_metrics.append({
            "customer_size": customer_size,
            "instance_index": instance_index,
            "group_sizes": [int(record["customer_count"]) for record in group_records],
            "max_group_size": max(int(record["customer_count"]) for record in group_records),
            "total_final_cost": float(sum(record["final_cost"] for record in group_records)),
            "total_set_tsp_runtime_seconds": float(
                sum(record["set_tsp_runtime_seconds"] for record in group_records)
            ),
            "max_group_set_tsp_runtime_seconds": float(
                max(record["set_tsp_runtime_seconds"] for record in group_records)
            ),
            "total_downstream_seconds": float(
                sum(
                    record["set_tsp_wall_seconds"] + record["phase3_seconds"]
                    for record in group_records
                )
            ),
            "max_group_downstream_seconds": float(
                max(
                    record["set_tsp_wall_seconds"] + record["phase3_seconds"]
                    for record in group_records
                )
            ),
            "timeout_count": sum(
                bool(record["set_tsp_timed_out"]) for record in group_records
            ),
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": _summarize_group_records(records),
        "by_customer_size": by_customer_size,
        "instances": instance_metrics,
    }


def write_probe_outputs(records, output_directory, run_id=None):
    """
    将探测明细和统计摘要分别写成 CSV 与 JSON。

    输入：明细记录、输出目录和可选运行标识。
    输出：`(csv_path, summary_path, summary)`。
    逻辑：CSV 供后续训练读取，JSON 供本轮人工判断时间预算。
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output_directory / f"set_tsp_probe_{run_id}.csv"
    summary_path = output_directory / f"set_tsp_probe_{run_id}_summary.json"

    records = [_jsonable(record) for record in records]
    fieldnames = list(records[0].keys()) if records else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            csv_record = {
                key: (
                    json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in record.items()
            }
            writer.writerow(csv_record)

    summary = summarize_probe_records(records)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(
            _jsonable(summary),
            summary_file,
            ensure_ascii=False,
            indent=2,
        )
    return csv_path, summary_path, summary
