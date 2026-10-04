"""用真实 Phase 2/3 求解器复核强化学习导出的客户划分。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cache import GroupEvaluationCache
from .candidates import (
    PartitionCandidate,
    canonical_partition,
    moved_customer_count,
    partition_key,
)
from .dataset import ExperimentInstance, load_instances
from .evaluator import evaluate_partition
from .features import action_delta_features
from .reporting import (
    append_jsonl,
    deduplicate_candidate_records,
    read_jsonl,
    write_json,
    write_jsonl,
)


@dataclass(frozen=True)
class ValidationCandidate:
    """保存待真实求解的划分，以及冻结监督代理当时给出的预测。"""

    candidate: PartitionCandidate
    surrogate_score: float | None
    surrogate_score_improvement: float | None
    surrogate_prediction: dict[str, float]
    surrogate_uncertainty: dict[str, float]


def _generator_family(name: str) -> str:
    """根据候选名称返回用于审计的生成方法族。"""
    if name == "mst":
        return "mst"
    if "ppo" in name or name == "best_training":
        return "rl_policy"
    if "greedy" in name:
        return "surrogate_greedy"
    if "random" in name:
        return "random_policy"
    return "external_partition"


def parse_validation_candidates(
    payload: dict[str, Any],
    instance: ExperimentInstance,
) -> list[ValidationCandidate]:
    """
    解析单实例候选文件并转换为统一候选对象。

    输入为 JSON 根对象和历史实例；输出按 MST 优先、完整分区去重后的候选列表。
    每个客户必须恰好归属一个原始仓库，防止服务器长时间求解错误分区。
    """
    rows = list(payload.get(instance.instance_id, ()))
    baseline = canonical_partition(instance.partition, instance.depots)
    baseline_key = partition_key(baseline, instance.depots)
    if not any(
        partition_key(
            canonical_partition(
                {int(key): value for key, value in row["partition"].items()},
                instance.depots,
            ),
            instance.depots,
        ) == baseline_key
        for row in rows
    ):
        rows.insert(0, {
            "name": "mst",
            "surrogate_score": None,
            "surrogate_score_improvement": 0.0,
            "surrogate_prediction": {},
            "surrogate_uncertainty": {},
            "partition": {str(key): list(value) for key, value in baseline.items()},
        })

    parsed: list[ValidationCandidate] = []
    seen_partitions = set()
    expected_customers = sorted(map(int, instance.cities))
    expected_depots = set(map(int, instance.depots))
    for row in rows:
        raw_partition = {
            int(key): tuple(map(int, values))
            for key, values in row["partition"].items()
        }
        if set(raw_partition) != expected_depots:
            raise ValueError(
                f"{instance.instance_id}/{row['name']} 的仓库集合与原实例不一致。"
            )
        partition = canonical_partition(raw_partition, instance.depots)
        assigned = [customer for group in partition.values() for customer in group]
        if sorted(assigned) != expected_customers or len(set(assigned)) != len(assigned):
            raise ValueError(
                f"{instance.instance_id}/{row['name']} 没有恰好覆盖全部客户。"
            )
        key = partition_key(partition, instance.depots)
        if key in seen_partitions:
            continue
        seen_partitions.add(key)
        moved = moved_customer_count(baseline, partition)
        name = "mst" if key == baseline_key else str(row["name"])
        parsed.append(ValidationCandidate(
            candidate=PartitionCandidate(
                name=name,
                kind="rl_true_validation",
                strength=moved / max(len(instance.cities), 1),
                partition=partition,
                moved_customers=moved,
                generator_family=_generator_family(name),
                generator_parameters=(
                    ("source", "validation_candidates.json"),
                ),
                action_trace=(f"external:{name}",),
                parent_name="mst",
            ),
            surrogate_score=(
                None
                if row.get("surrogate_score") is None
                else float(row["surrogate_score"])
            ),
            surrogate_score_improvement=(
                None
                if row.get("surrogate_score_improvement") is None
                else float(row["surrogate_score_improvement"])
            ),
            surrogate_prediction={
                str(key): float(value)
                for key, value in row.get("surrogate_prediction", {}).items()
            },
            surrogate_uncertainty={
                str(key): float(value)
                for key, value in row.get("surrogate_uncertainty", {}).items()
            },
        ))
    parsed.sort(key=lambda item: item.candidate.name != "mst")
    return parsed


def load_validation_instance(experiment, instance_id: str) -> ExperimentInstance:
    """根据稳定实例ID从实验允许的NPZ中精确加载一个历史实例。"""
    source_name, index_text = instance_id.rsplit("__", 1)
    source_path = next(
        (path for path in experiment.execution_files if path.stem == source_name),
        None,
    )
    if source_path is None:
        raise ValueError(f"当前路网与规模过滤条件不包含实例：{instance_id}")
    instance = load_instances(source_path, [int(index_text)])[0]
    if instance.instance_id != instance_id:
        raise RuntimeError(f"实例定位结果不一致：{instance.instance_id} != {instance_id}")
    return instance


def _record_key(record: dict[str, Any], depots: tuple[int, ...]) -> tuple:
    """把已完成记录转换成实例与规范分区组成的续跑键。"""
    partition = canonical_partition(
        {int(key): value for key, value in record["partition"].items()},
        depots,
    )
    return record["instance_id"], partition_key(partition, depots)


def evaluate_validation_candidates(
    experiment,
    payload: dict[str, Any],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """
    串行真实求解候选并支持候选级与仓库组级断点续跑。

    输入为已配置实验对象、候选JSON和输出目录；输出最终带实例内排名的兼容记录。
    每完成一个候选立即追加JSONL；若候选中途失败，SQLite仍保留已完成仓库组。
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / "candidate_records.jsonl"
    existing = deduplicate_candidate_records(read_jsonl(record_path))
    requested_ids = list(payload)
    instances = {
        instance_id: load_validation_instance(experiment, instance_id)
        for instance_id in requested_ids
    }
    specifications = {
        instance_id: parse_validation_candidates(payload, instance)
        for instance_id, instance in instances.items()
    }
    total = sum(len(items) for items in specifications.values())
    completed = 0
    print(f"[RL validation] 候选总数={total}，已有记录={len(existing)}", flush=True)

    with GroupEvaluationCache(experiment.cache_path) as cache:
        for instance_id in requested_ids:
            instance = instances[instance_id]
            candidates = specifications[instance_id]
            existing_lookup = {
                _record_key(record, instance.depots): record
                for record in existing
                if record["instance_id"] == instance_id
            }
            boundary, regions, distance, preparation = experiment._prepare_instance(instance)
            truck, drone = distance["truck"], distance["drone"]

            baseline_specification = candidates[0]
            baseline_key = (
                instance_id,
                partition_key(baseline_specification.candidate.partition, instance.depots),
            )
            baseline_record = existing_lookup.get(baseline_key)
            baseline_resumed = baseline_record is not None
            if baseline_record is None:
                baseline_evaluation = evaluate_partition(
                    instance_id=instance_id,
                    partition=baseline_specification.candidate.partition,
                    depots=instance.depots,
                    boundary=boundary,
                    regions=regions,
                    distance=distance,
                    drones_per_truck=instance.drones_per_truck,
                    drone_limit=instance.drone_limit,
                    drone_speed=instance.drone_speed,
                    cache=cache,
                    time_limit=experiment.solver_time_limit,
                    threads=experiment.solver_threads,
                    seed=experiment.solver_seed,
                    mip_gap=experiment.solver_mip_gap,
                    max_binary_variables=experiment.max_binary_variables,
                    evaluation_workers=1,
                )
                baseline_features = action_delta_features(
                    instance.partition,
                    baseline_specification.candidate,
                    instance.boundary_sizes,
                    truck.query,
                    drone.query,
                    graph_nodes=experiment._graph.number_of_nodes(),
                    graph_edges=experiment._graph.number_of_edges(),
                )
                baseline_record = experiment._candidate_record(
                    instance,
                    baseline_specification.candidate,
                    baseline_features,
                    baseline_evaluation,
                    baseline_evaluation,
                    preparation,
                    round_name="rl_true_validation",
                    acquisition=_acquisition(baseline_specification),
                )
                append_jsonl(record_path, [baseline_record])
                existing.append(baseline_record)
                existing_lookup[baseline_key] = baseline_record
            else:
                baseline_evaluation = baseline_record["evaluation"]
            completed += 1
            print(
                f"[RL validation] {completed}/{total} {instance_id} mst "
                f"{'resume' if baseline_resumed else 'done'}",
                flush=True,
            )

            for specification in candidates[1:]:
                candidate = specification.candidate
                key = instance_id, partition_key(candidate.partition, instance.depots)
                if key in existing_lookup:
                    completed += 1
                    print(
                        f"[RL validation] {completed}/{total} {instance_id} "
                        f"{candidate.name} resume",
                        flush=True,
                    )
                    continue
                evaluation = evaluate_partition(
                    instance_id=instance_id,
                    partition=candidate.partition,
                    depots=instance.depots,
                    boundary=boundary,
                    regions=regions,
                    distance=distance,
                    drones_per_truck=instance.drones_per_truck,
                    drone_limit=instance.drone_limit,
                    drone_speed=instance.drone_speed,
                    cache=cache,
                    time_limit=experiment.solver_time_limit,
                    threads=experiment.solver_threads,
                    seed=experiment.solver_seed,
                    mip_gap=experiment.solver_mip_gap,
                    max_binary_variables=experiment.max_binary_variables,
                    evaluation_workers=1,
                )
                features = action_delta_features(
                    instance.partition,
                    candidate,
                    instance.boundary_sizes,
                    truck.query,
                    drone.query,
                    graph_nodes=experiment._graph.number_of_nodes(),
                    graph_edges=experiment._graph.number_of_edges(),
                )
                record = experiment._candidate_record(
                    instance,
                    candidate,
                    features,
                    evaluation,
                    baseline_evaluation,
                    preparation,
                    round_name="rl_true_validation",
                    acquisition=_acquisition(specification),
                )
                append_jsonl(record_path, [record])
                existing.append(record)
                existing_lookup[key] = record
                completed += 1
                print(
                    f"[RL validation] {completed}/{total} {instance_id} "
                    f"{candidate.name} exact={record['labels']['candidate_exact']}",
                    flush=True,
                )

    # 只输出当前输入文件中的候选，并在全部完成后统一重算排名。
    final_records: list[dict[str, Any]] = []
    for instance_id in requested_ids:
        instance = instances[instance_id]
        lookup = {
            _record_key(record, instance.depots): record
            for record in existing
            if record["instance_id"] == instance_id
        }
        ordered = [
            lookup[(
                instance_id,
                partition_key(item.candidate.partition, instance.depots),
            )]
            for item in specifications[instance_id]
        ]
        final_records.extend(experiment._attach_instance_ranks(ordered))
    write_jsonl(record_path, final_records)
    return final_records


def _acquisition(specification: ValidationCandidate) -> dict[str, Any]:
    """把代理预测写入真实记录，供误差、排序和策略审计使用。"""
    return {
        "method": "rl_surrogate_validation",
        "surrogate_score": specification.surrogate_score,
        "surrogate_score_improvement": specification.surrogate_score_improvement,
        "surrogate_prediction": specification.surrogate_prediction,
        "surrogate_uncertainty": specification.surrogate_uncertainty,
    }


def _average_ranks(values: list[float]) -> np.ndarray:
    """输入越小越优的数值，输出处理并列值的平均名次。"""
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0 + 1.0
        position = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    """计算两个越小越优序列的 Spearman 相关；常量或不足两项时返回空。"""
    if len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if float(np.std(left_ranks)) == 0.0 or float(np.std(right_ranks)) == 0.0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def build_validation_report(
    records: list[dict[str, Any]],
    *,
    cost_limit: float = 0.10,
) -> dict[str, Any]:
    """汇总真实标签、代理误差、实例内排序和 PPO 候选确认结论。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["instance_id"])].append(record)
    instances = {}
    for instance_id, instance_records in grouped.items():
        rows = []
        surrogate_scores, true_times = [], []
        time_errors, cost_errors = [], []
        for record in instance_records:
            labels = record["labels"]
            acquisition = record.get("acquisition") or {}
            prediction = acquisition.get("surrogate_prediction", {})
            true_time = float(labels["downstream_total_seconds"])
            true_cost = float(labels["final_cost"])
            predicted_time = prediction.get("downstream_total_seconds")
            predicted_cost = prediction.get("final_cost")
            exact = bool(labels["candidate_exact"])
            cost_change = float(labels["cost_change_ratio"])
            time_saving = float(labels["downstream_time_saving_ratio"])
            if not exact:
                status = "censored_or_fallback"
            elif record["candidate_name"] == "mst":
                status = "baseline"
            elif time_saving > 0.0 and cost_change <= cost_limit:
                status = "confirmed"
            elif time_saving > 0.0:
                status = "time_improved_cost_exceeded"
            else:
                status = "not_improved"
            row = {
                "candidate_name": record["candidate_name"],
                "candidate_exact": exact,
                "verification_status": status,
                "true_downstream_total_seconds": true_time,
                "true_final_cost": true_cost,
                "true_time_saving_ratio": time_saving,
                "true_cost_change_ratio": cost_change,
                "true_time_rank": labels.get("time_rank"),
                "true_cost_rank": labels.get("cost_rank"),
                "true_pareto_front": bool(labels.get("pareto_front", False)),
                "surrogate_score": acquisition.get("surrogate_score"),
                "predicted_downstream_total_seconds": predicted_time,
                "predicted_final_cost": predicted_cost,
                "time_prediction_error_seconds": (
                    None if predicted_time is None else float(predicted_time) - true_time
                ),
                "cost_prediction_error": (
                    None if predicted_cost is None else float(predicted_cost) - true_cost
                ),
                "right_censored_groups": int(labels["right_censored_groups"]),
            }
            rows.append(row)
            score = acquisition.get("surrogate_score")
            if exact and score is not None:
                surrogate_scores.append(float(score))
                true_times.append(true_time)
            if exact and row["time_prediction_error_seconds"] is not None:
                time_errors.append(abs(float(row["time_prediction_error_seconds"])))
            if exact and row["cost_prediction_error"] is not None:
                cost_errors.append(abs(float(row["cost_prediction_error"])))
        ppo_row = next(
            (row for row in rows if row["candidate_name"] == "ppo_best"),
            None,
        )
        instances[instance_id] = {
            "candidate_count": len(rows),
            "exact_candidate_count": sum(row["candidate_exact"] for row in rows),
            "surrogate_true_time_spearman": _spearman(surrogate_scores, true_times),
            "time_prediction_mae_seconds": (
                float(np.mean(time_errors)) if time_errors else None
            ),
            "cost_prediction_mae": float(np.mean(cost_errors)) if cost_errors else None,
            "ppo_best_verdict": (
                ppo_row["verification_status"] if ppo_row else "missing"
            ),
            "candidates": rows,
        }
    return {
        "kind": "rl_partition_true_phase2_phase3_validation",
        "cost_limit": float(cost_limit),
        "instances": instances,
    }


def validation_report_markdown(report: dict[str, Any]) -> str:
    """把真实复核JSON转换成便于人工检查的中文Markdown报告。"""
    lines = [
        "# 强化学习划分真实 Phase 2/3 复核",
        "",
        "本报告中的真实时间和成本来自服务器求解器；删失或回退候选不用于确认结论。",
        "",
    ]
    for instance_id, summary in report["instances"].items():
        lines.extend([
            f"## {instance_id}",
            "",
            f"- 精确候选：{summary['exact_candidate_count']}/{summary['candidate_count']}",
            f"- PPO最佳候选结论：`{summary['ppo_best_verdict']}`",
            f"- 代理分数与真实时间 Spearman：{summary['surrogate_true_time_spearman']}",
            f"- 时间预测 MAE（秒）：{summary['time_prediction_mae_seconds']}",
            f"- 成本预测 MAE：{summary['cost_prediction_mae']}",
            "",
            "|候选|精确|结论|真实总时间|真实节省|真实成本|成本变化|时间排名|Pareto前沿|",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for row in summary["candidates"]:
            lines.append(
                f"|{row['candidate_name']}|{row['candidate_exact']}|"
                f"{row['verification_status']}|"
                f"{row['true_downstream_total_seconds']:.3f}|"
                f"{row['true_time_saving_ratio']:.2%}|"
                f"{row['true_final_cost']:.3f}|"
                f"{row['true_cost_change_ratio']:.2%}|"
                f"{row['true_time_rank']}|{row['true_pareto_front']}|"
            )
        lines.append("")
    return "\n".join(lines)


def save_validation_reports(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    cost_limit: float = 0.10,
) -> dict[str, Any]:
    """生成真实复核 JSON 与 Markdown，并返回内存中的报告。"""
    directory = Path(output_dir)
    report = build_validation_report(records, cost_limit=cost_limit)
    write_json(directory / "validation_report.json", report)
    (directory / "validation_report.md").write_text(
        validation_report_markdown(report),
        encoding="utf-8",
    )
    return report
