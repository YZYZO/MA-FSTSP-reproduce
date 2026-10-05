"""把公平代理搜索结果与真实Phase 2/3候选记录连接并计算最终指标。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def _partition_signature(partition: dict[Any, Iterable[int]]) -> tuple:
    """输入JSON或整数键划分，输出仓库和客户均稳定排序的完整划分签名。"""
    return tuple(
        (int(depot), tuple(sorted(map(int, customers))))
        for depot, customers in sorted(
            partition.items(), key=lambda item: int(item[0])
        )
    )


def _exact_record(record: dict[str, Any]) -> bool:
    """判断真实候选是否精确完成且没有删失、超时或回退。"""
    labels = record["labels"]
    return bool(
        labels.get("candidate_exact", False)
        and not labels.get("candidate_right_censored", False)
        and int(labels.get("right_censored_groups", 0)) == 0
        and int(labels.get("timeout_groups", 0)) == 0
        and int(labels.get("fallback_groups", 0)) == 0
    )


def build_true_search_comparison(
    search_report: dict[str, Any],
    candidate_records: Iterable[dict[str, Any]],
    *,
    cost_limit: float = 0.10,
) -> dict[str, Any]:
    """
    连接代理搜索最佳划分和真实候选记录，计算逐实例及跨实例指标。

    真实后悔值以本次实际复核候选中的最快成本可行解为Oracle；它不是全划分空间
    的全局最优值。代理判断可行但真实成本越界时保留为假可行，不事后改成MST。
    """
    by_instance: dict[str, dict[tuple, dict[str, Any]]] = defaultdict(dict)
    baseline_by_instance: dict[str, dict[str, Any]] = {}
    for record in candidate_records:
        instance_id = str(record["instance_id"])
        signature = _partition_signature(record["partition"])
        by_instance[instance_id][signature] = record
        if str(record.get("candidate_name")) == "mst":
            baseline_by_instance[instance_id] = record

    instance_results: dict[str, dict[str, Any]] = {}
    method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance_id, methods in search_report["instances"].items():
        true_lookup = by_instance.get(instance_id, {})
        baseline = baseline_by_instance.get(instance_id)
        if baseline is None:
            raise ValueError(f"实例 {instance_id} 的真实记录缺少MST。")
        exact_feasible = [
            record for record in true_lookup.values()
            if _exact_record(record)
            and float(record["labels"]["cost_change_ratio"]) <= cost_limit
        ]
        oracle = min(
            exact_feasible,
            key=lambda record: float(record["labels"]["downstream_total_seconds"]),
        ) if exact_feasible else None
        oracle_time = (
            float(oracle["labels"]["downstream_total_seconds"])
            if oracle is not None else float("nan")
        )

        method_results: dict[str, Any] = {}
        for method, search in methods.items():
            signature = _partition_signature(search["best"]["partition"])
            record = true_lookup.get(signature)
            if record is None:
                row = {
                    "method": method,
                    "true_record_found": False,
                    "true_exact": False,
                    "true_cost_feasible": False,
                    "true_feasible_improvement": False,
                    "true_cost_change_ratio": None,
                    "constraint_violation_ratio": None,
                    "true_time_saving_ratio": None,
                    "true_feasible_regret_ratio": None,
                    "proxy_false_feasible": None,
                }
            else:
                labels = record["labels"]
                exact = _exact_record(record)
                cost_change = float(labels["cost_change_ratio"])
                feasible = exact and cost_change <= cost_limit
                time_value = float(labels["downstream_total_seconds"])
                time_saving = float(labels["downstream_time_saving_ratio"])
                proxy_claimed_feasible = bool(search["best"]["feasible"])
                row = {
                    "method": method,
                    "true_record_found": True,
                    "true_candidate_name": str(record["candidate_name"]),
                    "true_exact": exact,
                    "true_cost_feasible": feasible,
                    "true_feasible_improvement": feasible and time_saving > 0.0,
                    "true_cost_change_ratio": cost_change,
                    "constraint_violation_ratio": (
                        max(0.0, cost_change - cost_limit) if exact else None
                    ),
                    "true_time_saving_ratio": time_saving if feasible else None,
                    "true_feasible_regret_ratio": (
                        (time_value - oracle_time) / max(oracle_time, 1e-9)
                        if feasible and oracle is not None else None
                    ),
                    "proxy_false_feasible": (
                        proxy_claimed_feasible and not feasible
                    ),
                }
            row.update({
                "surrogate_query_count": int(search["surrogate_query_count"]),
                "generation_seconds": float(search.get("generation_seconds", 0.0)),
                "proxy_returned_mst": bool(search["returned_mst"]),
                "predicted_time_saving_ratio": float(
                    search["predicted_time_saving_ratio"]
                ),
                "predicted_cost_ucb_change": float(
                    search["best"]["cost_ucb_change"]
                ),
            })
            method_results[method] = row
            method_rows[method].append(row)
        instance_results[instance_id] = {
            "true_evaluated_partition_count": len(true_lookup),
            "true_exact_feasible_partition_count": len(exact_feasible),
            "oracle_candidate_name": (
                str(oracle["candidate_name"]) if oracle is not None else None
            ),
            "oracle_downstream_total_seconds": (
                oracle_time if oracle is not None else None
            ),
            "methods": method_results,
        }

    aggregate: dict[str, Any] = {}
    for method, rows in sorted(method_rows.items()):
        found = [row for row in rows if row["true_record_found"]]
        feasible = [row for row in found if row["true_cost_feasible"]]
        improvements = [row for row in feasible if row["true_feasible_improvement"]]
        violations = [
            row["constraint_violation_ratio"] for row in found
            if row["constraint_violation_ratio"] is not None
        ]
        regrets = [
            row["true_feasible_regret_ratio"] for row in feasible
            if row["true_feasible_regret_ratio"] is not None
        ]
        savings = [
            row["true_time_saving_ratio"] for row in feasible
            if row["true_time_saving_ratio"] is not None
        ]
        aggregate[method] = {
            "instance_count": len(rows),
            "true_record_coverage": len(found) / max(len(rows), 1),
            "true_exact_fraction": float(np.mean([
                row["true_exact"] for row in found
            ])) if found else 0.0,
            "true_cost_feasible_fraction": len(feasible) / max(len(rows), 1),
            "true_feasible_improvement_fraction": len(improvements) / max(len(rows), 1),
            "proxy_false_feasible_fraction": float(np.mean([
                bool(row["proxy_false_feasible"]) for row in found
            ])) if found else 0.0,
            "mean_constraint_violation_ratio": (
                float(np.mean(violations)) if violations else None
            ),
            "mean_time_saving_given_feasible": (
                float(np.mean(savings)) if savings else None
            ),
            "best_feasible_time_saving": (
                float(np.max(savings)) if savings else None
            ),
            "mean_true_feasible_regret_ratio": (
                float(np.mean(regrets)) if regrets else None
            ),
            "returned_mst_fraction": float(np.mean([
                row["proxy_returned_mst"] for row in rows
            ])),
            "mean_surrogate_query_count": float(np.mean([
                row["surrogate_query_count"] for row in rows
            ])),
            "mean_generation_seconds": float(np.mean([
                row["generation_seconds"] for row in rows
            ])),
        }

    return {
        "kind": "partition_fair_search_true_comparison",
        "cost_limit": float(cost_limit),
        "oracle_scope": "server_validated_candidate_pool_only",
        "instances": instance_results,
        "aggregate": aggregate,
    }
