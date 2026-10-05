"""枚举PPO最终客户迁移的全部子集，并按保守成本约束筛选候选。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

from .candidates import Partition, canonical_partition, partition_key


@dataclass(frozen=True)
class CustomerMove:
    """保存一位客户相对MST发生的最终仓库迁移。"""

    customer: int
    source: int
    target: int


def infer_customer_moves(
    baseline: Partition,
    target: Partition,
) -> tuple[CustomerMove, ...]:
    """比较两个完整划分，返回按客户编号排序的最终归属变化。"""
    before = {
        int(customer): int(depot)
        for depot, customers in baseline.items()
        for customer in customers
    }
    after = {
        int(customer): int(depot)
        for depot, customers in target.items()
        for customer in customers
    }
    if set(before) != set(after):
        raise ValueError("目标划分与MST没有覆盖相同客户集合。")
    return tuple(
        CustomerMove(customer, before[customer], after[customer])
        for customer in sorted(before)
        if before[customer] != after[customer]
    )


def apply_move_subset(
    baseline: Partition,
    moves: Iterable[CustomerMove],
) -> Partition:
    """从MST同时应用给定最终迁移子集，返回规范完整划分。"""
    result = {
        int(depot): list(map(int, customers))
        for depot, customers in baseline.items()
    }
    for move in moves:
        result[move.source].remove(move.customer)
        result[move.target].append(move.customer)
    return canonical_partition(result, result)


def enumerate_move_subsets(
    baseline: Partition,
    target: Partition,
) -> list[dict[str, Any]]:
    """
    枚举MST到PPO最终划分的所有迁移子集。

    输入为两个完整划分；输出包含MST、全部中间组合和完整PPO的稳定列表。
    """
    moves = infer_customer_moves(baseline, target)
    rows = []
    seen = set()
    for move_count in range(len(moves) + 1):
        for selected in combinations(moves, move_count):
            partition = apply_move_subset(baseline, selected)
            key = partition_key(partition, baseline)
            if key in seen:
                continue
            seen.add(key)
            if move_count == 0:
                name = "mst"
            elif move_count == len(moves):
                name = "ppo_best"
            else:
                name = "ppo_subset_" + "_".join(
                    str(move.customer) for move in selected
                )
            rows.append({
                "name": name,
                "partition": partition,
                "moves": tuple(selected),
                "move_count": move_count,
                "already_true_validated": move_count in (0, len(moves)),
            })
    return rows


def estimate_partition_complexity(
    partition: Partition,
    boundary_sizes: dict[int, int],
) -> dict[str, float]:
    """按Set-TSP二元变量主导项估计各组复杂度，并返回总量与最大值。"""
    values = []
    for customers in partition.values():
        set_sizes = [1] + [int(boundary_sizes[customer]) for customer in customers]
        set_count = len(set_sizes)
        values.append(float(
            set_count * set_count
            + sum(size * size for size in set_sizes)
            + sum(set_sizes) ** 2
        ))
    return {
        "sum": float(sum(values)),
        "max": float(max(values, default=0.0)),
    }


def attach_surrogate_analysis(
    subset_rows: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    boundary_sizes: dict[int, int],
    *,
    kappa: float,
    thresholds: tuple[float, ...] = (0.05, 0.07, 0.10),
) -> list[dict[str, Any]]:
    """
    把统一代理输出、成本UCB和三档可行性标记附加到迁移子集。

    成本UCB相对MST代理点预测计算；真实硬约束仍由服务器求解结果判定。
    """
    if len(subset_rows) != len(outputs):
        raise ValueError("迁移子集数量与代理输出数量不一致。")
    baseline_output = next(
        output
        for row, output in zip(subset_rows, outputs)
        if row["name"] == "mst"
    )
    baseline_cost = max(
        float(baseline_output["prediction"]["final_cost"]),
        1e-9,
    )
    baseline_time = max(
        float(baseline_output["prediction"]["downstream_total_seconds"]),
        1e-9,
    )
    baseline_complexity = estimate_partition_complexity(
        next(row["partition"] for row in subset_rows if row["name"] == "mst"),
        boundary_sizes,
    )
    analyzed = []
    for row, output in zip(subset_rows, outputs):
        prediction = {
            key: float(value)
            for key, value in output["prediction"].items()
        }
        uncertainty = {
            key: float(value)
            for key, value in output["uncertainty"]["numeric"].items()
        }
        predicted_cost = prediction["final_cost"]
        cost_uncertainty = max(uncertainty.get("final_cost", 0.0), 0.0)
        predicted_time = prediction["downstream_total_seconds"]
        cost_ucb = predicted_cost + float(kappa) * cost_uncertainty
        predicted_cost_change = (predicted_cost - baseline_cost) / baseline_cost
        cost_ucb_change = (cost_ucb - baseline_cost) / baseline_cost
        complexity = estimate_partition_complexity(row["partition"], boundary_sizes)
        analyzed.append({
            **row,
            "surrogate_prediction": prediction,
            "surrogate_uncertainty": uncertainty,
            "predicted_time_change_ratio": (
                predicted_time - baseline_time
            ) / baseline_time,
            "predicted_cost_change_ratio_vs_mst": predicted_cost_change,
            "cost_ucb": cost_ucb,
            "cost_ucb_change_ratio_vs_mst": cost_ucb_change,
            "mean_cost_feasible": {
                f"{int(round(limit * 100))}pct": predicted_cost_change <= limit
                for limit in thresholds
            },
            "ucb_cost_feasible": {
                f"{int(round(limit * 100))}pct": cost_ucb_change <= limit
                for limit in thresholds
            },
            "set_tsp_complexity": {
                **complexity,
                "sum_change_ratio": (
                    complexity["sum"] - baseline_complexity["sum"]
                ) / max(baseline_complexity["sum"], 1.0),
                "max_change_ratio": (
                    complexity["max"] - baseline_complexity["max"]
                ) / max(baseline_complexity["max"], 1.0),
            },
        })
    return analyzed


def select_true_validation_candidates(
    analyzed_rows: list[dict[str, Any]],
    *,
    count: int = 4,
) -> list[dict[str, Any]]:
    """
    从未真实求解的子集中选择固定预算候选。

    优先级依次为10%成本UCB可行、5%/7%/10%点预测可行，最后按成本越界幅度
    和预测时间补齐；同一候选只选择一次。
    """
    pool = [row for row in analyzed_rows if not row["already_true_validated"]]
    selected: list[dict[str, Any]] = []
    selected_names = set()

    def add_rows(rows: list[dict[str, Any]], reason: str) -> None:
        """按预测时间加入尚未选择的候选，并保存选择原因。"""
        for row in sorted(
            rows,
            key=lambda item: (
                item["surrogate_prediction"]["downstream_total_seconds"],
                item["name"],
            ),
        ):
            if len(selected) >= count or row["name"] in selected_names:
                continue
            selected.append({**row, "selection_reason": reason})
            selected_names.add(row["name"])

    add_rows(
        [row for row in pool if row["ucb_cost_feasible"]["10pct"]],
        "ucb_cost_feasible_10pct",
    )
    for threshold in ("5pct", "7pct", "10pct"):
        add_rows(
            [row for row in pool if row["mean_cost_feasible"][threshold]],
            f"mean_cost_feasible_{threshold}",
        )
    remaining = sorted(
        pool,
        key=lambda row: (
            max(0.0, row["cost_ucb_change_ratio_vs_mst"] - 0.10),
            max(0.0, row["predicted_cost_change_ratio_vs_mst"] - 0.10),
            row["surrogate_prediction"]["downstream_total_seconds"],
            row["name"],
        ),
    )
    add_rows(remaining, "closest_to_conservative_cost_boundary")
    return selected[:count]


def serialize_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """把分析行转换成真实验证程序兼容的紧凑候选JSON。"""
    return {
        "name": row["name"],
        # 真实验证报告用该字段计算代理与真实时间排序，因此直接保存预测总时间。
        "surrogate_score": row["surrogate_prediction"][
            "downstream_total_seconds"
        ],
        "surrogate_score_improvement": -row["predicted_time_change_ratio"],
        "surrogate_prediction": row["surrogate_prediction"],
        "surrogate_uncertainty": row["surrogate_uncertainty"],
        "subset_metadata": {
            "moves": [
                {
                    "customer": move.customer,
                    "source": move.source,
                    "target": move.target,
                }
                for move in row["moves"]
            ],
            "predicted_cost_change_ratio_vs_mst": row[
                "predicted_cost_change_ratio_vs_mst"
            ],
            "cost_ucb": row["cost_ucb"],
            "cost_ucb_change_ratio_vs_mst": row[
                "cost_ucb_change_ratio_vs_mst"
            ],
            "selection_reason": row.get("selection_reason"),
            "set_tsp_complexity": row["set_tsp_complexity"],
        },
        "partition": {
            str(depot): list(customers)
            for depot, customers in row["partition"].items()
        },
    }
