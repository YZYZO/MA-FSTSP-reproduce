"""枚举PPO最终迁移子集，调用冻结代理并导出服务器真实复核候选。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.candidates import canonical_partition  # noqa: E402
from src.partition_learning.rl_subset_candidates import (  # noqa: E402
    attach_surrogate_analysis,
    enumerate_move_subsets,
    select_true_validation_candidates,
    serialize_candidate,
)
from src.partition_learning.surrogate_bundle import PartitionSurrogateBundle  # noqa: E402


DEFAULT_ROOT = PROJECT_ROOT / "results" / "服务器运行结果下载"


def parse_arguments() -> argparse.Namespace:
    """解析代理清单、已有候选、实例、成本UCB和输出参数。"""
    parser = argparse.ArgumentParser(description="生成PPO迁移子集候选")
    parser.add_argument(
        "--bundle-manifest",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "260930supervised_8stage"
            / "stage8_deployment"
            / "ensembles"
            / "deployment_bundle.json"
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_ROOT / "260826_55k_master_results",
    )
    parser.add_argument(
        "--input-candidates",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "261004partition_rl_stage1_local"
            / "boston_11k_100_016"
            / "validation_candidates.json"
        ),
    )
    parser.add_argument(
        "--instance-id",
        default="20260825-112859-boston_11k-100__016",
    )
    parser.add_argument("--target-name", default="ppo_best")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "261004partition_rl_stage2_subsets",
    )
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--selection-count", type=int, default=4)
    return parser.parse_args()


def _json_safe(row: dict[str, Any]) -> dict[str, Any]:
    """递归转换划分、迁移数据类和元组，生成可读JSON分析行。"""
    result = dict(row)
    result["partition"] = {
        str(depot): list(customers)
        for depot, customers in row["partition"].items()
    }
    result["moves"] = [
        {
            "customer": move.customer,
            "source": move.source,
            "target": move.target,
        }
        for move in row["moves"]
    ]
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """把关键代理预测、约束和复杂度指标写成便于比较的CSV。"""
    fieldnames = [
        "name",
        "move_count",
        "moves",
        "predicted_total_seconds",
        "predicted_time_change_ratio",
        "predicted_final_cost",
        "predicted_cost_change_ratio_vs_mst",
        "final_cost_uncertainty",
        "cost_ucb",
        "cost_ucb_change_ratio_vs_mst",
        "mean_feasible_5pct",
        "mean_feasible_7pct",
        "mean_feasible_10pct",
        "ucb_feasible_5pct",
        "ucb_feasible_7pct",
        "ucb_feasible_10pct",
        "time_p90",
        "timeout_probability",
        "complexity_sum_change_ratio",
        "complexity_max_change_ratio",
        "already_true_validated",
        "selected_for_true_validation",
        "selection_reason",
    ]
    selected = {
        row["name"]: row.get("selection_reason") for row in rows if row.get("selected")
    }
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            prediction = row["surrogate_prediction"]
            uncertainty = row["surrogate_uncertainty"]
            writer.writerow({
                "name": row["name"],
                "move_count": row["move_count"],
                "moves": ";".join(
                    f"{move.customer}:{move.source}->{move.target}"
                    for move in row["moves"]
                ),
                "predicted_total_seconds": prediction["downstream_total_seconds"],
                "predicted_time_change_ratio": row["predicted_time_change_ratio"],
                "predicted_final_cost": prediction["final_cost"],
                "predicted_cost_change_ratio_vs_mst": row[
                    "predicted_cost_change_ratio_vs_mst"
                ],
                "final_cost_uncertainty": uncertainty.get("final_cost"),
                "cost_ucb": row["cost_ucb"],
                "cost_ucb_change_ratio_vs_mst": row[
                    "cost_ucb_change_ratio_vs_mst"
                ],
                "mean_feasible_5pct": row["mean_cost_feasible"]["5pct"],
                "mean_feasible_7pct": row["mean_cost_feasible"]["7pct"],
                "mean_feasible_10pct": row["mean_cost_feasible"]["10pct"],
                "ucb_feasible_5pct": row["ucb_cost_feasible"]["5pct"],
                "ucb_feasible_7pct": row["ucb_cost_feasible"]["7pct"],
                "ucb_feasible_10pct": row["ucb_cost_feasible"]["10pct"],
                "time_p90": prediction["downstream_time_p90"],
                "timeout_probability": prediction["right_censored_probability"],
                "complexity_sum_change_ratio": row["set_tsp_complexity"][
                    "sum_change_ratio"
                ],
                "complexity_max_change_ratio": row["set_tsp_complexity"][
                    "max_change_ratio"
                ],
                "already_true_validated": row["already_true_validated"],
                "selected_for_true_validation": row["name"] in selected,
                "selection_reason": selected.get(row["name"]),
            })


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    selected_names: set[str],
) -> None:
    """输出人工可读的迁移子集代理筛选表。"""
    lines = [
        "# PPO迁移子集代理筛选",
        "",
        "成本UCB = 预测成本 + κ × 集成标准差；真实10%硬约束由服务器求解判定。",
        "",
        "|候选|迁移数|预测时间变化|预测成本变化|成本UCB变化|UCB≤10%|复杂度总量变化|真实复核|",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"|{row['name']}|{row['move_count']}|"
            f"{row['predicted_time_change_ratio']:.2%}|"
            f"{row['predicted_cost_change_ratio_vs_mst']:.2%}|"
            f"{row['cost_ucb_change_ratio_vs_mst']:.2%}|"
            f"{row['ucb_cost_feasible']['10pct']}|"
            f"{row['set_tsp_complexity']['sum_change_ratio']:.2%}|"
            f"{'已选' if row['name'] in selected_names else ''}|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """生成8个子集、调用代理、筛选4个新候选并写出服务器输入。"""
    arguments = parse_arguments()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(arguments.input_candidates.read_text(encoding="utf-8"))
    source_rows = payload[arguments.instance_id]
    baseline_row = next(row for row in source_rows if row["name"] == "mst")
    target_row = next(
        row for row in source_rows if row["name"] == arguments.target_name
    )
    depots = tuple(map(int, baseline_row["partition"]))
    baseline = canonical_partition(
        {int(key): value for key, value in baseline_row["partition"].items()},
        depots,
    )
    target = canonical_partition(
        {int(key): value for key, value in target_row["partition"].items()},
        depots,
    )
    subset_rows = enumerate_move_subsets(baseline, target)
    bundle = PartitionSurrogateBundle(
        arguments.bundle_manifest.resolve(),
        arguments.result_root.resolve(),
        arguments.instance_id,
    )
    outputs = bundle.predict_many(
        [row["partition"] for row in subset_rows],
        kinds=["rl_subset" for _ in subset_rows],
        strengths=[
            row["move_count"] / max(len(bundle.instance.cities), 1)
            for row in subset_rows
        ],
    )
    analyzed = attach_surrogate_analysis(
        subset_rows,
        outputs,
        bundle.instance.boundary_sizes,
        kappa=arguments.kappa,
    )
    selected = select_true_validation_candidates(
        analyzed,
        count=arguments.selection_count,
    )
    selected_names = {row["name"] for row in selected}
    rows_with_selection = [
        {
            **row,
            "selected": row["name"] in selected_names,
            "selection_reason": next(
                (
                    selected_row["selection_reason"]
                    for selected_row in selected
                    if selected_row["name"] == row["name"]
                ),
                None,
            ),
        }
        for row in analyzed
    ]
    report = {
        "kind": "ppo_final_move_subset_surrogate_screening",
        "instance_id": arguments.instance_id,
        "source_candidate_file": str(arguments.input_candidates.resolve()),
        "target_name": arguments.target_name,
        "kappa": arguments.kappa,
        "cost_thresholds": [0.05, 0.07, 0.10],
        "move_count": max(row["move_count"] for row in analyzed),
        "subset_count": len(analyzed),
        "selected_new_candidate_count": len(selected),
        "selected_names": sorted(selected_names),
        "candidates": [_json_safe(row) for row in rows_with_selection],
    }
    (output_dir / "subset_surrogate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "subset_surrogate_predictions.csv", rows_with_selection)
    _write_markdown(
        output_dir / "subset_surrogate_report.md",
        rows_with_selection,
        selected_names,
    )

    # MST与完整PPO作为已验证对照；服务器只新增求解选中的中间子集。
    baseline_analyzed = next(row for row in analyzed if row["name"] == "mst")
    target_analyzed = next(row for row in analyzed if row["name"] == "ppo_best")
    server_rows = [
        serialize_candidate(baseline_analyzed),
        serialize_candidate(target_analyzed),
        *[serialize_candidate(row) for row in selected],
    ]
    server_payload = {arguments.instance_id: server_rows}
    (output_dir / "subset_candidates_for_true_validation.json").write_text(
        json.dumps(server_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"迁移子集={len(analyzed)}，新增真实复核候选={len(selected)}："
        + ", ".join(sorted(selected_names)),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
