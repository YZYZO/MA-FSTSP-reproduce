"""从一个或多个候选 JSONL 训练划分性能预测器，不读取路网或 NPZ。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.dataset import select_instance_indices  # noqa: E402
from src.partition_learning.models import train_model_ensemble  # noqa: E402
from src.partition_learning.pipeline import (  # noqa: E402
    ThreeRoundExperiment,
    _family_stratified_subset,
)
from src.partition_learning.reporting import (  # noqa: E402
    deduplicate_candidate_records,
    deduplicate_partition_records,
    read_jsonl,
    write_json,
    write_jsonl,
    write_partition_algorithm_markdown,
)


def parse_arguments() -> argparse.Namespace:
    """解析候选记录、输出目录和可选的分层实例上限，返回命令行参数。"""
    parser = argparse.ArgumentParser(description="用已完成候选记录训练客户划分性能预测器")
    parser.add_argument(
        "--records",
        type=Path,
        nargs="+",
        required=True,
        help="candidate_records.jsonl 文件或包含该文件的实验目录，可同时传入多个来源。",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cost-limit", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=260915)
    parser.add_argument(
        "--candidates-per-instance",
        type=int,
        default=24,
        help="每实例按算法族分层保留的候选数；默认与新一轮实验一致为24。",
    )
    parser.add_argument(
        "--max-instances-per-stratum",
        type=int,
        help="每个“路网×客户规模”最多保留的实例数；不设置时使用全部已完成实例。",
    )
    return parser.parse_args()


def _resolve_record_path(path: Path) -> Path:
    """输入 JSONL 或实验目录，输出实际候选记录文件路径。"""
    if path.is_file():
        return path
    active = path / "candidate_records_active.jsonl"
    if active.is_file():
        return active
    return path / "candidate_records.jsonl"


def _select_balanced_instances(records: list[dict], limit: int | None) -> list[dict]:
    """
    按“路网×客户规模”分层选择完整实例，输出其全部候选。

    不设置上限时保留全部记录；设置上限时在每层按实例下标等距抽样，防止单一路网或规模
    数量过多而主导模型。这里按实例而非候选行抽样，保证同一实例的候选集合不会被拆散。
    """
    if limit is None:
        return list(records)
    strata: dict[tuple[str, int], dict[str, int]] = {}
    for record in records:
        key = (str(record["graph_name"]), int(record["customer_count"]))
        strata.setdefault(key, {})[str(record["instance_id"])] = int(record["instance_index"])
    selected_ids: set[str] = set()
    for instances in strata.values():
        ordered = sorted(instances, key=lambda instance_id: (instances[instance_id], instance_id))
        positions = select_instance_indices(len(ordered), min(limit, len(ordered)))
        selected_ids.update(ordered[position] for position in positions)
    return [record for record in records if str(record["instance_id"]) in selected_ids]


def _select_stratified_candidates(records: list[dict], limit: int) -> list[dict]:
    """输入多个实例的候选记录，输出每实例覆盖算法族与动作强度的固定预算子集。"""
    by_instance: dict[str, list[dict]] = {}
    for record in records:
        by_instance.setdefault(str(record["instance_id"]), []).append(record)
    selected: list[dict] = []
    for instance_records in by_instance.values():
        selected.extend(_family_stratified_subset(
            instance_records,
            limit,
            family_of=lambda row: row.get("generator", {}).get(
                "family", row.get("candidate_kind", "unknown")
            ),
            is_stay=lambda row: row.get("candidate_name") == "stay",
        ))
    return selected


def main() -> int:
    """合并、去重和分层候选，生成算法报告并在实例级切分上训练监督模型。"""
    arguments = parse_arguments()
    source_paths = [_resolve_record_path(path.resolve()) for path in arguments.records]
    records = deduplicate_candidate_records([
        record
        for path in source_paths
        for record in read_jsonl(path)
    ])
    records = _select_balanced_instances(records, arguments.max_instances_per_stratum)
    records = _select_stratified_candidates(records, arguments.candidates_per_instance)
    if not records:
        raise ValueError("没有读到候选记录，请检查 --records 路径。")

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "candidate_records_selected.jsonl", records)
    algorithm_report = ThreeRoundExperiment._algorithm_report(records, arguments.cost_limit)
    write_json(output_dir / "algorithm_report.json", algorithm_report)

    exact_instances = {
        str(record["instance_id"])
        for record in records
        if bool(record["labels"].get("relative_labels_exact", False))
    }
    if len(exact_instances) < 2:
        result = {
            "source_paths": list(map(str, source_paths)),
            "algorithms": algorithm_report,
            "training_skipped": "精确候选不足两个独立实例，暂时只能生成数据报告。",
        }
        write_json(output_dir / "experiment_summary.json", result)
        (output_dir / "experiment_summary.md").write_text(
            "# 划分监督数据阶段性报告\n\n"
            f"已读取 {len(records)} 条候选、{algorithm_report['instance_count']} 个实例。"
            "精确候选不足两个独立实例，尚不能进行实例级训练/测试切分。\n",
            encoding="utf-8",
        )
        return 0

    training_records = deduplicate_partition_records(records)
    _, training_report = train_model_ensemble(
        training_records,
        output_dir / "models_serial_only",
        random_seed=arguments.random_seed,
        cost_limit=arguments.cost_limit,
    )
    result = {
        "source_paths": list(map(str, source_paths)),
        "configuration": {
            "cost_limit": arguments.cost_limit,
            "max_instances_per_stratum": arguments.max_instances_per_stratum,
            "candidates_per_instance": arguments.candidates_per_instance,
            "random_seed": arguments.random_seed,
        },
        "bootstrap_record_count": 0,
        "algorithms": algorithm_report,
        "training_serial_only": training_report,
        # 独立训练入口没有混入旧轮次，两项指向同一报告以保持汇总格式兼容。
        "training_mixed_exploratory": training_report,
    }
    write_json(output_dir / "experiment_summary.json", result)
    write_partition_algorithm_markdown(output_dir / "experiment_summary.md", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
