"""把RL候选真实复核JSONL编码并增量合并进最新GNN训练缓存。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.deep_sets_data import build_deepsets_cache  # noqa: E402
from src.partition_learning.gnn_cache_merge import merge_partition_gnn_caches  # noqa: E402
from src.partition_learning.gnn_data import (  # noqa: E402
    build_partition_gnn_cache,
    load_partition_gnn_cache,
)


def parse_arguments() -> argparse.Namespace:
    """解析基础GNN缓存、原始NPZ目录、新增真实记录和输出路径。"""
    parser = argparse.ArgumentParser(description="合并RL真实复核标签到GNN缓存")
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-records",
        type=Path,
        action="append",
        required=True,
        help="可重复提供多个candidate_records.jsonl。",
    )
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    return parser.parse_args()


def _load_jsonl(paths: list[Path]) -> list[dict]:
    """输入一个或多个JSONL文件，输出保持来源顺序的候选记录。"""
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def main() -> int:
    """编码新增候选、按完整划分去重合并，并写出缓存及统计报告。"""
    arguments = parse_arguments()
    base_cache = load_partition_gnn_cache(arguments.base_cache.resolve())
    record_paths = [path.resolve() for path in arguments.candidate_records]
    rows = _load_jsonl(record_paths)
    output_cache = arguments.output_cache.resolve()
    output_cache.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="partition_rl_cache_", dir=output_cache.parent
    ) as temporary:
        temporary_dir = Path(temporary)
        deepsets = build_deepsets_cache(
            rows,
            arguments.result_root.resolve(),
            temporary_dir / "deepsets.pt",
            cost_limit=float(base_cache["cost_limit"]),
            distance_batch_size=arguments.distance_batch_size,
        )
        addition = build_partition_gnn_cache(
            deepsets,
            arguments.result_root.resolve(),
            temporary_dir / "gnn.pt",
            k_neighbors=int(base_cache["k_neighbors"]),
            distance_batch_size=arguments.distance_batch_size,
        )

    merged, report = merge_partition_gnn_caches(
        base_cache,
        [addition],
        source_names=[str(path) for path in record_paths],
    )
    torch.save(merged, output_cache)
    report_path = (
        arguments.report.resolve()
        if arguments.report is not None
        else output_cache.with_suffix(".merge_report.json")
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"GNN缓存合并完成：新增={report['added_record_count']}，"
        f"替换={report['replaced_record_count']}，跳过={report['skipped_duplicate_count']}，"
        f"输出={output_cache}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
