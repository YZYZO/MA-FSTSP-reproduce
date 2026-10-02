"""由现有k=8缓存快速生成k=4缓存，并按需重建更密的k=16缓存。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.deep_sets_data import load_deepsets_cache  # noqa: E402
from src.partition_learning.gnn_data import (  # noqa: E402
    build_partition_gnn_cache,
    load_partition_gnn_cache,
    subsample_partition_gnn_cache,
)


def parse_arguments() -> argparse.Namespace:
    """解析源缓存、Deep Sets缓存、NPZ根目录和结构消融缓存目录。"""
    parser = argparse.ArgumentParser(description="准备k=4/8/16候选划分GNN缓存")
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--deepsets-cache", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    """复用k=8，子采样k=4，仅对缺失的k=16执行道路距离重建。"""
    arguments = parse_arguments()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_cache = load_partition_gnn_cache(arguments.source_cache.resolve())
    source_k = int(source_cache["k_neighbors"])
    if source_k != 8:
        raise ValueError(f"当前结构消融以k=8为源缓存，实际为k={source_k}。")
    k4_path = output_dir / "partition_gnn_k4.pt"
    k8_path = arguments.source_cache.resolve()
    k16_path = output_dir / "partition_gnn_k16.pt"
    if not k4_path.is_file():
        subsample_partition_gnn_cache(source_cache, k4_path, k_neighbors=4)
    if not k16_path.is_file():
        deepsets_cache = load_deepsets_cache(arguments.deepsets_cache.resolve())
        build_partition_gnn_cache(
            deepsets_cache,
            arguments.result_root.resolve(),
            k16_path,
            k_neighbors=16,
            distance_batch_size=arguments.distance_batch_size,
        )
    print(f"k=4: {k4_path}", flush=True)
    print(f"k=8: {k8_path}", flush=True)
    print(f"k=16: {k16_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
