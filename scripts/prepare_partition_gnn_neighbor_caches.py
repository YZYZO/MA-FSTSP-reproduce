"""为完整 Deep Sets 数据集准备 k=4、k=8 与 k=16 的 GNN 邻居缓存。"""

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
    """解析数据缓存、NPZ 根目录与输出目录，返回命令行参数。"""
    parser = argparse.ArgumentParser(description="准备覆盖完整数据集的 k=4/8/16 GNN 邻居缓存")
    parser.add_argument(
        "--source-cache",
        type=Path,
        required=True,
        help="旧的 k=8 缓存，仅用于核对特征维度并兼容已有启动命令",
    )
    parser.add_argument("--deepsets-cache", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    return parser.parse_args()


def cache_matches_deepsets(
    gnn_cache: dict,
    deepsets_cache: dict,
    *,
    k_neighbors: int,
) -> bool:
    """判断 GNN 缓存是否覆盖 Deep Sets 的全部实例、候选记录和指定邻居数。"""
    gnn_instances = set(gnn_cache.get("instances", {}))
    deepsets_instances = set(deepsets_cache.get("instances", {}))
    return (
        int(gnn_cache.get("k_neighbors", -1)) == int(k_neighbors)
        and gnn_instances == deepsets_instances
        and len(gnn_cache.get("records", ())) == len(deepsets_cache.get("records", ()))
    )


def _load_matching_cache(
    cache_path: Path,
    deepsets_cache: dict,
    *,
    k_neighbors: int,
) -> dict | None:
    """读取已有缓存；若它没有覆盖完整数据集则返回空值并触发重建。"""
    if not cache_path.is_file():
        return None
    cache = load_partition_gnn_cache(cache_path)
    if cache_matches_deepsets(cache, deepsets_cache, k_neighbors=k_neighbors):
        return cache
    return None


def main() -> int:
    """先保证完整 k=16 缓存存在，再从中稳定生成完整的 k=8 与 k=4 缓存。"""
    arguments = parse_arguments()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 源缓存仅承担兼容检查；新增实例必须以扩展后的 Deep Sets 缓存为准。
    source_cache = load_partition_gnn_cache(arguments.source_cache.resolve())
    if int(source_cache["k_neighbors"]) != 8:
        raise ValueError("--source-cache 必须是 k=8 的 GNN 缓存。")
    deepsets_cache = load_deepsets_cache(arguments.deepsets_cache.resolve())

    k4_path = output_dir / "partition_gnn_k4.pt"
    k8_path = output_dir / "partition_gnn_k8.pt"
    k16_path = output_dir / "partition_gnn_k16.pt"

    # k=16 包含生成 k=8 和 k=4 所需的全部近邻，避免重复计算道路最短距离。
    k16_cache = _load_matching_cache(k16_path, deepsets_cache, k_neighbors=16)
    if k16_cache is None:
        build_partition_gnn_cache(
            deepsets_cache,
            arguments.result_root.resolve(),
            k16_path,
            k_neighbors=16,
            distance_batch_size=arguments.distance_batch_size,
        )
        k16_cache = load_partition_gnn_cache(k16_path)

    if _load_matching_cache(k8_path, deepsets_cache, k_neighbors=8) is None:
        subsample_partition_gnn_cache(k16_cache, k8_path, k_neighbors=8)
    if _load_matching_cache(k4_path, deepsets_cache, k_neighbors=4) is None:
        subsample_partition_gnn_cache(k16_cache, k4_path, k_neighbors=4)

    print(f"k=4: {k4_path}", flush=True)
    print(f"k=8: {k8_path}", flush=True)
    print(f"k=16: {k16_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
