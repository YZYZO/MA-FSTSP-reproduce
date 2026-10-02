"""生成候选划分GNN交叉验证的误差、集成和不确定性报告。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.cross_validation_analysis import save_analysis_reports  # noqa: E402
from src.partition_learning.gnn_data import load_partition_gnn_cache  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """解析GNN缓存、交叉验证目录和分析产物目录。"""
    parser = argparse.ArgumentParser(description="分析候选划分GNN的折外预测")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cross-validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """加载缓存并生成可复用CSV、机器可读JSON和浏览器HTML报告。"""
    arguments = parse_arguments()
    cache = load_partition_gnn_cache(arguments.cache.resolve())
    save_analysis_reports(
        cache,
        arguments.cross_validation_dir.resolve(),
        arguments.output_dir.resolve(),
    )
    print(f"交叉验证分析完成：{arguments.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
