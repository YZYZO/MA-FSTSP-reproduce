"""为下一轮真实求解选择跨路网、跨规模的独立多样实例。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.active_instance_selection import (  # noqa: E402
    select_diverse_unlabelled_instances,
)
from src.partition_learning.gnn_data import load_partition_gnn_cache  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """解析原始实例根目录、当前缓存、每来源新增数量和输出清单。"""
    parser = argparse.ArgumentParser(description="选择监督模型主动扩充的新实例")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--instances-per-source", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """选择未标注最远点实例并写出算法实验可直接使用的JSON清单。"""
    arguments = parse_arguments()
    cache = load_partition_gnn_cache(arguments.cache.resolve())
    report = select_diverse_unlabelled_instances(
        arguments.result_root.resolve(),
        cache,
        instances_per_source=arguments.instances_per_source,
    )
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"主动扩充实例清单已写入：{output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
