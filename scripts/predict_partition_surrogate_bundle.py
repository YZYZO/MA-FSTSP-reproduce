"""命令行调用最终三角色监督代理模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.surrogate_bundle import PartitionSurrogateBundle  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """解析部署清单、原始 NPZ 根目录、实例和可选划分 JSON。"""
    parser = argparse.ArgumentParser(description="统一预测任意客户划分的二三阶段性能")
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument(
        "--partition-json",
        type=Path,
        help="仓库到客户列表的 JSON；省略时预测实例的 MST 基线划分。",
    )
    return parser.parse_args()


def main() -> int:
    """加载三角色部署集成并把一个划分的统一预测打印为 JSON。"""
    arguments = parse_arguments()
    bundle = PartitionSurrogateBundle(
        arguments.bundle_manifest,
        arguments.result_root,
        arguments.instance_id,
    )
    if arguments.partition_json is None:
        partition = bundle.instance.partition
    else:
        payload = json.loads(arguments.partition_json.read_text(encoding="utf-8"))
        partition = {
            int(depot): tuple(map(int, customers))
            for depot, customers in payload.items()
        }
    result = bundle.predict(partition, kind="stay")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
