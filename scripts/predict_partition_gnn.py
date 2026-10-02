"""演示或命令行调用任意客户划分的GNN监督预测接口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.gnn_data import load_partition_gnn_cache  # noqa: E402
from src.partition_learning.gnn_inference import PartitionGNNPredictor  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """解析缓存、原始NPZ、实例、模型和可选JSON分区文件。"""
    parser = argparse.ArgumentParser(description="预测任意客户划分的二三阶段性能")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--partition-json",
        type=Path,
        help="仓库到客户列表的JSON；省略时预测该实例的MST基线分区。",
    )
    return parser.parse_args()


def main() -> int:
    """加载推理上下文，预测指定分区并将完整结果打印为JSON。"""
    arguments = parse_arguments()
    cache = load_partition_gnn_cache(arguments.cache.resolve())
    predictor = PartitionGNNPredictor(
        cache,
        arguments.result_root.resolve(),
        arguments.instance_id,
        [path.resolve() for path in arguments.checkpoint],
    )
    if arguments.partition_json is None:
        partition = predictor.instance.partition
    else:
        payload = json.loads(arguments.partition_json.read_text(encoding="utf-8"))
        partition = {int(depot): tuple(map(int, cities)) for depot, cities in payload.items()}
    result = predictor.predict(partition, kind="stay")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
