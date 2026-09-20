"""仅读取三轮 JSON/JSONL 产物，生成本地最终审计报告。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.reporting import (  # noqa: E402
    build_local_final_report,
    write_json,
    write_local_final_markdown,
)


def parse_arguments() -> argparse.Namespace:
    """解析实验目录和成本上限，返回命令行参数。"""
    parser = argparse.ArgumentParser(description="汇总客户分区学习三轮结果")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "partition_learning_260915",
    )
    parser.add_argument("--cost-limit", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    """读取已有轮次产物，写出 JSON 和 Markdown 总结并返回成功状态。"""
    arguments = parse_arguments()
    report = build_local_final_report(arguments.output_dir, arguments.cost_limit)
    write_json(arguments.output_dir / "local_final_report.json", report)
    write_local_final_markdown(arguments.output_dir / "local_final_report.md", report)
    print(arguments.output_dir / "local_final_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
