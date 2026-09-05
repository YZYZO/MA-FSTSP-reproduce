"""分析离线候选潜力或独立复测结果，生成 JSON、CSV、Markdown 与候选散点图。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.partition_repair.reporting import candidate_report, evaluation_report


def main():
    """读取结果目录参数，自动区分候选采集与独立复测，输出可阅读报告。"""
    parser = argparse.ArgumentParser(description='分析 MA-FSTSP 分区修复实验')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    output = args.output or args.input / 'report'
    analyze = evaluation_report if (args.input / 'evaluation_config.json').exists() else candidate_report
    analyze(args.input, output, args.bootstrap)
    print(f'报告已生成：{output.resolve()}')


if __name__ == '__main__':
    main()
