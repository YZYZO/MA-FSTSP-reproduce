"""分析离线候选潜力或独立复测结果，生成 JSON、CSV、Markdown 与候选散点图。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.partition_repair.reporting import candidate_report, evaluation_report
from src.partition_repair.settings import EvaluationOptions
from src.partition_repair.storage import read_json


def main():
    """读取结果目录参数，自动区分候选采集与独立复测，输出可阅读报告。"""
    parser = argparse.ArgumentParser(description='分析 MA-FSTSP 分区修复实验')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--cost-limit', type=float)
    parser.add_argument('--min-phase2-saving', type=float)
    args = parser.parse_args()
    is_evaluation = (args.input / 'evaluation_config.json').exists()
    configuration_path = args.input / ('evaluation_config.json' if is_evaluation else 'manifest.json')
    settings = read_json(configuration_path)['configuration'].get('evaluation', {})
    # 显式参数覆盖清单；清单没有门槛时采用当前 10% / 20% 默认值。
    options = EvaluationOptions(**settings)
    options = EvaluationOptions(args.cost_limit if args.cost_limit is not None else options.cost_limit,
                                args.min_phase2_saving if args.min_phase2_saving is not None else options.min_phase2_saving)
    output = args.output or args.input / f'report_cost_{options.cost_limit:g}_time_{options.min_phase2_saving:g}'
    analyze = evaluation_report if is_evaluation else candidate_report
    analyze(args.input, output, args.bootstrap, options)
    print(f'报告已生成：{output.resolve()}')


if __name__ == '__main__':
    main()
