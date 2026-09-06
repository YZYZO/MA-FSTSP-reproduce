"""服务器上的 CPU 分组开发诊断与冻结策略训练入口。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.partition_repair.learning_dataset import load_dataset
from src.partition_repair.learning_calibration import diagnose, fit_bundle
from src.partition_repair.settings import EvaluationOptions


def main():
    """解析已有标签目录，执行分组诊断或独立验证校准，将模型与报告写入指定目录。"""
    parser = argparse.ArgumentParser(description='成本增幅 10% 下的 CPU 候选选择学习')
    parser.add_argument('--mode', choices=('diagnose', 'fit'), default='diagnose')
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--validation', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=906030)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--cost-limit', type=float, default=.10)
    parser.add_argument('--min-phase2-saving', type=float, default=.20)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('输出目录非空，请使用新的目录，避免覆盖已冻结策略或混合诊断。')
    dataset = load_dataset(args.input)
    options = EvaluationOptions(args.cost_limit, args.min_phase2_saving)
    if args.mode == 'diagnose':
        diagnose(dataset, args.output, args.folds, args.inner_folds, args.seed, options, args.bootstrap)
    else:
        if not args.validation:
            parser.error('fit 必须指定与训练实例不重叠的 --validation 候选目录。')
        fit_bundle(dataset, load_dataset(args.validation), args.output, args.seed, options, args.bootstrap)
    print(f'完成：{args.output.resolve()}。查看 learning_report.md 和 learning_report.json。')


if __name__ == '__main__':
    main()
