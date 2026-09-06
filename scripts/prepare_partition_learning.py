"""整理阶段 C 标签索引，派生当前实验清单或采样独立新实例。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MANHATTAN1k_GRAPH_PATH
from src.partition_repair.learning_preparation import dataset_summary, derive_manifest, sample_manifest


def main():
    """按明确子命令准备数据或实例清单；准备过程不运行 Set-TSP 或第三阶段。"""
    parser = argparse.ArgumentParser(description='阶段 C 数据和清单准备')
    sub = parser.add_subparsers(dest='action', required=True)
    dataset = sub.add_parser('dataset')
    dataset.add_argument('--input', type=Path, nargs='+', required=True)
    dataset.add_argument('--output', type=Path, required=True)
    derive = sub.add_parser('derive')
    derive.add_argument('--source-manifest', type=Path, required=True)
    derive.add_argument('--output', type=Path, required=True)
    derive.add_argument('--graph', type=Path, default=MANHATTAN1k_GRAPH_PATH)
    sample = sub.add_parser('sample')
    sample.add_argument('--template-manifest', type=Path, required=True)
    sample.add_argument('--split', choices=('train', 'validation', 'test'), required=True)
    sample.add_argument('--instances-per-size', type=int, default=10)
    sample.add_argument('--seed', type=int, required=True)
    sample.add_argument('--exclude-manifests', type=Path, nargs='*', default=[])
    sample.add_argument('--output', type=Path, required=True)
    sample.add_argument('--graph', type=Path, default=MANHATTAN1k_GRAPH_PATH)
    args = parser.parse_args()
    if args.action == 'dataset':
        dataset_summary(args.input, args.output)
    elif args.action == 'derive':
        derive_manifest(args.source_manifest, args.output, args.graph)
    else:
        sample_manifest(args.template_manifest, args.output, args.graph, args.split,
                        args.instances_per_size, args.seed, args.exclude_manifests)
    print(f'已生成：{args.output.resolve()}')


if __name__ == '__main__':
    main()
