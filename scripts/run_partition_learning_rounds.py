"""从命令行运行基于 260826 八个 NPZ 的三轮分区学习实验。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.pipeline import ThreeRoundExperiment  # noqa: E402


DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "服务器运行结果下载" / "260826_55k_master_results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "partition_learning_260915"


def parse_arguments() -> argparse.Namespace:
    """解析数据路径、轮次与本机求解预算，返回 argparse 命名空间。"""
    parser = argparse.ArgumentParser(description="运行 MA-FSTSP 分区学习三轮实验")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--round",
        choices=("oracle", "supervised", "active", "algorithms", "all"),
        default="all",
    )
    parser.add_argument(
        "--solver-time-limit",
        type=float,
        default=9999.0,
        help="每个仓库组的 Set-TSP 求解参数；9999 表示不向 Gurobi 设置时间上限。",
    )
    parser.add_argument(
        "--max-binary-variables",
        type=int,
        default=0,
        help="复杂度保护阈值；0 表示关闭保护并尝试求解全部模型。",
    )
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--candidates-per-instance", type=int, default=12)
    parser.add_argument("--active-pool-size", type=int, default=64)
    parser.add_argument("--algorithm-instances-per-file", type=int, default=3)
    parser.add_argument("--algorithm-candidates-per-instance", type=int, default=36)
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=1,
        help="同一候选内的仓库组评价数；监督时间实验默认严格串行。",
    )
    parser.add_argument(
        "--include-55k",
        action="store_true",
        help="显式允许载入并求解 55K；本地默认关闭，仅应在内存充足服务器使用。",
    )
    parser.add_argument(
        "--only-graph",
        choices=("manhattan_1k", "boston_11k", "manhattan_55k"),
        help="只执行指定路网，服务器补跑 55K 时使用。",
    )
    parser.add_argument(
        "--customer-counts",
        type=int,
        nargs="+",
        choices=(50, 100, 150),
        help="只执行并汇总指定客户规模，例如 --customer-counts 50 100。",
    )
    return parser.parse_args()


def main() -> int:
    """构造可续跑实验对象，按用户选择执行一轮或全部三轮。"""
    arguments = parse_arguments()
    experiment = ThreeRoundExperiment(
        arguments.result_root,
        arguments.output_dir,
        solver_time_limit=arguments.solver_time_limit,
        max_binary_variables=arguments.max_binary_variables,
        distance_batch_size=arguments.distance_batch_size,
        candidates_per_instance=arguments.candidates_per_instance,
        active_pool_size=arguments.active_pool_size,
        include_55k=arguments.include_55k,
        only_graph=arguments.only_graph,
        customer_counts=tuple(arguments.customer_counts) if arguments.customer_counts else None,
        evaluation_workers=arguments.evaluation_workers,
        algorithm_instances_per_file=arguments.algorithm_instances_per_file,
        algorithm_candidates_per_instance=arguments.algorithm_candidates_per_instance,
    )
    if arguments.round == "oracle":
        experiment.run_oracle()
    elif arguments.round == "supervised":
        experiment.run_supervised()
    elif arguments.round == "active":
        experiment.run_active()
    elif arguments.round == "algorithms":
        experiment.run_partition_algorithms()
    else:
        experiment.run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
