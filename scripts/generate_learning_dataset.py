"""生成里程碑 B 的 Set-MST 邻域学习数据集。"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

# 允许直接从项目根目录运行本脚本，同时不依赖外部 PYTHONPATH 设置。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import MANHATTAN_1K_EXPERIMENT, RESULTS_DIR
from problem import prepare_manhattan_road_network, sample_multiagent_instances
from src.learning.cache import EvaluationCache
from src.learning.dataset import (
    DatasetGenerationSettings,
    generate_instance_dataset_records,
    write_learning_dataset,
)


DEFAULT_DATASET_DIRECTORY = RESULTS_DIR / "learning" / "dataset"
DEFAULT_CACHE_PATH = RESULTS_DIR / "learning" / "probe" / "manhattan_1k_group_cache.sqlite3"


def build_argument_parser():
    """
    构造学习数据生成脚本的命令行参数。

    输入：无。
    输出：带里程碑 B 默认值的参数解析器。
    逻辑：默认每个规模 5 个实例，每实例生成 3 个 relocate 和 1 个 swap 邻域。
    """
    parser = argparse.ArgumentParser(
        description="从 Manhattan 1K 的 Set-MST 划分生成求解器感知学习数据。"
    )
    parser.add_argument(
        "--customer-sizes",
        nargs="+",
        type=int,
        default=[50, 100, 150],
        help="每个原始实例的客户数量。",
    )
    parser.add_argument(
        "--instances-per-size",
        type=int,
        default=5,
        help="每个客户规模的原始实例数量；默认按 3/1/1 划分训练、验证、测试。",
    )
    parser.add_argument(
        "--relocate-samples",
        type=int,
        default=3,
        help="每个原始实例生成的 relocate 一步邻域数量。",
    )
    parser.add_argument(
        "--swap-samples",
        type=int,
        default=1,
        help="每个原始实例生成的 swap 一步邻域数量。",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="单客户组 Set-TSP 的 Gurobi 硬时间上限。",
    )
    parser.add_argument(
        "--phase2-budget",
        type=float,
        default=20.0,
        help="第二阶段墙钟预算标签阈值。",
    )
    parser.add_argument(
        "--downstream-budget",
        type=float,
        default=50.0,
        help="第二、第三阶段总时间预算标签阈值。",
    )
    parser.add_argument(
        "--edge-mode",
        choices=("mean", "min", "max", "legacy"),
        default="mean",
        help="Set-MST 有向道路距离合并方式。",
    )
    parser.add_argument("--seed", type=int, default=0, help="邻域采样和求解随机种子。")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_DATASET_DIRECTORY,
        help="学习数据 CSV 和摘要 JSON 输出目录。",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="与探测实验共享的单客户组求解缓存。",
    )
    parser.add_argument("--run-id", default=None, help="可选输出文件标识。")
    return parser


def _map_identifier(graph_path):
    """
    生成与探测脚本一致的 Manhattan 1K 缓存地图标识。

    输入：GraphML 文件路径。
    输出：文件名、大小和修改时间组成的字符串。
    逻辑：保持相同地图上的基准客户组可以直接命中已有 SQLite 缓存。
    """
    file_stats = graph_path.stat()
    return f"{graph_path.name}:{file_stats.st_size}:{file_stats.st_mtime_ns}"


def _print_record_progress(record):
    """
    输出一个唯一客户组标签完成后的进度。

    输入：扁平学习记录。
    输出：无；立即打印并刷新。
    逻辑：显示拆分、动作、组规模、状态、墙钟时间和缓存命中情况。
    """
    print(
        "  split={split} size={customer_size} instance={instance_index} "
        "action={action_type} role={group_role} group={customer_count:.0f} "
        "status={set_tsp_status} wall={set_tsp_wall_seconds:.3f}s "
        "phase3={phase3_seconds:.3f}s cache={cache_hit}".format(**record),
        flush=True,
    )


def main(argv=None):
    """
    生成完整里程碑 B 数据集并保存实例级拆分摘要。

    输入：可选命令行参数列表。
    输出：进程状态码，同时生成 CSV、JSON 和可继续使用的 SQLite 缓存。
    逻辑：路网距离只初始化一次；中断后重跑会从客户组缓存恢复并补算缺失标签。
    """
    args = build_argument_parser().parse_args(argv)
    graph_path = Path(MANHATTAN_1K_EXPERIMENT.graph_path)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    settings = DatasetGenerationSettings(
        drone_count=MANHATTAN_1K_EXPERIMENT.drone_count,
        edge_mode=args.edge_mode,
        relocate_samples_per_instance=args.relocate_samples,
        swap_samples_per_instance=args.swap_samples,
        set_tsp_time_limit_seconds=args.time_limit,
        phase2_wall_budget_seconds=args.phase2_budget,
        downstream_budget_seconds=args.downstream_budget,
        solver_seed=args.seed,
    )

    print(f"Preparing Manhattan 1K network from {graph_path} ...", flush=True)
    graph, distance, distance_stats = prepare_manhattan_road_network(
        graph_path,
        show_distance_progress=False,
    )
    print(
        "Distance preparation finished: truck={:.3f}s, drone={:.3f}s, total={:.3f}s".format(
            distance_stats["truck_apsp_seconds"],
            distance_stats["drone_pairwise_seconds"],
            distance_stats["distance_initialization_seconds"],
        ),
        flush=True,
    )

    map_id = _map_identifier(graph_path)
    records = []
    interrupted = False
    with EvaluationCache(args.cache_path) as cache:
        try:
            for customer_size in args.customer_sizes:
                depot_instances, city_instances = sample_multiagent_instances(
                    graph,
                    args.instances_per_size,
                    MANHATTAN_1K_EXPERIMENT.depot_count,
                    customer_size,
                    seed=args.seed,
                )
                print(
                    f"Generating size={customer_size}, "
                    f"instances={args.instances_per_size} ...",
                    flush=True,
                )
                for instance_index in range(args.instances_per_size):
                    instance_records = generate_instance_dataset_records(
                        graph,
                        depot_instances[instance_index],
                        city_instances[instance_index],
                        distance,
                        map_id=map_id,
                        customer_size=customer_size,
                        instance_index=instance_index,
                        instances_per_size=args.instances_per_size,
                        cache=cache,
                        settings=settings,
                        progress_callback=_print_record_progress,
                    )
                    records.extend(instance_records)
                    print(
                        f"Completed size={customer_size} instance={instance_index}: "
                        f"unique group samples={len(instance_records)}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            interrupted = True
            print(
                "Dataset generation interrupted; completed group labels remain cached.",
                flush=True,
            )

    if records:
        csv_path, summary_path, summary = write_learning_dataset(
            records,
            args.output_directory,
            settings,
            run_id=run_id,
        )
        print(
            "Dataset finished: rows={}, instances={}, splits={}, timeouts={}, "
            "phase2_budget_exceeded={}, downstream_budget_exceeded={}".format(
                summary["row_count"],
                summary["base_instance_count"],
                summary["split_row_counts"],
                summary["timeout_count"],
                summary["phase2_budget_exceeded_count"],
                summary["downstream_budget_exceeded_count"],
            ),
            flush=True,
        )
        print(f"CSV: {csv_path}", flush=True)
        print(f"Summary: {summary_path}", flush=True)
    else:
        print("No learning record was completed.", flush=True)

    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
