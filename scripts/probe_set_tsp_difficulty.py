"""运行获批的 Manhattan 1K 小规模 Set-TSP 难度探测。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

# 直接运行 `python scripts/...` 时，把项目根目录加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import MANHATTAN_1K_EXPERIMENT, RESULTS_DIR
from problem import prepare_manhattan_road_network, sample_multiagent_instances
from src.learning.cache import EvaluationCache
from src.learning.probe import ProbeSettings, probe_instance, write_probe_outputs


DEFAULT_OUTPUT_DIRECTORY = RESULTS_DIR / "learning" / "probe"
DEFAULT_CACHE_PATH = DEFAULT_OUTPUT_DIRECTORY / "manhattan_1k_group_cache.sqlite3"


def build_argument_parser():
    """
    构造探测脚本命令行参数。

    输入：无。
    输出：带推荐默认值的 `ArgumentParser`。
    逻辑：默认运行 50/100/150 客户、每档 2 个实例、每客户组 30 秒上限。
    """
    parser = argparse.ArgumentParser(
        description="探测 Manhattan 1K 不同客户组的 Set-TSP 时间和超时风险。"
    )
    parser.add_argument(
        "--customer-sizes",
        nargs="+",
        type=int,
        default=[50, 100, 150],
        help="需要探测的总客户规模。",
    )
    parser.add_argument(
        "--instances-per-size",
        type=int,
        default=2,
        help="每个客户规模采样的原始实例数。",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="每个客户组 Set-TSP 的 Gurobi 时间上限（秒）。",
    )
    parser.add_argument(
        "--edge-mode",
        choices=("mean", "min", "max", "legacy"),
        default="mean",
        help="MST 构图的有向距离合并方式。",
    )
    parser.add_argument("--seed", type=int, default=0, help="实例采样和求解随机种子。")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="CSV 和 JSON 汇总输出目录。",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="可恢复的单客户组 SQLite 缓存路径。",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="可选输出文件标识；默认使用当前时间。",
    )
    return parser


def _map_identifier(graph_path):
    """
    根据 Manhattan 1K 文件元数据生成缓存地图标识。

    输入：GraphML 文件路径。
    输出：包含文件名、大小和修改时间的字符串。
    逻辑：地图文件变化后自动使用新缓存键，避免误复用旧标签。
    """
    file_stats = graph_path.stat()
    return f"{graph_path.name}:{file_stats.st_size}:{file_stats.st_mtime_ns}"


def _print_group_progress(record):
    """
    输出一个客户组完成后的关键运行信息。

    输入：`probe_instance` 生成的扁平记录。
    输出：无；立即刷新标准输出。
    逻辑：长探测期间每组给出进度，便于判断长尾和缓存命中情况。
    """
    print(
        "  instance={instance_index} depot={depot_index} group={customer_count} "
        "V={set_tsp_complexity_proxy:.0f} status={set_tsp_status} "
        "t={set_tsp_runtime_seconds:.3f}s phase3={phase3_seconds:.3f}s "
        "cache={cache_hit}".format(**record),
        flush=True,
    )


def _print_summary(summary):
    """
    在终端输出足以决定下一阶段预算的紧凑统计。

    输入：`summarize_probe_records` 返回的汇总字典。
    输出：无；打印总体和各客户规模的 P50/P90/超时率。
    逻辑：完整内容保存在 JSON，终端只显示关键判断指标。
    """
    overall = summary["overall"]
    runtime = overall["set_tsp_runtime_seconds"]
    print(
        "Overall: groups={group_count}, timeouts={timeout_count} "
        "({timeout_rate:.1%}), runtime P50={p50:.3f}s, "
        "P90={p90:.3f}s, max={max:.3f}s".format(
            group_count=overall["group_count"],
            timeout_count=overall["timeout_count"],
            timeout_rate=overall["timeout_rate"],
            **runtime,
        )
    )
    for customer_size, size_summary in summary["by_customer_size"].items():
        size_runtime = size_summary["set_tsp_runtime_seconds"]
        print(
            "  size={}: groups={}, timeouts={}, group max={}, "
            "runtime P50={:.3f}s P90={:.3f}s max={:.3f}s".format(
                customer_size,
                size_summary["group_count"],
                size_summary["timeout_count"],
                size_summary["group_size"]["max"],
                size_runtime["p50"],
                size_runtime["p90"],
                size_runtime["max"],
            )
        )


def main(argv=None):
    """
    准备 Manhattan 1K 路网并执行推荐的小规模探测实验。

    输入：可选命令行参数列表。
    输出：进程状态码；同时生成 CSV、JSON 和 SQLite 缓存。
    逻辑：每张地图只初始化一次距离，按规模采样实例，逐客户组可恢复评估。
    """
    args = build_argument_parser().parse_args(argv)
    graph_path = Path(MANHATTAN_1K_EXPERIMENT.graph_path)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    settings = ProbeSettings(
        drone_count=MANHATTAN_1K_EXPERIMENT.drone_count,
        edge_mode=args.edge_mode,
        set_tsp_time_limit_seconds=args.time_limit,
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
                    f"Probing customer size {customer_size} "
                    f"with {args.instances_per_size} instances ...",
                    flush=True,
                )
                for instance_index in range(args.instances_per_size):
                    records.extend(
                        probe_instance(
                            graph,
                            depot_instances[instance_index],
                            city_instances[instance_index],
                            distance,
                            map_id=map_id,
                            customer_size=customer_size,
                            instance_index=instance_index,
                            cache=cache,
                            settings=settings,
                            progress_callback=_print_group_progress,
                        )
                    )
        except KeyboardInterrupt:
            interrupted = True
            print("Probe interrupted; completed groups remain in SQLite cache.", flush=True)

    if records:
        csv_path, summary_path, summary = write_probe_outputs(
            records,
            args.output_directory,
            run_id=run_id,
        )
        _print_summary(summary)
        print(f"CSV: {csv_path}", flush=True)
        print(f"Summary: {summary_path}", flush=True)
    else:
        print("No group record was completed.", flush=True)

    if interrupted:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

