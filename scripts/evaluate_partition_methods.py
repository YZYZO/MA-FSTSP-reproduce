"""独立运行里程碑 D 六种客户划分方法的真实下游对比实验。"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MANHATTAN_1K_EXPERIMENT,
    MANHATTAN_11K_EXPERIMENT,
    MANHATTAN_55K_EXPERIMENT,
    RESULTS_DIR,
)
from problem import manhattan, prepare_manhattan_road_network, sample_multiagent_instances
from src.learning.cache import EvaluationCache
from src.learning.evaluation import (
    evaluate_partition_result,
    write_partition_evaluation_outputs,
)
from src.learning.partition_methods import (
    EVALUATION_PARTITION_METHODS,
    build_partition_method_context,
    run_partition_method,
)
from src.learning.lazy_distance import build_lazy_distance


MAP_SPECS = {
    "manhattan1k": MANHATTAN_1K_EXPERIMENT,
    "manhattan11k": MANHATTAN_11K_EXPERIMENT,
    "manhattan55k": MANHATTAN_55K_EXPERIMENT,
}
DEFAULT_OUTPUT_DIRECTORY = RESULTS_DIR / "learning" / "evaluation"
DEFAULT_CACHE_PATH = DEFAULT_OUTPUT_DIRECTORY / "group_cache.sqlite3"


def _latest_checkpoint(directory, pattern):
    """
    在指定目录查找最近修改的模型检查点。

    输入：目录和 glob 模式。
    输出：最近文件或 `None`。
    逻辑：允许训练脚本使用时间戳或带语义的运行标识。
    """
    candidates = list(Path(directory).glob(pattern))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def build_argument_parser():
    """
    构造独立划分评估脚本的命令行参数。

    输入：无。
    输出：默认先运行 Manhattan 1K 单实例 50 客户的参数解析器。
    逻辑：11K/55K 必须由 `--maps` 显式加入，防止误启动大型全点对距离计算。
    """
    parser = argparse.ArgumentParser(
        description="真实评估 MST、均衡、贪心、单层 RL 和层次化 RL 客户划分。"
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        choices=tuple(MAP_SPECS),
        default=["manhattan1k"],
    )
    parser.add_argument("--customer-sizes", nargs="+", type=int, default=[50])
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=EVALUATION_PARTITION_METHODS,
        default=list(EVALUATION_PARTITION_METHODS),
    )
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument(
        "--distance-mode",
        choices=("auto", "eager", "lazy"),
        default="auto",
        help="auto 在 1K 使用全矩阵，在 11K/55K 使用按需距离。",
    )
    parser.add_argument(
        "--lazy-truck-cache-rows",
        type=int,
        default=64,
        help="大地图按需卡车距离最多缓存的单源 Dijkstra 行数。",
    )
    parser.add_argument("--cost-tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--surrogate-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--hrl-checkpoint", type=Path, default=None)
    parser.add_argument("--single-level-checkpoint", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--run-id", default=None)
    return parser


def _map_identifier(graph_path):
    """
    为单组真实标签缓存生成地图标识。

    输入：地图文件路径。
    输出：文件名、大小和修改时间组成的字符串。
    逻辑：不同地图或文件版本不会错误共享 Set-TSP 标签。
    """
    stats = graph_path.stat()
    return f"{graph_path.name}:{stats.st_size}:{stats.st_mtime_ns}"


def _resolve_checkpoints(args):
    """
    自动补全评估所需的代理、单层和分层检查点。

    输入：命令行参数。
    输出：三个可能为空的路径。
    逻辑：仅当所选方法需要对应模型时才要求文件存在。
    """
    surrogate = args.surrogate_checkpoint or _latest_checkpoint(
        RESULTS_DIR / "learning" / "surrogate",
        "surrogate_*.pt",
    )
    hrl = args.hrl_checkpoint or _latest_checkpoint(
        RESULTS_DIR / "learning" / "hrl",
        "hrl_policy_*.pt",
    )
    single = args.single_level_checkpoint or _latest_checkpoint(
        RESULTS_DIR / "learning" / "single_level",
        "single_level_policy_*.pt",
    )
    learning_methods = {
        "solver_aware_greedy",
        "single_level_rl",
        "solver_aware_hrl",
    }
    if learning_methods.intersection(args.methods) and surrogate is None:
        raise FileNotFoundError("学习型方法需要 surrogate checkpoint。")
    if "solver_aware_hrl" in args.methods and hrl is None:
        raise FileNotFoundError("未找到 HRL checkpoint。")
    if "single_level_rl" in args.methods and single is None:
        raise FileNotFoundError("未找到 single-level checkpoint。")
    return surrogate, hrl, single


def _print_record(record):
    """
    输出一个方法—实例真实评估完成后的关键指标。

    输入：详细评估记录。
    输出：无。
    逻辑：立即显示成本、Set-TSP 总时间、超时、策略耗时与缓存调用。
    """
    print(
        "  method={method:<23} groups={group_sizes} cost={final_cost:.4f} "
        "set_tsp={set_tsp_total_seconds:.3f}s timeouts={timeout_count} "
        "strategy={partition_strategy_seconds:.3f}s real_calls={total_real_solver_calls}".format(
            **record
        ),
        flush=True,
    )


def _print_group_progress(group_record):
    """
    输出当前方法中一个仓库客户组的真实求解进度。

    输入：刚完成的客户组记录。
    输出：无。
    逻辑：大地图上用于区分 MST 构图、Set-TSP 和第三阶段耗时。
    """
    print(
        "    depot={depot} group={customer_count} status={set_tsp_status} "
        "set_tsp={set_tsp_wall_seconds:.3f}s phase3={phase3_seconds:.3f}s "
        "cache={cache_hit}".format(**group_record),
        flush=True,
    )


def main(argv=None):
    """
    按地图、规模、实例和方法运行真实下游对比并保存统一报告。

    输入：可选命令行参数列表。
    输出：进程状态码以及详细 CSV、摘要 JSON、Markdown 报告。
    逻辑：每张地图只初始化一次距离；中断时仍保存已完成记录。
    """
    args = build_argument_parser().parse_args(argv)
    surrogate_checkpoint, hrl_checkpoint, single_checkpoint = _resolve_checkpoints(args)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    records = []
    interrupted = False
    with EvaluationCache(args.cache_path) as cache:
        try:
            for map_name in args.maps:
                spec = MAP_SPECS[map_name]
                graph_path = Path(spec.graph_path)
                print(f"Preparing {spec.dataset_label} from {graph_path} ...", flush=True)
                use_lazy_distance = (
                    args.distance_mode == "lazy"
                    or (args.distance_mode == "auto" and map_name != "manhattan1k")
                )
                if use_lazy_distance:
                    graph = manhattan(graph_path)
                    distance = build_lazy_distance(
                        graph,
                        max_cached_truck_rows=args.lazy_truck_cache_rows,
                    )
                    print(
                        "Distance preparation finished: lazy matrix, "
                        f"truck_cache_rows={args.lazy_truck_cache_rows}",
                        flush=True,
                    )
                else:
                    graph, distance, stats = prepare_manhattan_road_network(
                        graph_path,
                        show_distance_progress=False,
                    )
                    print(
                        "Distance preparation finished: total={:.3f}s".format(
                            stats["distance_initialization_seconds"]
                        ),
                        flush=True,
                    )
                map_id = _map_identifier(graph_path)
                for customer_size in args.customer_sizes:
                    depot_instances, city_instances = sample_multiagent_instances(
                        graph,
                        args.instances,
                        spec.depot_count,
                        customer_size,
                        seed=args.seed + customer_size * 1000,
                    )
                    for instance_index in range(args.instances):
                        context = build_partition_method_context(
                            graph,
                            depot_instances[instance_index],
                            city_instances[instance_index],
                            distance,
                            spec.drone_count,
                        )
                        print(
                            f"Evaluating map={spec.dataset_label} size={customer_size} "
                            f"instance={instance_index} ...",
                            flush=True,
                        )
                        for method in args.methods:
                            truck_distance = distance["truck"]
                            if hasattr(truck_distance, "clear_cache"):
                                truck_distance.clear_cache()
                            print(f"  partitioning method={method} ...", flush=True)
                            partition_result = run_partition_method(
                                context,
                                method,
                                surrogate_checkpoint=surrogate_checkpoint,
                                hrl_checkpoint=hrl_checkpoint,
                                single_level_checkpoint=single_checkpoint,
                            )
                            print(
                                "    partition finished: groups={} strategy={:.3f}s loading={:.3f}s".format(
                                    [
                                        len(partition_result.groups[depot])
                                        for depot in context.model.depots
                                    ],
                                    partition_result.partition_strategy_seconds,
                                    partition_result.model_loading_seconds,
                                ),
                                flush=True,
                            )
                            record = evaluate_partition_result(
                                context,
                                partition_result,
                                map_id=map_id,
                                map_label=spec.dataset_label,
                                customer_size=customer_size,
                                instance_index=instance_index,
                                cache=cache,
                                time_limit_seconds=args.time_limit,
                                solver_seed=args.seed,
                                group_progress_callback=_print_group_progress,
                            )
                            records.append(record)
                            _print_record(record)
        except KeyboardInterrupt:
            interrupted = True
            print("Evaluation interrupted; saving completed records.", flush=True)

    if not records:
        print("No evaluation record was completed.", flush=True)
        return 130 if interrupted else 1
    detail_path, summary_path, report_path, summary = write_partition_evaluation_outputs(
        records,
        args.output_directory,
        cost_tolerance=args.cost_tolerance,
        run_id=run_id,
    )
    print(
        f"Evaluation finished: records={summary['record_count']}, "
        f"summary_rows={len(summary['summary_rows'])}",
        flush=True,
    )
    print(f"Details: {detail_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Report: {report_path}", flush=True)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
