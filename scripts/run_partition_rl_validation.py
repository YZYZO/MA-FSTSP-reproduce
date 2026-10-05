"""在服务器真实求解RL导出的少量客户划分候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.evaluator import EVALUATOR_VERSION  # noqa: E402
from src.partition_learning.pipeline import ThreeRoundExperiment  # noqa: E402
from src.partition_learning.reporting import write_json  # noqa: E402
from src.partition_learning.rl_validation import (  # noqa: E402
    evaluate_validation_candidates,
    save_validation_reports,
)


DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT / "results" / "服务器运行结果下载" / "260826_55k_master_results"
)
DEFAULT_CANDIDATES = (
    PROJECT_ROOT
    / "results"
    / "服务器运行结果下载"
    / "261004partition_rl_stage1_local"
    / "boston_11k_100_016"
    / "validation_candidates.json"
)


def parse_arguments() -> argparse.Namespace:
    """解析候选文件、求解预算、路网过滤和输出目录。"""
    parser = argparse.ArgumentParser(description="真实复核强化学习客户划分")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--group-cache",
        type=Path,
        help="可选的既有group_evaluations.sqlite3；用于跨候选批次复用仓库组真值。",
    )
    parser.add_argument("--solver-time-limit", type=float, default=600.0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--solver-mip-gap", type=float, default=1e-4)
    parser.add_argument("--max-binary-variables", type=int, default=0)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--cost-limit", type=float, default=0.10)
    parser.add_argument(
        "--only-graph",
        choices=("manhattan_1k", "boston_11k", "manhattan_55k"),
    )
    parser.add_argument(
        "--customer-counts",
        type=int,
        nargs="+",
        choices=(50, 100, 150),
    )
    parser.add_argument("--include-55k", action="store_true")
    return parser.parse_args()


def _configuration(arguments: argparse.Namespace, candidate_bytes: bytes) -> dict:
    """生成决定真实标签的稳定配置，阻止不同预算误用同一续跑目录。"""
    return {
        "kind": "rl_partition_true_validation",
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_path": str(arguments.candidates.resolve()),
        "result_root": str(arguments.result_root.resolve()),
        "evaluator_version": EVALUATOR_VERSION,
        "solver_time_limit": float(arguments.solver_time_limit),
        "solver_threads": int(arguments.solver_threads),
        "solver_seed": int(arguments.solver_seed),
        "solver_mip_gap": float(arguments.solver_mip_gap),
        "max_binary_variables": int(arguments.max_binary_variables),
        "distance_batch_size": int(arguments.distance_batch_size),
        "cost_limit": float(arguments.cost_limit),
        "evaluation_workers": 1,
        "group_cache": (
            str(arguments.group_cache.resolve())
            if arguments.group_cache is not None
            else None
        ),
        "only_graph": arguments.only_graph,
        "customer_counts": arguments.customer_counts,
        "include_55k": bool(arguments.include_55k),
    }


def _write_or_check_configuration(path: Path, configuration: dict) -> None:
    """首次写入配置；续跑时要求求解预算和候选摘要完全一致。"""
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != configuration:
            raise RuntimeError("输出目录已有不同配置，请为新预算使用新的输出目录。")
        return
    write_json(path, configuration)


def _apply_group_cache(experiment, group_cache: Path | None) -> None:
    """验证外部SQLite缓存存在，并让真实复核跨候选批次复用该缓存。"""
    if group_cache is None:
        return
    resolved = group_cache.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"指定的仓库组缓存不存在：{resolved}")
    experiment.cache_path = resolved


def main() -> int:
    """加载候选，串行真实求解，写出兼容记录与确认报告。"""
    arguments = parse_arguments()
    candidate_bytes = arguments.candidates.read_bytes()
    payload = json.loads(candidate_bytes.decode("utf-8"))
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = _configuration(arguments, candidate_bytes)
    _write_or_check_configuration(
        output_dir / "run_configuration.json",
        configuration,
    )
    experiment = ThreeRoundExperiment(
        arguments.result_root,
        output_dir,
        solver_time_limit=arguments.solver_time_limit,
        solver_threads=arguments.solver_threads,
        solver_seed=arguments.solver_seed,
        solver_mip_gap=arguments.solver_mip_gap,
        max_binary_variables=arguments.max_binary_variables,
        distance_batch_size=arguments.distance_batch_size,
        include_55k=arguments.include_55k,
        only_graph=arguments.only_graph,
        customer_counts=(
            tuple(arguments.customer_counts)
            if arguments.customer_counts
            else None
        ),
        evaluation_workers=1,
    )
    _apply_group_cache(experiment, arguments.group_cache)
    records = evaluate_validation_candidates(experiment, payload, output_dir)
    report = save_validation_reports(
        records,
        output_dir,
        cost_limit=arguments.cost_limit,
    )
    for instance_id, summary in report["instances"].items():
        print(
            f"[RL validation] 完成 {instance_id}: "
            f"exact={summary['exact_candidate_count']}/{summary['candidate_count']} "
            f"ppo_best={summary['ppo_best_verdict']} "
            f"hard_constraint_success={summary['hard_constraint_success']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
