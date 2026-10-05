"""在同一代理查询预算下比较随机、贪心、束搜索、模拟退火与PPO。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.rl_environment import (  # noqa: E402
    ACTION_FEATURE_NAMES,
    STATE_FEATURE_NAMES,
    SurrogatePartitionEnvironment,
    SurrogateRewardConfig,
)
from src.partition_learning.rl_policy import PartitionActorCritic  # noqa: E402
from src.partition_learning.rl_search import (  # noqa: E402
    annealing_budget_search,
    beam_budget_search,
    greedy_budget_search,
    ppo_budget_search,
    random_budget_search,
)
from src.partition_learning.surrogate_bundle import PartitionSurrogateBundle  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """解析冻结代理、实例、搜索预算、信赖域、PPO检查点和输出目录。"""
    parser = argparse.ArgumentParser(description="公平比较代理辅助划分搜索方法")
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--instance-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-budget", type=int, default=200)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--seed", type=int, default=261004)
    parser.add_argument("--uncertainty-kappa", type=float, default=1.0)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--policy-hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--max-moved-fraction", type=float, default=0.08)
    parser.add_argument("--max-group-ratio", type=float, default=1.75)
    parser.add_argument("--complexity-growth-limit", type=float, default=1.15)
    parser.add_argument("--cost-limit", type=float, default=0.10)
    return parser.parse_args()


def _load_policy(
    path: Path | None,
    *,
    hidden_dim: int,
    device: str,
) -> PartitionActorCritic | None:
    """输入可选训练状态，输出评估模式PPO策略；未提供时返回None。"""
    if path is None:
        return None
    try:
        checkpoint = torch.load(path.resolve(), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path.resolve(), map_location=device)
    policy = PartitionActorCritic(
        len(STATE_FEATURE_NAMES), len(ACTION_FEATURE_NAMES), hidden_dim
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


def _validation_payload(
    reports: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """把各方法最佳划分去重转换成服务器真实复核候选文件。"""
    payload: dict[str, list[dict[str, Any]]] = {}
    for instance_id, methods in reports.items():
        rows = []
        seen = set()
        for method_name, result in methods.items():
            best = result["best"]
            key = json.dumps(best["partition"], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "name": f"fair_{method_name}",
                "partition": best["partition"],
                "surrogate_score": best["predicted_time"],
                "surrogate_score_improvement": result[
                    "predicted_time_saving_ratio"
                ],
                "surrogate_prediction": best["prediction"],
                "surrogate_uncertainty": best["uncertainty"],
                "subset_metadata": {
                    "method": method_name,
                    "query_budget": result["query_budget"],
                    "surrogate_query_count": result["surrogate_query_count"],
                    "cost_ucb_change": best["cost_ucb_change"],
                    "feasible": best["feasible"],
                    "trace": best["trace"],
                },
            })
        payload[instance_id] = rows
    return payload


def _run_timed(method) -> dict[str, Any]:
    """输入无参搜索函数，输出增加墙钟生成耗时的搜索结果。"""
    started = time.perf_counter()
    result = method()
    result["generation_seconds"] = time.perf_counter() - started
    return result


def _aggregate_methods(
    reports: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """输入逐实例方法结果，输出查询、耗时和代理可行改进的汇总指标。"""
    method_names = sorted({
        method for methods in reports.values() for method in methods
    })
    aggregate: dict[str, dict[str, float | int]] = {}
    for method in method_names:
        rows = [methods[method] for methods in reports.values() if method in methods]
        aggregate[method] = {
            "instance_count": len(rows),
            "mean_surrogate_query_count": float(np.mean([
                row["surrogate_query_count"] for row in rows
            ])),
            "mean_generation_seconds": float(np.mean([
                row["generation_seconds"] for row in rows
            ])),
            "predicted_feasible_improvement_fraction": float(np.mean([
                not row["returned_mst"] for row in rows
            ])),
            "mean_predicted_time_saving_ratio": float(np.mean([
                row["predicted_time_saving_ratio"] for row in rows
            ])),
            "returned_mst_fraction": float(np.mean([
                row["returned_mst"] for row in rows
            ])),
        }
    return aggregate


def main() -> int:
    """逐实例运行公平搜索，保存代理报告及可直接交给服务器的去重候选。"""
    arguments = parse_arguments()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(arguments.seed)
    policy = _load_policy(
        arguments.policy_checkpoint,
        hidden_dim=arguments.policy_hidden_dim,
        device=arguments.device,
    )
    reward_config = SurrogateRewardConfig(
        max_steps=arguments.max_steps,
        max_actions=arguments.max_actions,
        max_moved_fraction=arguments.max_moved_fraction,
        max_group_ratio=arguments.max_group_ratio,
        complexity_growth_limit=arguments.complexity_growth_limit,
        cost_limit=arguments.cost_limit,
    )
    instance_reports: dict[str, dict[str, Any]] = {}
    for instance_offset, instance_id in enumerate(arguments.instance_id):
        environment = SurrogatePartitionEnvironment(
            PartitionSurrogateBundle(
                arguments.bundle_manifest.resolve(),
                arguments.result_root.resolve(),
                instance_id,
            ),
            reward_config,
        )
        seed = arguments.seed + instance_offset * 100
        methods = {
            "random": _run_timed(lambda: random_budget_search(
                environment,
                np.random.default_rng(seed + 1),
                query_budget=arguments.query_budget,
                uncertainty_kappa=arguments.uncertainty_kappa,
            )),
            "greedy": _run_timed(lambda: greedy_budget_search(
                environment,
                query_budget=arguments.query_budget,
                uncertainty_kappa=arguments.uncertainty_kappa,
            )),
            "beam": _run_timed(lambda: beam_budget_search(
                environment,
                query_budget=arguments.query_budget,
                beam_width=arguments.beam_width,
                uncertainty_kappa=arguments.uncertainty_kappa,
            )),
            "simulated_annealing": _run_timed(lambda: annealing_budget_search(
                environment,
                np.random.default_rng(seed + 2),
                query_budget=arguments.query_budget,
                uncertainty_kappa=arguments.uncertainty_kappa,
            )),
        }
        if policy is not None:
            methods["ppo"] = _run_timed(lambda: ppo_budget_search(
                environment,
                policy,
                np.random.default_rng(seed + 3),
                query_budget=arguments.query_budget,
                uncertainty_kappa=arguments.uncertainty_kappa,
                device=arguments.device,
            ))
        instance_reports[instance_id] = methods
        print(
            f"[公平搜索] {instance_id} "
            + ", ".join(
                f"{name}:查询={result['surrogate_query_count']},"
                f"节省={result['predicted_time_saving_ratio']:.2%}"
                for name, result in methods.items()
            ),
            flush=True,
        )

    report = {
        "kind": "partition_fair_surrogate_search",
        "warning": "所有改进均为代理预测，论文结论必须使用导出的候选做真实Phase 2/3复核。",
        "configuration": {
            "bundle_manifest": str(arguments.bundle_manifest.resolve()),
            "result_root": str(arguments.result_root.resolve()),
            "instance_ids": list(arguments.instance_id),
            "query_budget": arguments.query_budget,
            "beam_width": arguments.beam_width,
            "uncertainty_kappa": arguments.uncertainty_kappa,
            "seed": arguments.seed,
            "policy_checkpoint": (
                str(arguments.policy_checkpoint.resolve())
                if arguments.policy_checkpoint is not None else None
            ),
            "reward_config": asdict(reward_config),
        },
        "instances": instance_reports,
        "aggregate": _aggregate_methods(instance_reports),
    }
    (output_dir / "fair_search_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "fair_search_candidates.json").write_text(
        json.dumps(
            _validation_payload(instance_reports), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(f"公平搜索报告：{output_dir / 'fair_search_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
