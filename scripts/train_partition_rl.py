"""在冻结三角色监督代理上训练受约束客户划分 PPO 策略。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from html import escape
import json
from pathlib import Path
import random
import sys
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
from src.partition_learning.rl_policy import (  # noqa: E402
    PPOConfig,
    PartitionActorCritic,
    finish_trajectory,
    ppo_update,
    save_rl_policy,
)
from src.partition_learning.rl_training import (  # noqa: E402
    greedy_rollout,
    policy_rollout,
    random_rollout,
    rollout_to_dict,
)
from src.partition_learning.surrogate_bundle import PartitionSurrogateBundle  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """解析代理模型、训练实例、奖励权重、PPO超参数和输出目录。"""
    parser = argparse.ArgumentParser(description="训练监督代理辅助客户划分 PPO")
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--instance-id",
        action="append",
        required=True,
        help="可重复指定；第一轮建议只使用一个已知实例。",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--random-evaluations", type=int, default=10)
    parser.add_argument(
        "--policy-evaluations",
        type=int,
        default=10,
        help="最终用随机策略采样的 PPO 轨迹数，与随机基线使用相同预算。",
    )
    parser.add_argument("--seed", type=int, default=261004)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--restart", action="store_true")

    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=24)
    parser.add_argument("--max-moved-fraction", type=float, default=0.20)
    parser.add_argument("--max-group-ratio", type=float, default=1.75)
    parser.add_argument("--complexity-growth-limit", type=float, default=1.15)
    parser.add_argument("--cost-limit", type=float, default=0.10)
    parser.add_argument("--cost-penalty", type=float, default=2.0)
    parser.add_argument("--uncertainty-weight", type=float, default=0.10)
    parser.add_argument("--p90-weight", type=float, default=0.05)
    parser.add_argument("--timeout-weight", type=float, default=0.25)
    parser.add_argument("--step-penalty", type=float, default=0.002)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--value-weight", type=float, default=0.50)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--update-epochs", type=int, default=4)
    return parser.parse_args()


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    """把逐回合训练摘要写成便于后续画图的 UTF-8 CSV。"""
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _save_training_state(
    path: Path,
    *,
    episode: int,
    policy: PartitionActorCritic,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, Any]],
    best_training_results: dict[str, dict[str, Any]],
    numpy_state: dict[str, Any],
) -> None:
    """保存断点恢复所需的策略、优化器、历史和随机数状态。"""
    torch.save(
        {
            "episode": episode,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_training_results": best_training_results,
            "torch_random_state": torch.get_rng_state(),
            "numpy_random_state": numpy_state,
        },
        path,
    )


def _reward_chart(history: list[dict[str, Any]]) -> str:
    """把逐回合奖励和十回合移动平均绘制成无需依赖的内嵌 SVG。"""
    if not history:
        return "<p>本次只执行筛查，没有训练回合。</p>"
    rewards = np.asarray(
        [float(row["total_reward"]) for row in history],
        dtype=np.float64,
    )
    window = min(10, len(rewards))
    moving = np.convolve(
        rewards,
        np.ones(window, dtype=np.float64) / window,
        mode="valid",
    )
    width, height, padding = 840.0, 280.0, 36.0
    lower = min(float(np.min(rewards)), float(np.min(moving)), 0.0)
    upper = max(float(np.max(rewards)), float(np.max(moving)), 0.0)
    span = max(upper - lower, 1e-6)

    def points(values: np.ndarray, offset: int = 0) -> str:
        """把数值序列缩放为 SVG 折线坐标。"""
        denominator = max(len(rewards) - 1, 1)
        return " ".join(
            f"{padding + (index + offset) / denominator * (width - 2 * padding):.1f},"
            f"{height - padding - (value - lower) / span * (height - 2 * padding):.1f}"
            for index, value in enumerate(values)
        )

    zero_y = height - padding - (0.0 - lower) / span * (height - 2 * padding)
    return f"""<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img"
aria-label="逐回合奖励与移动平均曲线">
<line x1="{padding}" y1="{zero_y:.1f}" x2="{width-padding}" y2="{zero_y:.1f}"
stroke="#9aa5b1" stroke-dasharray="5 5"/>
<polyline points="{points(rewards)}" fill="none" stroke="#a9c5e8" stroke-width="1.5"/>
<polyline points="{points(moving, window - 1)}" fill="none" stroke="#1464a5" stroke-width="3"/>
<text x="{padding}" y="18" fill="#52616b">浅蓝：逐回合　深蓝：{window} 回合移动平均</text>
<text x="{padding}" y="{height-8}" fill="#52616b">1</text>
<text x="{width-padding-20}" y="{height-8}" fill="#52616b">{len(rewards)}</text>
</svg>"""


def _report_html(
    report: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    """输入最终评价与训练历史，输出含训练曲线的独立本地 HTML。"""
    rows = []
    for instance_id, evaluation in report["instances"].items():
        random_metrics = evaluation["random"]
        for method in ("mst", "ppo_deterministic", "ppo_best", "greedy"):
            result = evaluation[method]
            prediction = result["final_prediction"]
            rows.append(
                "<tr>"
                f"<td>{escape(instance_id)}</td><td>{escape(method)}</td>"
                f"<td>{result['final_score']:.3f}</td>"
                f"<td>{result['score_improvement']:.3f}</td>"
                f"<td>{prediction['downstream_total_seconds']:.1f}</td>"
                f"<td>{prediction['final_cost']:.3f}</td>"
                f"<td>{prediction['downstream_time_p90']:.1f}</td>"
                f"<td>{prediction['right_censored_probability']:.1%}</td>"
                f"<td>{result['moved_customers']}</td>"
                "</tr>"
            )
        rows.append(
            "<tr>"
            f"<td>{escape(instance_id)}</td><td>random mean</td>"
            f"<td>{random_metrics['mean_final_score']:.3f}</td>"
            f"<td>{random_metrics['mean_score_improvement']:.3f}</td>"
            "<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>"
            "</tr>"
        )
    sampling_rows = []
    for instance_id, evaluation in report["instances"].items():
        for method, key in (("PPO采样", "ppo_sampled"), ("随机采样", "random")):
            metrics = evaluation[key]
            sampling_rows.append(
                "<tr>"
                f"<td>{escape(instance_id)}</td><td>{method}</td>"
                f"<td>{metrics['count']}</td>"
                f"<td>{metrics['mean_final_score']:.3f}</td>"
                f"<td>{metrics['best_final_score']:.3f}</td>"
                f"<td>{metrics['mean_score_improvement']:.3f}</td>"
                "</tr>"
            )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>代理辅助划分 PPO</title>
<style>body{{font-family:system-ui;margin:32px;background:#f7f8fa;color:#17202a}}
table{{border-collapse:collapse;background:white}}th,td{{padding:9px 11px;border:1px solid #d9dee7}}
th{{background:#eaf0f8}}.warning{{padding:12px;background:#fff3cd}}
.chart{{max-width:900px;background:white;border:1px solid #d9dee7;padding:12px;margin:16px 0}}
</style></head>
<body><h1>代理辅助客户划分 PPO</h1>
<p class="warning">所有指标均来自冻结监督代理；尚未经过真实 Phase 2/3 求解确认。</p>
<h2>训练奖励</h2><div class="chart">{_reward_chart(history)}</div>
<h2>最终候选</h2>
<table><thead><tr><th>实例</th><th>方法</th><th>风险分数</th><th>改善</th>
<th>预测时间</th><th>预测成本</th><th>P90</th><th>超时概率</th><th>移动客户</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>同预算采样比较</h2>
<table><thead><tr><th>实例</th><th>方法</th><th>轨迹数</th><th>平均风险分数</th>
<th>最佳风险分数</th><th>平均改善</th></tr></thead>
<tbody>{''.join(sampling_rows)}</tbody></table></body></html>"""


def _mst_result(environment: SurrogatePartitionEnvironment) -> dict[str, Any]:
    """把环境重置后的 MST 基线转换成与其他方法一致的结果结构。"""
    environment.reset()
    assert environment.current_prediction is not None
    prediction = environment.current_prediction["prediction"]
    uncertainty = environment.current_prediction["uncertainty"]["numeric"]
    return {
        "total_reward": 0.0,
        "initial_score": environment.current_score,
        "final_score": environment.current_score,
        "score_improvement": 0.0,
        "moved_customers": 0,
        "step_count": 0,
        "trace": [],
        "final_prediction": {
            name: float(value)
            for name, value in prediction.items()
        },
        "final_uncertainty": {
            name: float(value)
            for name, value in uncertainty.items()
        },
        "final_partition": {
            str(depot): list(customers)
            for depot, customers in environment.current.items()
        },
    }


def main() -> int:
    """训练可恢复 PPO，随后与 MST、随机和代理贪心基线进行同环境比较。"""
    arguments = parse_arguments()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    random.seed(arguments.seed)
    random_state = np.random.default_rng(arguments.seed)
    device = torch.device(arguments.device)

    reward_config = SurrogateRewardConfig(
        max_steps=arguments.max_steps,
        max_actions=arguments.max_actions,
        max_moved_fraction=arguments.max_moved_fraction,
        max_group_ratio=arguments.max_group_ratio,
        complexity_growth_limit=arguments.complexity_growth_limit,
        cost_limit=arguments.cost_limit,
        cost_penalty=arguments.cost_penalty,
        uncertainty_weight=arguments.uncertainty_weight,
        p90_weight=arguments.p90_weight,
        timeout_weight=arguments.timeout_weight,
        step_penalty=arguments.step_penalty,
    )
    ppo_config = PPOConfig(
        hidden_dim=arguments.hidden_dim,
        learning_rate=arguments.learning_rate,
        gamma=arguments.gamma,
        gae_lambda=arguments.gae_lambda,
        clip_ratio=arguments.clip_ratio,
        value_weight=arguments.value_weight,
        entropy_weight=arguments.entropy_weight,
        update_epochs=arguments.update_epochs,
    )
    configuration = {
        "bundle_manifest": str(arguments.bundle_manifest.resolve()),
        "result_root": str(arguments.result_root.resolve()),
        "instance_ids": list(arguments.instance_id),
        "episodes": arguments.episodes,
        "update_every": arguments.update_every,
        "random_evaluations": arguments.random_evaluations,
        "policy_evaluations": arguments.policy_evaluations,
        "seed": arguments.seed,
        "device": str(device),
        "reward_config": asdict(reward_config),
        "ppo_config": asdict(ppo_config),
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "action_feature_names": list(ACTION_FEATURE_NAMES),
    }
    (output_dir / "run_configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 每个实例只初始化一次三角色代理，跨回合保留划分预测缓存。
    environments = {
        instance_id: SurrogatePartitionEnvironment(
            PartitionSurrogateBundle(
                arguments.bundle_manifest.resolve(),
                arguments.result_root.resolve(),
                instance_id,
            ),
            reward_config,
        )
        for instance_id in arguments.instance_id
    }
    policy = PartitionActorCritic(
        len(STATE_FEATURE_NAMES),
        len(ACTION_FEATURE_NAMES),
        arguments.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=ppo_config.learning_rate,
    )
    state_path = output_dir / "training_state.pt"
    history: list[dict[str, Any]] = []
    best_training_results: dict[str, dict[str, Any]] = {}
    start_episode = 0
    if state_path.is_file() and not arguments.restart:
        checkpoint = torch.load(
            state_path,
            map_location=device,
            weights_only=False,
        )
        policy.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint["history"])
        best_training_results = dict(
            checkpoint.get("best_training_results", {})
        )
        start_episode = int(checkpoint["episode"])
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        random_state.bit_generator.state = checkpoint["numpy_random_state"]
        print(f"[resume] 从第 {start_episode + 1} 回合继续", flush=True)

    pending_experiences = []
    for episode_index in range(start_episode, arguments.episodes):
        instance_id = arguments.instance_id[
            episode_index % len(arguments.instance_id)
        ]
        environment = environments[instance_id]
        policy.train()
        rollout = policy_rollout(
            environment,
            policy,
            deterministic=False,
            device=str(device),
        )
        pending_experiences.extend(finish_trajectory(
            rollout.steps,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
        ))
        update_metrics = {
            "loss": float("nan"),
            "actor_loss": float("nan"),
            "value_loss": float("nan"),
            "entropy": float("nan"),
        }
        should_update = (
            (episode_index + 1) % arguments.update_every == 0
            or episode_index + 1 == arguments.episodes
        )
        if should_update:
            update_metrics = ppo_update(
                policy,
                optimizer,
                pending_experiences,
                ppo_config,
                device=device,
            )
            pending_experiences.clear()
        history.append({
            "episode": episode_index + 1,
            "instance_id": instance_id,
            "total_reward": rollout.total_reward,
            "initial_score": rollout.initial_score,
            "final_score": rollout.final_score,
            "score_improvement": rollout.initial_score - rollout.final_score,
            "moved_customers": rollout.moved_customers,
            "steps": len(rollout.trace),
            **update_metrics,
        })
        rollout_summary = rollout_to_dict(rollout)
        previous_best = best_training_results.get(instance_id)
        if (
            previous_best is None
            or rollout_summary["final_score"] < previous_best["final_score"]
        ):
            best_training_results[instance_id] = rollout_summary
        _write_history(output_dir / "training_history.csv", history)
        if should_update:
            _save_training_state(
                state_path,
                episode=episode_index + 1,
                policy=policy,
                optimizer=optimizer,
                history=history,
                best_training_results=best_training_results,
                numpy_state=random_state.bit_generator.state,
            )
        print(
            f"[PPO] episode={episode_index + 1}/{arguments.episodes} "
            f"reward={rollout.total_reward:.4f} "
            f"score={rollout.final_score:.4f} "
            f"steps={len(rollout.trace)} cache={len(environment.prediction_cache)}",
            flush=True,
        )

    policy.eval()
    evaluations: dict[str, Any] = {}
    for instance_id, environment in environments.items():
        ppo_deterministic = rollout_to_dict(policy_rollout(
            environment,
            policy,
            deterministic=True,
            device=str(device),
        ))
        ppo_sampled_results = [
            rollout_to_dict(policy_rollout(
                environment,
                policy,
                deterministic=False,
                device=str(device),
            ))
            for _ in range(max(arguments.policy_evaluations, 1))
        ]
        ppo_best = min(
            ppo_sampled_results,
            key=lambda result: result["final_score"],
        )
        greedy_result = rollout_to_dict(greedy_rollout(environment))
        random_results = [
            rollout_to_dict(random_rollout(environment, random_state))
            for _ in range(max(arguments.random_evaluations, 1))
        ]
        evaluations[instance_id] = {
            "mst": _mst_result(environment),
            "ppo_deterministic": ppo_deterministic,
            "ppo_best": ppo_best,
            "ppo_sampled": {
                "count": len(ppo_sampled_results),
                "mean_final_score": float(np.mean([
                    result["final_score"] for result in ppo_sampled_results
                ])),
                "best_final_score": float(ppo_best["final_score"]),
                "mean_score_improvement": float(np.mean([
                    result["score_improvement"]
                    for result in ppo_sampled_results
                ])),
                "runs": ppo_sampled_results,
            },
            "best_training": best_training_results.get(
                instance_id,
                ppo_deterministic,
            ),
            "greedy": greedy_result,
            "random": {
                "count": len(random_results),
                "mean_final_score": float(np.mean([
                    result["final_score"] for result in random_results
                ])),
                "best_final_score": float(np.min([
                    result["final_score"] for result in random_results
                ])),
                "mean_score_improvement": float(np.mean([
                    result["score_improvement"] for result in random_results
                ])),
                "runs": random_results,
            },
            "prediction_cache_size": len(environment.prediction_cache),
        }
    report = {
        "kind": "partition_surrogate_ppo_pilot",
        "warning": (
            "所有性能均由冻结监督代理给出，必须用真实 Phase 2/3 "
            "求解抽查后才能作为算法改进结论。"
        ),
        "configuration": configuration,
        "training_episode_count": len(history),
        "instances": evaluations,
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evaluation_report.html").write_text(
        _report_html(report, history),
        encoding="utf-8",
    )
    # 导出少量不重复划分，供服务器用真实 Phase 2/3 做最终抽查。
    validation_candidates: dict[str, list[dict[str, Any]]] = {}
    for instance_id, evaluation in evaluations.items():
        candidates = []
        seen_partitions = set()
        random_best = min(
            evaluation["random"]["runs"],
            key=lambda result: result["final_score"],
        )
        for name, result in (
            ("mst", evaluation["mst"]),
            ("ppo_deterministic", evaluation["ppo_deterministic"]),
            ("ppo_best", evaluation["ppo_best"]),
            ("best_training", evaluation["best_training"]),
            ("greedy", evaluation["greedy"]),
            ("random_best", random_best),
        ):
            key = json.dumps(result["final_partition"], sort_keys=True)
            if key in seen_partitions:
                continue
            seen_partitions.add(key)
            candidates.append({
                "name": name,
                "surrogate_score": result["final_score"],
                "surrogate_score_improvement": result["score_improvement"],
                "surrogate_prediction": result["final_prediction"],
                "surrogate_uncertainty": result["final_uncertainty"],
                "partition": result["final_partition"],
            })
        validation_candidates[instance_id] = candidates
    (output_dir / "validation_candidates.json").write_text(
        json.dumps(validation_candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_rl_policy(
        output_dir / "partition_rl_policy.pt",
        policy,
        ppo_config=ppo_config,
        metadata={
            "instance_ids": list(arguments.instance_id),
            "reward_config": asdict(reward_config),
            "bundle_manifest": str(arguments.bundle_manifest.resolve()),
            "training_episode_count": len(history),
        },
    )
    print(f"强化学习训练完成：{output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
