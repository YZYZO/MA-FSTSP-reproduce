"""训练里程碑 C 的求解器感知分层客户分区策略。"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

# 允许从任意工作目录直接运行脚本，同时保持项目内部导入方式不变。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import MANHATTAN_1K_EXPERIMENT, RESULTS_DIR
from problem import prepare_manhattan_road_network, sample_multiagent_instances
from src.fstsp import MultiAgentFlyingSidekickTSP
from src.learning.cache import EvaluationCache
from src.learning.evaluator import GroupEvaluator
from src.learning.hierarchical_policy import (
    ActorCriticTrainingConfig,
    HierarchicalPartitionPolicy,
    HierarchicalPolicyConfig,
    evaluate_hierarchical_policy,
    policy_checkpoint_payload,
    train_hierarchical_policy,
)
from src.learning.partition_env import (
    PartitionEnvironmentConfig,
    PartitionRepairEnvironment,
    SurrogateGroupScorer,
)
from src.learning.settings import SetTSPSolverSettings
from src.partitioning import partition_customers


DEFAULT_SURROGATE_DIRECTORY = RESULTS_DIR / "learning" / "surrogate"
DEFAULT_OUTPUT_DIRECTORY = RESULTS_DIR / "learning" / "hrl"
DEFAULT_CACHE_PATH = RESULTS_DIR / "learning" / "probe" / "manhattan_1k_group_cache.sqlite3"


def _latest_surrogate_checkpoint():
    """
    查找最近生成的里程碑 B 代理检查点。

    输入：无。
    输出：最近修改的 `.pt` 路径；目录为空时返回 `None`。
    逻辑：优先使用文件修改时间，兼容带 `-final` 后缀的人工确认版本。
    """
    candidates = list(DEFAULT_SURROGATE_DIRECTORY.glob("surrogate_*.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def build_argument_parser():
    """
    构造分层策略训练脚本参数。

    输入：无。
    输出：带里程碑 C 推荐默认值的 `ArgumentParser`。
    逻辑：默认使用 50/100 客户小规模训练，真实求解复核需显式给出正预算。
    """
    parser = argparse.ArgumentParser(
        description="在 Set-MST 初始划分上训练 relocate/swap/stop 分层策略。"
    )
    parser.add_argument(
        "--surrogate-checkpoint",
        type=Path,
        default=None,
        help="里程碑 B 的代理模型检查点；默认自动选择最新文件。",
    )
    parser.add_argument(
        "--customer-sizes",
        nargs="+",
        type=int,
        default=[50, 100],
        help="策略训练使用的客户规模。",
    )
    parser.add_argument(
        "--train-instances-per-size",
        type=int,
        default=2,
        help="每个规模的训练实例数。",
    )
    parser.add_argument(
        "--validation-instances-per-size",
        type=int,
        default=1,
        help="每个规模的验证实例数。",
    )
    parser.add_argument("--imitation-epochs", type=int, default=6)
    parser.add_argument("--reinforcement-episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--relocate-candidates", type=int, default=4)
    parser.add_argument("--swap-candidates", type=int, default=4)
    parser.add_argument(
        "--real-verification-budget",
        type=int,
        default=0,
        help="每个实例在整个训练期间最多调用真实单组求解器的次数；默认不新增昂贵标签。",
    )
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--phase2-budget", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--run-id", default=None)
    return parser


def _map_identifier(graph_path):
    """
    生成与里程碑 A/B 一致的地图缓存标识。

    输入：GraphML 文件路径。
    输出：文件名、大小和修改时间组成的字符串。
    逻辑：真实复核开启后可以继续复用已有 SQLite 单组结果。
    """
    file_stats = graph_path.stat()
    return f"{graph_path.name}:{file_stats.st_size}:{file_stats.st_mtime_ns}"


def _build_environment(
    graph,
    distance,
    depots,
    cities,
    surrogate_checkpoint,
    environment_config,
    map_id,
    cache,
    seed,
    time_limit_seconds,
):
    """
    从一个采样实例建立 Set-MST 初始分区及修复环境。

    输入：实例、代理路径、环境配置、地图标识、缓存和随机种子。
    输出：可训练的 `PartitionRepairEnvironment`。
    逻辑：候选集合和 MST 仅计算一次；真实评估器只在复核预算为正时挂载。
    """
    model = MultiAgentFlyingSidekickTSP(
        graph,
        depots,
        cities,
        distance,
        MANHATTAN_1K_EXPERIMENT.drone_count,
        theta=(0.5, 0.5),
    )
    convex_sets = model.get_boundary_convex_sets(model.theta[0])
    initial_groups = partition_customers(
        model.depots,
        model.cities,
        convex_sets,
        model.distance["truck"],
        model.distance["drone"],
        model.speed,
        edge_mode="mean",
        coefficient=model.const,
    )
    scorer = SurrogateGroupScorer.from_checkpoint(
        graph,
        distance,
        convex_sets,
        surrogate_checkpoint,
        mc_samples=environment_config.mc_samples,
    )
    evaluator = None
    if environment_config.real_verification_budget > 0:
        evaluator = GroupEvaluator(
            model,
            convex_sets,
            map_id=map_id,
            cache=cache,
            solver_settings=SetTSPSolverSettings(
                time_limit_seconds=time_limit_seconds,
                threads=1,
                seed=seed,
            ),
        )
    return PartitionRepairEnvironment(
        graph,
        distance,
        model.depots,
        convex_sets,
        initial_groups,
        scorer,
        config=environment_config,
        group_evaluator=evaluator,
    )


def _print_progress(row):
    """
    输出一轮模仿或强化学习训练进度。

    输入：训练历史行。
    输出：无。
    逻辑：只显示阶段、轮次、损失及可用的回报/验证改善，避免打印每个动作造成噪声。
    """
    if row["stage"] == "imitation":
        print(
            "  imitation epoch={:02d} loss={:.4f} teacher_reward={:.4f}".format(
                row["iteration"],
                row["loss"],
                row["mean_teacher_reward"],
            ),
            flush=True,
        )
        return
    validation = row.get("validation_score_improvement")
    suffix = f" validation_improvement={validation:.4f}" if validation is not None else ""
    if row["iteration"] == 1 or validation is not None:
        print(
            "  actor-critic episode={:03d} reward={:.4f} loss={:.4f}{}".format(
                row["iteration"],
                row["episode_reward"],
                row["loss"],
                suffix,
            ),
            flush=True,
        )


def _write_history(path, history):
    """
    将异构阶段训练历史写成统一 CSV。

    输入：输出路径和历史行列表。
    输出：无；创建父目录及 CSV 文件。
    逻辑：字段取全部行键的并集，便于直接用 pandas 绘制两阶段曲线。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main(argv=None):
    """
    执行里程碑 C 小规模训练并保存策略、历史和验证指标。

    输入：可选命令行参数列表。
    输出：进程状态码，同时生成 PyTorch 检查点、JSON 指标和 CSV 历史。
    逻辑：按原始实例分开训练/验证；验证集仅用于选择最佳 Actor-Critic 权重。
    """
    args = build_argument_parser().parse_args(argv)
    surrogate_checkpoint = args.surrogate_checkpoint or _latest_surrogate_checkpoint()
    if surrogate_checkpoint is None:
        raise FileNotFoundError("未找到里程碑 B 的代理模型检查点。")
    surrogate_checkpoint = Path(surrogate_checkpoint)
    graph_path = Path(MANHATTAN_1K_EXPERIMENT.graph_path)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")

    environment_config = PartitionEnvironmentConfig(
        max_steps=args.max_steps,
        relocate_candidates_per_direction=args.relocate_candidates,
        swap_candidates_per_pair=args.swap_candidates,
        phase2_budget_seconds=args.phase2_budget,
        real_verification_budget=args.real_verification_budget,
        mc_samples=1,
    )
    training_config = ActorCriticTrainingConfig(
        imitation_epochs=args.imitation_epochs,
        reinforcement_episodes=args.reinforcement_episodes,
        evaluation_interval=max(1, min(10, args.reinforcement_episodes)),
        seed=args.seed,
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

    training_environments = []
    validation_environments = []
    map_id = _map_identifier(graph_path)
    with EvaluationCache(args.cache_path) as cache:
        for customer_size in args.customer_sizes:
            instance_count = (
                args.train_instances_per_size
                + args.validation_instances_per_size
            )
            depot_instances, city_instances = sample_multiagent_instances(
                graph,
                instance_count,
                MANHATTAN_1K_EXPERIMENT.depot_count,
                customer_size,
                seed=args.seed + customer_size * 1000,
            )
            for instance_index in range(instance_count):
                environment = _build_environment(
                    graph,
                    distance,
                    depot_instances[instance_index],
                    city_instances[instance_index],
                    surrogate_checkpoint,
                    environment_config,
                    map_id,
                    cache,
                    seed=args.seed,
                    time_limit_seconds=args.time_limit,
                )
                if instance_index < args.train_instances_per_size:
                    training_environments.append(environment)
                    split = "train"
                else:
                    validation_environments.append(environment)
                    split = "validation"
                print(
                    f"  environment split={split} size={customer_size} "
                    f"groups={[len(environment.groups[d]) for d in environment.depots]}",
                    flush=True,
                )

        first_environment = training_environments[0]
        policy_config = HierarchicalPolicyConfig(
            group_observation_dim=first_environment.group_observation_dim,
            action_feature_dim=first_environment.lower_action_feature_dim,
            seed=args.seed,
        )
        policy = HierarchicalPartitionPolicy(policy_config)
        print(
            "Training hierarchical policy: train_instances={}, validation_instances={}, "
            "imitation_epochs={}, actor_critic_episodes={} ...".format(
                len(training_environments),
                len(validation_environments),
                training_config.imitation_epochs,
                training_config.reinforcement_episodes,
            ),
            flush=True,
        )
        policy, training_artifacts = train_hierarchical_policy(
            policy,
            training_environments,
            validation_environments,
            training_config=training_config,
            progress_callback=_print_progress,
        )
        training_metrics = evaluate_hierarchical_policy(policy, training_environments)
        validation_metrics = evaluate_hierarchical_policy(policy, validation_environments)

    quality_gate = {
        "validation_score_improved": (
            validation_metrics["mean_score_improvement"] > 0.0
        ),
        "validation_predicted_cost_within_2_percent": (
            validation_metrics["mean_final_cost_sum"]
            <= validation_metrics["mean_initial_cost_sum"] * 1.02
        ),
    }
    quality_gate["eligible_for_milestone_d_evaluation"] = all(quality_gate.values())
    metrics = {
        "run_id": run_id,
        "surrogate_checkpoint": str(surrogate_checkpoint),
        "graph_path": str(graph_path),
        "customer_sizes": args.customer_sizes,
        "train_instances_per_size": args.train_instances_per_size,
        "validation_instances_per_size": args.validation_instances_per_size,
        "training": training_metrics,
        "validation": validation_metrics,
        "quality_gate": quality_gate,
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_directory / f"hrl_policy_{run_id}.pt"
    metrics_path = args.output_directory / f"hrl_policy_{run_id}_metrics.json"
    history_path = args.output_directory / f"hrl_policy_{run_id}_history.csv"
    torch.save(
        policy_checkpoint_payload(
            policy,
            training_config,
            environment_config,
            surrogate_checkpoint,
            training_artifacts,
            metrics,
        ),
        checkpoint_path,
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_history(history_path, training_artifacts["history"])

    print(
        "Training finished: validation score improvement={:.4f}, "
        "predicted cost {:.4f} -> {:.4f}, eligible_for_D={}".format(
            validation_metrics["mean_score_improvement"],
            validation_metrics["mean_initial_cost_sum"],
            validation_metrics["mean_final_cost_sum"],
            quality_gate["eligible_for_milestone_d_evaluation"],
        ),
        flush=True,
    )
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    print(f"History: {history_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
