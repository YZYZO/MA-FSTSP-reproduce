"""训练里程碑 D 消融实验使用的单层 Actor-Critic 基线。"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import MANHATTAN_1K_EXPERIMENT, RESULTS_DIR
from problem import prepare_manhattan_road_network, sample_multiagent_instances
from src.fstsp import MultiAgentFlyingSidekickTSP
from src.learning.partition_env import (
    PartitionEnvironmentConfig,
    PartitionRepairEnvironment,
    SurrogateGroupScorer,
)
from src.learning.single_level_policy import (
    SingleLevelActorCritic,
    SingleLevelPolicyConfig,
    SingleLevelTrainingConfig,
    evaluate_single_level_policy,
    single_level_checkpoint_payload,
    train_single_level_policy,
)
from src.partitioning import partition_customers


DEFAULT_SURROGATE_DIRECTORY = RESULTS_DIR / "learning" / "surrogate"
DEFAULT_OUTPUT_DIRECTORY = RESULTS_DIR / "learning" / "single_level"
DEFAULT_CACHE_PATH = RESULTS_DIR / "learning" / "evaluation" / "group_cache.sqlite3"


def _latest_surrogate_checkpoint():
    """
    查找最近生成的里程碑 B 代理模型。

    输入：无。
    输出：最近修改的代理 `.pt` 文件或 `None`。
    逻辑：单层与分层策略必须使用同一个代理标签口径。
    """
    candidates = list(DEFAULT_SURROGATE_DIRECTORY.glob("surrogate_*.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def build_argument_parser():
    """
    构造单层策略训练参数。

    输入：无。
    输出：带公平消融默认设置的参数解析器。
    逻辑：客户规模、实例数、动作规模和轮数默认与已验收 HRL 相同。
    """
    parser = argparse.ArgumentParser(
        description="训练扁平化仓库对与客户操作的单层 Actor-Critic 基线。"
    )
    parser.add_argument("--surrogate-checkpoint", type=Path, default=None)
    parser.add_argument("--customer-sizes", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--train-instances-per-size", type=int, default=2)
    parser.add_argument("--validation-instances-per-size", type=int, default=1)
    parser.add_argument("--imitation-epochs", type=int, default=6)
    parser.add_argument("--reinforcement-episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--relocate-candidates", type=int, default=4)
    parser.add_argument("--swap-candidates", type=int, default=4)
    parser.add_argument("--phase2-budget", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--run-id", default=None)
    return parser


def _build_environment(
    graph,
    distance,
    depots,
    cities,
    surrogate_checkpoint,
    environment_config,
):
    """
    建立一个与 HRL 完全同口径的单层训练环境。

    输入：路网实例、代理模型和环境配置。
    输出：从 symmetric MST 开始的 `PartitionRepairEnvironment`。
    逻辑：不挂载真实求解器，确保两种 RL 训练都只消费相同代理预测。
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
    return PartitionRepairEnvironment(
        graph,
        distance,
        model.depots,
        convex_sets,
        initial_groups,
        scorer,
        config=environment_config,
    )


def _print_progress(row):
    """
    输出单层策略训练的低频进度。

    输入：一行训练历史。
    输出：无。
    逻辑：模仿阶段逐轮显示，强化阶段只显示首轮和验证轮。
    """
    if row["stage"] == "imitation":
        print(
            f"  imitation epoch={row['iteration']:02d} loss={row['loss']:.4f}",
            flush=True,
        )
    elif row["iteration"] == 1 or "validation_score_improvement" in row:
        validation = row.get("validation_score_improvement")
        suffix = f" validation_improvement={validation:.4f}" if validation is not None else ""
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
    将两阶段训练历史保存为 CSV。

    输入：输出路径和历史记录。
    输出：无。
    逻辑：按所有行字段的并集写表，保留验证轮的额外指标。
    """
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
    训练、验证并保存正式单层 RL 消融基线。

    输入：可选命令行参数列表。
    输出：进程状态码和检查点、指标、历史三个文件。
    逻辑：采样种子与 HRL 脚本一致，保证两种结构使用相同原始实例。
    """
    args = build_argument_parser().parse_args(argv)
    surrogate_checkpoint = args.surrogate_checkpoint or _latest_surrogate_checkpoint()
    if surrogate_checkpoint is None:
        raise FileNotFoundError("未找到里程碑 B 代理模型。")
    surrogate_checkpoint = Path(surrogate_checkpoint)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    environment_config = PartitionEnvironmentConfig(
        max_steps=args.max_steps,
        relocate_candidates_per_direction=args.relocate_candidates,
        swap_candidates_per_pair=args.swap_candidates,
        phase2_budget_seconds=args.phase2_budget,
        real_verification_budget=0,
        mc_samples=1,
    )
    training_config = SingleLevelTrainingConfig(
        imitation_epochs=args.imitation_epochs,
        reinforcement_episodes=args.reinforcement_episodes,
        evaluation_interval=max(1, min(10, args.reinforcement_episodes)),
        seed=args.seed,
    )

    graph_path = Path(MANHATTAN_1K_EXPERIMENT.graph_path)
    print(f"Preparing Manhattan 1K network from {graph_path} ...", flush=True)
    graph, distance, distance_stats = prepare_manhattan_road_network(
        graph_path,
        show_distance_progress=False,
    )
    print(
        "Distance preparation finished: total={:.3f}s".format(
            distance_stats["distance_initialization_seconds"]
        ),
        flush=True,
    )
    training_environments = []
    validation_environments = []
    for customer_size in args.customer_sizes:
        instance_count = args.train_instances_per_size + args.validation_instances_per_size
        depots, cities = sample_multiagent_instances(
            graph,
            instance_count,
            MANHATTAN_1K_EXPERIMENT.depot_count,
            customer_size,
            seed=args.seed + customer_size * 1000,
        )
        for index in range(instance_count):
            environment = _build_environment(
                graph,
                distance,
                depots[index],
                cities[index],
                surrogate_checkpoint,
                environment_config,
            )
            target = (
                training_environments
                if index < args.train_instances_per_size
                else validation_environments
            )
            target.append(environment)
            split = "train" if target is training_environments else "validation"
            print(
                f"  environment split={split} size={customer_size} "
                f"groups={[len(environment.groups[d]) for d in environment.depots]}",
                flush=True,
            )

    action_feature_dim = (
        training_environments[0].upper_pair_feature_dim
        + training_environments[0].lower_action_feature_dim
    )
    policy = SingleLevelActorCritic(
        SingleLevelPolicyConfig(
            action_feature_dim=action_feature_dim,
            seed=args.seed,
        )
    )
    print(
        "Training single-level policy: train_instances={}, validation_instances={} ...".format(
            len(training_environments),
            len(validation_environments),
        ),
        flush=True,
    )
    policy, artifacts = train_single_level_policy(
        policy,
        training_environments,
        validation_environments,
        config=training_config,
        progress_callback=_print_progress,
    )
    training_metrics = evaluate_single_level_policy(policy, training_environments)
    validation_metrics = evaluate_single_level_policy(policy, validation_environments)
    metrics = {
        "run_id": run_id,
        "surrogate_checkpoint": str(surrogate_checkpoint),
        "training": training_metrics,
        "validation": validation_metrics,
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_directory / f"single_level_policy_{run_id}.pt"
    metrics_path = args.output_directory / f"single_level_policy_{run_id}_metrics.json"
    history_path = args.output_directory / f"single_level_policy_{run_id}_history.csv"
    torch.save(
        single_level_checkpoint_payload(
            policy,
            training_config,
            environment_config,
            surrogate_checkpoint,
            artifacts,
            metrics,
        ),
        checkpoint_path,
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_history(history_path, artifacts["history"])
    print(
        "Training finished: validation score improvement={:.4f}, cost {:.4f} -> {:.4f}".format(
            validation_metrics["mean_score_improvement"],
            validation_metrics["mean_initial_cost_sum"],
            validation_metrics["mean_final_cost_sum"],
        ),
        flush=True,
    )
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    print(f"History: {history_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
