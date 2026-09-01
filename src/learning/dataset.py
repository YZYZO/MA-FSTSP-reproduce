"""从 Set-MST 初始划分生成 relocate/swap 学习数据集。"""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from src.fstsp import MultiAgentFlyingSidekickTSP
from src.learning.evaluator import GroupEvaluator
from src.learning.features import GROUP_FEATURE_NAMES, extract_group_feature_dict
from src.learning.settings import SetTSPSolverSettings
from src.partitioning import partition_customers


@dataclass(frozen=True)
class DatasetGenerationSettings:
    """
    保存学习数据生成时的划分、邻域和标签配置。

    输入：车队参数、MST 模式、邻域数量、求解上限、预算和随机种子。
    输出：不可变配置对象。
    逻辑：默认预算来自里程碑 A 的 Manhattan 1K 探测结果。
    """

    drone_count: int = 3
    theta: tuple = (0.5, 0.5)
    edge_mode: str = "mean"
    relocate_samples_per_instance: int = 3
    swap_samples_per_instance: int = 1
    set_tsp_time_limit_seconds: float = 30.0
    phase2_wall_budget_seconds: float = 20.0
    downstream_budget_seconds: float = 50.0
    solver_threads: int = 1
    solver_seed: int = 0


@dataclass(frozen=True)
class PartitionAction:
    """
    描述一个从 Set-MST 基准分区出发的一步邻域动作。

    输入：动作类型、源/目标仓库、源客户、可选目标客户和亲和力变化。
    输出：不可变动作对象。
    逻辑：`relocate` 只使用源客户，`swap` 同时交换两个客户。
    """

    action_type: str
    source_depot: object
    target_depot: object
    source_customer: object
    target_customer: object | None
    affinity_delta: float

    def to_dict(self):
        """
        将邻域动作转换成普通字典。

        输入：当前动作。
        输出：可写入 CSV/JSON 的字段字典。
        逻辑：保留动作来源，方便后续消融 relocate 与 swap。
        """
        return asdict(self)


def _stable_node_key(node):
    """
    生成节点编号的稳定排序键。

    输入：任意节点编号。
    输出：类型名与文本表示组成的元组。
    逻辑：邻域采样和样本哈希不依赖字典或数组遍历顺序。
    """
    return type(node).__name__, repr(node)


def _jsonable(value):
    """
    将数据集记录递归转换成 JSON 基础类型。

    输入：可能包含 NumPy 标量、数组、元组、路径和非有限浮点数的值。
    输出：可由标准 JSON 编码器处理的对象。
    逻辑：数据集不依赖 pickle，便于版本检查和跨进程读取。
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def assign_instance_split(instance_index, instances_per_size):
    """
    按原始实例编号分配训练、验证和测试集合。

    输入：当前实例编号和每个客户规模的实例总数。
    输出：`train/validation/test` 字符串。
    逻辑：前 60% 训练，中间 20% 验证，最后至少一个实例测试；同一实例的全部邻域保持同组。
    """
    train_end = max(1, int(math.ceil(instances_per_size * 0.6)))
    validation_end = max(train_end + 1, int(math.ceil(instances_per_size * 0.8)))
    validation_end = min(validation_end, instances_per_size - 1)
    if instance_index < train_end:
        return "train"
    if instance_index < validation_end:
        return "validation"
    return "test"


def _depot_affinity(customer, depot, truck_distance):
    """
    计算客户与仓库之间的双向平均道路距离。

    输入：客户、仓库和有向卡车距离表。
    输出：两个方向道路距离的均值。
    逻辑：邻域采样优先选择对源、目标仓库归属较模糊的边界客户。
    """
    return (
        truck_distance[depot][customer]
        + truck_distance[customer][depot]
    ) / 2.0


def _select_diverse_candidates(candidates, count, seed):
    """
    从按边界程度排序的动作候选中选择少量确定且多样的动作。

    输入：候选 `(排序值, 动作)` 列表、数量和随机种子。
    输出：不重复动作列表。
    逻辑：保留最靠近边界的动作，再从前部候选池随机补充，兼顾可解释性和多样性。
    """
    if count <= 0 or len(candidates) == 0:
        return []
    candidates = sorted(candidates, key=lambda item: item[0])
    selected = [candidates[0][1]]
    if count == 1:
        return selected

    pool_size = min(len(candidates), max(count * 12, 24))
    random_state = np.random.RandomState(seed)
    remaining_indices = list(range(1, pool_size))
    random_state.shuffle(remaining_indices)
    for index in remaining_indices:
        action = candidates[index][1]
        if action not in selected:
            selected.append(action)
        if len(selected) >= count:
            break
    return selected


def generate_neighbor_actions(
    groups,
    depots,
    truck_distance,
    relocate_count,
    swap_count,
    seed,
):
    """
    从一个 Set-MST 基准分区生成 relocate 和 swap 一步邻域动作。

    输入：客户组、仓库、有向道路距离、两类动作数量和随机种子。
    输出：`PartitionAction` 列表。
    逻辑：用双向仓库亲和力变化衡量边界程度，并在候选池中选择多样动作。
    """
    ordered_depots = list(depots)
    relocate_candidates = []
    for source_depot in ordered_depots:
        for customer in groups[source_depot]:
            source_affinity = _depot_affinity(
                customer,
                source_depot,
                truck_distance,
            )
            for target_depot in ordered_depots:
                if target_depot == source_depot:
                    continue
                target_affinity = _depot_affinity(
                    customer,
                    target_depot,
                    truck_distance,
                )
                affinity_delta = float(target_affinity - source_affinity)
                action = PartitionAction(
                    action_type="relocate",
                    source_depot=source_depot,
                    target_depot=target_depot,
                    source_customer=customer,
                    target_customer=None,
                    affinity_delta=affinity_delta,
                )
                sort_key = (
                    abs(affinity_delta),
                    affinity_delta,
                    _stable_node_key(source_depot),
                    _stable_node_key(target_depot),
                    _stable_node_key(customer),
                )
                relocate_candidates.append((sort_key, action))

    swap_candidates = []
    for source_index, source_depot in enumerate(ordered_depots):
        for target_depot in ordered_depots[source_index + 1:]:
            for source_customer in groups[source_depot]:
                for target_customer in groups[target_depot]:
                    before = (
                        _depot_affinity(source_customer, source_depot, truck_distance)
                        + _depot_affinity(target_customer, target_depot, truck_distance)
                    )
                    after = (
                        _depot_affinity(source_customer, target_depot, truck_distance)
                        + _depot_affinity(target_customer, source_depot, truck_distance)
                    )
                    affinity_delta = float(after - before)
                    action = PartitionAction(
                        action_type="swap",
                        source_depot=source_depot,
                        target_depot=target_depot,
                        source_customer=source_customer,
                        target_customer=target_customer,
                        affinity_delta=affinity_delta,
                    )
                    sort_key = (
                        abs(affinity_delta),
                        affinity_delta,
                        _stable_node_key(source_customer),
                        _stable_node_key(target_customer),
                    )
                    swap_candidates.append((sort_key, action))

    relocate_actions = _select_diverse_candidates(
        relocate_candidates,
        relocate_count,
        seed,
    )
    swap_actions = _select_diverse_candidates(
        swap_candidates,
        swap_count,
        seed + 100003,
    )
    return relocate_actions + swap_actions


def apply_partition_action(groups, action):
    """
    在客户组副本上应用一个 relocate 或 swap 动作。

    输入：基准分区和 `PartitionAction`。
    输出：动作后的新分区字典。
    逻辑：只改变源、目标两个客户组，并稳定排序组内客户。
    """
    updated_groups = {
        depot: list(customers)
        for depot, customers in groups.items()
    }
    updated_groups[action.source_depot].remove(action.source_customer)
    updated_groups[action.target_depot].append(action.source_customer)
    if action.action_type == "swap":
        updated_groups[action.target_depot].remove(action.target_customer)
        updated_groups[action.source_depot].append(action.target_customer)
    return {
        depot: sorted(customers, key=_stable_node_key)
        for depot, customers in updated_groups.items()
    }


def _sample_key(base_instance_id, depot, customers):
    """
    为一个实例内的唯一仓库客户组生成稳定样本编号。

    输入：原始实例标识、仓库和客户集合。
    输出：短 SHA-256 文本。
    逻辑：相同客户组在多个邻域分区中出现时只保存和求解一次。
    """
    canonical = json.dumps(
        _jsonable({
            "base_instance_id": base_instance_id,
            "depot": depot,
            "customers": sorted(customers, key=_stable_node_key),
        }),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def generate_instance_dataset_records(
    graph,
    depots,
    cities,
    distance,
    map_id,
    customer_size,
    instance_index,
    instances_per_size,
    cache,
    settings=None,
    progress_callback=None,
):
    """
    为一个原始实例生成基准组与 relocate/swap 邻域组标签。

    输入：路网实例、标识、拆分信息、缓存、配置和可选进度回调。
    输出：去重后的单客户组训练记录列表。
    逻辑：所有邻域从同一个 Set-MST 分区出发，只真实评估基准组及每个动作改变的两个组。
    """
    settings = settings or DatasetGenerationSettings()
    split = assign_instance_split(instance_index, instances_per_size)
    base_instance_id = f"manhattan1k-size{customer_size}-instance{instance_index}"

    model = MultiAgentFlyingSidekickTSP(
        graph,
        depots,
        cities,
        distance,
        settings.drone_count,
        theta=settings.theta,
    )
    boundary_start = time.perf_counter()
    convex_sets = model.get_boundary_convex_sets(model.theta[0])
    boundary_seconds = time.perf_counter() - boundary_start
    partition_start = time.perf_counter()
    base_groups = partition_customers(
        model.depots,
        model.cities,
        convex_sets,
        model.distance["truck"],
        model.distance["drone"],
        model.speed,
        edge_mode=settings.edge_mode,
        coefficient=model.const,
    )
    partition_seconds = time.perf_counter() - partition_start

    evaluator = GroupEvaluator(
        model,
        convex_sets,
        map_id=map_id,
        cache=cache,
        solver_settings=SetTSPSolverSettings(
            time_limit_seconds=settings.set_tsp_time_limit_seconds,
            threads=settings.solver_threads,
            seed=settings.solver_seed,
        ),
    )
    actions = generate_neighbor_actions(
        base_groups,
        model.depots,
        model.distance["truck"],
        settings.relocate_samples_per_instance,
        settings.swap_samples_per_instance,
        seed=settings.solver_seed + customer_size * 1000 + instance_index,
    )

    records_by_key = {}

    def evaluate_group(partition_id, action, depot, customers, group_role):
        """
        评估并登记一个唯一客户组，重复组直接复用本实例记录。

        输入：分区、动作、仓库、客户和组角色。
        输出：无；结果写入 `records_by_key`。
        逻辑：SQLite 避免跨运行重复求解，样本键避免同一运行重复行。
        """
        sample_key = _sample_key(base_instance_id, depot, customers)
        if sample_key in records_by_key:
            return
        features = extract_group_feature_dict(
            graph,
            distance,
            depot,
            customers,
            convex_sets,
        )
        evaluation = evaluator.evaluate(depot, customers)
        solver_result = evaluation.set_tsp_result
        downstream_seconds = (
            evaluation.set_tsp_wall_seconds + evaluation.phase3_seconds
        )
        action_values = action.to_dict() if action is not None else {
            "action_type": "base",
            "source_depot": None,
            "target_depot": None,
            "source_customer": None,
            "target_customer": None,
            "affinity_delta": 0.0,
        }
        record = {
            "sample_key": sample_key,
            "base_instance_id": base_instance_id,
            "split": split,
            "map_id": map_id,
            "customer_size": int(customer_size),
            "instance_index": int(instance_index),
            "partition_id": partition_id,
            "group_role": group_role,
            "depot_node": depot,
            "customers": list(evaluation.customers),
            "mst_edge_mode": settings.edge_mode,
            "boundary_construction_seconds": float(boundary_seconds),
            "mst_partition_seconds": float(partition_seconds),
            **action_values,
            **features,
            "target_final_cost": evaluation.final_cost,
            "target_log_set_tsp_time": math.log1p(
                evaluation.set_tsp_wall_seconds
            ),
            "target_set_tsp_timeout": int(solver_result.timed_out),
            "target_phase2_budget_exceeded": int(
                evaluation.set_tsp_wall_seconds
                > settings.phase2_wall_budget_seconds
            ),
            "target_downstream_total_seconds": downstream_seconds,
            "target_downstream_budget_exceeded": int(
                downstream_seconds > settings.downstream_budget_seconds
            ),
            "target_cost_is_approximate": int(solver_result.timed_out),
            "set_tsp_runtime_seconds": solver_result.runtime_seconds,
            "set_tsp_wall_seconds": evaluation.set_tsp_wall_seconds,
            "phase3_seconds": evaluation.phase3_seconds,
            "set_tsp_status": solver_result.status_name,
            "set_tsp_mip_gap": solver_result.mip_gap,
            "set_tsp_solution_count": solver_result.solution_count,
            "set_tsp_has_incumbent": int(solver_result.has_incumbent),
            "set_tsp_fallback_used": int(solver_result.fallback_used),
            "cache_hit": int(evaluation.cache_hit),
        }
        records_by_key[sample_key] = _jsonable(record)
        if progress_callback is not None:
            progress_callback(records_by_key[sample_key])

    for depot in model.depots:
        evaluate_group(
            partition_id="base",
            action=None,
            depot=depot,
            customers=base_groups[depot],
            group_role="base",
        )

    for action_index, action in enumerate(actions):
        partition_id = f"{action.action_type}-{action_index}"
        neighbor_groups = apply_partition_action(base_groups, action)
        evaluate_group(
            partition_id,
            action,
            action.source_depot,
            neighbor_groups[action.source_depot],
            "source_after_action",
        )
        evaluate_group(
            partition_id,
            action,
            action.target_depot,
            neighbor_groups[action.target_depot],
            "target_after_action",
        )
    return list(records_by_key.values())


def summarize_learning_dataset(records, settings):
    """
    汇总学习数据集的实例、动作、标签和缓存命中情况。

    输入：全部记录和数据生成配置。
    输出：JSON 可序列化摘要。
    逻辑：按实例而不是行统计拆分，显式检查相邻划分没有跨集合泄漏。
    """
    records = list(records)
    split_instances = {}
    split_rows = {}
    action_counts = {}
    for record in records:
        split = record["split"]
        split_instances.setdefault(split, set()).add(record["base_instance_id"])
        split_rows[split] = split_rows.get(split, 0) + 1
        action = record["action_type"]
        action_counts[action] = action_counts.get(action, 0) + 1

    instance_to_split = {}
    for record in records:
        instance_to_split.setdefault(record["base_instance_id"], set()).add(
            record["split"]
        )
    leakage_free = all(len(splits) == 1 for splits in instance_to_split.values())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(records),
        "base_instance_count": len(instance_to_split),
        "feature_names": list(GROUP_FEATURE_NAMES),
        "split_row_counts": split_rows,
        "split_instance_counts": {
            split: len(instances)
            for split, instances in split_instances.items()
        },
        "instance_split_leakage_free": leakage_free,
        "action_row_counts": action_counts,
        "timeout_count": sum(
            int(record["target_set_tsp_timeout"]) for record in records
        ),
        "phase2_budget_exceeded_count": sum(
            int(record["target_phase2_budget_exceeded"]) for record in records
        ),
        "downstream_budget_exceeded_count": sum(
            int(record["target_downstream_budget_exceeded"]) for record in records
        ),
        "cache_hit_count": sum(int(record["cache_hit"]) for record in records),
        "settings": _jsonable(asdict(settings)),
    }


def write_learning_dataset(records, output_directory, settings, run_id=None):
    """
    将学习数据明细与数据质量摘要写入 CSV 和 JSON。

    输入：记录、输出目录、配置和可选运行标识。
    输出：`(csv_path, summary_path, summary)`。
    逻辑：特征和标签使用扁平 CSV，生成参数与拆分检查使用 JSON。
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output_directory / f"learning_dataset_{run_id}.csv"
    summary_path = output_directory / f"learning_dataset_{run_id}_summary.json"
    records = [_jsonable(record) for record in records]
    fieldnames = list(records[0].keys()) if records else []

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: (
                    json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in record.items()
            })

    summary = summarize_learning_dataset(records, settings)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    return csv_path, summary_path, summary
