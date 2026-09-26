"""为层次化 Deep Sets 重建客户集合并生成可复用的张量缓存。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .dataset import ExperimentInstance, load_instances
from .deep_sets import DeepSetBatch, POSITIVE_TARGET_INDICES
from .features import group_variable_proxy
from .models import REGRESSION_TARGETS
from .reporting import deduplicate_partition_records
from .road import (
    DroneDistanceMatrix,
    TerminalRoadMatrix,
    build_road_csr,
    build_spatial_sets,
    compare_boundary_sizes,
    load_road_graph,
    precompute_terminal_distances,
)


BOUNDARY_FEATURE_NAMES = (
    "node_x", "node_y", "relative_city_x", "relative_city_y",
    "air_distance_to_city", "log_in_degree", "log_out_degree",
)
CUSTOMER_STATIC_FEATURE_NAMES = (
    "customer_x", "customer_y", "log_boundary_size", "log_in_degree", "log_out_degree",
)
ASSIGNMENT_FEATURE_NAMES = (
    "relative_depot_x", "relative_depot_y", "road_out", "road_in",
    "road_asymmetry", "road_stretch", "set_road_out_min", "set_road_out_mean",
    "set_road_in_min", "set_road_in_mean", "group_fraction", "moved_from_baseline",
)
GROUP_FEATURE_NAMES = (
    "depot_x", "depot_y", "log_in_degree", "log_out_degree", "customer_fraction",
    "boundary_fraction", "log_variable_proxy", "empty_group",
)


@dataclass(frozen=True)
class InstanceSplit:
    """保存互不重叠的训练、验证和测试实例编号。"""

    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def _partition_from_record(payload: dict[str, Sequence[int]]) -> dict[int, tuple[int, ...]]:
    """输入 JSON 记录中的分区对象，输出整数仓库键和客户元组。"""
    return {int(depot): tuple(map(int, customers)) for depot, customers in payload.items()}


def _owner_map(partition: dict[int, Sequence[int]]) -> dict[int, int]:
    """输入仓库到客户的分区，输出客户到仓库的反向映射。"""
    return {
        int(city): int(depot)
        for depot, customers in partition.items()
        for city in customers
    }


def _safe_distance(value: float, replacement: float) -> float:
    """输入可能为无穷的路网距离，输出有限值供神经网络训练。"""
    return float(value) if math.isfinite(float(value)) else float(replacement)


def _graph_coordinates(graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """输入路网，输出原坐标、逐维中心和非零标准差。"""
    coordinates = np.asarray(
        [graph.nodes[node]["pos"] for node in range(len(graph))], dtype=np.float64
    )
    center = np.mean(coordinates, axis=0)
    scale = np.std(coordinates, axis=0)
    scale[scale < 1e-9] = 1.0
    return coordinates, center, scale


def _road_scale(instance: ExperimentInstance, truck: TerminalRoadMatrix) -> float:
    """输入实例和终端距离矩阵，输出仓库—客户距离的稳健归一化尺度。"""
    values = []
    for depot in instance.depots:
        for city in instance.cities:
            values.extend((truck.query(depot, city), truck.query(city, depot)))
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return max(float(np.median(finite)) if len(finite) else 1.0, 1e-6)


def _static_instance_tensors(
    instance: ExperimentInstance,
    graph,
    road_csr,
    *,
    distance_batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    重建单实例的道路节点集合和静态客户特征。

    输入历史实例、路网及 CSR；输出可缓存的静态数组和仅在构造候选时使用的上下文。
    """
    boundary, _ = build_spatial_sets(
        graph,
        instance.cities,
        instance.drone_limit * instance.theta[0],
        instance.drone_limit / 2.0,
    )
    boundary_audit = compare_boundary_sizes(boundary, instance.boundary_sizes)
    if not boundary_audit["exact"]:
        raise RuntimeError(
            f"{instance.instance_id} 边界集合与原实验不一致：{boundary_audit['examples']}"
        )

    # Deep Sets 第一层只需要 Phase 2 的边界集合，不重复物化 Phase 3 服务区域节点。
    terminals = set(instance.depots) | set(instance.cities)
    for nodes in boundary.values():
        terminals.update(nodes)
    terminal_data = precompute_terminal_distances(
        road_csr, terminals, batch_size=distance_batch_size
    )
    truck = TerminalRoadMatrix(terminal_data)
    drone = DroneDistanceMatrix(graph)
    coordinates, coordinate_center, coordinate_scale = _graph_coordinates(graph)
    normalized_coordinates = (coordinates - coordinate_center) / coordinate_scale
    road_scale = _road_scale(instance, truck)

    cities = tuple(map(int, instance.cities))
    city_index = {city: index for index, city in enumerate(cities)}
    boundary_rows: list[list[float]] = []
    boundary_customer: list[int] = []
    customer_rows: list[list[float]] = []
    normalized_air_scale = max(instance.drone_limit * instance.theta[0], 1e-6)

    for customer_position, city in enumerate(cities):
        # 极端情况下用客户节点作为单元素集合，使空集合仍有确定的可学习表示。
        nodes = list(boundary[city]) or [city]
        city_coordinate = normalized_coordinates[city]
        for node in nodes:
            node_coordinate = normalized_coordinates[node]
            boundary_rows.append([
                float(node_coordinate[0]),
                float(node_coordinate[1]),
                float(node_coordinate[0] - city_coordinate[0]),
                float(node_coordinate[1] - city_coordinate[1]),
                float(drone.query(city, node) / normalized_air_scale),
                math.log1p(graph.in_degree(node)),
                math.log1p(graph.out_degree(node)),
            ])
            boundary_customer.append(customer_position)
        customer_rows.append([
            float(city_coordinate[0]),
            float(city_coordinate[1]),
            math.log1p(len(boundary[city])),
            math.log1p(graph.in_degree(city)),
            math.log1p(graph.out_degree(city)),
        ])

    context = {
        "instance": instance,
        "graph": graph,
        "boundary": boundary,
        "truck": truck,
        "drone": drone,
        "normalized_coordinates": normalized_coordinates,
        "road_scale": road_scale,
        "city_index": city_index,
    }
    baseline_assignment, baseline_group_index, baseline_group_features = _partition_features(
        instance.partition, instance.partition, context
    )
    static = {
        "boundary_features": np.asarray(boundary_rows, dtype=np.float32),
        "boundary_customer": np.asarray(boundary_customer, dtype=np.int64),
        "customer_static_features": np.asarray(customer_rows, dtype=np.float32),
        "baseline_assignment_features": baseline_assignment,
        "baseline_customer_group": baseline_group_index,
        "baseline_group_features": baseline_group_features,
        "customer_count": len(cities),
        "depot_count": len(instance.depots),
        "graph_name": instance.graph_name,
        "source_name": instance.source_name,
        "instance_index": instance.instance_index,
    }
    return static, context


def _partition_features(
    partition: dict[int, Sequence[int]],
    baseline: dict[int, Sequence[int]],
    context: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    把一个候选分区编码成逐客户归属特征、客户到组索引和逐组特征。

    输入候选、MST 基线和单实例上下文；输出固定客户顺序且固定仓库顺序的三个数组。
    """
    instance: ExperimentInstance = context["instance"]
    graph = context["graph"]
    boundary: dict[int, list[int]] = context["boundary"]
    truck: TerminalRoadMatrix = context["truck"]
    drone: DroneDistanceMatrix = context["drone"]
    normalized_coordinates: np.ndarray = context["normalized_coordinates"]
    road_scale = float(context["road_scale"])
    city_index: dict[int, int] = context["city_index"]
    depots = tuple(map(int, instance.depots))
    depot_index = {depot: index for index, depot in enumerate(depots)}
    owner = _owner_map(partition)
    baseline_owner = _owner_map(baseline)
    total_customers = max(len(instance.cities), 1)
    total_boundary = max(sum(instance.boundary_sizes.values()), 1)

    customer_group = np.empty(len(instance.cities), dtype=np.int64)
    assignment_rows = np.empty(
        (len(instance.cities), len(ASSIGNMENT_FEATURE_NAMES)), dtype=np.float32
    )
    replacement_distance = 10.0 * road_scale
    for city in instance.cities:
        position = city_index[int(city)]
        depot = owner[int(city)]
        group_position = depot_index[depot]
        customer_group[position] = group_position
        forward = _safe_distance(truck.query(depot, city), replacement_distance)
        backward = _safe_distance(truck.query(city, depot), replacement_distance)
        air = max(drone.query(depot, city), 1e-6)
        nodes = list(boundary[int(city)]) or [int(city)]
        set_out = np.asarray([
            _safe_distance(truck.query(depot, node), replacement_distance) for node in nodes
        ])
        set_in = np.asarray([
            _safe_distance(truck.query(node, depot), replacement_distance) for node in nodes
        ])
        delta_coordinate = normalized_coordinates[int(city)] - normalized_coordinates[depot]
        assignment_rows[position] = np.asarray([
            float(delta_coordinate[0]),
            float(delta_coordinate[1]),
            forward / road_scale,
            backward / road_scale,
            abs(forward - backward) / max(0.5 * (forward + backward), 1e-6),
            0.5 * (forward + backward) / air,
            float(np.min(set_out) / road_scale),
            float(np.mean(set_out) / road_scale),
            float(np.min(set_in) / road_scale),
            float(np.mean(set_in) / road_scale),
            len(partition.get(depot, ())) / total_customers,
            float(baseline_owner[int(city)] != depot),
        ], dtype=np.float32)

    group_rows = []
    for depot in depots:
        members = tuple(map(int, partition.get(depot, ())))
        depot_coordinate = normalized_coordinates[depot]
        boundary_load = sum(instance.boundary_sizes[city] for city in members)
        group_rows.append([
            float(depot_coordinate[0]),
            float(depot_coordinate[1]),
            math.log1p(graph.in_degree(depot)),
            math.log1p(graph.out_degree(depot)),
            len(members) / total_customers,
            boundary_load / total_boundary,
            math.log1p(group_variable_proxy(members, instance.boundary_sizes)),
            float(len(members) == 0),
        ])
    return assignment_rows, customer_group, np.asarray(group_rows, dtype=np.float32)


def _candidate_labels(record: dict, depot_order: Sequence[int], cost_limit: float) -> dict[str, Any]:
    """输入候选记录和固定仓库顺序，输出回归、分类及逐组监督标签。"""
    labels = record["labels"]
    evaluation = record.get("evaluation", {})
    phase2 = float(labels.get("phase2_serial_effective_seconds", evaluation.get(
        "phase2_serial_effective_seconds", evaluation.get("phase2_effective_seconds", 0.0)
    )))
    phase3 = float(labels.get("phase3_seconds", evaluation.get("phase3_seconds", 0.0)))
    downstream = float(labels.get(
        "downstream_total_seconds", evaluation.get("downstream_total_seconds", phase2 + phase3)
    ))
    target_values = np.asarray([
        float(labels["time_saving_ratio"]),
        phase2,
        phase3,
        downstream,
        float(labels.get("solver_work_sum", evaluation.get("solver_work_sum", 0.0))),
        float(labels["cost_change_ratio"]),
        float(labels.get("final_cost", evaluation.get("final_cost", 0.0))),
    ], dtype=np.float32)
    exact = bool(labels.get("candidate_exact", evaluation.get("candidate_exact", False)))
    relative_exact = bool(labels.get("relative_labels_exact", exact))
    target_mask = np.asarray([
        relative_exact,
        exact,
        exact,
        exact,
        exact,
        relative_exact,
        exact,
    ], dtype=bool)

    by_depot = {
        int(item["depot"]): item for item in evaluation.get("group_labels", [])
    }
    group_log_phase2, group_time_mask, group_censored = [], [], []
    for depot in depot_order:
        item = by_depot.get(int(depot), {})
        time_value = float(item.get("phase2_effective_seconds", 0.0))
        censored = bool(item.get("right_censored", item.get("censored", False)))
        valid = bool(item.get("runtime_label_valid", not censored))
        fallback = bool(item.get("fallback_used", False) or item.get("guard_fallback", False))
        group_log_phase2.append(math.log1p(max(time_value, 0.0)))
        group_time_mask.append(valid and not censored and not fallback)
        group_censored.append(censored)

    right_censored = bool(labels.get(
        "candidate_right_censored",
        labels.get("candidate_censored", evaluation.get("right_censored_groups", 0)),
    ))
    return {
        "targets_raw": target_values,
        "target_mask": target_mask,
        "right_censored": float(right_censored),
        "cost_feasible": float(float(labels["cost_change_ratio"]) <= cost_limit),
        "cost_feasible_mask": relative_exact,
        "group_log_phase2": np.asarray(group_log_phase2, dtype=np.float32),
        "group_time_mask": np.asarray(group_time_mask, dtype=bool),
        "group_right_censored": np.asarray(group_censored, dtype=np.float32),
    }


def _local_npz_paths(result_root: str | Path) -> dict[str, Path]:
    """输入历史结果根目录，输出以 NPZ 文件名主干索引的本地路径。"""
    return {path.stem: path.resolve() for path in Path(result_root).rglob("*.npz")}


def build_deepsets_cache(
    records: Iterable[dict],
    result_root: str | Path,
    output_path: str | Path,
    *,
    cost_limit: float = 0.10,
    distance_batch_size: int = 128,
) -> dict[str, Any]:
    """
    从候选记录和历史 NPZ 构造层次集合张量缓存并写入磁盘。

    只读取候选记录实际涉及的路网和实例；现有50/100/150实验不会触碰55K数据。
    输出缓存把实例静态集合与候选归属分开存储，避免24个候选重复保存边界节点特征。
    """
    rows = deduplicate_partition_records(list(records))
    if not rows:
        raise ValueError("没有候选记录可用于构造 Deep Sets 数据。")
    npz_paths = _local_npz_paths(result_root)
    global_feature_names = tuple(sorted({name for row in rows for name in row["features"]}))
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_instance[str(row["instance_id"])].append(row)

    # 每张路网只读取和稀疏化一次；每个实例仍单独计算其有限终端距离。
    graph_cache: dict[str, Any] = {}
    road_csr_cache: dict[str, Any] = {}
    instance_cache: dict[str, dict[str, Any]] = {}
    candidate_cache: list[dict[str, Any]] = []

    for position, (instance_id, instance_rows) in enumerate(sorted(by_instance.items()), start=1):
        first = instance_rows[0]
        source_name = str(first["source_name"])
        if source_name not in npz_paths:
            raise FileNotFoundError(f"未在 {result_root} 找到 {source_name}.npz")
        instance = load_instances(npz_paths[source_name], [int(first["instance_index"])])[0]
        graph_name = instance.graph_name
        if graph_name not in graph_cache:
            graph_cache[graph_name] = load_road_graph(instance.graph_path)
            road_csr_cache[graph_name] = build_road_csr(graph_cache[graph_name])
        print(
            f"[DeepSets cache] {position}/{len(by_instance)} {instance_id}", flush=True
        )
        static, context = _static_instance_tensors(
            instance,
            graph_cache[graph_name],
            road_csr_cache[graph_name],
            distance_batch_size=distance_batch_size,
        )
        instance_cache[instance_id] = static
        for row in instance_rows:
            partition = _partition_from_record(row["partition"])
            assignment, customer_group, group_features = _partition_features(
                partition, instance.partition, context
            )
            labels = _candidate_labels(row, instance.depots, cost_limit)
            candidate_cache.append({
                "instance_id": instance_id,
                "candidate_name": str(row.get("candidate_name", "")),
                "candidate_kind": str(row.get("candidate_kind", "unknown")),
                "graph_name": str(row["graph_name"]),
                "customer_count": int(row["customer_count"]),
                "candidate_assignment_features": assignment,
                "candidate_customer_group": customer_group,
                "candidate_group_features": group_features,
                "global_features": np.asarray([
                    float(row["features"].get(name, 0.0)) for name in global_feature_names
                ], dtype=np.float32),
                **labels,
            })

    cache = {
        "format_version": 1,
        "feature_names": {
            "boundary": BOUNDARY_FEATURE_NAMES,
            "customer_static": CUSTOMER_STATIC_FEATURE_NAMES,
            "assignment": ASSIGNMENT_FEATURE_NAMES,
            "group": GROUP_FEATURE_NAMES,
            "global": global_feature_names,
            "targets": REGRESSION_TARGETS,
        },
        "cost_limit": float(cost_limit),
        "instances": instance_cache,
        "records": candidate_cache,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return cache


def load_deepsets_cache(path: str | Path) -> dict[str, Any]:
    """输入可信的本地缓存路径，输出由 CPU 张量和 NumPy 数组组成的数据字典。"""
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def select_global_feature_indices(
    cache: dict[str, Any], *, include_method_features: bool
) -> tuple[np.ndarray, tuple[str, ...]]:
    """输入缓存和算法类别开关，输出使用的全局特征列下标及名称。"""
    names = tuple(cache["feature_names"]["global"])
    selected = tuple(
        index for index, name in enumerate(names)
        if include_method_features or not name.startswith("kind_")
    )
    return np.asarray(selected, dtype=np.int64), tuple(names[index] for index in selected)


def split_instance_ids(
    cache: dict[str, Any], *, test_fraction: float = 0.20, random_seed: int = 260915
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    按路网和客户规模分层切分独立实例。

    只从存在相对精确标签的实例中抽测试集；其余实例留在训练集用于绝对值和超时任务。
    """
    strata: dict[tuple[str, int], set[str]] = defaultdict(set)
    all_ids = set(cache["instances"])
    for record in cache["records"]:
        relative_mask = bool(record["target_mask"][0] and record["target_mask"][5])
        if relative_mask:
            strata[(record["graph_name"], int(record["customer_count"]))].add(
                record["instance_id"]
            )
    generator = random.Random(random_seed)
    test_ids: set[str] = set()
    for instance_ids in strata.values():
        ordered = sorted(instance_ids)
        generator.shuffle(ordered)
        count = max(1, int(round(test_fraction * len(ordered))))
        test_ids.update(ordered[:count])
    train_ids = all_ids - test_ids
    return tuple(sorted(train_ids)), tuple(sorted(test_ids))


def split_instance_ids_three_way(
    cache: dict[str, Any],
    *,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    random_seed: int = 260915,
) -> InstanceSplit:
    """
    按路网与客户规模分层生成训练、验证和测试实例。

    验证集只负责提前停止，测试集在最终模型确定前不会参与训练过程。若同一分层至少有
    三个包含超时候选的实例，则训练、验证和测试各保留一个，避免状态任务无法评估。
    回归标签缺失由候选级掩码处理，因此这些实例仍可用于评估超时分类。
    """
    strata: dict[tuple[str, int], set[str]] = defaultdict(set)
    all_ids = set(cache["instances"])
    for instance_id, instance in cache["instances"].items():
        strata[(instance["graph_name"], int(instance["customer_count"]))].add(instance_id)

    generator = random.Random(random_seed)
    validation_ids: set[str] = set()
    test_ids: set[str] = set()
    censored_counts: dict[str, int] = defaultdict(int)
    for record in cache["records"]:
        if bool(record["right_censored"]):
            censored_counts[record["instance_id"]] += 1
    censored_instance_ids = set(censored_counts)
    for stratum, instance_ids in sorted(strata.items()):
        test_count = max(1, int(round(test_fraction * len(instance_ids))))
        validation_count = max(1, int(round(validation_fraction * len(instance_ids))))
        if test_count + validation_count >= len(instance_ids):
            raise ValueError(f"分层 {stratum} 的实例数不足以同时划分训练、验证和测试集。")

        censored = sorted(
            set(instance_ids) & censored_instance_ids,
            key=lambda instance_id: (-censored_counts[instance_id], instance_id),
        )
        ordinary = sorted(set(instance_ids) - censored_instance_ids)
        generator.shuffle(ordinary)

        # 至少保留一个超时实例用于训练；其余优先覆盖测试和验证状态评价。
        available_censored = censored[1:]
        selected_test = available_censored[:1]
        selected_validation = available_censored[1:2]
        remaining = available_censored[2:] + ordinary
        generator.shuffle(remaining)
        test_needed = test_count - len(selected_test)
        selected_test.extend(remaining[:test_needed])
        remaining = remaining[test_needed:]
        validation_needed = validation_count - len(selected_validation)
        selected_validation.extend(remaining[:validation_needed])

        test_ids.update(selected_test)
        validation_ids.update(selected_validation)

    train_ids = all_ids - validation_ids - test_ids
    return InstanceSplit(
        train_ids=tuple(sorted(train_ids)),
        validation_ids=tuple(sorted(validation_ids)),
        test_ids=tuple(sorted(test_ids)),
    )


def global_feature_normalizer(
    cache: dict[str, Any], train_instance_ids: Sequence[str], feature_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """输入训练实例和全局特征列，输出仅由训练集计算的均值与标准差。"""
    if len(feature_indices) == 0:
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    allowed = set(train_instance_ids)
    matrix = np.stack([
        record["global_features"][feature_indices]
        for record in cache["records"]
        if record["instance_id"] in allowed
    ])
    mean = np.mean(matrix, axis=0).astype(np.float32)
    scale = np.std(matrix, axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


class DeepSetsCandidateDataset(Dataset):
    """按实例过滤缓存候选，并在读取时完成全局特征与回归标签变换。"""

    def __init__(
        self,
        cache: dict[str, Any],
        instance_ids: Sequence[str],
        feature_indices: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
    ):
        self.cache = cache
        allowed = set(instance_ids)
        self.record_indices = [
            index for index, record in enumerate(cache["records"])
            if record["instance_id"] in allowed
        ]
        self.feature_indices = feature_indices
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale

    def __len__(self) -> int:
        """返回当前训练或测试子集中的候选数量。"""
        return len(self.record_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """输入子集下标，输出一个候选及其共享实例静态张量。"""
        record = self.cache["records"][self.record_indices[index]]
        static = self.cache["instances"][record["instance_id"]]
        targets = np.array(record["targets_raw"], dtype=np.float32, copy=True)
        for target_index in POSITIVE_TARGET_INDICES:
            targets[target_index] = np.log1p(max(float(targets[target_index]), 0.0))
        return {
            **static,
            **record,
            "global_features": (
                record["global_features"][self.feature_indices] - self.feature_mean
            ) / self.feature_scale,
            "targets": targets,
        }


class InstanceBatchSampler(Sampler[list[int]]):
    """每批完整保留若干实例的所有候选，使实例内排序损失可以构造比较对。"""

    def __init__(
        self,
        dataset: DeepSetsCandidateDataset,
        *,
        instances_per_batch: int = 2,
        shuffle: bool = True,
        random_seed: int = 260915,
    ):
        self.instances_per_batch = instances_per_batch
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for dataset_index, record_index in enumerate(dataset.record_indices):
            instance_id = dataset.cache["records"][record_index]["instance_id"]
            grouped[instance_id].append(dataset_index)
        self.grouped = grouped

    def set_epoch(self, epoch: int) -> None:
        """输入当前轮次，更新确定性随机种子。"""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        """逐批输出若干完整实例对应的候选下标。"""
        instance_ids = sorted(self.grouped)
        if self.shuffle:
            random.Random(self.random_seed + self.epoch).shuffle(instance_ids)
        for start in range(0, len(instance_ids), self.instances_per_batch):
            selected = instance_ids[start:start + self.instances_per_batch]
            yield [index for instance_id in selected for index in self.grouped[instance_id]]

    def __len__(self) -> int:
        """返回每轮批次数量。"""
        return math.ceil(len(self.grouped) / self.instances_per_batch)


def collate_deepsets(samples: Sequence[dict[str, Any]]) -> DeepSetBatch:
    """
    把不同规模的候选层次集合打包成一个 DeepSetBatch。

    输入候选样本序列；输出通过偏移索引连接边界节点、客户、仓库组和样本的扁平张量。
    """
    boundary_features, boundary_customer = [], []
    customer_static, baseline_assignment, candidate_assignment = [], [], []
    baseline_customer_group, candidate_customer_group = [], []
    baseline_group_features, candidate_group_features = [], []
    baseline_group_sample, candidate_group_sample = [], []
    targets, target_masks, global_features = [], [], []
    right_censored, cost_feasible, cost_feasible_masks = [], [], []
    group_log_phase2, group_time_masks, group_right_censored = [], [], []
    instance_ids = {instance_id: index for index, instance_id in enumerate(sorted({
        sample["instance_id"] for sample in samples
    }))}
    instance_index, candidate_names = [], []

    customer_offset = baseline_group_offset = candidate_group_offset = 0
    for sample_index, sample in enumerate(samples):
        boundary_features.append(sample["boundary_features"])
        boundary_customer.append(sample["boundary_customer"] + customer_offset)
        customer_static.append(sample["customer_static_features"])
        baseline_assignment.append(sample["baseline_assignment_features"])
        candidate_assignment.append(sample["candidate_assignment_features"])
        baseline_customer_group.append(sample["baseline_customer_group"] + baseline_group_offset)
        candidate_customer_group.append(sample["candidate_customer_group"] + candidate_group_offset)
        baseline_group_features.append(sample["baseline_group_features"])
        candidate_group_features.append(sample["candidate_group_features"])
        baseline_group_sample.append(np.full(len(sample["baseline_group_features"]), sample_index))
        candidate_group_sample.append(np.full(len(sample["candidate_group_features"]), sample_index))
        targets.append(sample["targets"])
        target_masks.append(sample["target_mask"])
        global_features.append(sample["global_features"])
        right_censored.append(sample["right_censored"])
        cost_feasible.append(sample["cost_feasible"])
        cost_feasible_masks.append(sample["cost_feasible_mask"])
        group_log_phase2.append(sample["group_log_phase2"])
        group_time_masks.append(sample["group_time_mask"])
        group_right_censored.append(sample["group_right_censored"])
        instance_index.append(instance_ids[sample["instance_id"]])
        candidate_names.append(sample["candidate_name"])

        customer_offset += len(sample["customer_static_features"])
        baseline_group_offset += len(sample["baseline_group_features"])
        candidate_group_offset += len(sample["candidate_group_features"])

    return DeepSetBatch(
        boundary_features=torch.as_tensor(np.concatenate(boundary_features), dtype=torch.float32),
        boundary_customer=torch.as_tensor(np.concatenate(boundary_customer), dtype=torch.long),
        customer_static_features=torch.as_tensor(np.concatenate(customer_static), dtype=torch.float32),
        baseline_assignment_features=torch.as_tensor(np.concatenate(baseline_assignment), dtype=torch.float32),
        candidate_assignment_features=torch.as_tensor(np.concatenate(candidate_assignment), dtype=torch.float32),
        baseline_customer_group=torch.as_tensor(np.concatenate(baseline_customer_group), dtype=torch.long),
        candidate_customer_group=torch.as_tensor(np.concatenate(candidate_customer_group), dtype=torch.long),
        baseline_group_features=torch.as_tensor(np.concatenate(baseline_group_features), dtype=torch.float32),
        candidate_group_features=torch.as_tensor(np.concatenate(candidate_group_features), dtype=torch.float32),
        baseline_group_sample=torch.as_tensor(np.concatenate(baseline_group_sample), dtype=torch.long),
        candidate_group_sample=torch.as_tensor(np.concatenate(candidate_group_sample), dtype=torch.long),
        global_features=torch.as_tensor(np.stack(global_features), dtype=torch.float32),
        targets=torch.as_tensor(np.stack(targets), dtype=torch.float32),
        target_mask=torch.as_tensor(np.stack(target_masks), dtype=torch.bool),
        right_censored=torch.as_tensor(right_censored, dtype=torch.float32),
        cost_feasible=torch.as_tensor(cost_feasible, dtype=torch.float32),
        cost_feasible_mask=torch.as_tensor(cost_feasible_masks, dtype=torch.bool),
        group_log_phase2=torch.as_tensor(np.concatenate(group_log_phase2), dtype=torch.float32),
        group_time_mask=torch.as_tensor(np.concatenate(group_time_masks), dtype=torch.bool),
        group_right_censored=torch.as_tensor(np.concatenate(group_right_censored), dtype=torch.float32),
        instance_index=torch.as_tensor(instance_index, dtype=torch.long),
        instance_ids=tuple(sample["instance_id"] for sample in samples),
        candidate_names=tuple(candidate_names),
    )

