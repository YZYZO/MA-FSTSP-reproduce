"""为已知道路网内的候选划分GNN构造客户道路图缓存与批次。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .dataset import ExperimentInstance, load_instances
from .deep_sets import DeepSetBatch
from .deep_sets_data import collate_deepsets
from .road import (
    DroneDistanceMatrix,
    TerminalRoadMatrix,
    build_road_csr,
    load_road_graph,
    precompute_terminal_distances,
)


ROAD_EDGE_FEATURE_NAMES = (
    "log_road_forward",
    "log_road_reverse",
    "road_asymmetry",
    "log_air_distance",
    "log_road_stretch",
)


@dataclass
class PartitionGraphBatch(DeepSetBatch):
    """在层次集合批次基础上增加打包后的有向客户道路边。"""

    road_edge_index: Tensor
    road_edge_features: Tensor

    def to(self, device: torch.device | str) -> "PartitionGraphBatch":
        """输入目标设备，输出所有张量迁移后的候选图批次。"""
        values = {
            name: value.to(device) if isinstance(value, Tensor) else value
            for name, value in self.__dict__.items()
        }
        return PartitionGraphBatch(**values)


def _road_scale(instance: ExperimentInstance, truck: TerminalRoadMatrix) -> float:
    """输入实例和终端路网距离，输出仓库—客户距离的稳健归一化尺度。"""
    values = []
    for depot in instance.depots:
        for city in instance.cities:
            values.extend((truck.query(depot, city), truck.query(city, depot)))
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return max(float(np.median(finite)) if len(finite) else 1.0, 1e-6)


def build_customer_road_edges(
    instance: ExperimentInstance,
    truck: TerminalRoadMatrix,
    drone: DroneDistanceMatrix,
    *,
    k_neighbors: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构造客户之间的稀疏有向道路近邻图。

    输入实例、道路/空中距离和近邻数；输出局部客户下标边及方向相关的五维边特征。
    同时保留每个客户的出向和入向近邻，避免单向道路导致局部结构缺失。
    """
    cities = tuple(map(int, instance.cities))
    customer_count = len(cities)
    road = truck.pairwise(cities, cities).astype(np.float64, copy=True)
    scale = _road_scale(instance, truck)
    finite = road[np.isfinite(road)]
    replacement = max(float(np.max(finite)) if len(finite) else scale, 10.0 * scale)
    road[~np.isfinite(road)] = replacement
    np.fill_diagonal(road, np.inf)

    edge_pairs: set[tuple[int, int]] = set()
    neighbor_count = min(max(int(k_neighbors), 1), max(customer_count - 1, 1))
    for customer in range(customer_count):
        outgoing = np.argsort(road[customer])[:neighbor_count]
        incoming = np.argsort(road[:, customer])[:neighbor_count]
        edge_pairs.update((customer, int(target)) for target in outgoing if target != customer)
        edge_pairs.update((int(source), customer) for source in incoming if source != customer)

    ordered_edges = sorted(edge_pairs)
    edge_index = np.asarray(ordered_edges, dtype=np.int64).T
    air = drone.pairwise(cities, cities)
    features = []
    for source, target in ordered_edges:
        forward = float(road[source, target])
        reverse = float(road[target, source])
        air_distance = max(float(air[source, target]), 1e-6)
        features.append([
            math.log1p(forward / scale),
            math.log1p(reverse / scale),
            abs(forward - reverse) / max(0.5 * (forward + reverse), 1e-6),
            math.log1p(air_distance / scale),
            math.log1p(0.5 * (forward + reverse) / air_distance),
        ])
    return edge_index, np.asarray(features, dtype=np.float32)


def _npz_path_map(result_root: str | Path) -> dict[str, Path]:
    """输入历史结果根目录，输出以文件主干索引的NPZ路径。"""
    return {path.stem: path.resolve() for path in Path(result_root).rglob("*.npz")}


def build_partition_gnn_cache(
    deepsets_cache: dict[str, Any],
    result_root: str | Path,
    output_path: str | Path,
    *,
    k_neighbors: int = 8,
    distance_batch_size: int = 128,
) -> dict[str, Any]:
    """
    在现有Deep Sets缓存上补充客户道路图并保存GNN缓存。

    输入已有集合/标签缓存和原始NPZ根目录；只为缓存中的实例计算客户与仓库终端距离，
    不重建边界集合、不调用求解器，也不会读取未出现在缓存中的55K实例。
    """
    npz_paths = _npz_path_map(result_root)
    graph_cache: dict[str, Any] = {}
    road_csr_cache: dict[str, Any] = {}
    augmented_instances: dict[str, dict[str, Any]] = {}

    instance_items = sorted(deepsets_cache["instances"].items())
    for position, (instance_id, static) in enumerate(instance_items, start=1):
        source_name = str(static["source_name"])
        if source_name not in npz_paths:
            raise FileNotFoundError(f"未在 {result_root} 找到 {source_name}.npz")
        instance = load_instances(
            npz_paths[source_name], [int(static["instance_index"])]
        )[0]
        if instance.graph_name not in graph_cache:
            graph = load_road_graph(instance.graph_path)
            graph_cache[instance.graph_name] = graph
            road_csr_cache[instance.graph_name] = build_road_csr(graph)
        graph = graph_cache[instance.graph_name]
        terminals = tuple(instance.depots) + tuple(instance.cities)
        terminal_data = precompute_terminal_distances(
            road_csr_cache[instance.graph_name],
            terminals,
            batch_size=distance_batch_size,
        )
        edge_index, edge_features = build_customer_road_edges(
            instance,
            TerminalRoadMatrix(terminal_data),
            DroneDistanceMatrix(graph),
            k_neighbors=k_neighbors,
        )
        augmented_instances[instance_id] = {
            **static,
            "road_edge_index": edge_index,
            "road_edge_features": edge_features,
        }
        print(
            f"[GNN cache] {position}/{len(instance_items)} {instance_id} "
            f"edges={edge_index.shape[1]}",
            flush=True,
        )

    feature_names = dict(deepsets_cache["feature_names"])
    feature_names["road_edge"] = ROAD_EDGE_FEATURE_NAMES
    cache = {
        **deepsets_cache,
        "format_version": 2,
        "cache_type": "partition_gnn",
        "k_neighbors": int(k_neighbors),
        "feature_names": feature_names,
        "instances": augmented_instances,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return cache


def load_partition_gnn_cache(path: str | Path) -> dict[str, Any]:
    """输入可信的本地GNN缓存路径，输出CPU数组和张量组成的缓存字典。"""
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def subsample_partition_gnn_cache(
    cache: dict[str, Any],
    output_path: str | Path,
    *,
    k_neighbors: int,
) -> dict[str, Any]:
    """
    从较密的客户道路近邻图无损抽取较小k的出向/入向近邻缓存。

    输入已有GNN缓存、输出路径和目标k；输出保留相同标签与候选、仅减少道路边的新缓存。
    第一维道路边特征是单调变换后的正向距离，因此排序与原始构图完全一致。
    """
    source_k = int(cache["k_neighbors"])
    if k_neighbors > source_k:
        raise ValueError("子采样目标k不能大于源缓存k。")
    instances: dict[str, dict[str, Any]] = {}
    for instance_id, static in cache["instances"].items():
        edge_index = np.asarray(static["road_edge_index"], dtype=np.int64)
        edge_features = np.asarray(static["road_edge_features"], dtype=np.float32)
        source, target = edge_index
        customer_count = int(static["customer_count"])
        keep: set[int] = set()
        for customer in range(customer_count):
            outgoing = np.flatnonzero(source == customer)
            incoming = np.flatnonzero(target == customer)
            keep.update(outgoing[np.argsort(edge_features[outgoing, 0])[:k_neighbors]].tolist())
            keep.update(incoming[np.argsort(edge_features[incoming, 0])[:k_neighbors]].tolist())
        selected = np.asarray(sorted(keep), dtype=np.int64)
        instances[instance_id] = {
            **static,
            "road_edge_index": edge_index[:, selected],
            "road_edge_features": edge_features[selected],
        }
    result = {**cache, "k_neighbors": int(k_neighbors), "instances": instances}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result


def collate_partition_graphs(samples: Sequence[dict[str, Any]]) -> PartitionGraphBatch:
    """
    把不同实例的候选道路图打包为一个批次。

    输入候选样本；输出复用Deep Sets层次张量并按客户偏移拼接道路边的批次。
    """
    base_batch = collate_deepsets(samples)
    edge_indices, edge_features = [], []
    customer_offset = 0
    for sample in samples:
        edge_indices.append(np.asarray(sample["road_edge_index"], dtype=np.int64) + customer_offset)
        edge_features.append(np.asarray(sample["road_edge_features"], dtype=np.float32))
        customer_offset += len(sample["customer_static_features"])
    return PartitionGraphBatch(
        **base_batch.__dict__,
        road_edge_index=torch.as_tensor(np.concatenate(edge_indices, axis=1), dtype=torch.long),
        road_edge_features=torch.as_tensor(np.concatenate(edge_features), dtype=torch.float32),
    )
