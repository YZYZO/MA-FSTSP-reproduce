"""从未标注实例中选择结构上远离现有训练集的独立路网实例。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import discover_result_files, load_instances
from .road import load_road_graph


def _summary_features(instance, coordinates: np.ndarray) -> np.ndarray:
    """输入历史实例和标准化道路坐标，输出不依赖二三阶段标签的结构摘要。"""
    group_sizes = np.asarray([len(group) for group in instance.partition.values()], dtype=float)
    boundary_sizes = np.asarray(list(instance.boundary_sizes.values()), dtype=float)
    city_coordinates = coordinates[np.asarray(instance.cities, dtype=int)]
    depot_coordinates = coordinates[np.asarray(instance.depots, dtype=int)]
    return np.asarray([
        np.mean(group_sizes), np.std(group_sizes), np.max(group_sizes), np.min(group_sizes),
        np.mean(boundary_sizes), np.std(boundary_sizes), np.max(boundary_sizes),
        *np.mean(city_coordinates, axis=0), *np.std(city_coordinates, axis=0),
        *np.mean(depot_coordinates, axis=0), *np.std(depot_coordinates, axis=0),
    ], dtype=np.float64)


def farthest_point_indices(
    features: np.ndarray,
    existing_indices: list[int],
    selection_count: int,
) -> list[int]:
    """输入特征和已标注下标，输出标准化空间中逐步最远的新实例下标。"""
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (features - mean) / scale
    selected = list(map(int, existing_indices))
    available = [index for index in range(len(features)) if index not in selected]
    output = []
    for _ in range(min(selection_count, len(available))):
        reference = standardized[selected] if selected else standardized[[available[0]]]
        distances = [
            float(np.min(np.linalg.norm(reference - standardized[index], axis=1)))
            for index in available
        ]
        choice_position = int(np.argmax(distances))
        choice = available.pop(choice_position)
        output.append(choice)
        selected.append(choice)
    return output


def select_diverse_unlabelled_instances(
    result_root: str | Path,
    cache: dict[str, Any],
    *,
    instances_per_source: int = 1,
) -> dict[str, Any]:
    """
    按路网与客户规模分别选择未标注的最远点实例。

    输入原始NPZ根目录和当前GNN缓存；输出可供算法真值实验直接读取的下标清单。
    本地显式排除55K，仅遍历1K和11K的六个来源文件。
    """
    existing: dict[str, list[int]] = {}
    for static in cache["instances"].values():
        existing.setdefault(str(static["source_name"]), []).append(
            int(static["instance_index"])
        )
    graph_coordinates: dict[str, np.ndarray] = {}
    indices_by_source: dict[str, list[int]] = {}
    entries = []
    for path in discover_result_files(result_root):
        if "manhattan_55k" in path.name:
            continue
        with np.load(path, allow_pickle=True) as data:
            total = int(len(data["instance_indices"]))
        instances = load_instances(path, range(total))
        graph_name = instances[0].graph_name
        if graph_name not in graph_coordinates:
            graph = load_road_graph(instances[0].graph_path)
            coordinates = np.asarray(
                [graph.nodes[node]["pos"] for node in range(len(graph))], dtype=np.float64
            )
            scale = np.std(coordinates, axis=0)
            scale[scale < 1e-9] = 1.0
            graph_coordinates[graph_name] = (coordinates - np.mean(coordinates, axis=0)) / scale
        feature_matrix = np.stack([
            _summary_features(instance, graph_coordinates[graph_name])
            for instance in instances
        ])
        labelled = sorted(set(existing.get(path.stem, [])))
        selected = farthest_point_indices(
            feature_matrix, labelled, instances_per_source
        )
        indices_by_source[path.stem] = selected
        for index in selected:
            reference = feature_matrix[labelled] if labelled else feature_matrix[[index]]
            entries.append({
                "source_name": path.stem,
                "graph_name": graph_name,
                "customer_count": len(instances[index].cities),
                "instance_index": index,
                "instance_id": instances[index].instance_id,
                "distance_to_existing_raw": float(np.min(
                    np.linalg.norm(reference - feature_matrix[index], axis=1)
                )),
            })
    return {
        "selection_method": "stratified_farthest_point_v1",
        "instances_per_source": int(instances_per_source),
        "excluded_graphs": ["manhattan_55k"],
        "indices_by_source": indices_by_source,
        "instances": entries,
    }
