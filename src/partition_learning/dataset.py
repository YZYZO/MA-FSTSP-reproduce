"""读取八个历史 NPZ，并把原始 MST 分区恢复成统一实例对象。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from config import MANHATTAN1k_GRAPH_PATH, MANHATTAN11k_GRAPH_PATH, MANHATTAN55k_GRAPH_PATH


@dataclass(frozen=True)
class ExperimentInstance:
    """
    保存一个可重放的历史实例。

    输入由 NPZ 路径和实例下标共同确定；输出字段包含路网、参数、原始分区及历史标签。
    该对象只保留轻量元数据，不持有整张路网或距离矩阵。
    """

    instance_id: str
    source_path: Path
    source_name: str
    graph_path: Path
    graph_name: str
    instance_index: int
    depots: tuple[int, ...]
    cities: tuple[int, ...]
    partition: dict[int, tuple[int, ...]]
    boundary_sizes: dict[int, int]
    drones_per_truck: int
    drone_limit: float
    drone_speed: float
    theta: tuple[float, float]
    historical_cost: float
    historical_phase2_seconds: float


def discover_result_files(result_root: str | Path) -> list[Path]:
    """
    在给定目录中发现本实验支持的八个 NPZ。

    输入为 260826 结果根目录；输出按路网名称和客户规模稳定排序的路径列表。
    逻辑上排除其他 NPZ，并检查 1K 三个、11K 三个、55K 两个文件是否齐全。
    """
    root = Path(result_root)
    files = sorted(
        path for path in root.rglob("*.npz")
        if any(token in path.name for token in ("manhattan_1k", "boston_11k", "manhattan_55k"))
    )
    if len(files) != 8:
        raise ValueError(f"期望发现 8 个目标 NPZ，实际发现 {len(files)} 个：{root}")
    return files


def select_instance_indices(instance_count: int, count: int) -> list[int]:
    """
    从一个 NPZ 中按等距分位点选择实例。

    输入为总实例数和目标数量；输出不重复且覆盖首尾的稳定下标。
    该选择不查看成本、时间或分区结果，避免 Oracle 阶段发生结果导向抽样。
    """
    if count >= instance_count:
        return list(range(instance_count))
    return np.rint(np.linspace(0, instance_count - 1, count)).astype(int).tolist()


def oracle_subset_indices(supervised_indices: Iterable[int]) -> list[int]:
    """
    从每个文件的监督样本中选出三个 Oracle 实例。

    输入为第二轮的有序下标；输出首、中、尾三个下标，确保第一轮结果可直接回流第二轮。
    """
    values = list(supervised_indices)
    return [values[0], values[len(values) // 2], values[-1]]


def _graph_metadata(npz_name: str) -> tuple[str, Path]:
    """根据 NPZ 文件名返回规范路网名称和本地 GraphML 路径。"""
    if "manhattan_1k" in npz_name:
        return "manhattan_1k", MANHATTAN1k_GRAPH_PATH
    if "boston_11k" in npz_name:
        return "boston_11k", MANHATTAN11k_GRAPH_PATH
    if "manhattan_55k" in npz_name:
        return "manhattan_55k", MANHATTAN55k_GRAPH_PATH
    raise ValueError(f"无法从文件名识别路网：{npz_name}")


def _decode_partition(payload: str) -> dict[int, tuple[int, ...]]:
    """把 phase1_groups_json 的单实例文本恢复成以仓库节点为键的客户元组。"""
    records = json.loads(payload)
    return {
        int(record["depot_node"]): tuple(map(int, record["customers"]))
        for record in records
    }


def _decode_boundary_sizes(payload: str) -> dict[int, int]:
    """从阶段 2 记录恢复每位客户对应的固定边界集合大小。"""
    sizes: dict[int, int] = {}
    for record in json.loads(payload):
        customers = list(map(int, record["customers"]))
        convex_sizes = list(map(int, record["convex_set_sizes"]))
        for city, size in zip(customers, convex_sizes[1:]):
            sizes[city] = size
    return sizes


def load_instances(npz_path: str | Path, indices: Iterable[int]) -> list[ExperimentInstance]:
    """
    从一个 NPZ 读取指定实例。

    输入为 NPZ 路径和实例下标；输出轻量 ExperimentInstance 列表。
    逻辑上复用原始 MST 分区、边界规模、问题参数和历史基线标签，不读取 PDF 或重跑原算法。
    """
    path = Path(npz_path)
    graph_name, graph_path = _graph_metadata(path.name)
    instances: list[ExperimentInstance] = []
    with np.load(path, allow_pickle=True) as data:
        for index in indices:
            partition = _decode_partition(str(data["phase1_groups_json"][index]))
            boundary_sizes = _decode_boundary_sizes(str(data["phase2_orders_json"][index]))
            source_stem = path.stem
            instances.append(ExperimentInstance(
                instance_id=f"{source_stem}__{int(index):03d}",
                source_path=path.resolve(),
                source_name=source_stem,
                graph_path=Path(graph_path).resolve(),
                graph_name=graph_name,
                instance_index=int(index),
                depots=tuple(map(int, data["depots"][index])),
                cities=tuple(map(int, data["cities"][index])),
                partition=partition,
                boundary_sizes=boundary_sizes,
                drones_per_truck=int(data["drones_per_truck"]),
                drone_limit=float(data["drone_limit"]),
                drone_speed=float(data["drone_speed"]),
                theta=tuple(map(float, data["theta"])),
                historical_cost=float(data["stsp_cost"][index]),
                historical_phase2_seconds=float(np.sum(data["phase2_time"][index])),
            ))
    return instances
