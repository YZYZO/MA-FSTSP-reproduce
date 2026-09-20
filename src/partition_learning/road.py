"""重建客户边界集合，并为单实例预计算有限终端间的有向道路距离。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import time
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from problem import manhattan


EARTH_RADIUS_KM = 6371.0


def load_road_graph(graph_path):
    """
    读取并规范化一张实验路网。

    输入为 GraphML 路径；输出节点连续编号、边权为公里的 MultiDiGraph。
    具体节点顺序完全复用原实验的 problem.manhattan，保证 NPZ 节点编号仍然有效。
    """
    return manhattan(graph_path)


def _coordinate_array(graph: nx.Graph) -> np.ndarray:
    """输入连续编号路网，输出按节点编号排列的弧度制经纬度矩阵。"""
    coordinates = np.asarray([graph.nodes[node]["pos"] for node in range(len(graph))], dtype=np.float64)
    return np.radians(coordinates)


def build_spatial_sets(
    graph: nx.MultiDiGraph,
    cities: Iterable[int],
    boundary_radius_km: float,
    region_radius_km: float,
    *,
    chunk_size: int = 4096,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    按原论文代码的全局最近客户规则重建边界集合。

    输入为路网、同一实例的全部客户、全局边界半径和单客户服务半径；
    输出 `(boundary, regions)`。前者按最近客户互斥归属，后者按原算法严格小于半径独立覆盖。
    """
    city_list = list(map(int, cities))
    coordinates = _coordinate_array(graph)
    city_coordinates = coordinates[city_list]
    assignments = np.full(len(graph), -1, dtype=np.int32)
    regions = {city: [] for city in city_list}

    # 原实现使用 <= 并在完全相等时让后出现的客户覆盖，因此对反向客户轴取 argmin。
    for start in range(0, len(graph), chunk_size):
        stop = min(len(graph), start + chunk_size)
        points = coordinates[start:stop, None, :]
        targets = city_coordinates[None, :, :]
        x = (targets[..., 0] - points[..., 0]) * np.cos(0.5 * (targets[..., 1] + points[..., 1]))
        y = targets[..., 1] - points[..., 1]
        distances = EARTH_RADIUS_KM * np.sqrt(x * x + y * y)
        reverse_index = np.argmin(distances[:, ::-1], axis=1)
        closest_index = len(city_list) - 1 - reverse_index
        closest_distance = distances[np.arange(stop - start), closest_index]
        accepted = closest_distance <= boundary_radius_km
        assignments[start:stop][accepted] = closest_index[accepted]
        for city_index, city in enumerate(city_list):
            local_nodes = np.flatnonzero(distances[:, city_index] < region_radius_km) + start
            regions[city].extend(local_nodes.astype(int).tolist())

    boundary_mask = np.zeros(len(graph), dtype=bool)
    for source, target in graph.edges():
        owner = assignments[int(source)]
        if owner >= 0 and assignments[int(target)] != owner:
            boundary_mask[int(source)] = True

    boundary = {
        city: np.flatnonzero(boundary_mask & (assignments == city_index)).astype(int).tolist()
        for city_index, city in enumerate(city_list)
    }
    return boundary, regions


def build_boundary_sets(
    graph: nx.MultiDiGraph,
    cities: Iterable[int],
    radius_km: float,
    *,
    chunk_size: int = 4096,
) -> dict[int, list[int]]:
    """
    兼容只需要边界集合的调用。

    输入为路网、客户与半径；输出最近客户规则下的边界节点。
    单客户区域使用相同半径临时计算但不返回，正式实验应直接调用 build_spatial_sets 避免重复扫描。
    """
    boundary, _ = build_spatial_sets(
        graph,
        cities,
        radius_km,
        radius_km,
        chunk_size=chunk_size,
    )
    return boundary


def compare_boundary_sizes(
    boundary: Mapping[int, list[int]],
    expected_sizes: Mapping[int, int],
) -> dict[str, object]:
    """
    比较本地重建边界与 NPZ 保存规模。

    输入为实际集合和历史规模；输出是否完全一致、失配数量及少量示例，供每个实例写入审计记录。
    """
    mismatches = [
        {"city": int(city), "expected": int(expected_sizes[city]), "actual": len(boundary[city])}
        for city in expected_sizes
        if len(boundary.get(city, ())) != int(expected_sizes[city])
    ]
    return {
        "exact": not mismatches,
        "mismatch_count": len(mismatches),
        "examples": mismatches[:10],
    }


def build_road_csr(graph: nx.MultiDiGraph) -> csr_matrix:
    """
    把 MultiDiGraph 转成 SciPy 有向稀疏矩阵。

    输入为规范路网；输出每对平行边仅保留最小权重的 CSR。
    该矩阵让 11K/55K 使用 C 实现批量 Dijkstra，避免全图 N² 距离物化。
    """
    minimum_edges: dict[tuple[int, int], float] = {}
    for source, target, attributes in graph.edges(data=True):
        key = (int(source), int(target))
        weight = float(attributes["weight"])
        previous = minimum_edges.get(key)
        if previous is None or weight < previous:
            minimum_edges[key] = weight
    rows = np.fromiter((edge[0] for edge in minimum_edges), dtype=np.int32)
    columns = np.fromiter((edge[1] for edge in minimum_edges), dtype=np.int32)
    weights = np.fromiter(minimum_edges.values(), dtype=np.float64)
    return csr_matrix((weights, (rows, columns)), shape=(len(graph), len(graph)))


@dataclass
class TerminalDistanceData:
    """保存单实例有限终端的稠密有向道路距离和预计算耗时。"""

    terminals: tuple[int, ...]
    matrix: np.ndarray
    preprocessing_seconds: float


def precompute_terminal_distances(
    road_csr: csr_matrix,
    terminals: Iterable[int],
    *,
    batch_size: int = 128,
) -> TerminalDistanceData:
    """
    仅物化当前实例会用到的终端间有向最短路。

    输入为整图 CSR 和仓库/客户/边界节点集合；输出 m×m 距离矩阵及耗时。
    每批 Dijkstra 的 n 列结果在抽取终端列后立即释放，峰值内存约为 batch_size×n。
    """
    terminal_nodes = tuple(sorted(set(map(int, terminals))))
    matrix = np.empty((len(terminal_nodes), len(terminal_nodes)), dtype=np.float64)
    started_at = time.perf_counter()
    target_columns = np.asarray(terminal_nodes, dtype=np.int32)
    for start in range(0, len(terminal_nodes), batch_size):
        stop = min(len(terminal_nodes), start + batch_size)
        sources = np.asarray(terminal_nodes[start:stop], dtype=np.int32)
        full_rows = dijkstra(road_csr, directed=True, indices=sources, return_predecessors=False)
        matrix[start:stop] = full_rows[:, target_columns]
    return TerminalDistanceData(
        terminals=terminal_nodes,
        matrix=matrix,
        preprocessing_seconds=time.perf_counter() - started_at,
    )


class _TerminalRow(Mapping):
    """把固定源节点的一行暴露成与原嵌套距离字典兼容的只读映射。"""

    def __init__(self, parent: "TerminalRoadMatrix", source: int):
        self.parent = parent
        self.source = int(source)

    def __getitem__(self, target: int) -> float:
        return self.parent.query(self.source, int(target))

    def __iter__(self):
        return iter(self.parent.terminals)

    def __len__(self) -> int:
        return len(self.parent.terminals)


class TerminalRoadMatrix(Mapping):
    """为原算法提供 distance['truck'][u][v] 兼容访问。"""

    def __init__(self, data: TerminalDistanceData):
        self.terminals = data.terminals
        self.matrix = data.matrix
        self.indices = {node: index for index, node in enumerate(data.terminals)}

    def query(self, source: int, target: int) -> float:
        """输入两个预计算终端，输出有向道路最短路距离。"""
        return float(self.matrix[self.indices[source], self.indices[target]])

    def pairwise(self, sources: Iterable[int], targets: Iterable[int]) -> np.ndarray:
        """输入源、目标节点序列，输出保持方向的二维距离子矩阵。"""
        source_indices = np.fromiter((self.indices[int(node)] for node in sources), dtype=np.int32)
        target_indices = np.fromiter((self.indices[int(node)] for node in targets), dtype=np.int32)
        return self.matrix[np.ix_(source_indices, target_indices)]

    def __getitem__(self, source: int) -> _TerminalRow:
        return _TerminalRow(self, int(source))

    def __iter__(self):
        return iter(self.terminals)

    def __len__(self) -> int:
        return len(self.terminals)


class _DroneRow(Mapping):
    """固定无人机源节点的即时距离行。"""

    def __init__(self, parent: "DroneDistanceMatrix", source: int):
        self.parent = parent
        self.source = int(source)

    def __getitem__(self, target: int) -> float:
        return self.parent.query(self.source, int(target))

    def __iter__(self):
        return iter(range(len(self.parent.coordinates)))

    def __len__(self) -> int:
        return len(self.parent.coordinates)


class DroneDistanceMatrix(Mapping):
    """按原 equirectangular 公式即时计算无人机直线距离。"""

    def __init__(self, graph: nx.Graph):
        self.coordinates = _coordinate_array(graph)

    def query(self, source: int, target: int) -> float:
        """输入路网节点编号，输出公里制无人机直线距离。"""
        first = self.coordinates[source]
        second = self.coordinates[target]
        x = (second[0] - first[0]) * math.cos(0.5 * (second[1] + first[1]))
        y = second[1] - first[1]
        return EARTH_RADIUS_KM * math.sqrt(x * x + y * y)

    def pairwise(self, sources: Iterable[int], targets: Iterable[int]) -> np.ndarray:
        """输入源、目标节点序列，向量化输出公里制空中距离矩阵。"""
        source_coordinates = self.coordinates[np.asarray(list(sources), dtype=np.int32)][:, None, :]
        target_coordinates = self.coordinates[np.asarray(list(targets), dtype=np.int32)][None, :, :]
        x = (target_coordinates[..., 0] - source_coordinates[..., 0]) * np.cos(
            0.5 * (target_coordinates[..., 1] + source_coordinates[..., 1])
        )
        y = target_coordinates[..., 1] - source_coordinates[..., 1]
        return EARTH_RADIUS_KM * np.sqrt(x * x + y * y)

    def __getitem__(self, source: int) -> _DroneRow:
        return _DroneRow(self, int(source))

    def __iter__(self):
        return iter(range(len(self.coordinates)))

    def __len__(self) -> int:
        return len(self.coordinates)
