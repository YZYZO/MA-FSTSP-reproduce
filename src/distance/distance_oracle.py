"""
本模块提供与原嵌套距离字典兼容的只读距离查询接口。

阶段 1 仅实现两类距离后端：
1. `EagerDistanceMatrix`：为小图物化卡车全对最短距离，用作回归基线。
2. `GeographicDistanceMatrix`：只保存节点坐标，按需计算无人机直线距离。

H2H 原生后端将在后续阶段实现；当前遇到需要 H2H 的图时会明确报错，
不会静默退回逐次 Dijkstra 或逐渐缓存成全对矩阵。
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Any

import networkx as nx

from config import DISTANCE_BACKEND, EAGER_DISTANCE_MAX_NODES
from utils import haversine


SUPPORTED_DISTANCE_BACKENDS = frozenset({'auto', 'eager', 'h2h'})


class UnsupportedDistanceOperation(TypeError):
    """表示调用方尝试遍历、修改或物化只读距离矩阵。"""


def _normalize_node_id(node: Any) -> Any:
    """
    将 NumPy 整数等实现了整数协议的节点编号转换为 Python `int`。

    输入：
    - node: 调用方传入的节点编号，也允许保留非整数哈希节点标签。

    输出：
    - 整数节点返回 Python `int`；其他节点标签保持不变。

    实现逻辑：
    - 优先调用 `operator.index`，从而兼容 `numpy.int64` 等整数类型；
      不支持整数协议的对象原样返回，由后续成员检查给出明确异常。
    """
    try:
        return int(operator.index(node))
    except TypeError:
        return node


class ReadOnlyDistanceRow:
    """
    保存固定源节点，并将第二层下标查询转发给所属距离矩阵。

    输入：
    - matrix: 实际执行查询的只读距离矩阵。
    - source: 已经过矩阵校验的源节点。

    输出：
    - 行代理对象；使用 `row[target]` 时返回对应距离。
    """

    __slots__ = ('_matrix', '_source')

    def __init__(self, matrix: 'ReadOnlyDistanceMatrix', source: Any) -> None:
        self._matrix = matrix
        self._source = source

    def __getitem__(self, target: Any) -> float:
        """
        查询固定源节点到目标节点的距离并返回 Python `float`。

        输入：
        - target: 目标节点编号。

        输出：
        - `source -> target` 的距离。
        """
        return self._matrix.query(self._source, target)

    def __setitem__(self, target: Any, value: Any) -> None:
        """拒绝修改距离值；输入不产生输出，只抛出只读异常。"""
        raise UnsupportedDistanceOperation('距离矩阵是只读对象，不支持赋值。')

    def __delitem__(self, target: Any) -> None:
        """拒绝删除距离值；输入不产生输出，只抛出只读异常。"""
        raise UnsupportedDistanceOperation('距离矩阵是只读对象，不支持删除。')

    def __iter__(self):
        """拒绝遍历整行，防止调用方隐式物化所有目标节点距离。"""
        raise UnsupportedDistanceOperation('距离行不支持遍历，请使用 row[target] 查询单个距离。')

    def __len__(self) -> int:
        """拒绝把距离行当作完整映射使用。"""
        raise UnsupportedDistanceOperation('距离行不提供长度，避免误认为整行已物化。')

    def keys(self):
        """拒绝枚举目标节点键。"""
        raise UnsupportedDistanceOperation('距离行不支持 keys()，请按目标节点查询。')

    def values(self):
        """拒绝枚举整行距离值。"""
        raise UnsupportedDistanceOperation('距离行不支持 values()，请按目标节点查询。')

    def items(self):
        """拒绝枚举整行节点对与距离。"""
        raise UnsupportedDistanceOperation('距离行不支持 items()，请按目标节点查询。')


class ReadOnlyDistanceMatrix:
    """
    为距离后端提供统一的只读双下标协议和节点校验。

    输入：
    - nodes: 后端能够查询的节点标签集合。

    输出：
    - 抽象只读矩阵；子类需要实现 `query`。
    """

    def __init__(self, nodes) -> None:
        # `_nodes` 是后端认可的节点集合，用于在进入原生层前拦截未知编号。
        self._nodes = frozenset(_normalize_node_id(node) for node in nodes)

    @property
    def node_count(self) -> int:
        """返回可查询节点数量，不触发任何距离计算。"""
        return len(self._nodes)

    def _validate_node(self, node: Any, role: str) -> Any:
        """
        规范化并校验单个源点或终点编号。

        输入：
        - node: 待校验节点。
        - role: 用于错误信息的“源”或“目标”说明。

        输出：
        - 规范化后的节点标签。
        """
        normalized = _normalize_node_id(node)
        if normalized not in self._nodes:
            raise KeyError(f'未知{role}节点 {node!r}；距离后端包含 {self.node_count} 个节点。')
        return normalized

    def __getitem__(self, source: Any) -> ReadOnlyDistanceRow:
        """
        返回固定源节点的轻量行代理，不计算或物化整行距离。

        输入：
        - source: 源节点编号。

        输出：
        - `ReadOnlyDistanceRow` 行代理。
        """
        normalized = self._validate_node(source, '源')
        return ReadOnlyDistanceRow(self, normalized)

    def __setitem__(self, source: Any, value: Any) -> None:
        """拒绝替换距离行。"""
        raise UnsupportedDistanceOperation('距离矩阵是只读对象，不支持赋值。')

    def __delitem__(self, source: Any) -> None:
        """拒绝删除距离行。"""
        raise UnsupportedDistanceOperation('距离矩阵是只读对象，不支持删除。')

    def __iter__(self):
        """拒绝遍历全部源节点，避免形成全矩阵操作。"""
        raise UnsupportedDistanceOperation('距离矩阵不支持遍历，请使用 matrix[source][target]。')

    def __len__(self) -> int:
        """拒绝把距离矩阵当作已物化映射使用。"""
        raise UnsupportedDistanceOperation('距离矩阵不提供长度，请读取 node_count。')

    def keys(self):
        """拒绝枚举全部源节点。"""
        raise UnsupportedDistanceOperation('距离矩阵不支持 keys()。')

    def values(self):
        """拒绝枚举全部距离行。"""
        raise UnsupportedDistanceOperation('距离矩阵不支持 values()。')

    def items(self):
        """拒绝枚举全部节点对。"""
        raise UnsupportedDistanceOperation('距离矩阵不支持 items()。')

    def query(self, source: Any, target: Any) -> float:
        """
        查询一个有向节点对距离。

        输入：
        - source: 源节点。
        - target: 目标节点。

        输出：
        - Python `float` 距离；具体计算由子类实现。
        """
        raise NotImplementedError


class EagerDistanceMatrix(ReadOnlyDistanceMatrix):
    """
    保存小图全对最短距离，并通过只读双下标接口提供查询。

    输入：
    - distances: `source -> target -> distance` 嵌套映射。

    输出：
    - 可用于回归测试和小图求解的 eager 距离矩阵。
    """

    def __init__(self, distances: Mapping[Any, Mapping[Any, float]]) -> None:
        if not isinstance(distances, Mapping):
            raise TypeError('distances 必须是嵌套映射。')

        # `_distances` 保留小图物化结果；每个值转换为 float 以稳定公开接口。
        normalized_distances = {
            _normalize_node_id(source): {
                _normalize_node_id(target): float(distance)
                for target, distance in row.items()
            }
            for source, row in distances.items()
        }
        super().__init__(normalized_distances.keys())
        self._distances = normalized_distances

    @classmethod
    def from_graph(cls, graph: nx.Graph, weight: str = 'weight') -> 'EagerDistanceMatrix':
        """
        在小图上运行全对 Dijkstra 并构造 eager 距离矩阵。

        输入：
        - graph: 加权路网图。
        - weight: NetworkX 使用的边权字段。

        输出：
        - `EagerDistanceMatrix`。

        实现逻辑：
        - 仅用于受规模阈值保护的小图；先计算所有源点最短路，查询时再走代理。
        """
        return cls(dict(nx.all_pairs_dijkstra_path_length(graph, weight=weight)))

    def query(self, source: Any, target: Any) -> float:
        """
        从已物化的小图结果中读取一个有向最短距离。

        输入：源节点与目标节点。
        输出：Python `float`；不可达时抛出明确 `ValueError`。
        """
        normalized_source = self._validate_node(source, '源')
        normalized_target = self._validate_node(target, '目标')
        try:
            return float(self._distances[normalized_source][normalized_target])
        except KeyError as exc:
            raise ValueError(
                f'节点 {normalized_source!r} 无法到达节点 {normalized_target!r}。'
            ) from exc


class GeographicDistanceMatrix(ReadOnlyDistanceMatrix):
    """
    仅保存节点坐标，并按需计算无人机地理距离。

    输入：
    - graph: 节点必须包含统一的 `pos=[lon, lat]` 属性。

    输出：
    - 不包含任何节点对矩阵的坐标距离查询对象。
    """

    def __init__(self, graph: nx.Graph) -> None:
        # `_coordinates` 只占 O(n) 空间，并复制为不可变二元组以隔离图属性后续修改。
        coordinates = {}
        for node, attributes in graph.nodes(data=True):
            if 'pos' not in attributes:
                raise KeyError(f'节点 {node!r} 缺少 pos=[lon, lat] 坐标。')
            position = attributes['pos']
            if len(position) != 2:
                raise ValueError(f'节点 {node!r} 的 pos 必须恰好包含经度和纬度。')
            coordinates[_normalize_node_id(node)] = (float(position[0]), float(position[1]))

        super().__init__(coordinates.keys())
        self._coordinates = coordinates

    def query(self, source: Any, target: Any) -> float:
        """
        按项目现有 `haversine` 语义计算一个节点对的无人机距离。

        输入：源节点与目标节点。
        输出：单位为公里的 Python `float`。
        """
        normalized_source = self._validate_node(source, '源')
        normalized_target = self._validate_node(target, '目标')
        return float(haversine(
            self._coordinates[normalized_source],
            self._coordinates[normalized_target],
        ))


def _select_backend(graph: nx.Graph, backend: str | None) -> str:
    """
    根据显式参数和图规模选择当前距离后端。

    输入：
    - graph: 待查询路网。
    - backend: `auto`、`eager`、`h2h` 或 `None`。

    输出：
    - 解析后的 `eager` 或 `h2h`。
    """
    selected = DISTANCE_BACKEND if backend is None else backend
    if not isinstance(selected, str):
        raise TypeError('距离后端名称必须是字符串。')
    selected = selected.lower()
    if selected not in SUPPORTED_DISTANCE_BACKENDS:
        supported = ', '.join(sorted(SUPPORTED_DISTANCE_BACKENDS))
        raise ValueError(f'未知距离后端 {selected!r}；可选值为 {supported}。')
    if selected == 'auto':
        return 'eager' if graph.number_of_nodes() <= EAGER_DISTANCE_MAX_NODES else 'h2h'
    return selected


def build_distance_provider(
    graph: nx.Graph,
    backend: str | None = None,
    dataset_name: str | None = None,
    graph_path: str | None = None,
    index_dir: str | None = None,
    builder_path: str | None = None,
    library_path: str | None = None,
) -> dict[str, ReadOnlyDistanceMatrix]:
    """
    根据配置创建卡车和无人机距离提供器。

    输入：
    - graph: 已带 `weight` 边权与 `pos` 节点坐标的路网。
    - backend: `eager`、`h2h`、`auto` 或 `None`；`None` 使用项目配置。
    - dataset_name: 用于错误信息和后续缓存命名的数据集名称。
    - graph_path: 可选源 GraphML 路径，用于读取前后的本机大图保护。
    - index_dir: 可选 H2H 缓存根目录，主要用于隔离测试缓存。
    - builder_path/library_path: 可选原生构建器与查询动态库路径。

    输出：
    - 保持 `distance['truck'][u][v]` 与 `distance['drone'][u][v]` 协议的字典。

    实现逻辑：
    1. `auto` 在小图上选择 eager，超过阈值时选择 H2H。
    2. eager 只允许不超过阈值的小图构建全对距离。
    3. H2H 使用版本化图哈希缓存和只读 mmap，不静默回退 Dijkstra。
    4. 无人机距离始终按需计算，不物化全对矩阵。
    """
    if graph.number_of_nodes() == 0:
        raise ValueError('不能为无节点图构造距离提供器。')

    selected = _select_backend(graph, backend)
    label = dataset_name or '<unnamed>'
    if selected == 'h2h':
        # 延迟导入避免 h2h_backend 复用只读基类时形成模块初始化循环。
        from .h2h_backend import (
            H2HDistanceMatrix,
            enforce_local_graph_guard,
            ensure_h2h_index,
            native_artifact_paths,
        )
        enforce_local_graph_guard(graph=graph, graph_path=graph_path)
        cache = ensure_h2h_index(
            graph,
            dataset_name=label,
            index_dir=index_dir,
            builder_path=builder_path,
        )
        selected_library = library_path or str(native_artifact_paths()[1])
        return {
            'truck': H2HDistanceMatrix(
                cache.index_path,
                library_path=selected_library,
                node_count=cache.node_count,
                graph_hash=cache.graph_hash,
            ),
            'drone': GeographicDistanceMatrix(graph),
        }

    if graph.number_of_nodes() > EAGER_DISTANCE_MAX_NODES:
        raise ValueError(
            f'数据集 {label!r} 含 {graph.number_of_nodes()} 个节点，超过 eager 上限 '
            f'{EAGER_DISTANCE_MAX_NODES}；请使用 H2H 后端。'
        )

    # `truck` 为小图精确 APSP 基线，`drone` 仅保留 O(n) 坐标数据。
    return {
        'truck': EagerDistanceMatrix.from_graph(graph),
        'drone': GeographicDistanceMatrix(graph),
    }
