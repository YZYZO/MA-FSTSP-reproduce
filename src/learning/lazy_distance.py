"""为 11K/55K 路网实验提供不物化 N² 矩阵的按需距离接口。"""

from collections import OrderedDict
from collections.abc import Mapping

import networkx as nx
import numpy as np

from utils import haversine


class _DistanceRow(Mapping):
    """
    将固定源节点的一行距离暴露为只读映射。

    输入：父矩阵和源节点。
    输出：支持 `row[target]` 的 Mapping。
    逻辑：具体计算委托给父矩阵，保持与原嵌套字典相同的索引语法。
    """

    def __init__(self, matrix, source):
        """
        保存父矩阵和固定源节点。

        输入：距离矩阵对象及源节点。
        输出：行视图。
        逻辑：行对象本身不复制任何距离数据。
        """
        self.matrix = matrix
        self.source = source

    def __getitem__(self, target):
        """返回固定源到目标节点的距离。"""
        return self.matrix.distance(self.source, target)

    def __iter__(self):
        """按父矩阵稳定节点顺序迭代所有目标。"""
        return iter(self.matrix.nodes)

    def __len__(self):
        """返回目标节点总数。"""
        return len(self.matrix.nodes)


class _BaseLazyDistanceMatrix(Mapping):
    """
    定义与原 `{source: {target: value}}` 兼容的只读矩阵外层接口。

    输入：路网图。
    输出：通过 `matrix[source][target]` 查询距离。
    逻辑：外层只创建轻量行视图，子类实现实际距离计算。
    """

    def __init__(self, graph):
        """保存稳定节点列表和路网引用。"""
        self.graph = graph
        self.nodes = tuple(graph.nodes)

    def __getitem__(self, source):
        """返回一个固定源节点的只读距离行视图。"""
        if source not in self.graph:
            raise KeyError(source)
        return _DistanceRow(self, source)

    def __iter__(self):
        """迭代全部源节点。"""
        return iter(self.nodes)

    def __len__(self):
        """返回源节点数量。"""
        return len(self.nodes)


class LazyTruckDistanceMatrix(_BaseLazyDistanceMatrix):
    """
    使用有限 LRU 单源最短路行缓存的卡车距离矩阵。

    输入：有向路网和最大缓存行数。
    输出：与原卡车嵌套字典相同的查询接口。
    逻辑：首次查询一个源时运行一次 Dijkstra，并用稠密 NumPy 行控制内存。
    """

    def __init__(self, graph, max_cached_rows=64):
        """
        初始化节点索引、LRU 缓存和遥测计数器。

        输入：路网及最多保留的单源距离行数。
        输出：按需卡车距离矩阵。
        逻辑：55K 路网缓存 64 个 float64 行约占 28 MB，而不是物化全部 N²。
        """
        super().__init__(graph)
        self.max_cached_rows = max_cached_rows
        self.node_indices = {node: index for index, node in enumerate(self.nodes)}
        self.row_cache = OrderedDict()
        self.dijkstra_call_count = 0
        self.row_cache_hit_count = 0

    def _compute_row(self, source):
        """
        运行一次单源 Dijkstra 并压入定长 LRU 缓存。

        输入：源节点。
        输出：按稳定节点索引排列的 NumPy 距离行。
        逻辑：不可达位置保留正无穷；当前 Manhattan 图通常使用强连通分量。
        """
        lengths = nx.single_source_dijkstra_path_length(
            self.graph,
            source,
            weight="weight",
        )
        row = np.full(len(self.nodes), np.inf, dtype=np.float64)
        for target, value in lengths.items():
            row[self.node_indices[target]] = float(value)
        self.row_cache[source] = row
        self.row_cache.move_to_end(source)
        while len(self.row_cache) > self.max_cached_rows:
            self.row_cache.popitem(last=False)
        self.dijkstra_call_count += 1
        return row

    def distance(self, source, target):
        """
        查询一对节点的有向道路最短路距离。

        输入：源节点和目标节点。
        输出：浮点最短路距离。
        逻辑：命中时刷新 LRU 顺序，未命中时计算完整源行以服务后续密集查询。
        """
        if source in self.row_cache:
            row = self.row_cache[source]
            self.row_cache.move_to_end(source)
            self.row_cache_hit_count += 1
        else:
            row = self._compute_row(source)
        return float(row[self.node_indices[target]])

    def telemetry(self):
        """
        返回按需卡车距离计算的缓存遥测。

        输入：无。
        输出：Dijkstra 次数、行命中数、当前和最大缓存行数。
        逻辑：用于判断大地图实验是否因缓存过小发生过多重算。
        """
        return {
            "dijkstra_call_count": self.dijkstra_call_count,
            "row_cache_hit_count": self.row_cache_hit_count,
            "cached_row_count": len(self.row_cache),
            "max_cached_rows": self.max_cached_rows,
        }

    def clear_cache(self):
        """
        清空单源距离行并重置遥测计数。

        输入：无。
        输出：无。
        逻辑：独立方法评估前调用，使大地图策略计时不受前一个方法预热影响。
        """
        self.row_cache.clear()
        self.dijkstra_call_count = 0
        self.row_cache_hit_count = 0


class LazyDroneDistanceMatrix(_BaseLazyDistanceMatrix):
    """
    即时计算球面直线距离的无人机距离矩阵。

    输入：带 `pos` 坐标的路网。
    输出：与原无人机嵌套字典相同的查询接口。
    逻辑：单次 haversine 成本很低，不缓存 N×客户规模结果以控制 55K 内存。
    """

    def distance(self, source, target):
        """
        即时计算两个节点的球面直线距离。

        输入：源节点和目标节点。
        输出：浮点 haversine 距离。
        逻辑：完全复用原 `problem._pairwise_distance` 的无人机距离定义。
        """
        return float(
            haversine(
                self.graph.nodes[source]["pos"],
                self.graph.nodes[target]["pos"],
            )
        )


def build_lazy_distance(graph, max_cached_truck_rows=64):
    """
    为大型路网构造按需卡车和无人机距离对象。

    输入：标准化路网和卡车 LRU 最大行数。
    输出：含 `truck` 与 `drone` 的兼容距离字典。
    逻辑：调用方无需修改现有算法中的 `distance[type][u][v]` 访问方式。
    """
    return {
        "truck": LazyTruckDistanceMatrix(
            graph,
            max_cached_rows=max_cached_truck_rows,
        ),
        "drone": LazyDroneDistanceMatrix(graph),
    }
