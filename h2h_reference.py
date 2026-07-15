"""
有向 H2H 的纯 Python 小图参考实现。

本模块实现距离保持消元、分解树、出入向祖先标签、二进制提升 LCA 和
基于 LCA bag 的精确距离查询。实现优先保证公式清晰与可逐项核对，不面向
生产规模；节点数超过配置上限时会在任何消元工作开始前拒绝构建。
"""

from __future__ import annotations

import heapq
import math
import operator
from dataclasses import dataclass
from typing import Any

import networkx as nx

from config import H2H_REFERENCE_MAX_NODES


@dataclass(frozen=True)
class H2HReferenceStats:
    """保存参考索引的规模统计，供测试和原生实现逐项对照。"""

    node_count: int
    edge_count: int
    zero_weight_edges: int
    structural_fill_edges: int
    shortcut_arcs: int
    treewidth: int
    treeheight: int
    label_count: int
    position_count: int


def _as_node_id(node: Any) -> int:
    """
    将 Python/NumPy 整数节点标签转换为 Python `int`。

    输入：待转换节点标签。
    输出：Python 整数；非整数标签抛出 `TypeError`。
    """
    try:
        return int(operator.index(node))
    except TypeError as exc:
        raise TypeError(f'H2H 节点编号必须是整数，收到 {node!r}。') from exc


def normalize_directed_graph(graph: nx.Graph) -> tuple[list[dict[int, float]], int, int]:
    """
    将有向图规范化为按有序节点对保留最小边权的简单邻接表。

    输入：节点连续编号且边含 `weight` 的 DiGraph/MultiDiGraph。
    输出：`(out_edges, finite_edge_count, zero_weight_edge_count)`。

    实现逻辑：
    1. 校验有向性和 `[0, n)` 连续整数编号；
    2. 平行边只保留最小权重，但不合并相反方向；
    3. 拒绝负数及非有限权重，自环经校验后不进入消元邻接表；
    4. 验证规范化图强连通。
    """
    if not graph.is_directed():
        raise ValueError('H2H 只接受有向图，不能把道路方向隐式改成无向。')

    node_count = graph.number_of_nodes()
    if node_count == 0:
        raise ValueError('不能为无节点图构建 H2H 参考索引。')
    normalized_nodes = {_as_node_id(node) for node in graph.nodes}
    expected_nodes = set(range(node_count))
    if normalized_nodes != expected_nodes or len(normalized_nodes) != node_count:
        raise ValueError(f'H2H 节点必须连续编号为 [0, {node_count})。')

    # `out_edges[u][v]` 保存所有平行 `u -> v` 边中的最小有限权重。
    out_edges: list[dict[int, float]] = [dict() for _ in range(node_count)]
    zero_weight_pairs: set[tuple[int, int]] = set()
    for source, target, attributes in graph.edges(data=True):
        source_id = _as_node_id(source)
        target_id = _as_node_id(target)
        if 'weight' not in attributes:
            raise KeyError(f'边 {source_id} -> {target_id} 缺少 weight。')
        weight = float(attributes['weight'])
        if not math.isfinite(weight):
            raise ValueError(f'边 {source_id} -> {target_id} 的权重必须有限。')
        if weight < 0.0:
            raise ValueError(f'边 {source_id} -> {target_id} 含负权 {weight}。')
        if weight == 0.0:
            zero_weight_pairs.add((source_id, target_id))
        if source_id == target_id:
            continue
        previous = out_edges[source_id].get(target_id)
        if previous is None or weight < previous:
            out_edges[source_id][target_id] = weight

    normalized_graph = nx.DiGraph()
    normalized_graph.add_nodes_from(range(node_count))
    for source, row in enumerate(out_edges):
        normalized_graph.add_weighted_edges_from(
            (source, target, weight) for target, weight in row.items()
        )
    if not nx.is_strongly_connected(normalized_graph):
        component_sizes = sorted(
            (len(component) for component in nx.strongly_connected_components(normalized_graph)),
            reverse=True,
        )
        raise ValueError(
            f'H2H 要求强连通图；当前共有 {len(component_sizes)} 个强连通分量，'
            f'最大规模为 {component_sizes[0]}。'
        )

    edge_count = sum(len(row) for row in out_edges)
    return out_edges, edge_count, len(zero_weight_pairs)


class DirectedH2HReference:
    """
    为不超过配置上限的小型有向图构建可核对的 H2H 索引。

    输入：
    - graph: 已带非负 `weight` 的强连通有向图。
    - max_nodes: 可选的更严格节点上限，不能超过项目配置上限。

    输出：构建完成的参考索引对象，可调用 `query(source, target)`。
    """

    def __init__(self, graph: nx.Graph, max_nodes: int | None = None) -> None:
        configured_limit = H2H_REFERENCE_MAX_NODES
        selected_limit = configured_limit if max_nodes is None else int(max_nodes)
        if selected_limit <= 0 or selected_limit > configured_limit:
            raise ValueError(
                f'max_nodes 必须位于 [1, {configured_limit}]，收到 {selected_limit}。'
            )
        if graph.number_of_nodes() > selected_limit:
            raise ValueError(
                f'Python H2H 参考实现最多允许 {selected_limit} 个节点，'
                f'当前图含 {graph.number_of_nodes()} 个节点。'
            )

        normalized_edges, edge_count, zero_weight_edges = normalize_directed_graph(graph)
        self.node_count = len(normalized_edges)
        self._original_edges = [dict(row) for row in normalized_edges]

        elimination = self._eliminate(normalized_edges)
        self.rank = elimination['rank']
        self.bag_neighbors = elimination['bag_neighbors']
        self.star_out = elimination['star_out']
        self.star_in = elimination['star_in']
        self.parent, self.depth, root = self._build_decomposition_tree()
        self.root = root
        self.dis_out, self.dis_in, self.positions = self._build_labels()
        self.up = self._build_lca_table()

        label_count = sum(len(row) for row in self.dis_out)
        position_count = sum(len(row) for row in self.positions)
        self.stats = H2HReferenceStats(
            node_count=self.node_count,
            edge_count=edge_count,
            zero_weight_edges=zero_weight_edges,
            structural_fill_edges=elimination['structural_fill_edges'],
            shortcut_arcs=elimination['shortcut_arcs'],
            treewidth=max((len(row) for row in self.bag_neighbors), default=0),
            treeheight=max(self.depth) + 1,
            label_count=label_count,
            position_count=position_count,
        )

    def _eliminate(self, normalized_edges: list[dict[int, float]]) -> dict[str, Any]:
        """
        按动态最小结构度执行有向距离保持消元。

        输入：规范化的有限出边邻接表。
        输出：rank、bag、双向 star 权重及 fill-in 统计。

        实现逻辑：结构邻接表被补成 clique 以维持树分解性质；有限距离只在
        `u -> v` 与 `v -> w` 都存在时新增或缩短 `u -> w` shortcut。
        """
        node_count = self.node_count
        out_edges = [dict(row) for row in normalized_edges]
        # `structural[u]` 表示消元图中的无向结构邻居，可能包含无有限弧的填边。
        structural: list[set[int]] = [set() for _ in range(node_count)]
        for source, row in enumerate(out_edges):
            for target in row:
                structural[source].add(target)
                structural[target].add(source)

        active = [True] * node_count
        heap = [(len(structural[node]), node) for node in range(node_count)]
        heapq.heapify(heap)
        rank = [-1] * node_count
        bag_neighbors: list[list[int]] = [[] for _ in range(node_count)]
        star_out: list[dict[int, float]] = [dict() for _ in range(node_count)]
        star_in: list[dict[int, float]] = [dict() for _ in range(node_count)]
        structural_fill_edges = 0
        shortcut_arcs = 0

        for order in range(node_count):
            while heap:
                degree, node = heapq.heappop(heap)
                if active[node] and degree == len(structural[node]):
                    break
            else:  # pragma: no cover - 仅保护内部状态损坏
                raise RuntimeError('动态最小度堆意外耗尽。')

            neighbors = sorted(structural[node])
            rank[node] = order
            bag_neighbors[node] = neighbors
            for neighbor in neighbors:
                star_out[node][neighbor] = out_edges[node].get(neighbor, math.inf)
                star_in[node][neighbor] = out_edges[neighbor].get(node, math.inf)

            # 结构邻居补成 clique，保证 bag 中节点之后都位于父节点祖先链。
            for index, source in enumerate(neighbors):
                for target in neighbors[index + 1:]:
                    if target not in structural[source]:
                        structural[source].add(target)
                        structural[target].add(source)
                        structural_fill_edges += 1

            # 有限 shortcut 严格遵循前驱 -> 当前点 -> 后继的有向组合。
            for source in neighbors:
                incoming = out_edges[source].get(node)
                if incoming is None:
                    continue
                for target in neighbors:
                    if source == target:
                        continue
                    outgoing = out_edges[node].get(target)
                    if outgoing is None:
                        continue
                    candidate = incoming + outgoing
                    previous = out_edges[source].get(target)
                    if previous is None:
                        out_edges[source][target] = candidate
                        shortcut_arcs += 1
                    elif candidate < previous:
                        out_edges[source][target] = candidate

            active[node] = False
            for neighbor in neighbors:
                structural[neighbor].discard(node)
                out_edges[neighbor].pop(node, None)
            structural[node].clear()
            out_edges[node].clear()
            for neighbor in neighbors:
                heapq.heappush(heap, (len(structural[neighbor]), neighbor))

        return {
            'rank': rank,
            'bag_neighbors': bag_neighbors,
            'star_out': star_out,
            'star_in': star_in,
            'structural_fill_edges': structural_fill_edges,
            'shortcut_arcs': shortcut_arcs,
        }

    def _build_decomposition_tree(self) -> tuple[list[int], list[int], int]:
        """
        根据 bag 和消元 rank 构造一一映射的分解树。

        输入：使用当前对象的 rank 与 bag。
        输出：`(parent, depth, root)`；根的 parent 为自身。
        """
        parent = [-1] * self.node_count
        roots = []
        for node, neighbors in enumerate(self.bag_neighbors):
            if not neighbors:
                roots.append(node)
                parent[node] = node
                continue
            later_neighbors = [neighbor for neighbor in neighbors if self.rank[neighbor] > self.rank[node]]
            if len(later_neighbors) != len(neighbors):
                raise RuntimeError(f'节点 {node} 的 bag 含已提前消元节点。')
            parent[node] = min(later_neighbors, key=lambda neighbor: self.rank[neighbor])

        if len(roots) != 1:
            raise RuntimeError(f'分解树必须恰有一个根，当前为 {roots}。')
        root = roots[0]
        children: list[list[int]] = [[] for _ in range(self.node_count)]
        for node, parent_node in enumerate(parent):
            if node != root:
                if self.rank[parent_node] <= self.rank[node]:
                    raise RuntimeError('分解树父节点 rank 必须高于子节点。')
                children[parent_node].append(node)

        depth = [-1] * self.node_count
        depth[root] = 0
        stack = [root]
        while stack:
            node = stack.pop()
            for child in children[node]:
                if depth[child] != -1:
                    raise RuntimeError('分解树中检测到环。')
                depth[child] = depth[node] + 1
                stack.append(child)
        if any(value < 0 for value in depth):
            raise RuntimeError('分解树未覆盖全部节点。')
        return parent, depth, root

    def _build_labels(self) -> tuple[list[list[float]], list[list[float]], list[list[int]]]:
        """
        从根到叶构造每个节点到祖先链的出向/入向精确标签。

        输入：当前对象的分解树、bag 和 star 权重。
        输出：`(dis_out, dis_in, positions)` 三组逐节点数组。

        `dis_out[v][i]` 表示 `v -> ancestor_i`，`dis_in[v][i]` 表示
        `ancestor_i -> v`；`positions[v]` 保存 bag 节点在祖先链中的深度。
        """
        dis_out: list[list[float]] = [[] for _ in range(self.node_count)]
        dis_in: list[list[float]] = [[] for _ in range(self.node_count)]
        positions: list[list[int]] = [[] for _ in range(self.node_count)]
        ancestors: list[list[int]] = [[] for _ in range(self.node_count)]

        # rank 从大到小恰好保证父节点先于子节点处理。
        for node in sorted(range(self.node_count), key=lambda value: self.rank[value], reverse=True):
            if node == self.root:
                ancestors[node] = [node]
            else:
                ancestors[node] = ancestors[self.parent[node]] + [node]
            chain = ancestors[node]
            if len(chain) != self.depth[node] + 1:
                raise RuntimeError('祖先链长度与分解树深度不一致。')

            bag_nodes = [node] + self.bag_neighbors[node]
            node_positions = sorted({self.depth[bag_node] for bag_node in bag_nodes})
            for bag_node in bag_nodes:
                if chain[self.depth[bag_node]] != bag_node:
                    raise RuntimeError(f'节点 {node} 的 bag 节点 {bag_node} 不在祖先链对应位置。')
            positions[node] = node_positions

            out_label = [math.inf] * len(chain)
            in_label = [math.inf] * len(chain)
            out_label[self.depth[node]] = 0.0
            in_label[self.depth[node]] = 0.0

            for target_position, target in enumerate(chain[:-1]):
                for boundary in self.bag_neighbors[node]:
                    boundary_depth = self.depth[boundary]

                    if boundary_depth == target_position:
                        boundary_to_target = 0.0
                        target_to_boundary = 0.0
                    elif boundary_depth > target_position:
                        boundary_to_target = dis_out[boundary][target_position]
                        target_to_boundary = dis_in[boundary][target_position]
                    else:
                        boundary_to_target = dis_in[target][boundary_depth]
                        target_to_boundary = dis_out[target][boundary_depth]

                    out_label[target_position] = min(
                        out_label[target_position],
                        self.star_out[node][boundary] + boundary_to_target,
                    )
                    in_label[target_position] = min(
                        in_label[target_position],
                        target_to_boundary + self.star_in[node][boundary],
                    )

            if any(not math.isfinite(value) for value in out_label + in_label):
                raise RuntimeError(f'强连通图节点 {node} 的祖先标签出现无穷距离。')
            dis_out[node] = out_label
            dis_in[node] = in_label

        return dis_out, dis_in, positions

    def _build_lca_table(self) -> list[list[int]]:
        """
        构建二进制提升 LCA 表。

        输入：当前分解树的 parent/depth。
        输出：`up[level][node]` 二维列表，根节点向上仍指向自身。
        """
        level_count = max(1, max(self.depth).bit_length() + 1)
        first_level = [self.parent[node] for node in range(self.node_count)]
        up = [first_level]
        for _ in range(1, level_count):
            previous = up[-1]
            up.append([previous[previous[node]] for node in range(self.node_count)])
        return up

    def lca(self, first: int, second: int) -> int:
        """
        查询两个分解树节点的最近公共祖先。

        输入：两个合法连续节点编号。
        输出：最近公共祖先节点编号。
        """
        first_id = self._validate_node(first)
        second_id = self._validate_node(second)
        if self.depth[first_id] < self.depth[second_id]:
            first_id, second_id = second_id, first_id

        difference = self.depth[first_id] - self.depth[second_id]
        for level in range(len(self.up)):
            if difference & (1 << level):
                first_id = self.up[level][first_id]
        if first_id == second_id:
            return first_id
        for level in range(len(self.up) - 1, -1, -1):
            if self.up[level][first_id] != self.up[level][second_id]:
                first_id = self.up[level][first_id]
                second_id = self.up[level][second_id]
        return self.parent[first_id]

    def _validate_node(self, node: Any) -> int:
        """
        校验查询节点编号。

        输入：Python/NumPy 整数节点。
        输出：范围内 Python `int`，否则抛出明确异常。
        """
        node_id = _as_node_id(node)
        if not 0 <= node_id < self.node_count:
            raise KeyError(f'未知 H2H 节点 {node!r}；合法范围为 [0, {self.node_count})。')
        return node_id

    def query(self, source: Any, target: Any) -> float:
        """
        使用 LCA bag 的二跳标签查询有向最短距离。

        输入：源节点和目标节点。
        输出：Python `float`；相同节点直接返回 0。
        """
        source_id = self._validate_node(source)
        target_id = self._validate_node(target)
        if source_id == target_id:
            return 0.0
        ancestor = self.lca(source_id, target_id)
        result = min(
            self.dis_out[source_id][position] + self.dis_in[target_id][position]
            for position in self.positions[ancestor]
        )
        if not math.isfinite(result):
            raise RuntimeError(f'H2H 查询 {source_id} -> {target_id} 返回无穷距离。')
        return float(result)

    def query_batch(self, sources, targets) -> list[float]:
        """
        批量执行等长节点序列的参考查询。

        输入：等长 sources 与 targets 可迭代对象。
        输出：保持输入顺序的 Python `float` 列表。
        """
        source_list = list(sources)
        target_list = list(targets)
        if len(source_list) != len(target_list):
            raise ValueError('sources 与 targets 长度必须一致。')
        return [self.query(source, target) for source, target in zip(source_list, target_list)]
