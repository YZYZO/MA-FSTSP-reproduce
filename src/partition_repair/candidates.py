"""对称 Set-MST 与有限预算下不同力度的分区候选。"""

from dataclasses import dataclass
from itertools import combinations
import math

import networkx as nx
import numpy as np

from utils import mst_partition
from .features import binary_count, canonical_partition, partition_key
from .settings import RepairOptions


@dataclass
class Candidate:
    """保存完整候选分区、产生方式和修复力度，输入输出均不包含求解后标签。"""

    name: str
    kind: str
    strength: float
    partition: dict


def symmetric_set_graph(model, boundary):
    """输入实例和固定边界，对每对仓库/客户显式计算两个方向并取均值，返回无向图。"""
    depots = set(map(int, model.depots))
    nodes = list(map(int, model.depots)) + sorted(map(int, model.cities))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    def directed(first, second):
        """计算一个方向的集合接驳代价，仓库只使用自身节点，客户使用全局边界。"""
        starts = [first] if first in depots else boundary[first]
        ends = [second] if second in depots else boundary[second]
        best = model.distance['truck'][first][second]
        for start in starts:
            first_flight = 0 if first in depots else model.distance['drone'][first][start]
            for end in ends:
                last_flight = 0 if second in depots else model.distance['drone'][end][second]
                best = min(best, model.distance['truck'][start][end] +
                           (first_flight + last_flight) / model.speed * model.const)
        return float(best)

    for first, second in combinations(nodes, 2):
        graph.add_edge(first, second, weight=(directed(first, second) + directed(second, first)) / 2)
    return graph


def symmetric_mst(model, boundary):
    """输入完整实例与边界，输出显式对称 Set-MST 分区；复用现有树形 DP。"""
    graph = symmetric_set_graph(model, boundary)
    return canonical_partition(mst_partition(graph, np.asarray(model.depots), np.asarray(model.cities)), model.depots)


def relocate(partition, source, target, customers):
    """输入分区、源组、目标组和迁移客户，输出独立新分区，不修改输入。"""
    result = canonical_partition(partition, partition)
    moving = set(customers)
    result[source] = [c for c in result[source] if c not in moving]
    result[target] = sorted(result[target] + list(moving))
    return result


def swap(partition, source, target, first, second):
    """输入两组各一个客户，交换其归属并返回客户稳定排序的新分区。"""
    result = canonical_partition(partition, partition)
    result[source].remove(first)
    result[target].remove(second)
    result[source] = sorted(result[source] + [second])
    result[target] = sorted(result[target] + [first])
    return result


def shortlisted_customers(context, group, source, target, limit):
    """按归属接近程度和边界负担交替取客户，输出有限列表，保留两类动作覆盖。"""
    affinity = sorted(group, key=lambda c: (context.affinity[target, c] - context.affinity[source, c], c))
    burden = sorted(group, key=lambda c: (-len(context.boundary[c]), c))
    selected = []
    for first, second in zip(affinity, burden):
        for city in (first, second):
            if city not in selected:
                selected.append(city)
            if len(selected) >= limit:
                return selected
    return selected


def nearby_depots(context, source, choices, limit):
    """输入源仓库和可选仓库，按对称道路距离截断目标组，避免全量动作组合。"""
    return sorted(choices, key=lambda d: (context.road(source, d), d))[:limit]


def repair_score(context, before, after, weight):
    """输入前后分区，用相对变量节省减去几何增量评价；返回无需求解器的手工分数。"""
    old, new = context.partition(before), context.partition(after)
    saving = (old['binary_sum'] - new['binary_sum']) / max(old['binary_sum'], 1)
    geometry = (new['geometry_sum'] - old['geometry_sum']) / max(old['geometry_sum'], 1e-12)
    return saving - weight * geometry


def count_repair_path(context, baseline, options):
    """输入初始分区，沿人数过多组到不足组的迁移生成完整均衡路径，返回各步分区。"""
    depots = context.depots
    quotient, remainder = divmod(len(context.cities), len(depots))
    # 优先把较大目标容量留给当前较大的组，减少达到均衡所需迁移量。
    ranked = sorted(depots, key=lambda d: (-len(baseline[d]), d))
    targets = {d: quotient + int(d in ranked[:remainder]) for d in depots}
    current = canonical_partition(baseline, depots)
    path = []
    while any(len(current[d]) > targets[d] for d in depots):
        sources = [d for d in depots if len(current[d]) > targets[d]]
        receivers = [d for d in depots if len(current[d]) < targets[d]]
        moves = []
        for source in sources:
            for target in nearby_depots(context, source, receivers, options.destination_count):
                for city in shortlisted_customers(context, current[source], source, target, options.customers_per_group):
                    moves.append((context.affinity[target, city] - context.affinity[source, city], source, target, city))
        _, source, target, city = min(moves)
        current = relocate(current, source, target, [city])
        path.append(current)
    return path


def burden_repair_path(context, baseline, options):
    """按变量负担减少和几何增量选择有限迁移，返回最多四分之一客户数步的路径。"""
    current = canonical_partition(baseline, context.depots)
    path = []
    for _ in range(max(1, math.ceil(len(context.cities) * 0.25))):
        choices = []
        sources = sorted(context.depots, key=lambda d: (-binary_count(current[d], context.boundary), d))[:2]
        for source in sources:
            targets = nearby_depots(context, source, [d for d in context.depots if d != source], options.destination_count)
            for target in targets:
                for city in shortlisted_customers(context, current[source], source, target, options.customers_per_group):
                    after = relocate(current, source, target, [city])
                    old_load = binary_count(current[source], context.boundary) + binary_count(current[target], context.boundary)
                    new_load = binary_count(after[source], context.boundary) + binary_count(after[target], context.boundary)
                    if new_load < old_load:
                        choices.append((repair_score(context, current, after, options.geometry_weight), source, target, city, after))
        if not choices:
            break
        current = max(choices, key=lambda item: (item[0], -item[1], -item[2], -item[3]))[-1]
        path.append(current)
    return path


def connected_cluster(tree, seed, size):
    """从种子沿 MST 的最短边扩展，输出恰好指定大小的连通客户簇或空列表。"""
    selected = {seed}
    while len(selected) < size:
        frontier = [(tree[a][b]['weight'], b) for a in sorted(selected) for b in tree.neighbors(a) if b not in selected]
        if not frontier:
            return []
        selected.add(min(frontier)[1])
    return sorted(selected)


def generate_candidates(context, baseline, options=None):
    """输入初始分区与求解前上下文，生成最多十二个不同候选，按类型预留名额。"""
    options = options or RepairOptions()
    candidates, seen = [], set()

    def add(name, kind, strength, partition):
        """将新分区加入有序候选集；重复分区和超过总预算的候选直接忽略。"""
        key = partition_key(partition, context.depots)
        if key not in seen and len(candidates) < options.max_candidates:
            seen.add(key)
            candidates.append(Candidate(name, kind, strength, canonical_partition(partition, context.depots)))

    add('stay', 'stay', 0.0, baseline)
    path = count_repair_path(context, baseline, options)
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        if path:
            add(f'count_{fraction:g}', 'count', fraction, path[math.ceil(fraction * len(path)) - 1])
    path = burden_repair_path(context, baseline, options)
    for fraction in (0.1, 0.25):
        if path:
            index = min(len(path), max(1, math.ceil(len(context.cities) * fraction))) - 1
            add(f'burden_{fraction:g}', 'burden', fraction, path[index])

    sources = sorted(context.depots, key=lambda d: (-binary_count(baseline[d], context.boundary), d))[:2]
    for size in (2, 4, 8):
        best = None
        for source in sources:
            if len(baseline[source]) < size:
                continue
            tree = context.customer_tree(baseline[source])
            targets = nearby_depots(context, source, [d for d in context.depots if d != source], options.destination_count)
            for target in targets:
                seeds = shortlisted_customers(context, baseline[source], source, target, options.cluster_seeds)
                for seed in seeds:
                    cluster = connected_cluster(tree, seed, size)
                    after = relocate(baseline, source, target, cluster)
                    score = repair_score(context, baseline, after, options.geometry_weight)
                    if best is None or score > best[0]:
                        best = (score, after)
        if best is not None:
            add(f'cluster_{size}', 'cluster', size, best[1])

    # 局部名额比较迁移和交换；先截断每组客户，再枚举小列表的笛卡尔积。
    best = None
    for source, target in combinations(context.depots, 2):
        left = shortlisted_customers(context, baseline[source], source, target, options.customers_per_group)
        right = shortlisted_customers(context, baseline[target], target, source, options.customers_per_group)
        local = [('relocate', relocate(baseline, source, target, [a])) for a in left]
        local += [('relocate', relocate(baseline, target, source, [b])) for b in right]
        local += [('swap', swap(baseline, source, target, a, b)) for a in left for b in right]
        for kind, after in local:
            if partition_key(after, context.depots) in seen:
                continue
            score = repair_score(context, baseline, after, options.geometry_weight)
            if best is None or score > best[0]:
                best = (score, kind, after)
    if best is not None:
        add('local_' + best[1], best[1], 1, best[2])
    return candidates
