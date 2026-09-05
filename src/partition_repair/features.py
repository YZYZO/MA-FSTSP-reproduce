"""只依赖求解前信息的共享距离摘要、车组特征和分区差分。"""

from itertools import combinations
import time

import networkx as nx
import numpy as np


def canonical_partition(partition, depots):
    """输入分区与仓库顺序，输出客户稳定排序的独立列表，避免候选互相修改。"""
    return {int(depot): sorted(int(city) for city in partition[depot]) for depot in depots}


def partition_key(partition, depots):
    """将分区转成稳定可哈希结构，用于去重；仓库顺序具有固定含义。"""
    return tuple((int(depot), tuple(sorted(map(int, partition[depot])))) for depot in depots)


def binary_count(customers, boundary):
    """输入客户和边界集合，返回当前稠密 Set-TSP 的二元变量数；空组返回零。"""
    if not customers:
        return 0
    sizes = [len(boundary[city]) for city in customers]
    return (len(customers) + 1) ** 2 + 1 + sum(b * b for b in sizes) + (1 + sum(sizes)) ** 2


class FeatureContext:
    """输入固定实例和全局边界，缓存求解前的组摘要，输出可复用特征。"""

    def __init__(self, model, boundary):
        """初始化只读距离与区域上下文；缓存只保存结构特征，不保存求解结果。"""
        self.model = model
        self.boundary = boundary
        self.depots = list(map(int, model.depots))
        self.cities = sorted(map(int, model.cities))
        self.truck = model.distance['truck']
        self.drone = model.distance['drone']
        self.group_cache = {}
        self.compute_seconds = 0.0
        # 往返亲和力同时考虑道路方向和边界点上的无人机接驳。
        self.affinity = {}
        for depot in self.depots:
            for city in self.cities:
                outward = min([self.truck[depot][city]] + [
                    self.truck[depot][node] + self.drone[node][city] / model.speed * model.const
                    for node in boundary[city]
                ])
                inward = min([self.truck[city][depot]] + [
                    self.truck[node][depot] + self.drone[city][node] / model.speed * model.const
                    for node in boundary[city]
                ])
                self.affinity[depot, city] = float(outward + inward)

    def road(self, first, second):
        """输入两个客户，返回双向卡车距离均值，作为几何摘要而非真实配送成本。"""
        return float((self.truck[first][second] + self.truck[second][first]) / 2)

    def customer_tree(self, customers):
        """输入组成员，返回客户间对称道路距离的 MST，供连通小簇和结构摘要共用。"""
        graph = nx.Graph()
        graph.add_nodes_from(sorted(customers))
        for first, second in combinations(sorted(customers), 2):
            graph.add_edge(first, second, weight=self.road(first, second))
        return nx.minimum_spanning_tree(graph)

    def group(self, depot, customers):
        """输入仓库和客户，计算候选负担、道路方向与区域特征；返回缓存的数值字典。"""
        key = (int(depot), tuple(sorted(customers)))
        if key in self.group_cache:
            return self.group_cache[key]
        start = time.perf_counter()
        customers = key[1]
        sizes = [len(self.boundary[city]) for city in customers]
        regions = [len(self.model.regions[city]) for city in customers]
        tree = self.customer_tree(customers)
        edges = [data['weight'] for _, _, data in tree.edges(data=True)]
        outward = [self.truck[depot][city] for city in customers]
        inward = [self.truck[city][depot] for city in customers]
        mst_length = sum(edges)
        # 几何代理是接入仓库的往返代价加两倍组内 MST 长度，不充当质量真值。
        geometry = 2 * mst_length + min((self.affinity[depot, c] for c in customers), default=0.0)
        result = {
            'customers': len(customers), 'boundary_total': sum(sizes),
            'boundary_squared': sum(b * b for b in sizes),
            'boundary_max': max(sizes, default=0),
            'binary': binary_count(customers, self.boundary),
            'region_total': sum(regions), 'region_squared': sum(r * r for r in regions),
            'region_max': max(regions, default=0),
            'depot_out_total': float(sum(outward)), 'depot_in_total': float(sum(inward)),
            'direction_difference': float(sum(abs(a - b) for a, b in zip(outward, inward))),
            'affinity_total': sum(self.affinity[depot, c] for c in customers),
            'mst_length': float(mst_length), 'mst_max_edge': float(max(edges, default=0.0)),
            'geometry': float(geometry),
        }
        self.group_cache[key] = result
        self.compute_seconds += time.perf_counter() - start
        return result

    def partition(self, partition):
        """输入完整分区，汇总所有车组的和、最大值及人数方差，输出定长特征。"""
        groups = [self.group(depot, partition[depot]) for depot in self.depots]
        result = {}
        for name in groups[0]:
            values = [group[name] for group in groups]
            result[name + '_sum'] = float(sum(values))
            result[name + '_max'] = float(max(values))
        result.update(
            customer_count=len(self.cities), depot_count=len(self.depots),
            drone_count=self.model.drone, drone_speed=self.model.speed,
            drone_limit=self.model.limit,
            customers_variance=float(np.var([len(partition[d]) for d in self.depots])),
        )
        return result

    def difference(self, baseline, candidate):
        """输入前后分区，输出前值、后值及差分；全部可在运行 Set-TSP 前计算。"""
        before, after = self.partition(baseline), self.partition(candidate)
        result = {}
        for name in before:
            result['before_' + name] = before[name]
            result['after_' + name] = after[name]
            result['delta_' + name] = after[name] - before[name]
        result['changed_groups'] = sum(baseline[d] != candidate[d] for d in self.depots)
        result['moved_customers'] = sum(len(set(baseline[d]) - set(candidate[d])) for d in self.depots)
        return result
