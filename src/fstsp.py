"""
本文件实现论文中的主算法 `MultiAgentFlyingSidekickTSP`。

主要内容：
1. 先为客户构造候选服务区域（文中称 convex sets）。
2. 再用 MST 思路把客户划分给不同仓库。
3. 对每个仓库组求一个集合化 TSP 顺序。
4. 最后通过动态规划近似联合优化卡车与多架无人机的同步路线。
"""

import elkai
import gurobipy as gp
from gurobipy import GRB
import math
import networkx as nx
import numpy as np
from .baseline import Baseline
from utils import mst_partition, stable_node_key


class MultiAgentFlyingSidekickTSP(Baseline):
    """
    论文主算法类。

    输入：
    - graph: 路网图。
    - depots: 仓库集合。
    - cities: 客户集合。
    - distance: 距离矩阵。
    - drone: 每辆卡车携带的无人机数量。
    - limit: 无人机最大飞行距离。
    - speed: 无人机速度参数。
    - theta: 两个启发式阈值，分别用于区域构造和集合 TSP。

    输出：
    - 无显式返回值，初始化算法内部状态。

    实现逻辑：
    1. 初始化公共输入。
    2. 为每个仓库准备客户分组容器。
    3. 预计算每个客户的可服务区域。
    """
    # Phase 1 方法标识会随实验结果保存，用于区分原版覆盖行为和修正版结果。
    PHASE1_PARTITION_METHOD = 'bidirectional-mean-set-mst-v1'

    def __init__(self, graph, depots, cities, distance, drone, limit=1.5, speed=1.6, theta=(0.5, 0.5)):
        super().__init__(graph, depots, cities, distance, drone, limit, speed)
        self.groups = {depot: [] for depot in depots}
        self.solution = []
        self.cost = 0
        self.theta = theta
        self.const = math.sqrt(2)
        self.phase1_partition_method = self.PHASE1_PARTITION_METHOD

        #语法：解包并合并两个字典，键相同时后面覆盖前面
        #作用：首先对客户节点（city），搜索整个图，保留无人机航程内的地图节点，组成键值对，构成客户集合
        #      其次对仓库节点（depot），只保留自己本身，组成键值对，构成仓库集合
        #      最后将客户集合与仓库集合合并
        self.regions = {**{ city: [node for node in self.graph.nodes if
                                    self.distance['drone'][node][city] < self.limit / 2] 
                            for city in cities},
                        **{ depot: [depot] for depot in depots}}
        #city：外层遍历所有客户节点（cities）作为键，内层条件遍历所有地图节点作为值，构成键值对
        #内层条件为无人机距离矩阵中 node 和 city 距离小于飞行距离上限的一半 的node
        #depot：遍历所有仓库，键和值相同

    def set_nn(self, theta):
        """
        用最近邻启发式将客户分配给仓库。

        输入：
        - theta: 区域截断阈值。

        输出：
        - 无显式返回值，结果写入 `self.groups`。

        实现逻辑：
        1. 对每个客户遍历所有仓库。
        2. 以“仓库到中继点的卡车距离 + 中继点到客户的无人机距离”为准则。
        3. 将客户分给代价最小的仓库。
        """
        for city in self.cities:
            distance = float('inf')
            depot = None
            for _depot in self.depots:
                _distance = min([self.distance['truck'][_depot][mid] + self.distance['drone'][mid][city] / self.speed
                                 for mid in self.graph.nodes if self.distance['drone'][mid][city] <= self.limit * theta])
                if _distance < distance:
                    distance = _distance
                    depot = _depot
            self.groups[depot].append(city)

    def _directed_set_distance(self, source, target, service_sets):
        """
        计算论文公式 (2) 语义下从一个服务集合到另一个服务集合的有向距离。

        输入：
        - source: 起始客户或仓库节点。
        - target: 目标客户或仓库节点。
        - service_sets: 客户使用边界候选点、仓库使用自身单点的集合映射。

        输出：
        - 从 `source` 服务完成后转移到 `target` 的最小标准化成本。

        实现逻辑：
        1. 以卡车直接从起点到终点的距离作为保底方案。
        2. 枚举两个服务集合中的候选点，叠加起点无人机段、卡车段和终点无人机段。
        3. 保留最小有限成本；方向反转时会独立重新计算，不复用当前结果。
        """
        try:
            best_distance = float(self.distance['truck'][source][target])
            source_nodes = service_sets[source]
            target_nodes = service_sets[target]
        except KeyError as exc:
            raise ValueError(
                f'缺少 Set-MST 距离或候选集合数据：{source!r} -> {target!r}。'
            ) from exc

        for source_node in source_nodes:
            for target_node in target_nodes:
                try:
                    candidate = (
                        float(self.distance['drone'][source][source_node])
                        / self.speed
                        * self.const
                        + float(
                            self.distance['truck'][source_node][target_node]
                        )
                        + float(self.distance['drone'][target_node][target])
                        / self.speed
                        * self.const
                    )
                except KeyError as exc:
                    raise ValueError(
                        '缺少 Set-MST 候选点间距离：'
                        f'{source!r}/{source_node!r} -> '
                        f'{target_node!r}/{target!r}。'
                    ) from exc
                if math.isfinite(candidate):
                    best_distance = min(best_distance, candidate)

        if not math.isfinite(best_distance):
            raise ValueError(
                f'Set-MST 有向距离 {source!r} -> {target!r} 不是有限数。'
            )
        return best_distance

    def _validate_set_mst_input(self, convex_sets):
        """
        校验 Set-MST 的仓库、客户和候选集合输入。

        输入：
        - convex_sets: 客户到边界候选点列表的映射。

        输出：
        - `(ordered_entities, service_sets)`，分别是稳定节点序列和统一服务集合。

        实现逻辑：拒绝重复节点、仓库客户重叠、图外节点和缺失客户集合，并把仓库
        规范成单点集合，使三类边能够使用同一套距离公式。
        """
        depots = list(self.depots)
        cities = list(self.cities)
        if len(depots) == 0:
            raise ValueError('Set-MST 至少需要一个仓库。')
        if len(set(depots)) != len(depots):
            raise ValueError('仓库数组中存在重复节点。')
        if len(set(cities)) != len(cities):
            raise ValueError('客户数组中存在重复节点。')
        if set(depots) & set(cities):
            raise ValueError('仓库节点与客户节点不能重叠。')

        graph_nodes = set(self.graph.nodes)
        problem_nodes = set(depots) | set(cities)
        missing_graph_nodes = problem_nodes - graph_nodes
        if missing_graph_nodes:
            raise ValueError(
                '仓库或客户不在路网中：'
                f'{sorted(missing_graph_nodes, key=stable_node_key)!r}。'
            )

        missing_sets = set(cities) - set(convex_sets)
        if missing_sets:
            raise ValueError(
                '缺少客户候选集合：'
                f'{sorted(missing_sets, key=stable_node_key)!r}。'
            )

        # `service_sets` 统一表示论文中的集合 S；空边界仍允许使用卡车直达保底成本。
        service_sets = {depot: [depot] for depot in depots}
        for city in cities:
            candidate_nodes = list(convex_sets[city])
            invalid_nodes = set(candidate_nodes) - graph_nodes
            if invalid_nodes:
                raise ValueError(
                    f'客户 {city!r} 的候选集合包含图外节点：'
                    f'{sorted(invalid_nodes, key=stable_node_key)!r}。'
                )
            service_sets[city] = sorted(
                set(candidate_nodes),
                key=stable_node_key,
            )

        ordered_entities = sorted(problem_nodes, key=stable_node_key)
        return ordered_entities, service_sets

    def set_mst(self, convex_sets):
        """
        基于候选区域构造完全图，并用 MST 分区给客户分组。

        输入：
        - convex_sets: 每个客户对应的候选服务点集合。

        输出：
        - 无显式返回值，分组结果写入 `self.groups`。

        实现逻辑：
        1. 在仓库和客户之间构造一个完全图。
        2. 边权采用论文中的近似代价公式。
        3. 调用 `mst_partition` 完成仓库-客户划分。
        """
        ordered_entities, service_sets = self._validate_set_mst_input(
            convex_sets
        )

        # 构造语义明确的无向完全图；每个无序节点对只添加一次，不存在覆盖写入。
        graph = nx.Graph()
        graph.graph['phase1_partition_method'] = self.phase1_partition_method
        graph.add_nodes_from(ordered_entities)
        for source_index, source in enumerate(ordered_entities):
            for target in ordered_entities[source_index + 1:]:
                forward_weight = self._directed_set_distance(
                    source,
                    target,
                    service_sets,
                )
                reverse_weight = self._directed_set_distance(
                    target,
                    source,
                    service_sets,
                )
                # 双向平均保留两个方向的道路代价，同时为无向 MST 定义唯一权重。
                symmetric_weight = (forward_weight + reverse_weight) / 2.0
                if not math.isfinite(symmetric_weight):
                    raise ValueError(
                        f'Set-MST 无向边 ({source!r}, {target!r}) 权重不是有限数。'
                    )
                graph.add_edge(
                    source,
                    target,
                    weight=symmetric_weight,
                    forward_weight=forward_weight,
                    reverse_weight=reverse_weight,
                )
        self.groups = mst_partition(graph, self.depots, self.cities)

    @staticmethod
    def cut_off(x, y):
        """
        将超过阈值的值截断为一个大惩罚。

        输入：
        - x: 待比较的数值。
        - y: 阈值。

        输出：
        - 若 `x <= y` 则返回 `x`，否则返回一个大罚值。
        """
        return x if x <= y else 100000

    def lkh(self, depot, cities):  # solve tsp via LKH
        """
        调用 `elkai` 求解单仓库组的 TSP 顺序。

        输入：
        - depot: 当前仓库。
        - cities: 属于该仓库的客户列表。

        输出：
        - 一个 TSP 访问顺序列表，索引相对于 `[depot] + cities`。

        实现逻辑：
        - 当客户数很少时直接返回显然答案，否则调用 `elkai`。
        """
        if len(cities) == 0:
            return [0, 0]
        elif len(cities) == 1:
            return [0, 1, 0]
        else:
            nodes = [depot] + cities
            int_matrix = [[self.distance['truck'][start][end] for end in nodes] for start in nodes]
            route = elkai.DistanceMatrix(int_matrix).solve_tsp()
            return route

    @staticmethod
    def set_tsp(convex_sets, distance, convex_set_distance):
        """
        求解集合化 TSP 顺序问题。

        输入：
        - convex_sets: 各客户或仓库对应的候选点集合。
        - distance: 不同集合间候选点两两距离。
        - convex_set_distance: 同一集合内部起点/终点切换代价。

        输出：
        - 一个访问顺序 `seq`。

        实现逻辑：
        1. 建立集合层面的 TSP 顺序变量。
        2. 建立集合内部节点选择变量和集合间连接变量。
        3. 用 Gurobi 求解后恢复访问顺序。
        """
        n = len(convex_sets)
        model = gp.Model('Set-TSP')
        #日志输出设置
        model.setParam("OutputFlag", 0)
        # first write a tsp for the visiting order of convex sets using GG model
        select = model.addMVar((n, n), vtype=GRB.BINARY)
        model.addConstrs(select[u, u] == 0 for u in range(n))
        model.addConstrs(np.ones((n,)) @ select[:, v] == 1 for v in range(n))
        model.addConstrs(np.ones((n,)) @ select[u, :] == 1 for u in range(n))
        flow = model.addMVar((n, n), vtype=GRB.CONTINUOUS)
        model.addConstrs(flow[u, v] <= n * select[u, v] for u in range(n) for v in range(n))
        model.addConstr(np.ones((n,)) @ flow[0, :] == n - 1)
        model.addConstr(np.ones((n,)) @ flow[:, 0] == 0)
        model.addConstrs(flow[u, u] == 0 for u in range(n))
        model.addConstrs(np.ones((n,)) @ flow[:, v] - np.ones((n,)) @ flow[v, :] == 1 for v in range(1, n))

        # internal is the selection of node pair inside each convex set
        internal = [[[model.addVar(vtype=GRB.BINARY) for _ in convex_set] for _ in convex_set] for convex_set in
                    convex_sets]
        # external is the selection of node pair between two convex sets
        external = [[[[model.addVar(vtype=GRB.BINARY) for _ in v] for _ in u] for v in convex_sets] for u in
                    convex_sets]
        model.addConstrs(gp.quicksum([internal[i][j][k] for j in range(len(convex_sets[i]))
                                      for k in range(len(convex_sets[i]))]) == 1 for i in range(n))
        model.addConstrs(gp.quicksum([external[u][v][i][j] for i in range(len(convex_sets[u]))
                                      for j in range(len(convex_sets[v]))]) == select[u, v]
                         for u in range(n) for v in range(n))
        # node j in convex sets v should have same out degree internal and in degree external
        model.addConstrs(gp.quicksum([external[u][v][i][j] for u in range(n) for i in range(len(convex_sets[u]))]) ==
                         gp.quicksum([internal[v][j][k] for k in range(len(convex_sets[v]))]) for v in range(n)
                         for j in range(len(convex_sets[v])))
        # node i in convex sets u should have same in degree internal and out degree external
        model.addConstrs(gp.quicksum([external[u][v][i][j] for v in range(n) for j in range(len(convex_sets[v]))]) ==
                         gp.quicksum([internal[u][k][i] for k in range(len(convex_sets[u]))]) for u in range(n)
                         for i in range(len(convex_sets[u])))
        model.setObjective(gp.quicksum([convex_set_distance[i][j][k] * internal[i][j][k] for i in range(n)
                                        for j in range(len(convex_sets[i])) for k in range(len(convex_sets[i]))]) +
                           gp.quicksum([distance[u][v][i][j] * external[u][v][i][j] for u in range(n) for v in range(n)
                                        for i in range(len(convex_sets[u])) for j in range(len(convex_sets[v]))]),
                           GRB.MINIMIZE)
        model.optimize()

        seq = [0]
        while seq.count(0) < 2:
            for j in range(n):
                if select[seq[-1], j].X > 0.99:
                    seq.append(j)
                    break
        return seq

    def local_search_multi_drone_appr(self, seq, depot):
        """
        在给定客户顺序下，用动态规划近似优化多无人机联合调度。

        输入：
        - seq: 客户访问顺序，首尾均为仓库。
        - depot: 当前仓库节点。

        输出：
        - `(solution, cost)`：
          - `solution`: 当前仓库的卡车/无人机联合路线。
          - `cost`: 对应完工时间。

        实现逻辑：
        1. 为每个阶段维护最优值函数 `value`。
        2. 用 `appr` 近似估计同时投放多架无人机时的代价。
        3. 通过前缀 DP 累积卡车与无人机联合路线。
        """



        # value as defined in Eq. (13)
        value = [{node: float('inf') for node in self.graph.nodes} for _ in range(2 * len(seq) - 2)]
        value[0][depot] = 0
        # appr is the time defined in Eq. (9)
        appr = [[{node: {} for node in self.regions[city]} for _ in range(self.drone)] for city in seq[1:-1]]
        # tour record the optimal tour to reach each time (appr value) defined in Eq. (9)
        tour = [[{node: {} for node in self.regions[city]} for _ in range(self.drone)] for city in seq[1:-1]]
        # group record the number of customers before visited together with the current customer
        group = [1 for _ in seq[1:-1]]  # seq[i] has the same departure node as seq[i - group[i - 1] + 1]
        # prefix record the optimal tour to reach the corresponding value in Eq. (13)
        prefix = [{node: {'truck': [], 'drone': []} for node in self.graph.nodes} for _ in range(2 * len(seq) - 2)]
        prefix[0][depot]['truck'] = [depot]

        for i in range(1, len(seq) - 1):
            for node in self.regions[seq[i]]:
                for _node in self.regions[seq[i]]:
                    # initialize appr as Eq. (8)
                    drone_time = self.distance['drone'][seq[i]][_node] + self.distance['drone'][seq[i]][node]
                    truck_time = self.distance['truck'][_node][seq[i]] + self.distance['truck'][seq[i]][node]

                    tour[i - 1][0][node][_node] = {'truck': [], 'drone': []}
                    if drone_time <= self.limit:
                        appr[i - 1][0][node][_node] = max(drone_time / self.speed, self.distance['truck'][_node][node])
                        tour[i - 1][0][node][_node]['truck'] = [_node, node]
                        tour[i - 1][0][node][_node]['drone'] = [[_node, seq[i], node]]
                    else:
                        appr[i - 1][0][node][_node] = truck_time
                        tour[i - 1][0][node][_node]['truck'] = [_node, seq[i], node]

            # track the number of possible customers to be visited together
            while i - 1 - group[i - 1] >= 0 and group[i - 1] <= group[i - 2] and group[i - 1] < self.drone:
                if self.distance['drone'][seq[i - group[i - 1]]][seq[i]] < 2 * self.limit:
                    group[i - 1] += 1
                else:
                    break
            for j in range(1, group[i - 1]):
                # assume throw all together, then
                for node in self.regions[seq[i]]:
                    for _node in self.regions[seq[i - j]]:
                        # Eq. (9)
                        appr[i - 1][j][node][_node] = float('inf')
                        tour[i - 1][j][node][_node] = {'truck': [], 'drone': []}

                        consumption = self.distance['drone'][seq[i]][_node] + self.distance['drone'][seq[i]][node]
                        if consumption > self.limit:
                            continue
                        for _mid in self.regions[seq[i - 1]]:
                            cost = max(appr[i - 2][j - 1][_mid][_node] + self.distance['truck'][_mid][node],
                                       consumption / self.speed)
                            if cost < appr[i - 1][j][node][_node]:
                                appr[i - 1][j][node][_node] = cost
                                tour[i - 1][j][node][_node]['truck'] = tour[i - 2][j - 1][_mid][_node][
                                                                           'truck'].copy() + [node]
                                tour[i - 1][j][node][_node]['drone'] = tour[i - 2][j - 1][_mid][_node][
                                                                           'drone'].copy() + [[_node, seq[i], node]]
            #print("@")

        for i in range(1, len(seq) - 1):
            for node in self.regions[seq[i]]:
                for _node in self.regions[seq[i - 1]]:
                    # initialize value as Eq. (12)
                    if value[2 * i - 2][_node] + self.distance['truck'][_node][node] < value[2 * i - 1][node]:
                        value[2 * i - 1][node] = value[2 * i - 2][_node] + self.distance['truck'][_node][node]
                        prefix[2 * i - 1][node]['truck'] = prefix[2 * i - 2][_node]['truck'].copy() + [node]
                        prefix[2 * i - 1][node]['drone'] = prefix[2 * i - 2][_node]['drone'].copy()

            for j in range(min(self.drone, i)):
                for node in self.regions[seq[i]]:
                    for _node in self.regions[seq[i - j]]:
                        # approximation method to estimate the time consumption as Eq. (13)
                        if _node not in appr[i - 1][j][node].keys():
                            continue
                        if value[2 * i - 2 * j - 1][_node] + appr[i - 1][j][node][_node] < value[2 * i][node]:
                            value[2 * i][node] = value[2 * i - 2 * j - 1][_node] + appr[i - 1][j][node][_node]
                            prefix[2 * i][node]['truck'] = prefix[2 * i - 2 * j - 1][_node]['truck'].copy() + \
                                                           tour[i - 1][j][node][_node]['truck'].copy()
                            prefix[2 * i][node]['drone'] = prefix[2 * i - 2 * j - 1][_node]['drone'].copy() + \
                                                           tour[i - 1][j][node][_node]['drone'].copy()

        for node in self.regions[seq[-2]]:
            # after visiting the last customer, return to the depot
            if value[2 * len(seq) - 4][node] + self.distance['truck'][node][depot] < value[2 * len(seq) - 3][depot]:
                value[2 * len(seq) - 3][depot] = value[2 * len(seq) - 4][node] + self.distance['truck'][node][depot]
                prefix[2 * len(seq) - 3][depot]['truck'] = prefix[2 * len(seq) - 4][node]['truck'].copy() + [depot]
                prefix[2 * len(seq) - 3][depot]['drone'] = prefix[2 * len(seq) - 4][node]['drone'].copy()

        return prefix[2 * len(seq) - 3][depot], value[2 * len(seq) - 3][depot]

    def get_seq(self, depot, convex_sets):
        """
        获取当前仓库组的客户访问顺序。

        输入：
        - depot: 当前仓库。
        - convex_sets: 当前仓库组的候选点集合列表。

        输出：
        - 客户顺序索引序列。

        实现逻辑：
        - 若关闭集合 TSP，则调用 `lkh`；
          否则构造集合化距离并调用 `set_tsp`。
        """
        if self.theta[1] == 0:
            seq = self.lkh(depot, self.groups[depot])
        else:
            set_distance = [[[max(self.distance['truck'][k][j],
                                  self.cut_off((self.distance['drone'][j][city] + self.distance['drone'][city][k]),
                                               self.limit)) / self.speed for j in convex_set] for k in convex_set]
                            for convex_set, city in zip(convex_sets, [depot] + self.groups[depot])]
            distance = [[[[self.distance['truck'][i][j] for j in v] for i in u]
                         for v in convex_sets] for u in convex_sets]
            seq = self.set_tsp(convex_sets, distance, set_distance)
        return seq

    def single_solution(self, depot, convex_sets):
        """
        求解单个仓库组的联合配送路线。

        输入：
        - depot: 当前仓库。
        - convex_sets: 当前仓库及其客户的候选点集合。

        输出：
        - 当前仓库组的内部解表示。

        实现逻辑：
        1. 先生成客户访问顺序。
        2. 再调用多无人机动态规划近似模块。
        3. 把成本累加到总成本中。
        """
        cities = self.groups[depot]
        if len(cities) == 0:
            return {'truck': [depot, depot], 'drone': []}

        seq = self.get_seq(depot, convex_sets)
        seq = [depot] + [cities[i - 1] for i in seq[1:-1]] + [depot]
        solution, cost = self.local_search_multi_drone_appr(seq, depot)
        self.cost += cost
        return solution

    def convert(self, solution):
        """
        将内部解表示转成统一输出格式。

        输入：
        - solution: 当前仓库组的内部解。

        输出：
        - 统一格式的路线字典。
        """
        route = {'truck': solution['truck'], 'drone': [[] for _ in range(self.drone)]}
        route['drone'][0] = solution['drone']
        return route

    # not necessary a convex set, a wrong name
    def get_convex_sets(self, theta):
        """
        为每个客户构造候选服务点集合。

        输入：
        - theta: 允许的区域半径比例。

        输出：
        - `convex_sets` 字典，键为客户，值为候选点列表。

        实现逻辑：
        - 对图中每个节点，找到满足阈值条件且最近的客户，并将节点归到该客户名下。
        """
        # 客户采用稳定节点顺序，等距候选不再由输入数组中“最后出现者”决定。
        ordered_cities = sorted(set(self.cities), key=stable_node_key)
        convex_sets = {city: [] for city in ordered_cities}
        # 遍历所有点，对每个点按“距离、稳定客户键”选择唯一最近客户。
        for node in self.graph.nodes:
            closest_city = None
            closest_key = None
            for city in ordered_cities:
                city_distance = self.distance['drone'][node][city]
                if city_distance > self.limit * theta:
                    continue
                candidate_key = (city_distance, stable_node_key(city))
                if closest_key is None or candidate_key < closest_key:
                    closest_city = city
                    closest_key = candidate_key
            if closest_city is not None:
                convex_sets[closest_city].append(node)
        return convex_sets

    def get_boundary_convex_sets(self, theta):
        """
        从候选服务点集合中提取边界点集合。

        输入：
        - theta: 生成候选区域时使用的阈值。

        输出：
        - `boundary_convex_sets` 字典。

        实现逻辑：
        - 若某节点属于某客户区域，但存在邻居不属于该区域，则把该节点视为边界点。
        """
        #计算得到convex_sets字典，表示每个客户服务范围内的地图节点，键为客户，值为该客户服务范围内的点
        convex_sets = self.get_convex_sets(theta)
        #初始化边界点集合
        #语法：字典推导式，遍历所有客户，创建键值对，键为客户（city），值为空
        boundary_convex_sets = {city: [] for city in convex_sets}
        #遍历每个客户的，服务范围内的点的，邻居节点
        for city in convex_sets:
            for node in convex_sets[city]:
                for neighbor in nx.neighbors(self.graph, node):
                    #若该点的邻居点不在服务范围（集合）内，则说明该点是边界点
                    if neighbor not in convex_sets[city]:
                        boundary_convex_sets[city].append(node)
                        break
        return boundary_convex_sets

    def solve(self):
        """
        执行论文主算法的完整求解流程。

        输入：
        - 无显式输入。

        输出：
        - `(self.solution, self.cost)`。

        实现逻辑：
        1. 先构造边界候选点集合。
        2. 再用 MST 给客户分仓库。
        3. 对每个仓库单独求解联合路线。
        4. 汇总所有仓库的结果。
        """
        #获取客户服务范围（集合）内的边界点，加速计算
        convex_sets = self.get_boundary_convex_sets(self.theta[0])
        #用mst算法为仓库分配对应的客户
        self.set_mst(convex_sets)
        #对每个仓库（每个客户分组） 
        for depot in self.depots:     
            #创建当前仓库及其对应客户的点集合 的集合convex_set
            convex_set = [[depot]] + [convex_sets[city] for city in self.groups[depot]]
            #对当前仓库求解
            solution = self.single_solution(depot, convex_set) 
            #将解统一格式后返回
            self.solution.append(self.convert(solution)) 
        return self.solution, self.cost

    # function that helps fast generate the results for the ablation study of the drone numbers
    def solve_multiple_drones(self):
        """
        固定客户顺序，快速评估不同无人机数量对应的成本。

        输入：
        - 无显式输入。

        输出：
        - 一个列表，按 0 到 5 架无人机依次记录成本。

        实现逻辑：
        1. 先生成仓库分组与访问顺序。
        2. 在相同顺序下重复调用动态规划近似模块。
        3. 输出不同无人机数下的成本。
        """
        costs = []
        convex_sets = self.get_boundary_convex_sets(self.theta[0])
        self.set_mst(convex_sets)
        for depot in self.depots:
            convex_set = [[depot]] + [convex_sets[city] for city in self.groups[depot]]
            cities = self.groups[depot]
            if len(cities) > 0:
                seq = self.get_seq(depot, convex_set)
                seq = [depot] + [cities[i - 1] for i in seq[1:-1]] + [depot]
                for drone in range(6):
                    self.drone = drone
                    _, cost = self.local_search_multi_drone_appr(seq, depot)
                    costs.append(cost)
        return costs
