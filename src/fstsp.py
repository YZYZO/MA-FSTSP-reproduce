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
import time
import networkx as nx
import numpy as np
from .baseline import Baseline
from utils import mst_partition


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
    def __init__(self, graph, depots, cities, distance, drone, limit=1.5, speed=1.6, theta=(0.5, 0.5)):
        """输入完整实例，初始化共享路网与服务区域，并记录实例构造耗时。"""
        initialization_start = time.perf_counter()
        super().__init__(graph, depots, cities, distance, drone, limit, speed)
        self.groups = {depot: [] for depot in depots}
        self.solution = []
        self.cost = 0
        self.theta = theta
        self.const = math.sqrt(2)

        #语法：解包并合并两个字典，键相同时后面覆盖前面
        #作用：首先对客户节点（city），搜索整个图，保留无人机航程内的地图节点，组成键值对，构成客户集合
        #      其次对仓库节点（depot），只保留自己本身，组成键值对，构成仓库集合
        #      最后将客户集合与仓库集合合并
        self.regions = {**{ city: [node for node in self.graph.nodes if
                                    self.distance['drone'][node][city] < self.limit / 2] 
                            for city in cities},
                        **{ depot: [depot] for depot in depots}}
        self.initialization_seconds = time.perf_counter() - initialization_start
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
        # construct the fully connected graph between convex sets and depots
        graph = nx.Graph()
        for depot in self.depots:
            graph.add_node(depot)
        for city in self.cities:
            graph.add_node(city)
        # distance computed as Eq. (3) in the paper
        for depot in self.depots:
            for city in self.cities:
                weight = self.distance['truck'][depot][city]
                for node in convex_sets[city]:
                    weight = min(weight, self.distance['truck'][depot][node] +
                                 self.distance['drone'][node][city] / self.speed * self.const)
                graph.add_edge(depot, city, weight=weight)
            for _depot in self.depots:
                graph.add_edge(depot, _depot, weight=self.distance['truck'][depot][_depot])
        for city in self.cities:
            for _city in self.cities:
                weight = self.distance['truck'][city][_city]
                for node in convex_sets[city]:
                    for _node in convex_sets[_city]:
                        weight = min(weight, self.distance['truck'][node][_node] +
                                     self.distance['drone'][city][node] / self.speed * self.const
                                     + self.distance['drone'][_city][_node] / self.speed * self.const)
                graph.add_edge(city, _city, weight=weight)
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
    def set_tsp(convex_sets, distance, convex_set_distance, *, solver_options=None, return_info=False):
        """
        求解集合化 TSP 顺序问题。

        输入：
        - convex_sets: 各客户或仓库对应的候选点集合。
        - distance: 不同集合间候选点两两距离。
        - convex_set_distance: 同一集合内部起点/终点切换代价。
        - solver_options: 优化预算、线程、种子和间隙配置；默认使用共同的实验配置。
        - return_info: 是否同时返回建模/优化/恢复计时和求解状态。

        输出：
        - 默认返回访问顺序；return_info=True 时返回 `(seq, info)`，无可行解时 seq 为 None。

        实现逻辑：
        1. 建立集合层面的 TSP 顺序变量。
        2. 建立集合内部节点选择变量和集合间连接变量。
        3. 用 Gurobi 求解后恢复访问顺序。
        """
        from .partition_repair.settings import SolverOptions

        options = solver_options or SolverOptions()
        build_start = time.perf_counter()
        n = len(convex_sets)
        with gp.Model('Set-TSP') as model:
            # 统一各候选的优化预算和资源设置，日志输出关闭。
            model.setParam("OutputFlag", 0)
            model.setParam("Threads", options.threads)
            model.setParam("Seed", options.seed)
            model.setParam("MIPGap", options.mip_gap)
            if options.time_limit is not None:
                model.setParam("TimeLimit", options.time_limit)
            # select 表示集合访问边；flow 用单商品流约束消除不经过仓库的子环。
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

            # internal 表示每个集合内部的进出节点对。
            internal = [[[model.addVar(vtype=GRB.BINARY) for _ in convex_set] for _ in convex_set] for convex_set in
                        convex_sets]
            # external 表示不同集合之间的有向节点连接。
            external = [[[[model.addVar(vtype=GRB.BINARY) for _ in v] for _ in u] for v in convex_sets] for u in
                        convex_sets]
            model.addConstrs(gp.quicksum([internal[i][j][k] for j in range(len(convex_sets[i]))
                                          for k in range(len(convex_sets[i]))]) == 1 for i in range(n))
            model.addConstrs(gp.quicksum([external[u][v][i][j] for i in range(len(convex_sets[u]))
                                          for j in range(len(convex_sets[v]))]) == select[u, v]
                             for u in range(n) for v in range(n))
            # 集合入口的外部入度与内部出度保持一致。
            model.addConstrs(gp.quicksum([external[u][v][i][j] for u in range(n) for i in range(len(convex_sets[u]))]) ==
                             gp.quicksum([internal[v][j][k] for k in range(len(convex_sets[v]))]) for v in range(n)
                             for j in range(len(convex_sets[v])))
            # 集合出口的内部入度与外部出度保持一致。
            model.addConstrs(gp.quicksum([external[u][v][i][j] for v in range(n) for j in range(len(convex_sets[v]))]) ==
                             gp.quicksum([internal[u][k][i] for k in range(len(convex_sets[u]))]) for u in range(n)
                             for i in range(len(convex_sets[u])))
            model.setObjective(gp.quicksum([convex_set_distance[i][j][k] * internal[i][j][k] for i in range(n)
                                            for j in range(len(convex_sets[i])) for k in range(len(convex_sets[i]))]) +
                               gp.quicksum([distance[u][v][i][j] * external[u][v][i][j] for u in range(n) for v in range(n)
                                            for i in range(len(convex_sets[u])) for j in range(len(convex_sets[v]))]),
                               GRB.MINIMIZE)
            # 先提交延迟更新，将 Python 建模与优化调用的时间分开。
            model.update()
            info = {
                'phase2_build_seconds': time.perf_counter() - build_start,
                'num_vars': model.NumVars,
                'num_binary': model.NumBinVars,
                'num_constrs': model.NumConstrs,
            }
            optimize_start = time.perf_counter()
            model.optimize()
            info['phase2_optimize_seconds'] = time.perf_counter() - optimize_start
            extract_start = time.perf_counter()
            has_incumbent = model.SolCount > 0
            gap = float(model.MIPGap) if has_incumbent else None
            info.update({
                'status': int(model.Status),
                'has_incumbent': has_incumbent,
                'solution_count': int(model.SolCount),
                'timeout': model.Status == GRB.TIME_LIMIT,
                'gap': gap if gap is not None and math.isfinite(gap) else None,
                'gap_is_finite': gap is not None and math.isfinite(gap),
                'set_tsp_objective': float(model.ObjVal) if has_incumbent else None,
                'work': float(model.Work),
                'solver_runtime_seconds': float(model.Runtime),
            })
            seq = None
            if has_incumbent:
                # 仅在存在可行解时读取 X，并限制恢复步数，避免不完整环导致死循环。
                successors = np.argmax(select.X, axis=1)
                seq = [0]
                for _ in range(n):
                    seq.append(int(successors[seq[-1]]))
                if seq[-1] != 0 or sorted(seq[:-1]) != list(range(n)):
                    raise RuntimeError('Set-TSP 可行解未恢复为完整客户环。')
            info['phase2_extract_seconds'] = time.perf_counter() - extract_start
        if return_info:
            return seq, info
        if seq is None:
            raise RuntimeError('Set-TSP 没有可行解；请通过 get_seq 使用统一回退。')
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

    def get_seq(self, depot, convex_sets, *, cities=None, solver_options=None, return_info=False):
        """
        获取当前仓库组的客户访问顺序。

        输入：
        - depot: 当前仓库。
        - convex_sets: 当前仓库组的候选点集合列表。
        - cities: 可显式传入客户组，避免依赖或修改模型中的当前分区。
        - solver_options: 所有分区共用的求解配置。
        - return_info: 是否返回距离准备、求解、回退的计时与状态。

        输出：
        - 客户顺序索引序列，或 `(序列, 计时与状态字典)`。

        实现逻辑：
        - 若关闭集合 TSP，则调用 `lkh`；
          否则构造集合化距离并调用 `set_tsp`。
        """
        # 显式客户参数使固定分区评价无需修改 self.groups。
        cities = list(self.groups[depot] if cities is None else cities)
        info = {
            'phase2_distance_seconds': 0.0, 'phase2_build_seconds': 0.0,
            'phase2_optimize_seconds': 0.0, 'phase2_extract_seconds': 0.0,
            'phase2_fallback_seconds': 0.0, 'fallback_used': False,
            'status': 'EMPTY', 'has_incumbent': False, 'solution_count': 0,
            'timeout': False, 'gap': None, 'gap_is_finite': False, 'set_tsp_objective': None,
            'num_vars': 0, 'num_binary': 0, 'num_constrs': 0, 'work': 0.0,
            'solver_runtime_seconds': 0.0, 'set_tsp_solver': 'none',
        }
        if not cities:
            seq = [0, 0]
        elif self.theta[1] == 0:
            start = time.perf_counter()
            seq = self.lkh(depot, cities)
            info.update(status='LKH', set_tsp_solver='LKH')
            info['phase2_optimize_seconds'] = time.perf_counter() - start
        elif any(not nodes for nodes in convex_sets):
            seq = None
            info.update(status='EMPTY_BOUNDARY', set_tsp_solver='Set-TSP')
        else:
            start = time.perf_counter()
            set_distance = [[[max(self.distance['truck'][k][j],
                                  self.cut_off((self.distance['drone'][j][city] + self.distance['drone'][city][k]),
                                               self.limit)) / self.speed for j in convex_set] for k in convex_set]
                            for convex_set, city in zip(convex_sets, [depot] + cities)]
            distance = [[[[self.distance['truck'][i][j] for j in v] for i in u]
                         for v in convex_sets] for u in convex_sets]
            info['phase2_distance_seconds'] = time.perf_counter() - start
            seq, solver_info = self.set_tsp(
                convex_sets, distance, set_distance,
                solver_options=solver_options, return_info=True,
            )
            info.update(solver_info, set_tsp_solver='Set-TSP')
        if seq is None:
            start = time.perf_counter()
            seq = self.nearest_neighbor_sequence(depot, cities)
            info.update(fallback_used=True, fallback_method='directed_nearest_neighbor')
            info['phase2_fallback_seconds'] = time.perf_counter() - start
        return (seq, info) if return_info else seq

    def nearest_neighbor_sequence(self, depot, cities):
        """输入仓库和客户，按有向卡车距离贪心生成闭环索引；距离相同时按索引确定顺序。"""
        nodes = [depot] + list(cities)
        remaining = set(range(1, len(nodes)))
        sequence = [0]
        while remaining:
            chosen = min(remaining, key=lambda i: (self.distance['truck'][nodes[sequence[-1]]][nodes[i]], i))
            remaining.remove(chosen)
            sequence.append(chosen)
        return sequence + [0]

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
        #构造空字典，键为客户，值为空，存储候选点
        convex_sets = {city: [] for city in self.cities}
        #遍历所有点，对每个点找到最近的客户，并将点归到该客户名下
        for node in self.graph.nodes:
            closest_city = None
            closest_distance = self.limit * theta
            for city in self.cities:
                if self.distance['drone'][node][city] <= closest_distance:
                    closest_city = city
                    closest_distance = self.distance['drone'][node][city]
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
        boundary_convex_sets = {city: [] for city in self.cities}
        #遍历每个客户的，服务范围内的点的，邻居节点
        for city in self.cities:
            for node in convex_sets[city]:
                for neighbor in nx.neighbors(self.graph, node):
                    #若该点的邻居点不在服务范围（集合）内，则说明该点是边界点
                    if neighbor not in convex_sets[city]:
                        boundary_convex_sets[city].append(node)
                        break
        return boundary_convex_sets

    def solve(self, *, partition=None, partition_strategy='original_mst', solver_options=None, repair_options=None):
        """
        执行论文主算法的完整求解流程。

        输入：
        - partition: 可选完整分区；传入时直接评价该分区。
        - partition_strategy: 分区方法名称，默认 original_mst。
        - solver_options、repair_options: 求解配置与候选生成配置。

        输出：
        - `(self.solution, self.cost)`。

        实现逻辑：
        1. 先构造边界候选点集合。
        2. 再用 MST 给客户分仓库。
        3. 对每个仓库单独求解联合路线。
        4. 汇总所有仓库的结果。
        """
        from .partition_repair.evaluator import solve_with_records

        solution, cost, _ = solve_with_records(
            self, partition=partition, partition_strategy=partition_strategy,
            solver_options=solver_options, repair_options=repair_options,
        )
        return solution, cost

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
