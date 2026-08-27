"""
本文件实现论文中的主算法 `MultiAgentFlyingSidekickTSP`。

主要内容：
1. 先为客户构造候选服务区域（文中称 convex sets）。
2. 使用配置的 SMST 或 Directed Set-GTDS 把客户划分给不同仓库。
3. 对每个仓库组求一个集合化 TSP 顺序。
4. 最后通过动态规划近似联合优化卡车与多架无人机的同步路线。
"""

import elkai
import math
import time
import networkx as nx
import numpy as np
from .baseline import Baseline
from .partition import normalize_candidate_sets, set_gtds_partition
from .set_tsp_solver import SetTSPSolveResult, solve_set_tsp
from utils import mst_partition


class InstanceTimeLimitExceeded(TimeoutError):
    """表示实例级时间预算已在不可中断的 Phase 3 计算中耗尽。"""


def _check_instance_deadline(deadline):
    """检查绝对截止时间；输入为 ``perf_counter`` 时间戳，超时则抛出异常。"""

    if deadline is not None and time.perf_counter() >= deadline:
        raise InstanceTimeLimitExceeded('实例级时间预算已耗尽')


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
    def __init__(
        self,
        graph,
        depots,
        cities,
        distance,
        drone,
        limit=1.5,
        speed=1.6,
        theta=(0.5, 0.5),
        partition_method='smst_original',
        partition_epsilon=0.01,
        partition_scale=1000,
        partition_min_active_depots=None,
        partition_active_depot_policy='all',
        partition_drone_cost_mode='paper',
        set_tsp_time_limit=None,
        precomputed_partition_result=None,
    ):
        super().__init__(graph, depots, cities, distance, drone, limit, speed)
        self.groups = {depot: [] for depot in depots}
        self.solution = []
        self.cost = 0
        self.theta = theta
        self.const = math.sqrt(2)
        # 第一阶段默认继续使用公开代码中的 SMST，实验入口可显式切换到新方法。
        self.partition_method = partition_method
        self.partition_epsilon = partition_epsilon
        self.partition_scale = partition_scale
        # 显式整数约束用于兼容旧实验；未指定时由 ``all/free`` 策略统一决定。
        self.partition_min_active_depots = (
            None
            if partition_min_active_depots is None
            else int(partition_min_active_depots)
        )
        self.partition_active_depot_policy = partition_active_depot_policy
        # GTDS 主实验遵循论文式（3）的 1/speed；sqrt(2)/speed 仅用于敏感性实验。
        self.partition_drone_cost_mode = partition_drone_cost_mode
        # 默认不设置 Set-TSP 上限；实验协议可以显式传入正数以研究超时行为。
        self.set_tsp_time_limit = set_tsp_time_limit
        self.partition_result = precomputed_partition_result
        self.precomputed_partition_result = precomputed_partition_result

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

    def set_gtds(
        self,
        candidate_sets,
        apply_model_budget=True,
        epsilon=None,
        min_active_depots=None,
        active_depot_policy=None,
        drone_cost_mode=None,
    ):
        """
        使用 Directed Set-GTDS 将客户划分给各仓库。

        输入：正规化候选集合、模型预算开关、epsilon、活跃仓库策略和无人机代价模式。
        输出：无显式返回值；分组写入 ``self.groups``，诊断结果写入
        ``self.partition_result``。
        逻辑：把当前模型的仓库、客户、距离与速度交给独立分区模块，Phase 2/3
        仍沿用本类原有实现。
        """

        # 显式参数优先于模型默认值，使具名消融方法不会隐式修改模型配置。
        effective_epsilon = (
            self.partition_epsilon if epsilon is None else float(epsilon)
        )
        effective_min_active = (
            self.partition_min_active_depots
            if min_active_depots is None
            else int(min_active_depots)
        )
        effective_policy = (
            self.partition_active_depot_policy
            if active_depot_policy is None
            else active_depot_policy
        )
        effective_drone_cost_mode = (
            self.partition_drone_cost_mode
            if drone_cost_mode is None
            else drone_cost_mode
        )
        result = set_gtds_partition(
            depots=list(self.depots),
            cities=list(self.cities),
            candidate_sets=candidate_sets,
            truck_distance=self.distance['truck'],
            drone_distance=self.distance['drone'],
            speed=self.speed,
            epsilon=effective_epsilon,
            scale=self.partition_scale,
            apply_model_budget=apply_model_budget,
            min_active_depots=effective_min_active,
            active_depot_policy=effective_policy,
            drone_cost_mode=effective_drone_cost_mode,
        )
        self.groups = result.groups
        self.partition_result = result

    def partition_customers(self, candidate_sets):
        """
        按配置选择第一阶段客户划分方法。

        输入：Phase 1 与 Phase 2 共用的正规化候选集合。
        输出：无显式返回值；更新 ``self.groups`` 和可选的分区诊断结果。
        逻辑：默认调用原始 SMST；新方法和无预算消融只通过显式方法名启用。
        """

        self.groups = {depot: [] for depot in self.depots}
        if self.precomputed_partition_result is not None:
            self.partition_result = self.precomputed_partition_result
            self.groups = {
                depot: list(self.partition_result.groups.get(depot, []))
                for depot in self.depots
            }
            return
        self.partition_result = None
        if self.partition_method == 'smst_original':
            self.set_mst(candidate_sets)
        elif self.partition_method == 'snn':
            self.set_nn(self.theta[0])
        elif self.partition_method == 'directed_set_gtds':
            self.set_gtds(
                candidate_sets,
                apply_model_budget=True,
                active_depot_policy='all',
                drone_cost_mode='paper',
            )
        elif self.partition_method == 'set_gtds_no_budget':
            self.set_gtds(
                candidate_sets,
                apply_model_budget=False,
                active_depot_policy='all',
                drone_cost_mode='paper',
            )
        elif self.partition_method == 'gtds_sqrt2':
            self.set_gtds(
                candidate_sets,
                apply_model_budget=True,
                active_depot_policy='all',
                drone_cost_mode='smst_compatible',
            )
        elif self.partition_method == 'gtds_free_eps01':
            self.set_gtds(
                candidate_sets,
                epsilon=0.01,
                active_depot_policy='free',
            )
        elif self.partition_method == 'gtds_free_eps05':
            self.set_gtds(candidate_sets, epsilon=0.05, min_active_depots=1)
        elif self.partition_method == 'gtds_free_eps10':
            self.set_gtds(candidate_sets, epsilon=0.10, min_active_depots=1)
        elif self.partition_method == 'gtds_all_eps01':
            self.set_gtds(
                candidate_sets,
                epsilon=0.01,
                active_depot_policy='all',
            )
        elif self.partition_method.startswith('gtds_all_eps'):
            epsilon_codes = {
                '000': 0.0,
                '005': 0.005,
                '010': 0.01,
                '020': 0.02,
                '050': 0.05,
            }
            epsilon_code = self.partition_method.removeprefix('gtds_all_eps')
            if epsilon_code not in epsilon_codes:
                raise ValueError(f'未知的 epsilon 敏感性方法：{self.partition_method!r}')
            self.set_gtds(
                candidate_sets,
                epsilon=epsilon_codes[epsilon_code],
                active_depot_policy='all',
                drone_cost_mode='paper',
            )
        else:
            raise ValueError(f'未知的第一阶段划分方法：{self.partition_method!r}')

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

    def local_search_multi_drone_appr(self, seq, depot, deadline=None):
        """
        在给定客户顺序下，用动态规划近似优化多无人机联合调度。

        输入：
        - seq: 客户访问顺序，首尾均为仓库。
        - depot: 当前仓库节点。
        - deadline: 可选的 ``perf_counter`` 绝对截止时间。

        输出：
        - `(solution, cost)`：
          - `solution`: 当前仓库的卡车/无人机联合路线。
          - `cost`: 对应完工时间。

        实现逻辑：
        1. 为每个阶段维护最优值函数 `value`。
        2. 用 `appr` 近似估计同时投放多架无人机时的代价。
        3. 通过前缀 DP 累积卡车与无人机联合路线。
        """



        _check_instance_deadline(deadline)
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
            # Phase 3 不依赖外部求解器，因此在主要 DP 层之间主动检查实例级截止时间。
            _check_instance_deadline(deadline)
            for node in self.regions[seq[i]]:
                _check_instance_deadline(deadline)
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
                _check_instance_deadline(deadline)
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
            _check_instance_deadline(deadline)
            for node in self.regions[seq[i]]:
                _check_instance_deadline(deadline)
                for _node in self.regions[seq[i - 1]]:
                    # initialize value as Eq. (12)
                    if value[2 * i - 2][_node] + self.distance['truck'][_node][node] < value[2 * i - 1][node]:
                        value[2 * i - 1][node] = value[2 * i - 2][_node] + self.distance['truck'][_node][node]
                        prefix[2 * i - 1][node]['truck'] = prefix[2 * i - 2][_node]['truck'].copy() + [node]
                        prefix[2 * i - 1][node]['drone'] = prefix[2 * i - 2][_node]['drone'].copy()

            for j in range(min(self.drone, i)):
                _check_instance_deadline(deadline)
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

        _check_instance_deadline(deadline)
        for node in self.regions[seq[-2]]:
            # after visiting the last customer, return to the depot
            if value[2 * len(seq) - 4][node] + self.distance['truck'][node][depot] < value[2 * len(seq) - 3][depot]:
                value[2 * len(seq) - 3][depot] = value[2 * len(seq) - 4][node] + self.distance['truck'][node][depot]
                prefix[2 * len(seq) - 3][depot]['truck'] = prefix[2 * len(seq) - 4][node]['truck'].copy() + [depot]
                prefix[2 * len(seq) - 3][depot]['drone'] = prefix[2 * len(seq) - 4][node]['drone'].copy()

        return prefix[2 * len(seq) - 3][depot], value[2 * len(seq) - 3][depot]


    @staticmethod
    def set_tsp(
        convex_sets,
        distance,
        convex_set_distance,
        time_limit=None,
    ):
        """
        求解 Set-TSP 并返回访问序列的兼容接口。

        输入为候选集合、两类代价和可选时限；内部调用独立求解模块。输出访问序列；
        若没有 incumbent 则抛出带状态信息的异常，正式实验应使用 ``get_seq_result``
        读取结构化失败状态。
        """

        result = solve_set_tsp(
            convex_sets,
            distance,
            convex_set_distance,
            time_limit=time_limit,
        )
        if result.sequence is None:
            raise RuntimeError(
                f'Set-TSP 未产生可用解：{result.status}; '
                f'{result.error_message or "无额外错误信息"}'
            )
        return result.sequence

    def get_seq_result(self, depot, convex_sets, time_limit=None):
        """
        求解当前仓库的访问顺序并返回统一遥测对象。

        输入为仓库节点、该仓库的候选集合列表和可选本次时限；LKH 分支生成零模型
        规模的启发式遥测，Set-TSP 分支记录真实建模/优化状态。输出 ``SetTSPSolveResult``。
        """

        if self.theta[1] == 0:
            start = time.perf_counter()
            sequence = self.lkh(depot, self.groups[depot])
            return SetTSPSolveResult(
                sequence=sequence,
                status='heuristic_complete',
                objective=None,
                build_seconds=0.0,
                optimize_seconds=time.perf_counter() - start,
                num_bin_vars=0,
                num_vars=0,
                num_constraints=0,
                node_count=None,
                mip_gap=None,
                solution_count=1,
                time_limit_reached=False,
                has_incumbent=True,
            )

        set_distance = [
            [
                [
                    max(
                        self.distance['truck'][leave][entry],
                        self.cut_off(
                            self.distance['drone'][entry][city]
                            + self.distance['drone'][city][leave],
                            self.limit,
                        ),
                    ) / self.speed
                    for entry in convex_set
                ]
                for leave in convex_set
            ]
            for convex_set, city in zip(
                convex_sets,
                [depot] + self.groups[depot],
            )
        ]
        distance = [
            [
                [
                    [self.distance['truck'][start][end] for end in target_set]
                    for start in source_set
                ]
                for target_set in convex_sets
            ]
            for source_set in convex_sets
        ]
        effective_time_limit = (
            self.set_tsp_time_limit if time_limit is None else time_limit
        )
        return solve_set_tsp(
            convex_sets,
            distance,
            set_distance,
            time_limit=effective_time_limit,
        )

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
        result = self.get_seq_result(depot, convex_sets)
        if result.sequence is None:
            raise RuntimeError(
                f'Set-TSP 未产生可用解：{result.status}; '
                f'{result.error_message or "无额外错误信息"}'
            )
        return result.sequence

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

    def solve(self):
        """
        执行论文主算法的完整求解流程。

        输入：
        - 无显式输入。

        输出：
        - `(self.solution, self.cost)`。

        实现逻辑：
        1. 先构造边界候选点集合。
        2. 再用配置的第一阶段方法给客户分仓库。
        3. 对每个仓库单独求解联合路线。
        4. 汇总所有仓库的结果。
        """
        #获取客户服务范围（集合）内的边界点，加速计算
        raw_sets = self.get_boundary_convex_sets(self.theta[0])
        # Phase 1 和 Phase 2 共享同一份非空候选集合，空边界退化为客户自身。
        convex_sets = normalize_candidate_sets(self.cities, raw_sets)
        self.partition_customers(convex_sets)
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
        raw_sets = self.get_boundary_convex_sets(self.theta[0])
        convex_sets = normalize_candidate_sets(self.cities, raw_sets)
        self.partition_customers(convex_sets)
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
