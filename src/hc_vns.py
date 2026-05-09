"""
本文件实现基线算法 Hill Climbing + Variable Neighborhood Search。

主要内容：
1. 先按最近仓库原则对客户做分组。
2. 为每组仓库/客户生成一条初始卡车访问序列。
3. 通过多种邻域操作持续改进解。
4. 最终将内部表示转换为统一的卡车-无人机路径格式。
"""

# 导入 `networkx`，用于最短路和近似 TSP。
import networkx as nx
# 导入 `numpy`，用于拼接 depot/city 列表以及数组计算。
import numpy as np
# 导入 `random`，用于随机选择邻域操作和随机节点。
import random
# 导入算法基类。
from .baseline import Baseline
# 导入球面距离函数，用于生成客户可服务邻居集合。
from utils import haversine


class HillClimbingVariableNeighborhoodSearch(Baseline):
    """
    基于爬山法和变邻域搜索的启发式算法。

    输入：
    - graph: 路网图。
    - depots: 仓库节点集合。
    - cities: 客户节点集合。
    - distance: 距离矩阵字典。
    - drone: 无人机数量。
    - limit: 无人机最大飞行距离。
    - speed: 无人机速度参数。
    - rounds: 邻域搜索迭代轮数。

    输出：
    - 无显式返回值，只初始化算法状态。

    实现逻辑：
    1. 初始化公共字段。
    2. 为每个仓库准备客户分组容器。
    3. 预先为每个仓库/客户构造可起降邻居集合，降低后续搜索成本。
    """

    def __init__(self, graph, depots, cities, distance, drone, limit=1.5, speed=1.6, rounds=1000):
        """
        初始化 HC-VNS 算法。

        输入：
        - graph: 路网图。
        - depots: 仓库集合。
        - cities: 客户集合。
        - distance: 卡车/无人机距离矩阵。
        - drone: 无人机数量。
        - limit: 飞行距离限制。
        - speed: 无人机速度参数。
        - rounds: 迭代次数。

        输出：
        - 无。

        实现逻辑：
        - 保存公共输入，并预处理每个客户附近可用于起飞/降落的路网节点。
        """
        # 调用父类初始化公共属性。
        super().__init__(graph, depots, cities, distance, drone, limit, speed)
        # 初始化仓库到客户列表的映射。
        self.groups = {depot: [] for depot in self.depots}
        # 保存局部搜索轮数。
        self.rounds = rounds
        # 初始化最终解列表。
        self.solution = []
        # 初始化总成本。
        self.cost = 0
        # 为每个仓库和客户预处理“在距离限制内的可达路网节点”。
        self.neighbors = {city: [node for node in self.graph.nodes
                                 if haversine(self.graph.nodes[city]['pos'], self.graph.nodes[node]['pos']) < limit]
                          for city in np.concatenate((depots, cities))}

    def partition(self):
        """
        按最近仓库原则将客户分配给不同卡车。

        输入：
        - 无显式输入，使用当前实例中的 `depots` 和 `cities`。

        输出：
        - 无显式返回值，结果写入 `self.groups`。

        实现逻辑：
        1. 对每个客户遍历所有仓库。
        2. 比较卡车从仓库到该客户的最短路距离。
        3. 将客户分给距离最近的仓库。
        """
        # 逐个处理所有客户。
        for city in self.cities:
            # 初始化当前客户到最近仓库的距离。
            distance = float('inf')
            # 初始化最近仓库为空。
            closest_depot = None
            # 遍历所有仓库，寻找最近者。
            for depot in self.depots:
                # 计算当前仓库到客户的路网最短路长度。
                path_length = nx.shortest_path_length(self.graph, source=depot, target=city, weight='weight')
                # 如果更短，则更新最优仓库。
                if path_length < distance:
                    # 更新最短距离。
                    distance = path_length
                    # 更新最近仓库。
                    closest_depot = depot
            # 将当前客户加入最近仓库的分组中。
            self.groups[closest_depot].append(city)

    def init_solution(self, nodes):
        """
        为给定的一组节点生成初始访问解。

        输入：
        - nodes: 一个仓库以及其所属客户节点组成的列表。

        输出：
        - 一个内部解表示列表，每个元素是 `(关键节点, 访问者编号, 客户标记)`。

        实现逻辑：
        1. 若节点很少，则直接写出显然解。
        2. 否则先构造完全图，边权取原图最短路。
        3. 再调用 `networkx` 的贪心 TSP 近似生成初始环路。
        """
        # 如果只有一个节点，则它自己出发再回自己。
        if len(nodes) == 1:
            return [(nodes[0], -1, nodes[0])]
        # 如果只有两个节点，则按仓库-客户-仓库构造显然路径。
        elif len(nodes) == 2:
            return [(nodes[0], -1, nodes[0]), (nodes[1], -1, nodes[1]), (nodes[0], -1, nodes[0])]
        # 新建一个完全图，用于近似 TSP。
        graph = nx.Graph()
        # 枚举所有节点对。
        for i in nodes:
            for j in nodes:
                # 以原图最短路长度作为完全图边权。
                graph.add_edge(i, j, weight=nx.dijkstra_path_length(self.graph, i, j, weight='weight'))
        # 在完全图上调用贪心 TSP 近似算法。
        solution = nx.approximation.traveling_salesman_problem(graph, weight='weight', nodes=nodes,
                                                               method=nx.approximation.greedy_tsp)
        # 初始时默认所有客户都由卡车访问，因此访问者编号记为 `-1`。
        solution = [(i, -1, i) for i in solution]
        # 返回初始解。
        return solution

    def neighborhood_search(self, solution):
        """
        对当前解随机选择一种邻域操作进行改进。

        输入：
        - solution: 当前内部解表示。

        输出：
        - 改进后的解；若没有找到更优邻域，则返回原解。

        实现逻辑：
        1. 随机在三种核心邻域中选一种。
        2. 尝试修改起降点、改为无人机服务、或交换访问顺序。
        3. 只有当成本下降时才接受新解。
        """
        # 随机选择一种邻域编号。
        num = random.randint(1, 5)
        # 计算当前解成本。
        cost = self.solution_cost(solution)
        # 邻域 1：改变无人机起降节点。
        if num == 1:
            # 找出所有由无人机服务的节点位置。
            free_nodes = [i for i in range(len(solution)) if solution[i][1] > -1]
            # 若当前没有无人机任务，则无法应用该邻域。
            if len(free_nodes) == 0:
                return solution
            # 随机选择一个无人机任务位置。
            index = random.choice(free_nodes)
            # 复制当前解，准备尝试修改。
            _solution = solution.copy()
            # 遍历该任务的相邻候选起降节点。
            for node in self.graph.neighbors(solution[index][0]):
                # 用新节点替换当前任务的关键节点。
                _solution[index] = (node, solution[index][1], solution[index][2])
                # 计算修改后的成本。
                _cost = self.solution_cost(_solution)
                # 若成本更低，则接受这个新解。
                if _cost < cost:
                    solution = _solution
                    cost = _cost
        # 邻域 2：尝试把一个卡车服务客户改成无人机服务。
        if num == 2:
            # 找出当前仍由卡车直接服务的内部节点。
            truck_nodes = [i for i in range(1, len(solution) - 1) if solution[i][1] == -1]
            # 若不存在可替换的卡车节点，则返回原解。
            if len(truck_nodes) == 0:
                return solution
            # 随机选择一个待尝试替换的客户位置。
            index = random.choice(truck_nodes)
            # 初始化每架无人机的空闲状态。
            drones_on_sky = [0 for _ in range(self.drone)]
            # 扫描到当前索引之前的任务，判断无人机是否仍在空中。
            for i in range(index):
                _, d, city = solution[i]
                if d > -1:
                    drones_on_sky[d] += 1 if city > 0 else -1
            # 收集当前可用的无人机编号。
            free_drones = [i for i in range(self.drone) if drones_on_sky[i] == 0]
            # 若没有空闲无人机，则无法应用该邻域。
            if len(free_drones) == 0:
                return solution
            # 随机选一架空闲无人机。
            drone = random.choice(free_drones)
            # 当前待替换客户编号。
            city = solution[index][2]
            # 最优起飞/降落点尚未确定。
            best_pair = None
            # 枚举所有可能的起飞点。
            for start in self.neighbors[city]:
                # 枚举所有可能的降落点。
                for end in self.neighbors[city]:
                    # 构造替换后的新解。
                    _solution = solution[:index].copy() + [(start, drone, city), (end, drone, -city)] + \
                                solution[index + 1:].copy()
                    # 计算替换后的成本。
                    _cost = self.solution_cost(_solution)
                    # 若更优，则更新最优起降点。
                    if _cost < cost:
                        cost = _cost
                        best_pair = (start, end)
            # 若找到有效改进，则用最佳起降点更新解。
            if best_pair is not None:
                solution = solution[:index].copy() + [(best_pair[0], drone, city), (best_pair[1], drone, -city)] + \
                           solution[index + 1:].copy()
        # 邻域 3：交换相邻访问顺序。
        if num == 3:
            # 初始化最优交换位置为空。
            best_pair = None
            # 枚举所有相邻元素对。
            for i in range(1, len(solution) - 2):
                # 尝试交换当前位置和后一位置。
                _solution = solution[:i].copy() + [solution[i + 1]] + [solution[i]] + solution[i + 2:].copy()
                # 计算交换后的成本。
                _cost = self.solution_cost(_solution)
                # 若更优，则记录该交换。
                if _cost < cost:
                    cost = _cost
                    best_pair = (i, i + 1)
            # 若找到更优交换，则正式更新解。
            if best_pair is not None:
                solution = solution[:best_pair[0]].copy() + [solution[best_pair[1]]] + [solution[best_pair[0]]] + \
                           solution[best_pair[1] + 1:].copy()
        # 返回经过邻域搜索后的解。
        return solution

    def solution_cost(self, solution):
        """
        计算内部解表示对应的完成时间成本。

        输入：
        - solution: 内部路线表示列表。

        输出：
        - 路线最终完工时间。

        实现逻辑：
        1. 顺着卡车主路线累计卡车时间。
        2. 当发现某个无人机回收事件时，再回头找到对应起飞事件。
        3. 用“卡车时间”和“无人机飞行时间”取最大值，体现同步约束。
        """
        # 为解中每个事件位置初始化一个累计时间。
        cost = [0 for _ in solution]
        # 从第二个事件开始顺序计算累计时间。
        for i in range(1, len(solution)):
            # 先加上卡车从上一个关键节点到当前关键节点的时间。
            cost[i] = cost[i - 1] + self.distance['truck'][solution[i - 1][0]][solution[i][0]]
            # 如果当前事件是无人机降落事件，则需要考虑无人机同步约束。
            if solution[i][1] > -1 and solution[i][2] < 0:
                # 向前寻找与之配对的起飞事件。
                for j in range(1, i):
                    # 若起飞事件和降落事件的城市编号相反，则说明它们是一对。
                    if solution[j][-1] + solution[i][-1] == 0:
                        # 计算无人机从起飞点到客户再到降落点的飞行距离。
                        drone_distance = self.distance['drone'][solution[j][0]][solution[j][-1]] + \
                                         self.distance['drone'][solution[j][-1]][solution[i][0]]
                        # 若超过续航上限，则用极大罚值惩罚。
                        drone_distance = 1000000 if drone_distance > self.limit else drone_distance
                        # 当前时间必须不早于卡车时间和无人机完成时间两者的最大值。
                        cost[i] = max(cost[i], cost[j] + drone_distance / self.speed)
        # 返回整条路线的最终完工时间。
        return cost[-1]

    def convert(self, solution):
        """
        将内部解表示转换为统一的输出格式。

        输入：
        - solution: 内部事件列表。

        输出：
        - `{'truck': ..., 'drone': ...}` 格式的统一解。

        实现逻辑：
        1. 直接提取卡车关键节点序列。
        2. 扫描所有起飞/降落成对事件。
        3. 把每一对事件转换为 `[起飞点, 客户, 降落点]` 三元组。
        """
        # 初始化统一格式解，其中卡车路线直接取每个事件的关键节点。
        route = {'truck': [node for node, _, _ in solution], 'drone': [[] for d in range(self.drone)]}
        # 枚举所有可能的起飞事件位置。
        for i in range(1, len(solution) - 1):
            # 枚举其后的所有可能降落事件位置。
            for j in range(i + 1, len(solution) - 1):
                # 若两者城市编号互为相反数，则是一对无人机任务。
                if solution[i][-1] + solution[j][-1] == 0:
                    # 取正值作为客户编号。
                    city = max(solution[i][-1], solution[j][-1])
                    # 将该无人机任务加入对应无人机的路线中。
                    route['drone'][solution[i][1]].append([solution[i][0], city, solution[j][0]])
        # 返回统一格式结果。
        return route

    def single_solution(self, depot):
        """
        求解单个仓库分组对应的局部路线。

        输入：
        - depot: 当前仓库节点。

        输出：
        - 无显式返回值，结果写入 `self.solution` 与 `self.cost`。

        实现逻辑：
        1. 为该仓库生成初始解。
        2. 反复执行邻域搜索。
        3. 将最终内部解转换为统一格式并加入总解。
        """
        # 基于该仓库及其客户生成初始解。
        solution = self.init_solution([depot] + self.groups[depot])
        # 按设定轮数迭代执行邻域搜索。
        for _ in range(self.rounds):
            solution = self.neighborhood_search(solution)
        # 将当前仓库对应的最终成本累加到总成本中。
        self.cost += self.solution_cost(solution)
        # 将内部解转换为统一输出格式。
        _solution = self.convert(solution)
        # 将当前仓库的解加入总解列表。
        self.solution.append(_solution.copy())

    def solve(self):
        """
        运行 HC-VNS 整体求解流程。

        输入：
        - 无显式输入，使用实例中的图与节点集合。

        输出：
        - `(self.solution, self.cost)`。

        实现逻辑：
        1. 先按最近仓库将客户分组。
        2. 对每个仓库独立执行一次局部搜索求解。
        3. 汇总所有仓库的解与成本。
        """
        # 先为所有客户完成仓库分组。
        self.partition()
        # 对每个仓库分别求解其局部路线。
        for depot in self.depots:
            self.single_solution(depot)
        # 返回总解和总成本。
        return self.solution, self.cost
