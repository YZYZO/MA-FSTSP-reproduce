"""
本文件存放整个项目的通用工具函数。

主要内容：
1. 基础距离函数，如欧氏距离与球面距离。
2. 一些简单辅助函数，如符号函数、最近邻搜索、进制转换。
3. 与图算法相关的工具，如 MST 分组和 ATSP 近似求解。
"""

# 导入 `math`，用于三角函数、平方根和弧度转换。
import math
# 导入 `networkx`，用于图算法与近似 TSP。
import networkx as nx
# 导入 `numpy`，用于数组与最小值索引计算。
import numpy as np
from pathlib import Path

from config import RESULTS_DIR


def result_path(*parts) -> Path:
    """
    构造项目结果目录下的路径，但不创建任何目录或文件。

    输入：
    - parts: 依次追加到结果根目录的路径片段。

    输出：
    - 以 `RESULTS_DIR` 为根的 `Path` 对象。
    """
    return RESULTS_DIR.joinpath(*parts)


def ensure_dir(path) -> Path:
    """
    确保目标目录及其父目录存在，并返回规范化后的路径对象。

    输入：
    - path: 待创建的字符串路径或 `Path` 对象。

    输出：
    - 已存在的目录 `Path`；目录缺失时递归创建。
    """
    # 统一转换为 `Path`，使调用方既可以传字符串，也可以传路径对象。
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def euclidean(a, b):
    """
    计算二维平面上两点之间的欧氏距离。

    输入：
    - a: 第一个点，形如 `(x, y)` 或 `[x, y]`。
    - b: 第二个点，形如 `(x, y)` 或 `[x, y]`。

    输出：
    - 两点之间的欧氏距离，浮点数。

    实现逻辑：
    - 直接套用二维欧氏距离公式。
    """
    # 返回两点之间的欧氏距离。
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def haversine(pos1, pos2):
    """
    根据经纬度近似计算地球表面两点间距离，单位为公里。

    输入：
    - pos1: 第一个位置，形如 `[lon, lat]`。
    - pos2: 第二个位置，形如 `[lon, lat]`。

    输出：
    - 两点之间的球面距离，单位公里。

    实现逻辑：
    1. 先将经纬度从角度转换为弧度。
    2. 用局部近似形式计算平面化后的球面距离。
    3. 用地球半径将其换算为公里。
    """
    # 解析第一个点的经度与纬度。
    lon1, lat1 = pos1
    # 解析第二个点的经度与纬度。
    lon2, lat2 = pos2

    # 定义地球平均半径，单位公里。
    radius = 6371.0

    # 将第一个点经度转换成弧度。
    lon1_rad = math.radians(lon1)
    # 将第一个点纬度转换成弧度。
    lat1_rad = math.radians(lat1)
    # 将第二个点经度转换成弧度。
    lon2_rad = math.radians(lon2)
    # 将第二个点纬度转换成弧度。
    lat2_rad = math.radians(lat2)

    # 计算经度方向上的局部投影差值。
    x = (lon2_rad - lon1_rad) * math.cos(0.5 * (lat2_rad + lat1_rad))
    # 计算纬度方向上的差值。
    y = lat2_rad - lat1_rad

    # 返回最终距离。
    return radius * math.sqrt(x ** 2 + y ** 2)


def sign(number):
    """
    返回一个数的符号。

    输入：
    - number: 任意实数。

    输出：
    - 若小于 0 返回 `-1`，否则返回 `1`。

    实现逻辑：
    - 使用简单条件表达式判断符号。
    """
    # 若 number 小于 0，则返回 -1，否则返回 1。
    return -1 if number < 0 else 1


def nearest_node(graph, location):
    """
    在图中找到距离给定位置最近的节点。

    输入：
    - graph: 图对象，节点上需要有 `pos` 属性。
    - location: 给定位置，经纬度形式。

    输出：
    - 最近节点的编号。

    实现逻辑：
    1. 遍历所有节点。
    2. 计算给定位置到每个节点的球面距离。
    3. 返回最小距离对应的节点。
    """
    # 初始化当前最小距离为正无穷。
    distance = float('inf')
    # 初始化最近节点为空。
    nearest = None
    # 遍历图中的每一个节点。
    for node in graph.nodes:
        # 如果当前节点更近，则更新答案。
        if haversine(location, graph.nodes[node]['pos']) < distance:
            # 更新最小距离。
            distance = haversine(location, graph.nodes[node]['pos'])
            # 更新最近节点编号。
            nearest = node
    # 断言最终一定找到最近节点。
    assert nearest is not None, "no node closer than infinity, check the code"
    # 返回最近节点。
    return nearest


def nearest_node_except_self(graph, name):
    """
    在图中找到某个节点的最近邻节点，但不允许返回它本身。

    输入：
    - graph: 图对象。
    - name: 目标节点编号。

    输出：
    - 与该节点直接相邻且边权最小的邻居节点编号。

    实现逻辑：
    1. 遍历目标节点的所有邻居。
    2. 比较与邻居的边权。
    3. 返回边权最小的那个邻居。
    """
    # 初始化最小距离和最近邻为空。
    distance, nearest = float('inf'), None
    # 遍历目标节点的所有邻居。
    for neighbor in graph.neighbors(name):
        # 如果当前边权更小，则更新最近邻。
        if graph.edges[name, neighbor]['weight'] < distance:
            # 更新最近邻编号。
            nearest = neighbor
            # 更新最小边权。
            distance = graph.edges[name, neighbor]['weight']
    # 断言一定找到至少一个邻居。
    assert nearest is not None, "have no neighbor or all neighbors are further than infinity"
    # 断言最近邻不是它自己。
    assert nearest != name, "find self to be neighbor"
    # 返回最近邻节点。
    return nearest


def base_convert(number, i, j):
    """
    进行简单的进制拆分辅助，但当前实现只生成 `number` 在 `i` 进制下的数位列表。

    输入：
    - number: 十进制整数。
    - i: 拆分时使用的基数。
    - j: 保留参数，当前实现未实际使用。

    输出：
    - 一个列表，表示 `number` 用 `i` 进制拆分后的数字序列。

    实现逻辑：
    1. 持续对 `number` 取模并整除。
    2. 收集得到的低位到高位数字。
    3. 最后反转列表得到正常顺序。
    """
    # 初始化结果列表。
    ans = []
    # 当 number 仍大于 0 时持续拆分。
    while number > 0:
        # 记录当前最低位数字。
        ans.append(number % i)
        # 去掉最低位。
        number = number // i
    # 将列表反转成高位在前的顺序。
    ans.reverse()
    # 返回结果列表。
    return ans


def mst_partition(graph, depots, cities):
    """
    基于最小生成树将客户分配给不同仓库。

    输入：
    - graph: 完全图或带权图，节点中同时包含仓库和客户。
    - depots: 仓库节点数组。
    - cities: 客户节点数组。

    输出：
    - `groups` 字典，键为仓库节点，值为分配给该仓库的客户列表。

    实现逻辑：
    1. 先在图上构造最小生成树。
    2. 通过递归 DP 计算每个子树连接到仓库的最佳方式。
    3. 再通过第二次递归回溯每个客户应属于哪个仓库。
    """
    # 在输入图上构造最小生成树。
    tree = nx.minimum_spanning_tree(graph)

    # 初始化每个节点的父节点标记，`-1` 表示尚未设置。
    for i in tree.nodes:
        tree.nodes[i]['parent'] = -1

    def rooted_tree(node):
        """
        对最小生成树做一次自底向上的动态规划。

        输入：
        - node: 当前递归处理的节点。

        输出：
        - `(con, ncon)`：
          - `con`: 当前子树内部与某个仓库连通时的最小代价。
          - `ncon`: 当前子树不与内部仓库连通时的最小代价。

        实现逻辑：
        1. 递归处理所有孩子。
        2. 统计每个孩子子树的连通/非连通代价。
        3. 选择最优孩子作为与仓库相连的通道。
        """
        # `cons` 存储子树在“与仓库连通”情况下的代价。
        cons, ncons, index, diff = [], [], [], []
        # 遍历当前节点的所有邻居。
        for n in tree.neighbors(node):
            # 跳过父节点，只处理孩子。
            if n != tree.nodes[node]['parent']:
                # 先假设孩子子树通过父边与仓库连通。
                tree.nodes[n]['pcon'] = True
                # 记录孩子的父节点。
                tree.nodes[n]['parent'] = node
                # 递归求出孩子子树的两种代价。
                con, ncon = rooted_tree(n)
                # 记录孩子的连通代价。
                tree.nodes[n]['con'] = con
                # 记录孩子的非连通代价。
                tree.nodes[n]['ncon'] = ncon
                # 如果孩子本身就是仓库，则其连通方式比较特殊。
                if n in depots:
                    # 记录当前孩子走父边连接时的总代价。
                    cons.append(con + tree[node][n]['weight'])
                    # 记录孩子不走父边时的代价。
                    ncons.append(con)
                    # 记录连通与非连通方案的差值。
                    diff.append(cons[-1] - ncons[-1])
                    # 记录对应的孩子编号。
                    index.append(n)
                # 如果孩子是客户，且它的子树内部存在仓库。
                elif con != -1:
                    # 记录孩子走父边连接时的代价。
                    cons.append(con + tree[node][n]['weight'])
                    # 记录孩子不走父边时的最优代价。
                    ncons.append(min(ncon + tree[node][n]['weight'], con))
                    # 计算两类方案的差值。
                    diff.append(cons[-1] - ncons[-1])
                    # 保存对应的孩子编号。
                    index.append(n)
                    # 如果不连父边更优，则标记它不会向上连接。
                    if ncons[-1] == con:
                        tree.nodes[n]['pcon'] = False
                else:
                    # 如果孩子子树内没有仓库，只能把该子树整体向上带走。
                    ncons.append(ncon + tree[node][n]['weight'])
        # 如果当前节点就是仓库，则它天然连通到自己。
        if node in depots:
            # 当前子树连通代价为所有孩子非连通代价之和。
            con = sum(ncons)
            # 仓库节点不需要定义“非连通”方案。
            ncon = -1
        elif len(diff) == 0:
            # 如果没有任何能接到仓库的孩子，则当前子树无法内部连通到仓库。
            con = -1
            # 非连通代价就是所有孩子非连通代价之和。
            ncon = sum(ncons)
        else:
            # 选择差值最小的那个孩子作为仓库连接通道。
            id = np.argmin(np.array(diff))
            # 当前节点“连通”时的最优代价。
            con = min(diff) + sum(ncons)
            # 当前节点“非连通”时的代价。
            ncon = sum(ncons)
            # 记录最佳通道孩子。
            tree.nodes[node]['child'] = index[id]
        # 返回两种状态的最优值。
        return con, ncon

    # 从第一个仓库开始做一次树形 DP。
    con, ncon = rooted_tree(depots[0])
    # 记录根仓库的连通代价。
    tree.nodes[depots[0]]['con'] = con
    # 记录根仓库的非连通代价。
    tree.nodes[depots[0]]['ncon'] = ncon
    # 根节点没有父边，因此不向上连通。
    tree.nodes[depots[0]]['pcon'] = False

    def assign_group(node, value):
        """
        根据前一阶段 DP 的结果，将每个客户真正归属到某个仓库。

        输入：
        - node: 当前处理的节点。
        - value: 当前节点在递归路径上采用的状态值。

        输出：
        - 当前节点所属仓库组的编号。

        实现逻辑：
        1. 按仓库节点、连接到子树内仓库的客户节点、连接到外部仓库的客户节点三类分别处理。
        2. 沿着最优通路继续递归，给所有节点写入 `group` 字段。
        """
        # 如果当前节点本身是仓库。
        if node in depots:
            # 将其分组编号设为它在 `depots` 中的位置。
            tree.nodes[node]['group'] = np.where(depots == node)[0].item()
            # 遍历它的所有孩子。
            for n in tree.neighbors(node):
                # 跳过父节点。
                if n != tree.nodes[node]['parent']:
                    # 如果孩子也是仓库，则它自成一组。
                    if n in depots:
                        tree.nodes[n]['group'] = np.where(depots == n)[0].item()
                        assign_group(n, tree.nodes[n]['con'])
                    # 如果孩子通过当前节点向上连接，则它继承当前仓库组。
                    elif tree.nodes[n]['pcon']:
                        tree.nodes[n]['group'] = tree.nodes[node]['group']
                        assign_group(n, tree.nodes[n]['ncon'])
                    else:
                        # 否则它会连接到自身子树中的某个仓库。
                        tree.nodes[n]['group'] = assign_group(n, tree.nodes[n]['con'])

        # 如果当前节点是客户，且它连接到自身子树中的仓库。
        elif value == tree.nodes[node]['con']:
            # 找到最佳通路上的那个孩子。
            n = tree.nodes[node]['child']
            # 若该孩子本身就是仓库，则两者归入同组。
            if n in depots:
                index = np.where(depots == n)[0].item()
                tree.nodes[node]['group'] = index
                tree.nodes[n]['group'] = index
                assign_group(n, tree.nodes[n]['con'])
            else:
                # 若该孩子是客户，则继续递归直到找到仓库。
                tree.nodes[node]['group'] = assign_group(n, tree.nodes[n]['con'])
            # 处理除最佳通道外的其他孩子。
            for n in tree.neighbors(node):
                if n != tree.nodes[node]['parent'] and n != tree.nodes[node]['child']:
                    # 仓库孩子自成一组。
                    if n in depots:
                        tree.nodes[n]['group'] = np.where(depots == n)[0].item()
                        assign_group(n, tree.nodes[n]['con'])
                    # 若孩子通过当前节点向上连接，则继承当前组。
                    elif tree.nodes[n]['pcon']:
                        tree.nodes[n]['group'] = tree.nodes[node]['group']
                        assign_group(n, tree.nodes[n]['ncon'])
                    else:
                        # 否则它会连接到自己子树中的仓库。
                        tree.nodes[n]['group'] = assign_group(n, tree.nodes[n]['con'])

        else:
            # 当前节点连接到外部仓库时，继续把信息传给它的孩子。
            for n in tree.neighbors(node):
                # 跳过父节点。
                if n != tree.nodes[node]['parent']:
                    # 仓库孩子自成一组。
                    if n in depots:
                        tree.nodes[n]['group'] = np.where(depots == n)[0].item()
                        assign_group(n, tree.nodes[n]['con'])
                    # 若孩子通过当前节点向上连接，则继承当前组。
                    elif tree.nodes[n]['pcon']:
                        tree.nodes[n]['group'] = tree.nodes[node]['group']
                        assign_group(n, tree.nodes[n]['ncon'])
                    else:
                        # 否则递归处理其内部连通方案。
                        tree.nodes[n]['group'] = assign_group(n, tree.nodes[n]['con'])
        # 返回当前节点所属分组编号。
        return tree.nodes[node]['group']

    # 从第一个仓库开始回溯分组结果。
    assign_group(depots[0], tree.nodes[depots[0]]['con'])

    # 初始化最终分组字典。
    groups = {depot: [] for depot in depots}
    # 遍历树中的所有节点。
    for node in tree.nodes:
        # 只把客户加入最终结果。
        if node not in depots:
            groups[depots[tree.nodes[node]['group']]].append(node)
    # 返回仓库到客户列表的映射。
    return groups


def asymmetric_traveling_salesman_problem(graph, nodes_to_visit):
    """
    用一个近似技巧把 ATSP 转换为对称图，再调用 Christofides 求近似解。

    输入：
    - graph: 原始路网图。
    - nodes_to_visit: 需要访问的节点列表。

    输出：
    - 一个访问顺序列表。

    实现逻辑：
    1. 将每个节点复制成“入点/出点”两份。
    2. 用大罚值填满非允许边，从而逼近 ATSP 结构。
    3. 在构造出的对称图上调用 Christofides。
    4. 最后把虚拟节点过滤掉。
    """
    # 新建一个无向图，用于承载转换后的 TSP 实例。
    new_graph = nx.Graph()
    # 为每个原节点创建真实点和复制点。
    for node in nodes_to_visit:
        new_graph.add_node(node)
        new_graph.add_node(node + 1000000)
    # 为任意两个待访问节点建立对应边。
    for node in nodes_to_visit:
        for _node in nodes_to_visit:
            # 原节点到复制点的边权取原图最短路距离。
            new_graph.add_edge(node, _node + 1000000, weight=nx.dijkstra_path_length(graph, node, _node))
            # 再补一条镜像边，保持对称结构。
            new_graph.add_edge(node + 1000000, _node, weight=new_graph.edges[node, _node + 1000000]['weight'])
    # 对所有未定义边填入极大惩罚值。
    for node in new_graph.nodes:
        for _node in new_graph.nodes:
            # 已有边则跳过。
            if new_graph.has_edge(node, _node):
                continue
            # 无效边赋予超大权重。
            new_graph.add_edge(node, _node, weight=100000000)
    # 在转换后的图上调用 Christofides 近似算法。
    path = nx.approximation.christofides(new_graph)
    # 去掉虚拟复制节点，只保留原始节点序列。
    path = [node for node in path if node < 1000000]
    # 返回最终访问顺序。
    return path
