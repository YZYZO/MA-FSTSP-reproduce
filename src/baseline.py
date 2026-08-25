"""
本文件定义所有求解算法共享的基类 `Baseline`。

主要内容：
1. 统一保存图、仓库、客户、距离矩阵等公共输入。
2. 约定所有算法都应实现 `solve` 方法。
3. 约定所有算法可选择实现 `convert` 方法，将内部解格式转成统一输出格式。
"""


class Baseline:
    """
    所有算法类的公共父类。

    输入：
    - graph: 路网图。
    - depots: 仓库节点集合。
    - cities: 客户节点集合。
    - distance: 距离矩阵字典。
    - drone: 无人机数量。
    - limit: 无人机最大飞行距离限制。
    - speed: 无人机速度相对参数。

    输出：
    - 无显式返回值，仅完成成员变量初始化。

    实现逻辑：
    - 将所有子类都会使用的输入参数保存为实例属性。
    """

    def __init__(self, graph, depots, cities, distance, drone, limit, speed):
        """
        初始化算法基类中的公共字段。

        输入：
        - graph: 路网图。
        - depots: 仓库集合。
        - cities: 客户集合。
        - distance: 卡车/无人机距离字典。
        - drone: 无人机数量。
        - limit: 最大飞行距离。
        - speed: 无人机速度参数。

        输出：
        - 无。

        实现逻辑：
        - 将输入逐项存入实例属性，供所有子类复用。
        """
        self.graph = graph
        self.depots = depots
        self.cities = cities
        self.distance = distance
        self.name = self.__class__.__name__
        self.drone = drone
        self.limit = limit
        self.speed = speed

    def convert(self, solution):
        """
        将算法内部解表示转换成统一输出格式。

        输入：
        - solution: 子类内部定义的路线表示。

        输出：
        - 统一格式的路线结构；基类中不实现具体逻辑。

        实现逻辑：
        - 该方法作为接口保留，子类若有需要可自行重写。
        """
        # 统一输出格式约定如下：
        # {'truck': [node1, node2, ...], 'drone': [[[node1, city, node2], ...], ...]}
        # 基类不提供默认实现，因此直接留空。
        pass

    def solve(self):
        """
        求解当前问题实例。

        输入：
        - 无显式输入，使用实例初始化时保存的数据。

        输出：
        - 由各子类自行定义；通常是 `(solution, cost)`。

        实现逻辑：
        - 基类只定义接口，不提供具体求解逻辑。
        """
        # 强制要求子类实现该方法。
        raise NotImplementedError
