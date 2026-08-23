"""
验证 Set-MST 在有向非对称路网上是否受客户输入顺序影响。

本测试只构造测试数据并调用原版求解器，不修改或替换生产算法。测试实例使用
强连通 MultiDiGraph，卡车距离由有向 Dijkstra 计算，以复现真实单行路网的距离
非对称性。
"""

import unittest
from unittest import mock

import networkx as nx
import numpy as np

import src.fstsp as fstsp_module
from src.fstsp import MultiAgentFlyingSidekickTSP


# 两个仓库与三个客户的节点编号；顺序反转时，问题实例本身保持不变。
DEPOTS = np.array([0, 1])
FORWARD_CITIES = np.array([2, 3, 4])
REVERSED_CITIES = FORWARD_CITIES[::-1].copy()

# 有向路网的原始弧长。该图强连通，且多组节点之间的最短路距离明显不对称。
DIRECTED_ARC_WEIGHTS = {
    (0, 1): 8,
    (0, 2): 21,
    (0, 3): 21,
    (0, 4): 7,
    (1, 0): 28,
    (1, 2): 12,
    (1, 3): 31,
    (1, 4): 18,
    (2, 0): 15,
    (2, 1): 6,
    (2, 3): 7,
    (2, 4): 10,
    (3, 0): 3,
    (3, 1): 27,
    (3, 2): 37,
    (3, 4): 11,
    (4, 0): 10,
    (4, 1): 1,
    (4, 2): 13,
    (4, 3): 4,
}


def _build_directed_instance():
    """
    构造用于顺序不变性测试的最小有向路网与距离矩阵。

    输入：无。
    输出：`(road_graph, distance)`，分别为强连通 MultiDiGraph 和卡车/无人机
    距离字典。

    实现逻辑：先按固定弧长建立有向图，再用 NetworkX 有向 Dijkstra 生成卡车
    全源最短路；无人机距离采用一维欧氏距离，并配合较小航程确保每个客户的
    Phase 1 候选边界仅包含客户节点自身。
    """
    road_graph = nx.MultiDiGraph()
    road_graph.add_nodes_from(range(5))
    for (start, end), weight in DIRECTED_ARC_WEIGHTS.items():
        road_graph.add_edge(start, end, weight=weight)

    # 卡车距离保留路网方向性，满足有向最短路的三角不等式。
    truck_distance = {
        start: dict(lengths)
        for start, lengths in nx.all_pairs_dijkstra_path_length(
            road_graph,
            weight="weight",
        )
    }

    # 节点间隔远大于测试航程，使客户只能使用自己的路网节点作为候选点。
    drone_distance = {
        start: {
            end: abs(start - end) * 100.0
            for end in road_graph.nodes
        }
        for start in road_graph.nodes
    }
    return road_graph, {
        "truck": truck_distance,
        "drone": drone_distance,
    }


def _canonical_groups(groups):
    """
    将分仓结果转换成与客户列表内部顺序无关的可比较形式。

    输入：原算法返回的 `{depot: [customers...]}` 字典。
    输出：仓库编号到有序客户元组的字典。

    实现逻辑：统一把 NumPy 整数转换成 Python 整数，并仅对每个仓库内部的客户
    排序；仓库归属关系不会被改写。
    """
    return {
        int(depot): tuple(sorted(int(city) for city in cities))
        for depot, cities in groups.items()
    }


def _canonical_edge_weights(graph):
    """
    提取 Phase 1 无向图实际保留下来的边权。

    输入：`set_mst()` 传给 `mst_partition()` 的 NetworkX 图。
    输出：以有序节点对为键、浮点边权为值的字典。

    实现逻辑：忽略算法生成的自环，并规范化无向边端点顺序，从而直接比较两次
    构图最终保存的边属性。
    """
    return {
        tuple(sorted((int(start), int(end)))): float(data["weight"])
        for start, end, data in graph.edges(data=True)
        if start != end
    }


def _run_original_solver(city_order):
    """
    按给定客户顺序运行一次未经修改的三阶段求解器并采集诊断量。

    输入：`city_order`，包含相同客户节点的一维 NumPy 数组。
    输出：包含边界集合、Phase 1 实际边权、分仓集合和最终总成本的字典。

    实现逻辑：测试侧仅包装模块已导入的 `mst_partition()` 以复制其输入图，随后
    仍调用原函数完成分组；Phase 2 使用项目原有 LKH 路径，Phase 3 使用原动态
    规划逻辑。
    """
    road_graph, distance = _build_directed_instance()
    model = MultiAgentFlyingSidekickTSP(
        graph=road_graph,
        depots=DEPOTS.copy(),
        cities=city_order.copy(),
        distance=distance,
        drone=1,
        limit=1.0,
        speed=1.6,
        # theta[1] 为 0 时走项目原有 LKH 分支，避免调用商业求解器。
        theta=(0.5, 0),
    )

    # `captured_graph` 保存 Phase 1 完成全部覆盖写入后的实际无向图。
    captured_graph = {}
    original_mst_partition = fstsp_module.mst_partition

    def capture_then_partition(graph, depots, cities):
        """
        复制 Phase 1 图后调用原版 MST 分组函数。

        输入：原 `set_mst()` 传入的图、仓库数组与客户数组。
        输出：原 `mst_partition()` 的分仓结果。

        实现逻辑：图副本仅供测试读取边权，不干预传给生产函数的原图及其执行。
        """
        captured_graph["phase1"] = graph.copy()
        return original_mst_partition(graph, depots, cities)

    with mock.patch.object(
        fstsp_module,
        "mst_partition",
        side_effect=capture_then_partition,
    ):
        _, total_cost = model.solve()

    return {
        "boundary_sets": {
            int(city): tuple(int(node) for node in nodes)
            for city, nodes in model.get_boundary_convex_sets(model.theta[0]).items()
        },
        "phase1_edge_weights": _canonical_edge_weights(captured_graph["phase1"]),
        "groups": _canonical_groups(model.groups),
        "total_cost": float(total_cost),
    }


class TestSetMstCustomerOrderInvariance(unittest.TestCase):
    """集中验证测试前提以及用户要求的三项顺序不变性。"""

    @classmethod
    def setUpClass(cls):
        """
        对正序与逆序客户各运行一次原始求解流程。

        输入：unittest 自动传入测试类。
        输出：无；结果保存在类级 `forward` 与 `reversed` 字段中供各测试复用。

        实现逻辑：两次运行共享相同图定义、仓库和客户集合，唯一差异是客户数组
        的遍历顺序。
        """
        cls.forward = _run_original_solver(FORWARD_CITIES)
        cls.reversed = _run_original_solver(REVERSED_CITIES)

    def test_fixture_only_changes_customer_order(self):
        """
        验证测试夹具没有借候选区域变化制造伪阳性。

        输入：无显式输入。
        输出：断言两次运行的客户集合和规范化边界集合完全相同。

        实现逻辑：先忽略字典插入顺序再比较边界，再确认卡车 Dijkstra 距离确实
        非对称，因此测试同时满足“同一实例”和“有向路网”两个前提。
        """
        self.assertEqual(set(FORWARD_CITIES), set(REVERSED_CITIES))
        self.assertEqual(
            self.forward["boundary_sets"],
            self.reversed["boundary_sets"],
        )

        _, distance = _build_directed_instance()
        self.assertNotEqual(
            distance["truck"][2][3],
            distance["truck"][3][2],
        )

    @unittest.expectedFailure
    def test_phase1_edge_weights_are_order_invariant(self):
        """
        回归检查一：客户反转后 Phase 1 的最终边权应保持不变。

        输入：无显式输入。
        输出：当前 master 因无向边覆盖而预期失败；修复模型后应通过。

        实现逻辑：比较 `set_mst()` 完成构图后真正传入 MST 的全部非自环边权。
        """
        self.assertEqual(
            self.forward["phase1_edge_weights"],
            self.reversed["phase1_edge_weights"],
        )

    @unittest.expectedFailure
    def test_depot_customer_sets_are_order_invariant(self):
        """
        回归检查二：客户反转后每个仓库的客户集合应保持不变。

        输入：无显式输入。
        输出：当前 master 因 Phase 1 图变化而预期失败；修复模型后应通过。

        实现逻辑：仓库内客户先排序再比较，因此只检测客户换仓，不检测无意义的
        列表排列差异。
        """
        self.assertEqual(
            self.forward["groups"],
            self.reversed["groups"],
        )

    @unittest.expectedFailure
    def test_final_total_cost_is_order_invariant(self):
        """
        回归检查三：客户反转后完整求解得到的最终总成本应保持不变。

        输入：无显式输入。
        输出：当前 master 因分仓变化而预期失败；修复模型后应通过。

        实现逻辑：对两次原始三阶段结果做高精度浮点比较，避免只检查中间状态。
        """
        self.assertAlmostEqual(
            self.forward["total_cost"],
            self.reversed["total_cost"],
            places=9,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)