"""
验证有向非对称路网上，客户输入顺序是否会沿论文三阶段流程传播。

本文件是独立诊断测试：只构造数据、包装观测点并调用原版求解器，不修改生产代码。
与 `tests/test.py` 不同，本测试保留默认的 Set-TSP 分支，从而覆盖论文描述的完整
Phase 1（Set-MST）、Phase 2（Set-TSP）和 Phase 3（动态规划）流程。
"""

import unittest
from unittest import mock

import networkx as nx
import numpy as np

import src.fstsp as fstsp_module
from src.fstsp import MultiAgentFlyingSidekickTSP


# 固定两个仓库和三个客户；反转时只改变客户数组的枚举顺序，不改变问题实例。
DEPOTS = np.array([0, 1])
FORWARD_CITIES = np.array([2, 3, 4])
REVERSED_CITIES = FORWARD_CITIES[::-1].copy()

# 弧权刻意设置为非对称值，同时保证路网强连通。
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
    构造最小强连通有向路网及卡车、无人机距离矩阵。

    输入：无。
    输出：二元组 `(road_graph, distance)`，其中前者是 `MultiDiGraph`，后者包含
    `truck` 和 `drone` 两类全点对距离。

    实现逻辑：卡车距离由有向 Dijkstra 计算，保留单行路造成的非对称性；无人机
    距离设置得足够大，使每位客户的候选边界只含自身，排除 Phase 1 区域划分的
    并列判定对本测试的干扰。
    """
    road_graph = nx.MultiDiGraph()
    road_graph.add_nodes_from(range(5))
    for (start, end), weight in DIRECTED_ARC_WEIGHTS.items():
        road_graph.add_edge(start, end, weight=weight)

    # `truck_distance` 是有向最短路闭包，也是 Set-MST 使用的实际卡车距离。
    truck_distance = {
        start: dict(lengths)
        for start, lengths in nx.all_pairs_dijkstra_path_length(
            road_graph,
            weight="weight",
        )
    }
    # 节点间无人机距离远大于航程阈值，仅保留客户自身作为候选点。
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
    规范化仓库客户集合，忽略同一仓库内部无意义的列表顺序。

    输入：生产算法返回的 `{depot: [customers...]}`。
    输出：仓库编号到有序客户元组的映射。
    """
    return {
        int(depot): tuple(sorted(int(city) for city in cities))
        for depot, cities in groups.items()
    }


def _canonical_edge_weights(graph):
    """
    提取 Set-MST 最终无向图中的非自环边权。

    输入：`set_mst()` 交给 `mst_partition()` 的 NetworkX 图。
    输出：规范化无向节点对到浮点边权的映射。
    """
    return {
        tuple(sorted((int(start), int(end)))): float(data["weight"])
        for start, end, data in graph.edges(data=True)
        if start != end
    }


def _run_three_phase_solver(city_order):
    """
    按给定客户顺序运行原版三阶段求解器并采集各阶段诊断量。

    输入：`city_order`，包含相同客户节点、但枚举顺序可不同的一维数组。
    输出：记录边界集合、Phase 1 图和分仓、Phase 2 访问顺序、Phase 3 分仓成本
    以及最终总成本的字典。

    实现逻辑：仅通过 mock 包装原函数以复制输入或返回值，包装后仍调用原函数；
    `theta[1]` 使用非零默认值，确保 Phase 2 实际进入论文的 Set-TSP 分支。
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
        theta=(0.5, 0.5),
    )

    # 三个容器分别保存各阶段的只读观测结果，不参与生产算法决策。
    captured_graph = {}
    phase2_orders = {}
    phase3_costs = {}
    original_mst_partition = fstsp_module.mst_partition
    original_get_seq = model.get_seq
    original_phase3 = model.local_search_multi_drone_appr

    def capture_then_partition(graph, depots, cities):
        """
        复制 Phase 1 最终图后继续执行原版 MST 分仓。

        输入：Set-MST 图、仓库数组和客户数组。
        输出：原版 `mst_partition()` 返回的分仓结果。
        """
        captured_graph["phase1"] = graph.copy()
        return original_mst_partition(graph, depots, cities)

    def capture_then_get_seq(depot, convex_sets):
        """
        记录 Phase 2 的客户访问顺序后原样返回索引序列。

        输入：当前仓库及其候选集合。
        输出：原版 Set-TSP 生成的索引序列。
        """
        sequence = original_get_seq(depot, convex_sets)
        cities = model.groups[depot]
        phase2_orders[int(depot)] = tuple(
            [int(depot)]
            + [int(cities[index - 1]) for index in sequence[1:-1]]
            + [int(depot)]
        )
        return sequence

    def capture_then_optimize(sequence, depot):
        """
        记录 Phase 3 针对单个仓库的动态规划成本。

        输入：Phase 2 客户序列和当前仓库。
        输出：原版动态规划返回的 `(solution, cost)`。
        """
        solution, cost = original_phase3(sequence, depot)
        phase3_costs[int(depot)] = float(cost)
        return solution, cost

    with mock.patch.object(
        fstsp_module,
        "mst_partition",
        side_effect=capture_then_partition,
    ), mock.patch.object(
        model,
        "get_seq",
        side_effect=capture_then_get_seq,
    ), mock.patch.object(
        model,
        "local_search_multi_drone_appr",
        side_effect=capture_then_optimize,
    ):
        _, total_cost = model.solve()

    # 空仓库不会进入 Phase 2/3，显式补零便于逐仓库比较传播结果。
    for depot in DEPOTS:
        phase2_orders.setdefault(int(depot), (int(depot), int(depot)))
        phase3_costs.setdefault(int(depot), 0.0)

    return {
        "boundary_sets": {
            int(city): tuple(sorted(int(node) for node in nodes))
            for city, nodes in model.get_boundary_convex_sets(
                model.theta[0]
            ).items()
        },
        "phase1_edge_weights": _canonical_edge_weights(
            captured_graph["phase1"]
        ),
        "groups": _canonical_groups(model.groups),
        "phase2_orders": phase2_orders,
        "phase3_costs": phase3_costs,
        "total_cost": float(total_cost),
    }


class TestSetMstDirectedThreePhaseOrderInvariance(unittest.TestCase):
    """验证同一有向实例在客户逆序前后应保持算法结果不变。"""

    @classmethod
    def setUpClass(cls):
        """
        对正序和逆序客户各运行一次论文三阶段主流程。

        输入：unittest 自动传入测试类。
        输出：无；两次结果保存在类属性中供各测试复用。
        """
        cls.forward = _run_three_phase_solver(FORWARD_CITIES)
        cls.reversed = _run_three_phase_solver(REVERSED_CITIES)

    def test_fixture_only_changes_customer_order(self):
        """
        验证两次运行的问题实例及候选边界完全相同。

        输入：无显式输入。
        输出：断言客户集合、强连通性、非对称性和候选边界满足测试前提。
        """
        road_graph, distance = _build_directed_instance()
        self.assertEqual(set(FORWARD_CITIES), set(REVERSED_CITIES))
        self.assertTrue(nx.is_strongly_connected(road_graph))
        self.assertNotEqual(
            distance["truck"][2][3],
            distance["truck"][3][2],
        )
        self.assertEqual(
            self.forward["boundary_sets"],
            self.reversed["boundary_sets"],
        )

    @unittest.expectedFailure
    def test_phase1_edge_weights_are_order_invariant(self):
        """
        验证 Phase 1 的最终图边权不应依赖客户数组顺序。

        当前 master 预期因无向边的方向覆盖而失败；模型修复后应通过。
        """
        self.assertEqual(
            self.forward["phase1_edge_weights"],
            self.reversed["phase1_edge_weights"],
        )

    @unittest.expectedFailure
    def test_phase1_groups_are_order_invariant(self):
        """
        验证 Phase 1 每个仓库的客户集合不应依赖输入顺序。

        当前 master 预期因 MST 输入图改变而失败；模型修复后应通过。
        """
        self.assertEqual(
            self.forward["groups"],
            self.reversed["groups"],
        )

    @unittest.expectedFailure
    def test_phase2_orders_are_order_invariant(self):
        """
        验证 Phase 2 的逐仓库访问顺序不应因枚举顺序改变。

        本反例中 Phase 1 分仓已经改变，因而 Set-TSP 接收到不同子问题并预期失败。
        """
        self.assertEqual(
            self.forward["phase2_orders"],
            self.reversed["phase2_orders"],
        )

    @unittest.expectedFailure
    def test_phase3_costs_are_order_invariant(self):
        """
        验证 Phase 3 的逐仓库动态规划成本不应依赖客户数组顺序。

        本反例中 Phase 2 输入已经改变，因此逐仓库成本预期随之改变。
        """
        self.assertEqual(
            self.forward["phase3_costs"],
            self.reversed["phase3_costs"],
        )

    @unittest.expectedFailure
    def test_final_total_cost_is_order_invariant(self):
        """
        验证完整三阶段求解器的最终总成本不应依赖客户数组顺序。

        当前 master 预期失败；修复后应在高精度浮点比较下通过。
        """
        self.assertAlmostEqual(
            self.forward["total_cost"],
            self.reversed["total_cost"],
            places=9,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
