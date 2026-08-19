"""在独立小型有向图上测试改进模型的完整求解流程。"""

import math
import unittest

import networkx as nx
import numpy as np

from src.mst_improvement.model import (
    ImprovedMSTMultiAgentFlyingSidekickTSP,
    MSTImprovementConfig,
)


def _small_directed_instance():
    """
    构造无需外部地图、H2H 索引或 Gurobi Set-TSP 的六节点实例。

    输出：有向完全图、仓库、客户及卡车/无人机距离。
    """
    nodes = list(range(6))
    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(node, pos=(float(node) * 0.001, 0.0))
    truck = {start: {} for start in nodes}
    drone = {start: {} for start in nodes}
    for start in nodes:
        for end in nodes:
            if start == end:
                truck[start][end] = 0.0
                drone[start][end] = 0.0
            else:
                # 正反方向略有差别，用来覆盖有向成本构造路径。
                truck[start][end] = abs(start - end) + (0.1 if start < end else 0.2)
                drone[start][end] = abs(start - end) * 0.1
                graph.add_edge(start, end, weight=truck[start][end])
    return (
        graph,
        np.asarray([0, 5]),
        np.asarray([1, 2, 3, 4]),
        {'truck': truck, 'drone': drone},
    )


class ImprovedModelIntegrationTests(unittest.TestCase):
    """验证改进模型保持原模型输出协议并覆盖全部客户。"""

    def test_rooted_msf_end_to_end_without_set_tsp_milp(self):
        """关闭 Set-TSP MILP 和局部搜索后，完整流程应返回有限可行结果。"""
        graph, depots, cities, distance = _small_directed_instance()
        model = ImprovedMSTMultiAgentFlyingSidekickTSP(
            graph,
            depots,
            cities,
            distance,
            drone=1,
            limit=10.0,
            theta=(0.5, 0.0),
            improvement_config=MSTImprovementConfig(
                partition_method='rooted_msf',
                enable_relocate=False,
                enable_swap=False,
                exact_candidate_count=1,
                max_iterations=0,
                time_limit_seconds=5.0,
            ),
        )
        solution, cost = model.solve()

        assigned = [city for group in model.groups.values() for city in group]
        self.assertCountEqual(assigned, list(cities))
        self.assertEqual(len(solution), len(depots))
        self.assertTrue(math.isfinite(cost))
        self.assertGreaterEqual(cost, 0.0)
        self.assertEqual(model.telemetry['search']['stop_reason'], 'disabled')


if __name__ == '__main__':
    unittest.main()

