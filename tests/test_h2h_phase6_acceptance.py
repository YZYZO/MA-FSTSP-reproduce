"""阶段 6 独立验收：小图全对、中图 100k、吞吐、重载与双 worker。"""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import networkx as nx

from config import BOSTON_GRAPH_PATH, MANHATTAN_BASELINE_GRAPH_PATH
from h2h_acceptance import (
    benchmark_query_paths,
    benchmark_spawn_workers,
    build_query_workload,
    validate_query_workload,
)
from h2h_backend import H2HDistanceMatrix, ensure_h2h_index, read_h2h_index_statistics
from problem import cambridge, manhattan
from tests.h2h_native_test_utils import ensure_native_built
from tests.h2h_test_fixtures import (
    build_h2h_acceptance_graphs,
    build_random_strongly_connected_digraph,
)


RUN_MEDIUM_ACCEPTANCE = os.environ.get('H2H_RUN_LOCAL_ACCEPTANCE') == '1'


class H2HPhase6SmallGraphAcceptanceTests(unittest.TestCase):
    """通过生产缓存和生产 Python 包装重新执行阶段 6 小图全节点对验收。"""

    @classmethod
    def setUpClass(cls):
        """构建一次 Release 原生产物，供所有小图临时索引复用。"""
        cls.builder_path, cls.library_path = ensure_native_built('release')

    def test_all_nodes_pairs_match_dijkstra_through_production_wrapper(self):
        """全部基础图及随机图的每个有序节点对必须满足 1e-10 误差规则。"""
        graphs = build_h2h_acceptance_graphs()
        graphs['random-sparse-80'] = build_random_strongly_connected_digraph(
            80, seed=20260715
        )
        with tempfile.TemporaryDirectory(prefix='h2h-phase6-small-') as temporary:
            for name, graph in graphs.items():
                with self.subTest(graph=name):
                    cache = ensure_h2h_index(
                        graph,
                        dataset_name=name,
                        index_dir=temporary,
                        builder_path=self.builder_path,
                    )
                    matrix = H2HDistanceMatrix(
                        cache.index_path,
                        self.library_path,
                        cache.node_count,
                        cache.graph_hash,
                    )
                    try:
                        expected = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
                        for source in graph.nodes:
                            for target in graph.nodes:
                                self.assertTrue(math.isclose(
                                    matrix[source][target],
                                    float(expected[source][target]),
                                    rel_tol=1e-10,
                                    abs_tol=1e-10,
                                ))
                    finally:
                        matrix.close()


@unittest.skipUnless(
    RUN_MEDIUM_ACCEPTANCE,
    '设置 H2H_RUN_LOCAL_ACCEPTANCE=1 后运行阶段 6 中图 100k 验收。',
)
class H2HPhase6MediumGraphAcceptanceTests(unittest.TestCase):
    """在 4,333/8,313 节点真实标准化图上执行一次显式慢速验收。"""

    @classmethod
    def setUpClass(cls):
        """在独立临时目录构建两个中图索引，不接触 datasets/indexes。"""
        cls.builder_path, cls.library_path = ensure_native_built('release')
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix='h2h-phase6-medium-')
        cls.graphs = {
            'manhattan-4333': manhattan(MANHATTAN_BASELINE_GRAPH_PATH),
            'boston-8313': cambridge(BOSTON_GRAPH_PATH),
        }
        cls.caches = {}
        cls.matrices = {}
        for name, graph in cls.graphs.items():
            cache = ensure_h2h_index(
                graph,
                dataset_name=name,
                index_dir=cls.temporary_directory.name,
                builder_path=cls.builder_path,
            )
            cls.caches[name] = cache
            cls.matrices[name] = H2HDistanceMatrix(
                cache.index_path,
                cls.library_path,
                cache.node_count,
                cache.graph_hash,
            )

    @classmethod
    def tearDownClass(cls):
        """关闭两个 mmap，并清理临时索引和地图函数缓存。"""
        for matrix in cls.matrices.values():
            matrix.close()
        cls.temporary_directory.cleanup()
        manhattan.cache_clear()
        cambridge.cache_clear()

    def test_medium_graphs_match_100000_grouped_dijkstra_queries(self):
        """两个真实中图各 100,000 个有序节点对必须零失败。"""
        for offset, (name, graph) in enumerate(self.graphs.items()):
            with self.subTest(dataset=name):
                workload = build_query_workload(
                    graph.number_of_nodes(),
                    query_count=100_000,
                    source_count=200,
                    seed=20260715 + offset,
                )
                report = validate_query_workload(
                    graph, self.matrices[name], workload, name
                )
                print(f'PHASE6_VALIDATION {report.to_dict()}', flush=True)
                self.assertEqual(report.failure_count, 0)
                self.assertLessEqual(report.max_absolute_error, 1e-10)

    def test_release_query_throughput_and_memory_are_acceptable(self):
        """完整双下标路径达到 100k qps，连续复跑不产生明显 RSS 线性增长。"""
        name = 'manhattan-4333'
        graph = self.graphs[name]
        workload = build_query_workload(
            graph.number_of_nodes(), 100_000, 200, seed=20260717
        )
        report = benchmark_query_paths(self.matrices[name], workload)
        print(f'PHASE6_PERFORMANCE {report.to_dict()}', flush=True)
        self.assertGreaterEqual(report.double_indexed_queries_per_second, 100_000)
        if report.stable_rss_growth_bytes is not None:
            self.assertLessEqual(report.stable_rss_growth_bytes, 32 * 1024 * 1024)

    def test_cache_reload_statistics_and_two_spawn_workers(self):
        """中图缓存二次命中、索引统计、DLL 重开和两个 spawn worker 必须同时通过。"""
        name = 'manhattan-4333'
        graph = self.graphs[name]
        first_cache = self.caches[name]
        second_cache = ensure_h2h_index(
            graph,
            dataset_name=name,
            index_dir=self.temporary_directory.name,
            builder_path=self.builder_path,
        )
        self.assertFalse(second_cache.built)
        self.assertEqual(second_cache.cache_dir, first_cache.cache_dir)
        statistics = read_h2h_index_statistics(second_cache.cache_dir)
        self.assertEqual(statistics['node_count'], 4_333)
        self.assertGreater(statistics['treewidth'], 0)
        self.assertGreater(statistics['label_count'], 0)

        matrix = self.matrices[name]
        matrix.close()
        expected = nx.dijkstra_path_length(graph, 0, 100, weight='weight')
        self.assertAlmostEqual(matrix[0][100], expected, places=10)
        workload = build_query_workload(4_333, 10_000, 20, seed=20260718)
        worker_report = benchmark_spawn_workers(matrix, workload, worker_count=2)
        print(f'PHASE6_WORKERS {worker_report}', flush=True)
        self.assertEqual(worker_report['query_count'], 10_000)
        self.assertLess(worker_report['pickle_bytes'], 2_048)


if __name__ == '__main__':
    unittest.main()
