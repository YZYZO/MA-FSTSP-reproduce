"""验证显式 Manhattan/Boston 路径、标准化规模和旧缓存只读迁移行为。"""

import unittest

from config import BOSTON_GRAPH_PATH, MANHATTAN_BASELINE_GRAPH_PATH
from problem import CAMBRIDGE_CACHE, MANHATTAN_CACHE, cambridge, manhattan


class ProblemGraphLoadingTests(unittest.TestCase):
    """确认地图加载只标准化图，不触发 APSP 或改写历史缓存。"""

    def setUp(self):
        """每项测试前清理 lru_cache，确保真实执行加载路径。"""
        manhattan.cache_clear()
        cambridge.cache_clear()

    def tearDown(self):
        """测试后清除图对象引用。"""
        manhattan.cache_clear()
        cambridge.cache_clear()

    def test_baseline_manhattan_loads_4333_scc_without_touching_json(self):
        """4,426 原始节点应标准化为 4,333，旧 JSON 的时间和大小不能变化。"""
        if not MANHATTAN_BASELINE_GRAPH_PATH.is_file():
            self.skipTest(f'缺少 {MANHATTAN_BASELINE_GRAPH_PATH}')
        before = MANHATTAN_CACHE.stat() if MANHATTAN_CACHE.is_file() else None
        graph = manhattan(MANHATTAN_BASELINE_GRAPH_PATH)
        after = MANHATTAN_CACHE.stat() if MANHATTAN_CACHE.is_file() else None
        self.assertEqual(graph.number_of_nodes(), 4333)
        self.assertEqual(graph.graph['source_path'], str(MANHATTAN_BASELINE_GRAPH_PATH.resolve()))
        if before is not None:
            self.assertEqual((after.st_size, after.st_mtime_ns), (before.st_size, before.st_mtime_ns))

    def test_boston_loads_8313_scc_without_opening_or_touching_pickle(self):
        """8,412 原始节点应标准化为 8,313，1.65 GB pickle 保持完全不变。"""
        if not BOSTON_GRAPH_PATH.is_file():
            self.skipTest(f'缺少 {BOSTON_GRAPH_PATH}')
        before = CAMBRIDGE_CACHE.stat() if CAMBRIDGE_CACHE.is_file() else None
        graph = cambridge(BOSTON_GRAPH_PATH)
        after = CAMBRIDGE_CACHE.stat() if CAMBRIDGE_CACHE.is_file() else None
        self.assertEqual(graph.number_of_nodes(), 8313)
        self.assertEqual(graph.graph['source_path'], str(BOSTON_GRAPH_PATH.resolve()))
        if before is not None:
            self.assertEqual((after.st_size, after.st_mtime_ns), (before.st_size, before.st_mtime_ns))


if __name__ == '__main__':
    unittest.main()
