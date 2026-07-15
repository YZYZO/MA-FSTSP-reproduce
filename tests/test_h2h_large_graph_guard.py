"""验证本机选择 55k NYC 或未知 50k 图时在任何构建/读取前硬拦截。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from config import MANHATTAN_GRAPH_PATH
from distance_oracle import build_distance_provider
from problem import manhattan


class H2HLargeGraphGuardTests(unittest.TestCase):
    """确认本机保护不会创建 graph/index 临时文件或启动 builder。"""

    def tearDown(self):
        """清除带参数的地图读取缓存，避免测试状态泄漏。"""
        manhattan.cache_clear()

    def test_default_nyc_path_is_blocked_before_graphml_read(self):
        """默认 NYC 路径必须在 `nx.read_graphml` 和 subprocess 前直接失败。"""
        manhattan.cache_clear()
        with patch('problem.nx.read_graphml') as read_graphml, \
                patch('h2h_backend.subprocess.run') as run_builder:
            with self.assertRaisesRegex(RuntimeError, 'H2H_ENABLE_55K|200 GB'):
                manhattan(MANHATTAN_GRAPH_PATH)
            read_graphml.assert_not_called()
            run_builder.assert_not_called()

    def test_unknown_50000_node_graph_is_blocked_before_cache_directory(self):
        """非 NYC 路径但达到阈值的图也必须在规范化和建目录前停止。"""
        graph = nx.DiGraph()
        graph.add_nodes_from(range(50_000))
        with tempfile.TemporaryDirectory(prefix='h2h-large-guard-') as temp:
            with patch('h2h_backend.subprocess.run') as run_builder:
                with self.assertRaisesRegex(RuntimeError, '50000|服务器'):
                    build_distance_provider(
                        graph,
                        backend='h2h',
                        dataset_name='unknown-large',
                        index_dir=temp,
                    )
                run_builder.assert_not_called()
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
