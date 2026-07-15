"""验证 builder 日志统计和缓存索引统计可稳定进入阶段 6/7 报告。"""

import tempfile
import unittest

from h2h_backend import (
    ensure_h2h_index,
    parse_h2h_build_statistics,
    read_h2h_index_statistics,
)
from tests.h2h_native_test_utils import ensure_native_built
from tests.h2h_test_fixtures import build_fixed_20_node_graph


class H2HBuildStatisticsTests(unittest.TestCase):
    """确认新旧成功日志都可解析，索引头统计优先于日志重复字段。"""

    def test_parser_accepts_linux_peak_rss_and_legacy_windows_log(self):
        """Linux 日志读取峰值 RSS，旧 Windows 日志缺失该字段时安全记为零。"""
        prefix = (
            'H2H_BUILD_OK nodes=20 treewidth=3 treeheight=8 fill_edges=12 '
            'shortcut_arcs=17 labels=99 positions=88 seconds=0.125 '
        )
        linux = parse_h2h_build_statistics(
            prefix + 'peak_rss_bytes=1048576 index_bytes=4096\n'
        )
        legacy = parse_h2h_build_statistics(prefix + 'index_bytes=4096\n')
        self.assertEqual(linux['peak_rss_bytes'], 1_048_576)
        self.assertEqual(legacy['peak_rss_bytes'], 0)
        self.assertEqual(linux['fill_edges'], 12)
        self.assertAlmostEqual(linux['build_seconds'], 0.125)

    def test_cache_statistics_match_real_index_header(self):
        """真实 Release 缓存必须报告节点、树宽、标签、fill-in、耗时和文件大小。"""
        builder_path, _ = ensure_native_built('release')
        graph = build_fixed_20_node_graph()
        with tempfile.TemporaryDirectory(prefix='h2h-statistics-') as temporary:
            cache = ensure_h2h_index(
                graph,
                dataset_name='statistics',
                index_dir=temporary,
                builder_path=builder_path,
            )
            statistics = read_h2h_index_statistics(cache.cache_dir)
            self.assertEqual(statistics['node_count'], 20)
            self.assertGreater(statistics['treewidth'], 0)
            self.assertGreater(statistics['treeheight'], 0)
            self.assertGreater(statistics['label_count'], 0)
            self.assertGreater(statistics['index_bytes'], 160)
            self.assertGreaterEqual(statistics['fill_edges'], 0)
            self.assertGreaterEqual(statistics['build_seconds'], 0.0)


if __name__ == '__main__':
    unittest.main()
