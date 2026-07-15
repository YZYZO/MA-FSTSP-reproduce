"""验证阶段 7 服务器脚本在任何 GraphML/编译/缓存动作前执行安全检查。"""

import tempfile
import unittest
from pathlib import Path

from scripts.run_h2h_server_acceptance import (
    parse_arguments,
    validate_server_preconditions,
)


class H2HServerAcceptanceSafetyTests(unittest.TestCase):
    """确认 55k 脚本必须显式授权、Linux 且满足物理内存阈值。"""

    def _arguments(self, graph_path: Path, *extra: str):
        """输入临时图路径和附加选项，输出完整服务器参数命名空间。"""
        return parse_arguments(['--graph', str(graph_path), *extra])

    def test_missing_confirmation_fails_first(self):
        """未确认时即使路径不存在也必须先报告没有触碰图、编译和索引。"""
        arguments = self._arguments(Path('does-not-exist.graphml'))
        with self.assertRaisesRegex(PermissionError, '没有读取|confirm-server-55k'):
            validate_server_preconditions(
                arguments,
                system_name='Linux',
                total_memory_bytes=200 * 1024 ** 3,
            )

    def test_non_linux_and_low_memory_are_rejected(self):
        """显式确认不能绕过 Linux 平台和最低内存两道保护。"""
        with tempfile.TemporaryDirectory(prefix='h2h-server-safety-') as temporary:
            graph_path = Path(temporary) / 'nyc.graphml'
            graph_path.write_text('<graphml/>', encoding='utf-8')
            arguments = self._arguments(graph_path, '--confirm-server-55k')
            with self.assertRaisesRegex(RuntimeError, 'Linux'):
                validate_server_preconditions(
                    arguments,
                    system_name='Windows',
                    total_memory_bytes=200 * 1024 ** 3,
                )
            with self.assertRaisesRegex(RuntimeError, '低于要求'):
                validate_server_preconditions(
                    arguments,
                    system_name='Linux',
                    total_memory_bytes=16 * 1024 ** 3,
                )

    def test_confirmed_linux_server_returns_environment_summary(self):
        """确认、平台、内存和文件均满足时才返回后续报告所需环境信息。"""
        with tempfile.TemporaryDirectory(prefix='h2h-server-ready-') as temporary:
            graph_path = Path(temporary) / 'nyc.graphml'
            graph_path.write_text('<graphml/>', encoding='utf-8')
            arguments = self._arguments(
                graph_path,
                '--confirm-server-55k',
                '--worker-counts', '1,4,8,16',
                '--customer-counts', '20,50,100,150',
            )
            environment = validate_server_preconditions(
                arguments,
                system_name='Linux',
                total_memory_bytes=200 * 1024 ** 3,
            )
            self.assertEqual(environment['platform'], 'Linux')
            self.assertEqual(environment['total_memory_bytes'], 200 * 1024 ** 3)
            self.assertEqual(arguments.worker_counts, (1, 4, 8, 16))
            self.assertEqual(arguments.customer_counts, (20, 50, 100, 150))


if __name__ == '__main__':
    unittest.main()
