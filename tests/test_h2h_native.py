"""编译并验证阶段 3 C++ H2H builder、mmap 查询和稳定 C ABI。"""

from __future__ import annotations

import ctypes
import math
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from h2h_reference import DirectedH2HReference, normalize_directed_graph
from tests.h2h_native_test_utils import ensure_native_built
from tests.h2h_test_fixtures import (
    build_h2h_acceptance_graphs,
    build_random_strongly_connected_digraph,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_graph_binary(graph: nx.Graph, path: Path) -> None:
    """
    按阶段 3 固定格式写出已规范化的小图 graph.bin。

    输入：强连通有向图和临时输出路径。
    输出：无；平行边按最小权重归并，自环不写入消元输入。
    """
    out_edges, edge_count, _ = normalize_directed_graph(graph)
    payload = bytearray(struct.pack(
        '<8sIIIQ', b'H2HGRPH1', 1, 0x01020304, len(out_edges), edge_count
    ))
    for source, row in enumerate(out_edges):
        for target in sorted(row):
            payload.extend(struct.pack('<IId', source, target, row[target]))
    path.write_bytes(payload)


class NativeH2HLibrary:
    """测试专用的最小 ctypes 包装；生产包装按计划留到阶段 4。"""

    def __init__(self, library_path: Path) -> None:
        """
        加载动态库并声明 C ABI 参数类型。

        输入：h2h_query.dll/libh2h_query.so 路径。
        输出：可打开索引和执行标量/批量查询的测试对象。
        """
        self.library = ctypes.CDLL(str(library_path))
        self.library.h2h_api_version.argtypes = []
        self.library.h2h_api_version.restype = ctypes.c_uint32
        self.library.h2h_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.library.h2h_open.restype = ctypes.c_void_p
        self.library.h2h_query.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        self.library.h2h_query.restype = ctypes.c_double
        self.library.h2h_query_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        ]
        self.library.h2h_query_batch.restype = ctypes.c_int
        self.library.h2h_close.argtypes = [ctypes.c_void_p]
        self.library.h2h_close.restype = None
        if self.library.h2h_api_version() != 1:
            raise RuntimeError('测试只支持 H2H C ABI v1。')

    def open(self, index_path: Path) -> int:
        """
        打开 mmap 索引并在失败时转成包含原生消息的 Python 异常。

        输入：index.bin 路径。
        输出：非空原生句柄整数。
        """
        error_buffer = ctypes.create_string_buffer(2048)
        handle = self.library.h2h_open(
            str(index_path).encode('utf-8'), error_buffer, len(error_buffer)
        )
        if not handle:
            message = error_buffer.value.decode('utf-8', errors='replace')
            raise RuntimeError(message)
        return handle

    def query(self, handle: int, source: int, target: int) -> float:
        """输入原生句柄和节点对，输出一个 Python float 距离。"""
        value = float(self.library.h2h_query(handle, source, target))
        if not math.isfinite(value):
            raise RuntimeError(f'原生查询 {source} -> {target} 返回 NaN/Inf。')
        return value

    def query_batch(self, handle: int, sources: list[int], targets: list[int]) -> list[float]:
        """
        调用一次 C ABI 批量查询。

        输入：句柄和等长节点列表。
        输出：保持输入顺序的 Python float 列表。
        """
        if len(sources) != len(targets):
            raise ValueError('sources 与 targets 长度必须一致。')
        count = len(sources)
        source_array = (ctypes.c_uint32 * count)(*sources)
        target_array = (ctypes.c_uint32 * count)(*targets)
        output_array = (ctypes.c_double * count)()
        status = self.library.h2h_query_batch(
            handle, source_array, target_array, count, output_array
        )
        if status != 0:
            raise RuntimeError(f'原生批量查询失败，状态码 {status}。')
        return [float(value) for value in output_array]

    def close(self, handle: int) -> None:
        """关闭输入句柄；无返回值。"""
        self.library.h2h_close(handle)


class H2HNativeCorrectnessTests(unittest.TestCase):
    """逐项比较 C++ mmap 查询、Python 参考实现和 Dijkstra。"""

    @classmethod
    def setUpClass(cls):
        """用项目指定 Python 执行一键指定模式构建并加载动态库。"""
        # 默认验证 Release；阶段 3 内存检查可通过独立进程环境切换为 debug。
        build_mode = os.environ.get('H2H_NATIVE_TEST_MODE', 'release').lower()
        if build_mode not in {'release', 'debug'}:
            raise ValueError('H2H_NATIVE_TEST_MODE 只能是 release 或 debug。')
        cls.builder_path, library_path = ensure_native_built(build_mode)
        cls.native = NativeH2HLibrary(library_path)

    def _build_index(self, graph: nx.Graph, directory: Path, *extra_arguments: str) -> Path:
        """
        在测试临时目录写 graph.bin 并调用独立 builder 生成 index.bin。

        输入：图、临时目录和可选资源限制参数。
        输出：成功生成的 index.bin 路径。
        """
        graph_path = directory / 'graph.bin'
        index_path = directory / 'index.bin'
        _write_graph_binary(graph, graph_path)
        command = [
            str(self.builder_path), '--graph', str(graph_path), '--output', str(index_path),
            '--progress-interval', '0', *extra_arguments,
        ]
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            check=False,
        )
        if result.returncode != 0:
            self.fail(f'builder 失败：\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}')
        self.assertIn('H2H_BUILD_OK', result.stdout)
        self.assertTrue(index_path.is_file())
        return index_path

    def test_native_matches_reference_on_all_acceptance_graphs(self):
        """阶段 2 全部人工图和随机图的每个有序节点对必须逐项一致。"""
        graphs = build_h2h_acceptance_graphs()
        graphs.update({
            f'random-{seed}': build_random_strongly_connected_digraph(40, seed=seed)
            for seed in (260715, 260716, 260717)
        })
        for name, graph in graphs.items():
            with self.subTest(graph=name), tempfile.TemporaryDirectory(prefix='h2h-native-') as temp:
                index_path = self._build_index(graph, Path(temp), '--max-nodes', '200')
                handle = self.native.open(index_path)
                try:
                    reference = DirectedH2HReference(graph)
                    expected = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
                    for source in graph.nodes:
                        for target in graph.nodes:
                            actual = self.native.query(handle, source, target)
                            self.assertAlmostEqual(actual, reference.query(source, target), places=10)
                            self.assertAlmostEqual(actual, expected[source][target], places=10)
                finally:
                    self.native.close(handle)

    def test_batch_100000_queries_match_grouped_dijkstra(self):
        """单次 C ABI 批量 100,000 查询必须与按源分组的 Dijkstra 基准一致。"""
        graph = build_random_strongly_connected_digraph(60, seed=260718)
        with tempfile.TemporaryDirectory(prefix='h2h-native-batch-') as temp:
            index_path = self._build_index(graph, Path(temp), '--max-nodes', '200')
            handle = self.native.open(index_path)
            try:
                query_count = 100_000
                sources = [(index * 17 + 3) % 60 for index in range(query_count)]
                targets = [(index * 29 + 11) % 60 for index in range(query_count)]
                actual = self.native.query_batch(handle, sources, targets)
                expected_rows = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
                for index, value in enumerate(actual):
                    self.assertAlmostEqual(
                        value, expected_rows[sources[index]][targets[index]], places=10
                    )
            finally:
                self.native.close(handle)

    def test_resource_limit_stops_builder_before_index_write(self):
        """节点资源上限必须让 builder 非零退出，且不能留下可加载索引。"""
        graph = build_random_strongly_connected_digraph(20, seed=260719)
        with tempfile.TemporaryDirectory(prefix='h2h-native-limit-') as temp:
            directory = Path(temp)
            graph_path = directory / 'graph.bin'
            index_path = directory / 'index.bin'
            _write_graph_binary(graph, graph_path)
            result = subprocess.run(
                [
                    str(self.builder_path), '--graph', str(graph_path), '--output', str(index_path),
                    '--max-nodes', '10', '--progress-interval', '0',
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='replace',
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('超过 --max-nodes', result.stderr)
            self.assertFalse(index_path.exists())

    def test_mmap_rejects_missing_and_truncated_index(self):
        """动态库必须拒绝不存在或被截断的索引，不能越界读取映射。"""
        with tempfile.TemporaryDirectory(prefix='h2h-native-invalid-') as temp:
            directory = Path(temp)
            with self.assertRaises(RuntimeError):
                self.native.open(directory / 'missing-index.bin')

            graph = build_h2h_acceptance_graphs()['asymmetric-two']
            index_path = self._build_index(graph, directory)
            truncated_path = directory / 'truncated-index.bin'
            payload = index_path.read_bytes()
            truncated_path.write_bytes(payload[: len(payload) // 2])
            with self.assertRaisesRegex(RuntimeError, '短于|大小|边界'):
                self.native.open(truncated_path)


if __name__ == '__main__':
    unittest.main()
