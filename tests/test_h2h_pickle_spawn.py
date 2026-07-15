"""验证 H2HDistanceMatrix pickle 不含原生标签，并可在 spawn worker 重开。"""

import multiprocessing as mp
import pickle
import tempfile
import unittest

import networkx as nx

from h2h_backend import H2HDistanceMatrix, ensure_h2h_index
from tests.h2h_native_test_utils import ensure_native_built
from tests.h2h_test_fixtures import build_fixed_20_node_graph


def _spawn_query_worker(payload: bytes, source: int, target: int, result_queue) -> None:
    """
    在 spawn worker 反序列化距离对象并执行一次查询。

    输入：pickle、节点对和结果队列。
    输出：Queue 中的成功距离或异常文本。
    """
    try:
        matrix = pickle.loads(payload)
        result_queue.put((True, matrix[source][target], matrix._handle is not None))
        matrix.close()
    except Exception as exc:
        result_queue.put((False, type(exc).__name__, str(exc)))


class H2HPickleSpawnTests(unittest.TestCase):
    """确认 Windows 多进程只传路径和配置，索引页由各 worker 只读共享。"""

    def test_pickle_state_is_small_and_spawn_worker_reopens_index(self):
        """已打开对象 pickle 后不含句柄，当前进程和 worker 都能重新查询。"""
        builder_path, library_path = ensure_native_built('release')
        graph = build_fixed_20_node_graph()
        expected = nx.dijkstra_path_length(graph, 0, 10, weight='weight')
        with tempfile.TemporaryDirectory(prefix='h2h-pickle-') as temp:
            cache = ensure_h2h_index(
                graph, 'pickle', index_dir=temp, builder_path=builder_path
            )
            matrix = H2HDistanceMatrix(
                cache.index_path,
                library_path,
                cache.node_count,
                cache.graph_hash,
            )
            self.assertAlmostEqual(matrix[0][10], expected, places=10)
            payload = pickle.dumps(matrix, protocol=pickle.HIGHEST_PROTOCOL)
            state = matrix.__getstate__()
            self.assertEqual(set(state), {
                'index_path', 'library_path', 'node_count', 'graph_hash',
                'backend_version', 'stats_enabled',
            })
            self.assertLess(len(payload), 2048)

            restored = pickle.loads(payload)
            self.assertIsNone(restored._handle)
            self.assertAlmostEqual(restored[0][10], expected, places=10)
            restored.close()

            context = mp.get_context('spawn')
            result_queue = context.Queue()
            process = context.Process(
                target=_spawn_query_worker,
                args=(payload, 0, 10, result_queue),
            )
            process.start()
            result = result_queue.get(timeout=60)
            process.join(timeout=60)
            self.assertEqual(process.exitcode, 0)
            self.assertTrue(result[0], msg=str(result))
            self.assertTrue(result[2])
            self.assertAlmostEqual(result[1], expected, places=10)
            matrix.close()


if __name__ == '__main__':
    unittest.main()
