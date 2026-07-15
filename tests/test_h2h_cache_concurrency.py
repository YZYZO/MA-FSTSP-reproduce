"""验证两个 spawn 进程不会同时构建或写入同一 H2H 索引。"""

import multiprocessing as mp
import tempfile
import unittest
from pathlib import Path

from h2h_backend import ensure_h2h_index, validate_h2h_cache
from tests.h2h_native_test_utils import ensure_native_built
from tests.h2h_test_fixtures import build_fixed_20_node_graph


def _concurrent_cache_worker(graph, index_dir, builder_path, start_event, result_queue):
    """
    等待统一起跑信号后访问同一缓存，并把构建状态写回父进程。

    输入：可 pickle 图、缓存路径、builder、Event 和 Queue。
    输出：Queue 中的 `(ok, built, cache_dir, hash)` 或错误文本。
    """
    try:
        start_event.wait(timeout=30)
        result = ensure_h2h_index(
            graph,
            'concurrent',
            index_dir=index_dir,
            builder_path=builder_path,
        )
        result_queue.put((True, result.built, str(result.cache_dir), result.graph_hash))
    except Exception as exc:
        result_queue.put((False, type(exc).__name__, str(exc), ''))


class H2HCacheConcurrencyTests(unittest.TestCase):
    """确认跨进程 O_EXCL 锁只允许一个 builder 获胜。"""

    def test_two_spawn_processes_share_one_finished_cache(self):
        """两个同时访问者应得到同一路径，且 `built=True` 恰好出现一次。"""
        builder_path, _ = ensure_native_built('release')
        graph = build_fixed_20_node_graph()
        context = mp.get_context('spawn')
        start_event = context.Event()
        result_queue = context.Queue()
        with tempfile.TemporaryDirectory(prefix='h2h-cache-concurrent-') as temp:
            processes = [
                context.Process(
                    target=_concurrent_cache_worker,
                    args=(graph, temp, str(builder_path), start_event, result_queue),
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            start_event.set()
            results = [result_queue.get(timeout=60) for _ in processes]
            for process in processes:
                process.join(timeout=60)
                self.assertEqual(process.exitcode, 0)

            self.assertTrue(all(result[0] for result in results), msg=str(results))
            self.assertEqual(sorted(result[1] for result in results), [False, True])
            self.assertEqual(len({result[2] for result in results}), 1)
            self.assertEqual(len({result[3] for result in results}), 1)
            cache_dir = Path(results[0][2])
            self.assertTrue(validate_h2h_cache(cache_dir, results[0][3]))
            self.assertEqual(list(Path(temp).glob('.*.lock')), [])


if __name__ == '__main__':
    unittest.main()
