"""单客户组 SQLite 结果缓存测试。"""

from pathlib import Path
import tempfile
import unittest

from src.learning.cache import EvaluationCache


class EvaluationCacheTest(unittest.TestCase):
    """验证缓存键稳定性、写入读取和记录计数。"""

    def test_customer_order_does_not_change_cache_key(self):
        """
        验证相同客户集合的不同顺序使用同一缓存记录。

        输入：顺序相反的两个客户列表。
        输出：相同 SHA-256 和规范输入 JSON。
        逻辑：局部搜索动作不应因列表顺序重复调用昂贵求解器。
        """
        first = EvaluationCache.make_key("map-a", 0, [2, 3], {"limit": 5})
        second = EvaluationCache.make_key("map-a", 0, [3, 2], {"limit": 5})
        self.assertEqual(first, second)

    def test_sqlite_round_trip(self):
        """
        验证评估结果可以写入并从 SQLite 读取。

        输入：临时数据库和一个嵌套结果字典。
        输出：缓存命中内容与原结果一致，记录数为一。
        逻辑：使用测试目录下临时文件，测试结束自动清理。
        """
        tests_directory = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_directory) as temporary_directory:
            database_path = Path(temporary_directory) / "evaluation.sqlite3"
            with EvaluationCache(database_path) as cache:
                cache_key, input_json = cache.make_key(
                    "map-a",
                    0,
                    [2, 3],
                    {"limit": 5},
                )
                result = {"cost": 12.5, "solver": {"timed_out": False}}
                self.assertIsNone(cache.get(cache_key))
                cache.set(cache_key, input_json, result)
                self.assertEqual(cache.get(cache_key), result)
                self.assertEqual(cache.count(), 1)


if __name__ == "__main__":
    unittest.main()

