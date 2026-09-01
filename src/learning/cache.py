"""单客户组真实求解结果的 SQLite 缓存。"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


def _jsonable(value):
    """
    将节点编号和嵌套结果转换成稳定的 JSON 基础类型。

    输入：可能包含 NumPy 标量、元组、字典的任意值。
    输出：仅包含字典、列表、字符串、数值、布尔值和空值的对象。
    逻辑：递归转换 NumPy 标量与容器，供哈希和 SQLite 存储共同使用。
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _node_sort_key(node):
    """
    生成客户节点的稳定排序键。

    输入：任意节点编号。
    输出：类型名与文本表示组成的元组。
    逻辑：缓存键只取决于客户集合，不取决于调用方传入顺序。
    """
    return type(node).__name__, repr(node)


class EvaluationCache:
    """
    使用 SQLite 保存昂贵的单客户组真实评估结果。

    输入：SQLite 文件路径。
    输出：支持 `get/set/count/close` 的缓存对象。
    逻辑：以规范化输入的 SHA-256 作为主键，重复评估时直接恢复 JSON 结果。
    """

    def __init__(self, path):
        """
        打开缓存数据库并创建结果表。

        输入：数据库文件路径。
        输出：初始化后的缓存对象。
        逻辑：创建父目录，启用 WAL，再建立以哈希为主键的单表结构。
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS group_evaluations (
                cache_key TEXT PRIMARY KEY,
                input_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def make_key(map_id, depot, customers, parameters):
        """
        为一组客户及其求解参数生成稳定缓存键。

        输入：地图标识、仓库、客户集合和算法参数字典。
        输出：`(SHA-256, canonical_json)`。
        逻辑：客户先稳定排序，随后使用排序键 JSON 编码并计算摘要。
        """
        canonical_input = {
            "map_id": map_id,
            "depot": depot,
            "customers": sorted(customers, key=_node_sort_key),
            "parameters": parameters,
        }
        input_json = json.dumps(
            _jsonable(canonical_input),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        cache_key = hashlib.sha256(input_json.encode("utf-8")).hexdigest()
        return cache_key, input_json

    def get(self, cache_key):
        """
        读取一个缓存结果。

        输入：由 `make_key` 生成的 SHA-256。
        输出：结果字典；未命中时返回 `None`。
        逻辑：按主键查询并解析 JSON。
        """
        row = self.connection.execute(
            "SELECT result_json FROM group_evaluations WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, cache_key, input_json, result):
        """
        新增或覆盖一个缓存结果。

        输入：缓存键、规范输入 JSON 和结果字典。
        输出：无，数据提交到 SQLite。
        逻辑：使用 UPSERT 保证相同客户组只保留最新的完整记录。
        """
        result_json = json.dumps(
            _jsonable(result),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.connection.execute(
            """
            INSERT INTO group_evaluations(cache_key, input_json, result_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                input_json = excluded.input_json,
                result_json = excluded.result_json,
                created_at = excluded.created_at
            """,
            (
                cache_key,
                input_json,
                result_json,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.connection.commit()

    def count(self):
        """
        返回缓存中的客户组评估数量。

        输入：无。
        输出：整数记录数。
        逻辑：执行单个 `COUNT(*)` 查询。
        """
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM group_evaluations"
            ).fetchone()[0]
        )

    def close(self):
        """
        提交剩余事务并关闭数据库连接。

        输入：无。
        输出：无。
        逻辑：显式释放 SQLite 文件句柄，便于测试和训练任务安全结束。
        """
        self.connection.commit()
        self.connection.close()

    def __enter__(self):
        """进入上下文管理器并返回当前缓存对象。"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文管理器时关闭数据库连接。"""
        self.close()

