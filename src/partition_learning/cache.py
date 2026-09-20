"""使用 SQLite 持久化昂贵的仓库组真实求解结果。"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading


class GroupEvaluationCache:
    """按稳定键读写组评价 JSON，使三轮实验可中断续跑。"""

    def __init__(self, path: str | Path):
        """输入数据库路径，创建父目录、表和当前连接。"""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.RLock()
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS group_evaluations (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.commit()

    def get(self, cache_key: str) -> dict | None:
        """输入组缓存键，命中时返回评价字典，否则返回 None。"""
        with self.lock:
            row = self.connection.execute(
                "SELECT payload FROM group_evaluations WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, cache_key: str, payload: dict) -> None:
        """输入缓存键与评价字典，原子写入或覆盖该组记录。"""
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO group_evaluations(cache_key, payload) VALUES (?, ?)",
                (cache_key, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )
            self.connection.commit()

    def close(self) -> None:
        """提交待写事务并关闭数据库连接。"""
        with self.lock:
            self.connection.commit()
            self.connection.close()

    def __enter__(self) -> "GroupEvaluationCache":
        """返回上下文管理器自身。"""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """退出上下文时关闭连接。"""
        self.close()
