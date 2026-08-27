"""配对实验清单、输入哈希与原子检查点存储。"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

import numpy as np


def to_jsonable(value):
    """
    将实验对象递归转换成严格 JSON 可序列化数据。

    输入可包含 NumPy 标量/数组、Path、字典和非有限浮点数；输出只包含基础类型，
    非有限值转为 ``None``，避免检查点因 NaN/Infinity 破坏跨工具兼容性。
    """

    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(to_jsonable(key)): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def stable_instance_hash(dataset_key, depots, cities, graph_hash=None):
    """
    为一个采样实例生成与方法无关的稳定 SHA-256 标识。

    输入为数据集键、仓库序列、客户序列和可选图文件哈希；输出十六进制哈希。
    序列顺序保留，图内容变化也会产生新标识，所有对照方法共享同一哈希。
    """

    payload = {
        'dataset_key': str(dataset_key),
        'depots': to_jsonable(depots),
        'cities': to_jsonable(cities),
        'graph_hash': graph_hash,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path, payload):
    """
    通过同目录临时文件和原子替换写入一个 JSON 检查点。

    输入为目标路径和记录对象；输出目标 ``Path``。进程中断最多留下无效临时文件，
    不会覆盖已有完整记录，因此正式实验可以按“实例×方法”安全续跑。
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
    text = json.dumps(
        to_jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return target


def read_json(path):
    """
    读取一个 UTF-8 JSON 检查点。

    输入为文件路径；输出反序列化对象。解析异常直接向上抛出，防止损坏记录被误判
    为已完成实验。
    """

    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


class PairedExperimentStore:
    """
    管理一批配对实验的不可变清单与逐方法检查点。

    构造输入为批次目录和 manifest；首次创建时写入，续跑时要求 protocol_signature
    完全一致。记录以 instance_id/method 分层保存，输出可被分析脚本直接扫描。
    """

    def __init__(self, root, manifest):
        self.root = Path(root)
        self.records_root = self.root / 'records'
        self.manifest_path = self.root / 'manifest.json'
        self.root.mkdir(parents=True, exist_ok=True)
        expected = to_jsonable(manifest)
        if self.manifest_path.exists():
            actual = read_json(self.manifest_path)
            if actual.get('protocol_signature') != expected.get('protocol_signature'):
                raise ValueError('续跑目录中的实验协议与当前配置不一致。')
            self.manifest = actual
        else:
            atomic_write_json(self.manifest_path, expected)
            self.manifest = expected

    @staticmethod
    def _safe_component(value):
        """
        将方法名或实例名转换为安全的相对路径组件。

        输入为任意标识；输出仅含字母、数字、点、下划线和连字符的字符串。
        """

        component = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))
        if not component or component in {'.', '..'}:
            raise ValueError(f'非法检查点名称：{value!r}')
        return component

    def record_path(self, instance_id, method):
        """
        计算一个实例-方法检查点的确定路径。

        输入为 instance_id 与方法名；输出位于当前批次 ``records`` 目录内的 Path。
        """

        return (
            self.records_root
            / self._safe_component(instance_id)
            / f'{self._safe_component(method)}.json'
        )

    def has_record(self, instance_id, method):
        """
        判断指定实例-方法是否已有完整检查点。

        输入为两级标识；输出布尔值。失败记录也视为已完成观测，避免续跑无限重试。
        """

        return self.record_path(instance_id, method).is_file()

    def write_record(self, instance_id, method, record):
        """
        原子保存一个实例-方法记录并强制写入配对标识。

        输入为实例 ID、方法和记录数据；输出记录路径。调用方不能用正文覆盖这两个键。
        """

        payload = dict(record)
        payload['instance_id'] = str(instance_id)
        payload['method'] = str(method)
        return atomic_write_json(self.record_path(instance_id, method), payload)

    def read_record(self, instance_id, method):
        """
        读取指定实例-方法记录。

        输入为实例 ID 与方法；输出检查点字典。
        """

        return read_json(self.record_path(instance_id, method))
