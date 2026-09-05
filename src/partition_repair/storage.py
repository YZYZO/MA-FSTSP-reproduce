"""实例清单、离线标签与运行记录的持久化，不在正式计时中复用已求解结果。"""

from dataclasses import asdict, is_dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import numpy as np


def native(value):
    """递归将数组、标量、路径和配置转为 JSON 数据，保留原始数值而不生成字符串标签。"""
    if is_dataclass(value):
        return native(asdict(value))
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def fingerprint(value):
    """输入可序列化上下文，输出稳定 SHA256，用于防止实例或求解策略混用。"""
    payload = json.dumps(native(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def file_fingerprint(path):
    """输入文件路径，分块计算内容指纹，地图路径变化不会改变相同地图的身份。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime_metadata(root):
    """输入项目根目录，输出算法源码、依赖和机器记录，隔离不同环境的时间标签。"""
    root = Path(root)
    paths = [root / name for name in ('src/fstsp.py', 'src/baseline.py', 'utils.py', 'problem.py')]
    paths += sorted((root / 'src/partition_repair').glob('*.py'))
    return {
        'source_fingerprint': fingerprint({str(p.relative_to(root)): file_fingerprint(p) for p in paths}),
        'packages': {name: version(name) for name in ('numpy', 'networkx', 'gurobipy', 'elkai', 'scipy')},
        'python': platform.python_version(), 'platform': platform.platform(),
        'machine': platform.node(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
        'distance_semantics': 'directed_endpoint_haversine_edges_v1',
        'phase2_semantics': 'original_max_truck_drone_divided_by_speed_v1',
        'fallback': 'directed_nearest_neighbor_v1',
    }


def context_fingerprint(model, boundary, graph_fingerprint, runtime):
    """输入完整实例、两类区域、地图和运行元数据，返回组标签可复用的全局上下文键。"""
    return fingerprint({
        'graph': graph_fingerprint, 'runtime': runtime,
        'depots': list(model.depots), 'cities': list(model.cities),
        'boundary': boundary, 'regions': model.regions,
        'drone': model.drone, 'limit': model.limit, 'speed': model.speed, 'theta': model.theta,
    })


def group_cache_key(context_id, depot, customers, options, repeat=0):
    """输入全局上下文、组成员、求解策略和重复编号，输出稳定键；不同重复不得互相复用。"""
    return fingerprint({'context': context_id, 'depot': depot, 'customers': sorted(customers),
                        'solver': options, 'repeat': repeat})


def save_json(path, value):
    """将对象原子写入指定 JSON 文件，完整替换目标，避免中断后留下半份结果。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(native(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def read_json(path):
    """读取一个 UTF-8 JSON 文件并返回原始数据对象。"""
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


class RecordTable:
    """以每条记录独立 JSON 文件持久化、以清单 JSON 导出，支持可靠的组级续跑。"""

    def __init__(self, directory, name):
        """输入结果目录和表名，载入已有记录；各记录的 id 是幂等保存键。"""
        self.directory = Path(directory) / name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records = {path.stem: read_json(path) for path in sorted(self.directory.glob('*.json'))}

    def put(self, record):
        """输入带 id 的记录，原子保存并更新内存索引，返回该记录。"""
        record = native(record)
        save_json(self.directory / (record['id'] + '.json'), record)
        self.records[record['id']] = record
        return record

    def export(self):
        """将当前表导出成单一 JSON 数组，供分析脚本或后续学习使用。"""
        save_json(self.directory.parent / (self.directory.name + '.json'), list(self.records.values()))
