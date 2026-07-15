"""
H2H 原生索引的图规范化、版本化缓存、跨进程构建调度与 ctypes 查询包装。

阶段 4 的生产流程为：规范化图并生成 SHA-256，获取同一缓存的跨进程锁，
在独立 `.building-*` 目录调用 builder，完成校验后写 READY 并原子重命名；
查询对象只保存路径和少量配置，pickle 后在 worker 中延迟重新 mmap 索引。
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import operator
import os
import platform
import re
import struct
import subprocess
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from config import (
    H2H_AUTO_BUILD,
    H2H_BUILDER_MAX_SHORTCUT_ARCS,
    H2H_BUILDER_MAX_STRUCTURAL_EDGES,
    H2H_BUILDER_PROGRESS_INTERVAL,
    H2H_BUILD_LOCK_POLL_SECONDS,
    H2H_BUILD_LOCK_TIMEOUT_SECONDS,
    H2H_ENABLE_55K,
    H2H_INDEX_DIR,
    H2H_LARGE_GRAPH_MIN_NODES,
    H2H_NATIVE_BUILD_DIR,
    H2H_QUERY_STATS,
    MANHATTAN_GRAPH_PATH,
)
from distance_oracle import ReadOnlyDistanceMatrix


GRAPH_FORMAT_VERSION = 1
INDEX_FORMAT_VERSION = 1
H2H_API_VERSION = 1
ENDIAN_MARKER = 0x01020304
DISTANCE_SEMANTICS_VERSION = 'endpoint-equirectangular-v1'
GRAPH_HEADER = struct.Struct('<8sIIIQ')
GRAPH_EDGE = struct.Struct('<IId')
INDEX_HEADER = struct.Struct('<8s8I11Q32s')

# builder 成功行采用稳定的 key=value 格式；服务器报告只解析已知数值字段。
_BUILD_STATISTICS_PATTERN = re.compile(
    r'H2H_BUILD_OK\s+'
    r'nodes=(?P<nodes>\d+)\s+'
    r'treewidth=(?P<treewidth>\d+)\s+'
    r'treeheight=(?P<treeheight>\d+)\s+'
    r'fill_edges=(?P<fill_edges>\d+)\s+'
    r'shortcut_arcs=(?P<shortcut_arcs>\d+)\s+'
    r'labels=(?P<label_count>\d+)\s+'
    r'positions=(?P<position_count>\d+)\s+'
    r'seconds=(?P<build_seconds>[0-9.eE+-]+)\s+'
    r'(?:peak_rss_bytes=(?P<peak_rss_bytes>\d+)\s+)?'
    r'index_bytes=(?P<index_bytes>\d+)'
)


@dataclass(frozen=True)
class NormalizedGraphData:
    """保存构建和哈希共用的连续节点坐标及最小有向边。"""

    node_count: int
    coordinates: tuple[tuple[float, float], ...]
    edges: tuple[tuple[int, int, float], ...]
    zero_weight_edges: int


@dataclass(frozen=True)
class H2HCacheResult:
    """描述一次缓存命中或新建结果。"""

    cache_dir: Path
    graph_path: Path
    index_path: Path
    metadata_path: Path
    graph_hash: str
    node_count: int
    built: bool


def _node_id(node: Any) -> int:
    """
    将 Python/NumPy 整数节点转换为 Python `int`。

    输入：任意节点标签。
    输出：整数编号；不支持整数协议时抛出 `TypeError`。
    """
    try:
        return int(operator.index(node))
    except TypeError as exc:
        raise TypeError(f'H2H 节点必须是连续整数，收到 {node!r}。') from exc


def _validate_strong_connectivity(
    node_count: int,
    edges: tuple[tuple[int, int, float], ...],
) -> None:
    """
    用正向/反向遍历验证规范化边强连通，不额外创建 NetworkX 图。

    输入：节点数量和有向简单边。
    输出：无；非强连通时报告两个方向的可达规模。
    """
    forward = [[] for _ in range(node_count)]
    reverse = [[] for _ in range(node_count)]
    for source, target, _ in edges:
        forward[source].append(target)
        reverse[target].append(source)

    def reachable(adjacency: list[list[int]]) -> int:
        """从节点 0 遍历输入邻接表并返回可达节点数。"""
        seen = bytearray(node_count)
        seen[0] = 1
        stack = [0]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = 1
                    stack.append(neighbor)
        return sum(seen)

    forward_count = reachable(forward)
    reverse_count = reachable(reverse)
    if forward_count != node_count or reverse_count != node_count:
        raise ValueError(
            f'H2H 要求强连通图；从节点 0 正向可达 {forward_count}/{node_count}，'
            f'反向可达 {reverse_count}/{node_count}。'
        )


def normalize_graph_for_h2h(graph: nx.Graph) -> NormalizedGraphData:
    """
    将标准化 DiGraph/MultiDiGraph 转为原生 builder 所需的确定性数据。

    输入：连续整数节点、`pos=[lon, lat]`、非负有限 `weight` 的有向图。
    输出：坐标和按 `(source, target)` 排序的最小平行边。

    实现逻辑：反向道路保持独立；自环经权重校验后忽略；平行边只保留最小值；
    最后以 O(n+m) 正反遍历确认强连通。
    """
    if not graph.is_directed():
        raise ValueError('H2H 不能把道路图隐式转成无向图。')
    node_count = graph.number_of_nodes()
    if node_count == 0:
        raise ValueError('不能为无节点图构建 H2H 索引。')

    normalized_nodes = {_node_id(node) for node in graph.nodes}
    if normalized_nodes != set(range(node_count)) or len(normalized_nodes) != node_count:
        raise ValueError(f'H2H 节点必须连续编号为 [0, {node_count})。')

    coordinates = []
    for node in range(node_count):
        attributes = graph.nodes[node]
        if 'pos' not in attributes or len(attributes['pos']) != 2:
            raise ValueError(f'节点 {node} 缺少合法 pos=[lon, lat]。')
        longitude = float(attributes['pos'][0])
        latitude = float(attributes['pos'][1])
        if not math.isfinite(longitude) or not math.isfinite(latitude):
            raise ValueError(f'节点 {node} 坐标必须为有限 float64。')
        coordinates.append((longitude, latitude))

    # `minimum_edges` 是唯一一份 O(m) 规范化工作数据，避免构造全对或复制原图。
    minimum_edges: dict[tuple[int, int], float] = {}
    for source, target, attributes in graph.edges(data=True):
        source_id = _node_id(source)
        target_id = _node_id(target)
        if 'weight' not in attributes:
            raise KeyError(f'边 {source_id} -> {target_id} 缺少 weight。')
        weight = float(attributes['weight'])
        if not math.isfinite(weight):
            raise ValueError(f'边 {source_id} -> {target_id} 权重必须有限。')
        if weight < 0.0:
            raise ValueError(f'边 {source_id} -> {target_id} 含负权 {weight}。')
        if source_id == target_id:
            continue
        key = (source_id, target_id)
        previous = minimum_edges.get(key)
        if previous is None or weight < previous:
            minimum_edges[key] = weight

    edges = tuple(
        (source, target, minimum_edges[(source, target)])
        for source, target in sorted(minimum_edges)
    )
    _validate_strong_connectivity(node_count, edges)
    zero_weight_edges = sum(weight == 0.0 for _, _, weight in edges)
    if zero_weight_edges:
        warnings.warn(
            f'H2H 规范化图包含 {zero_weight_edges} 条零权边。',
            RuntimeWarning,
            stacklevel=2,
        )
    return NormalizedGraphData(
        node_count=node_count,
        coordinates=tuple(coordinates),
        edges=edges,
        zero_weight_edges=zero_weight_edges,
    )


def compute_graph_hash(normalized: NormalizedGraphData) -> str:
    """
    流式计算覆盖坐标、拓扑、权重、方向和语义版本的 SHA-256。

    输入：已规范化且排序的数据。
    输出：64 个小写十六进制字符。
    """
    digest = hashlib.sha256()
    digest.update(b'MA-FSTSP-H2H-GRAPH\0')
    digest.update(struct.pack('<IIQ?', INDEX_FORMAT_VERSION, normalized.node_count,
                              len(normalized.edges), True))
    semantics = DISTANCE_SEMANTICS_VERSION.encode('utf-8')
    digest.update(struct.pack('<I', len(semantics)))
    digest.update(semantics)
    for node, (longitude, latitude) in enumerate(normalized.coordinates):
        digest.update(struct.pack('<Idd', node, longitude, latitude))
    for source, target, weight in normalized.edges:
        digest.update(GRAPH_EDGE.pack(source, target, weight))
    return digest.hexdigest()


def graph_fingerprint(graph: nx.Graph) -> str:
    """
    计算一张标准化图的 H2H SHA-256。

    输入：路网图。
    输出：图哈希；主要用于加载日志和测试。
    """
    return compute_graph_hash(normalize_graph_for_h2h(graph))


def _slugify_dataset_name(dataset_name: str | None) -> str:
    """
    将数据集名称转换为安全、稳定且有长度上限的缓存目录前缀。

    输入：可选数据集名称。
    输出：仅含小写字母、数字和连字符的前缀。
    """
    raw = (dataset_name or 'dataset').strip().lower()
    slug = re.sub(r'[^a-z0-9]+', '-', raw).strip('-')
    return (slug or 'dataset')[:48]


def native_artifact_paths(build_dir: Path | None = None) -> tuple[Path, Path]:
    """
    返回当前平台 builder 和查询动态库的默认路径。

    输入：可选原生构建目录。
    输出：`(builder_path, library_path)`。
    """
    root = Path(build_dir or H2H_NATIVE_BUILD_DIR)
    if platform.system() == 'Windows':
        return root / 'h2h_builder.exe', root / 'h2h_query.dll'
    return root / 'h2h_builder', root / 'libh2h_query.so'


def enforce_local_graph_guard(
    graph: nx.Graph | None = None,
    graph_path: str | Path | None = None,
) -> None:
    """
    在读取目标 NYC 文件前或未知大图读取后执行本机硬保护。

    输入：可选图对象和源文件路径。
    输出：无；保护开启时抛出包含服务器建议的 `RuntimeError`。
    """
    if H2H_ENABLE_55K:
        return
    if graph_path is not None:
        selected = Path(graph_path).expanduser().resolve()
        protected = Path(MANHATTAN_GRAPH_PATH).expanduser().resolve()
        if selected == protected:
            raise RuntimeError(
                f'本机禁止读取或构建 55k 目标图 {selected}。请仅在约 200 GB RAM 的服务器'
                '将 config.H2H_ENABLE_55K 设为 True 后运行；当前未启动 APSP、builder 或求解。'
            )
    if graph is not None and graph.number_of_nodes() >= H2H_LARGE_GRAPH_MIN_NODES:
        raise RuntimeError(
            f'本机禁止处理含 {graph.number_of_nodes()} 个节点的大图。请在服务器显式设置 '
            'config.H2H_ENABLE_55K=True；当前不会回退 Dijkstra。'
        )


def _write_graph_binary(path: Path, normalized: NormalizedGraphData) -> None:
    """
    流式写出 C++ builder 的 graph.bin 并刷新到磁盘。

    输入：临时 graph.bin 路径和规范化数据。
    输出：无。
    """
    with path.open('wb') as output:
        output.write(GRAPH_HEADER.pack(
            b'H2HGRPH1', GRAPH_FORMAT_VERSION, ENDIAN_MARKER,
            normalized.node_count, len(normalized.edges),
        ))
        for source, target, weight in normalized.edges:
            output.write(GRAPH_EDGE.pack(source, target, weight))
        output.flush()
        os.fsync(output.fileno())


def _write_text_fsync(path: Path, content: str) -> None:
    """
    以 UTF-8 写文本并在返回前 fsync。

    输入：目标路径和文本。
    输出：无。
    """
    with path.open('w', encoding='utf-8', newline='\n') as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _read_index_header(index_path: Path) -> tuple[Any, ...]:
    """
    读取并解包固定 160 字节索引头。

    输入：index.bin 路径。
    输出：`INDEX_HEADER.unpack` 元组；短文件或 magic 错误时抛出异常。
    """
    with index_path.open('rb') as index_file:
        payload = index_file.read(INDEX_HEADER.size)
    if len(payload) != INDEX_HEADER.size:
        raise ValueError(f'索引头长度不是 {INDEX_HEADER.size} 字节。')
    header = INDEX_HEADER.unpack(payload)
    if header[0] != b'H2HIDX01':
        raise ValueError('索引 magic 不匹配。')
    if header[1] != INDEX_FORMAT_VERSION or header[2] != ENDIAN_MARKER:
        raise ValueError('索引版本或端序不匹配。')
    if header[3] != H2H_API_VERSION or header[4] != INDEX_HEADER.size:
        raise ValueError('索引 API 版本或头部大小不匹配。')
    return header


def _cache_validation_error(cache_dir: Path, expected_hash: str | None = None) -> str | None:
    """
    校验 READY、metadata、graph.bin 和 index.bin 的相互一致性。

    输入：缓存目录与可选预期图哈希。
    输出：有效时为 `None`，否则为可读错误文本。
    """
    ready_path = cache_dir / 'READY'
    metadata_path = cache_dir / 'metadata.json'
    graph_path = cache_dir / 'graph.bin'
    index_path = cache_dir / 'index.bin'
    for path in (ready_path, metadata_path, graph_path, index_path, cache_dir / 'build.log'):
        if not path.is_file():
            return f'缺少 {path.name}'
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        return f'metadata.json 无效：{exc}'
    graph_hash = metadata.get('graph_hash')
    if not isinstance(graph_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', graph_hash):
        return 'metadata graph_hash 无效'
    if expected_hash is not None and graph_hash != expected_hash:
        return 'metadata graph_hash 与当前图不一致'
    if ready_path.read_text(encoding='utf-8').strip() != graph_hash:
        return 'READY 内容与 graph_hash 不一致'
    if metadata.get('index_format_version') != INDEX_FORMAT_VERSION:
        return 'metadata 索引版本不一致'
    if metadata.get('distance_semantics') != DISTANCE_SEMANTICS_VERSION:
        return 'metadata 距离语义不一致'
    try:
        header = _read_index_header(index_path)
        if header[5] != metadata.get('node_count'):
            return '索引节点数与 metadata 不一致'
        if header[19] != index_path.stat().st_size:
            return '索引头文件大小与实际不一致'
        if header[20].hex() != graph_hash:
            return '索引头 metadata 哈希与图哈希不一致'
        # 只读取固定图头，缓存校验不能把 graph.bin 整体复制进 Python 内存。
        with graph_path.open('rb') as graph_file:
            graph_header_payload = graph_file.read(GRAPH_HEADER.size)
        graph_header = GRAPH_HEADER.unpack(graph_header_payload)
        if graph_header[0] != b'H2HGRPH1' or graph_header[3] != metadata.get('node_count'):
            return 'graph.bin 头部与 metadata 不一致'
        if graph_header[4] != metadata.get('edge_count'):
            return 'graph.bin 边数与 metadata 不一致'
        expected_graph_size = GRAPH_HEADER.size + graph_header[4] * GRAPH_EDGE.size
        if graph_path.stat().st_size != expected_graph_size:
            return 'graph.bin 文件大小与边数不一致'
    except (OSError, ValueError, struct.error) as exc:
        return f'二进制缓存校验失败：{exc}'
    return None


def validate_h2h_cache(cache_dir: str | Path, expected_hash: str | None = None) -> bool:
    """
    判断一个缓存是否具备完整 READY 和一致的版本/哈希。

    输入：缓存目录及可选预期哈希。
    输出：布尔值；不会加载动态库或启动 builder。
    """
    return _cache_validation_error(Path(cache_dir), expected_hash) is None


def parse_h2h_build_statistics(build_log: str) -> dict[str, int | float]:
    """
    从 builder 的成功日志解析服务器验收所需统计量。

    输入：完整 build.log 文本。
    输出：treewidth、treeheight、fill-in、shortcut、标签、耗时、峰值 RSS 等数值；
    日志不含成功行时抛出明确异常。
    """
    match = _BUILD_STATISTICS_PATTERN.search(build_log)
    if match is None:
        raise ValueError('build.log 不含可解析的 H2H_BUILD_OK 统计行。')
    integer_fields = (
        'nodes', 'treewidth', 'treeheight', 'fill_edges', 'shortcut_arcs',
        'label_count', 'position_count', 'peak_rss_bytes', 'index_bytes',
    )
    statistics: dict[str, int | float] = {}
    for field in integer_fields:
        value = match.group(field)
        statistics[field] = int(value) if value is not None else 0
    statistics['build_seconds'] = float(match.group('build_seconds'))
    return statistics


def read_h2h_index_statistics(cache_dir: str | Path) -> dict[str, int | float | str]:
    """
    读取一份完整缓存的索引头、构建日志和图身份统计。

    输入：包含 READY、metadata.json、build.log 与 index.bin 的缓存目录。
    输出：可直接写入 JSON 验收报告的扁平字典；不会 mmap 标签或执行距离查询。
    """
    directory = Path(cache_dir).expanduser().resolve()
    validation_error = _cache_validation_error(directory)
    if validation_error is not None:
        raise ValueError(f'H2H 缓存无效，无法读取统计：{validation_error}')
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    header = _read_index_header(directory / 'index.bin')
    statistics: dict[str, int | float | str] = {
        'graph_hash': str(metadata['graph_hash']),
        'node_count': int(header[5]),
        'treeheight': int(header[7]),
        'treewidth': int(header[8]),
        'label_count': int(header[9]),
        'position_count': int(header[10]),
        'index_bytes': int(header[19]),
    }
    # 构建日志保留 fill-in、shortcut 和峰值 RSS；索引头字段覆盖重复统计。
    log_statistics = parse_h2h_build_statistics(
        (directory / 'build.log').read_text(encoding='utf-8')
    )
    statistics.update(log_statistics)
    statistics.update({
        'node_count': int(header[5]),
        'treeheight': int(header[7]),
        'treewidth': int(header[8]),
        'label_count': int(header[9]),
        'position_count': int(header[10]),
        'index_bytes': int(header[19]),
    })
    return statistics


class H2HBuildLock:
    """使用 O_EXCL 锁文件协调不同 Python 进程构建同一索引。"""

    def __init__(self, path: Path, timeout: float, poll_interval: float) -> None:
        """保存锁路径与等待参数；尚不创建文件。"""
        self.path = path
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self.token = uuid.uuid4().hex
        self.acquired = False

    def __enter__(self) -> 'H2HBuildLock':
        """
        原子创建锁文件；已存在时轮询到超时。

        输出：已持锁对象。不会因超时删除其他进程的锁。
        """
        deadline = time.monotonic() + self.timeout
        payload = json.dumps({
            'pid': os.getpid(),
            'token': self.token,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False)
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f'等待 H2H 构建锁超时：{self.path}。请确认持锁进程是否仍在运行。'
                    )
                time.sleep(self.poll_interval)
                continue
            with os.fdopen(descriptor, 'w', encoding='utf-8') as lock_file:
                lock_file.write(payload)
                lock_file.flush()
                os.fsync(lock_file.fileno())
            self.acquired = True
            return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """仅当文件 token 仍属于当前对象时删除锁文件。"""
        if not self.acquired:
            return
        try:
            payload = json.loads(self.path.read_text(encoding='utf-8'))
            if payload.get('token') == self.token:
                self.path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            # 不确定归属时保留锁，避免误删新持有者的文件。
            pass
        self.acquired = False


def _quarantine_invalid_cache(cache_dir: Path) -> Path:
    """
    将无效最终目录原子改名保留，给同一哈希重新构建腾出目标名称。

    输入：已存在但校验失败的缓存目录。
    输出：带 `.invalid-*` 后缀的保留目录。
    """
    quarantine = cache_dir.with_name(
        f'{cache_dir.name}.invalid-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    )
    cache_dir.replace(quarantine)
    return quarantine


def ensure_h2h_index(
    graph: nx.Graph,
    dataset_name: str | None = None,
    index_dir: str | Path | None = None,
    builder_path: str | Path | None = None,
) -> H2HCacheResult:
    """
    命中或原子构建一份与当前图哈希完全一致的 H2H 缓存。

    输入：图、数据集名称、可选缓存根和 builder 路径。
    输出：包含缓存/索引路径、哈希和是否本次构建的结果。

    实现逻辑：大图保护先于目录创建；哈希相同的进程共享 O_EXCL 锁；构建发生在
    唯一临时目录，校验 metadata/index 哈希后写 READY，最后原子重命名。
    """
    enforce_local_graph_guard(graph=graph)
    normalized = normalize_graph_for_h2h(graph)
    graph_hash = compute_graph_hash(normalized)
    cache_root = Path(index_dir or H2H_INDEX_DIR).expanduser().resolve()
    cache_name = f'{_slugify_dataset_name(dataset_name)}-{graph_hash}-h2h-v{INDEX_FORMAT_VERSION}'
    cache_dir = cache_root / cache_name
    result_arguments = {
        'cache_dir': cache_dir,
        'graph_path': cache_dir / 'graph.bin',
        'index_path': cache_dir / 'index.bin',
        'metadata_path': cache_dir / 'metadata.json',
        'graph_hash': graph_hash,
        'node_count': normalized.node_count,
    }
    if validate_h2h_cache(cache_dir, graph_hash):
        return H2HCacheResult(**result_arguments, built=False)
    if not H2H_AUTO_BUILD:
        raise FileNotFoundError(
            f'H2H 缓存缺失或无效：{cache_dir}。请启用 H2H_AUTO_BUILD 或手工构建。'
        )

    selected_builder = Path(builder_path) if builder_path else native_artifact_paths()[0]
    selected_builder = selected_builder.expanduser().resolve()
    if not selected_builder.is_file():
        raise FileNotFoundError(
            f'H2H builder 不存在：{selected_builder}。请先运行 '
            r'D:\Anaconda3\envs\MA-FSTSP\python.exe scripts\build_h2h_native.py --release'
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / f'.{cache_name}.lock'
    with H2HBuildLock(
        lock_path,
        timeout=H2H_BUILD_LOCK_TIMEOUT_SECONDS,
        poll_interval=H2H_BUILD_LOCK_POLL_SECONDS,
    ):
        if validate_h2h_cache(cache_dir, graph_hash):
            return H2HCacheResult(**result_arguments, built=False)
        if cache_dir.exists():
            _quarantine_invalid_cache(cache_dir)

        temporary_dir = cache_root / (
            f'{cache_name}.building-{os.getpid()}-{uuid.uuid4().hex[:8]}'
        )
        temporary_dir.mkdir(parents=False, exist_ok=False)
        graph_path = temporary_dir / 'graph.bin'
        index_path = temporary_dir / 'index.bin'
        build_log_path = temporary_dir / 'build.log'
        _write_graph_binary(graph_path, normalized)

        command = [
            str(selected_builder), '--graph', str(graph_path), '--output', str(index_path),
            '--metadata-hash', graph_hash,
            '--progress-interval', str(H2H_BUILDER_PROGRESS_INTERVAL),
        ]
        if H2H_BUILDER_MAX_STRUCTURAL_EDGES:
            command.extend(('--max-structural-edges', str(H2H_BUILDER_MAX_STRUCTURAL_EDGES)))
        if H2H_BUILDER_MAX_SHORTCUT_ARCS:
            command.extend(('--max-shortcut-arcs', str(H2H_BUILDER_MAX_SHORTCUT_ARCS)))
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding='utf-8',
            errors='replace',
            check=False,
        )
        _write_text_fsync(build_log_path, process.stdout)
        if process.returncode != 0:
            raise RuntimeError(
                f'H2H builder 失败，退出码 {process.returncode}；临时目录保留在 '
                f'{temporary_dir}，日志为 {build_log_path}。'
            )

        build_statistics = parse_h2h_build_statistics(process.stdout)

        header = _read_index_header(index_path)
        if header[5] != normalized.node_count or header[20].hex() != graph_hash:
            raise RuntimeError('builder 生成的索引节点数或 metadata 哈希与当前图不一致。')
        metadata = {
            'api_version': H2H_API_VERSION,
            'built_at_utc': datetime.now(timezone.utc).isoformat(),
            'builder_path': str(selected_builder),
            'dataset_name': dataset_name or 'dataset',
            'directed': True,
            'distance_dtype': 'float64',
            'distance_semantics': DISTANCE_SEMANTICS_VERSION,
            'edge_count': len(normalized.edges),
            'graph_format_version': GRAPH_FORMAT_VERSION,
            'graph_hash': graph_hash,
            'index_format_version': INDEX_FORMAT_VERSION,
            'node_count': normalized.node_count,
            'zero_weight_edges': normalized.zero_weight_edges,
            'build_statistics': build_statistics,
        }
        _write_text_fsync(
            temporary_dir / 'metadata.json',
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        )
        _write_text_fsync(temporary_dir / 'READY', graph_hash + '\n')
        validation_error = _cache_validation_error(temporary_dir, graph_hash)
        if validation_error is not None:
            raise RuntimeError(f'H2H 临时缓存内部校验失败：{validation_error}')
        temporary_dir.replace(cache_dir)

    if not validate_h2h_cache(cache_dir, graph_hash):
        raise RuntimeError(f'H2H 缓存原子发布后校验失败：{cache_dir}')
    return H2HCacheResult(**result_arguments, built=True)


_LIBRARY_CACHE: dict[Path, ctypes.CDLL] = {}
_LIBRARY_CACHE_LOCK = threading.Lock()


def _load_native_library(library_path: Path) -> ctypes.CDLL:
    """
    每个 Python 进程按绝对路径加载并配置一次 H2H 动态库。

    输入：DLL/SO 路径。
    输出：已声明 C ABI 类型的 `ctypes.CDLL`。
    """
    resolved = library_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f'H2H 查询动态库不存在：{resolved}。请先运行 scripts/build_h2h_native.py。'
        )
    with _LIBRARY_CACHE_LOCK:
        library = _LIBRARY_CACHE.get(resolved)
        if library is not None:
            return library
        library = ctypes.CDLL(str(resolved))
        library.h2h_api_version.argtypes = []
        library.h2h_api_version.restype = ctypes.c_uint32
        library.h2h_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        library.h2h_open.restype = ctypes.c_void_p
        library.h2h_query.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        library.h2h_query.restype = ctypes.c_double
        library.h2h_query_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        ]
        library.h2h_query_batch.restype = ctypes.c_int
        library.h2h_close.argtypes = [ctypes.c_void_p]
        library.h2h_close.restype = None
        version = int(library.h2h_api_version())
        if version != H2H_API_VERSION:
            raise RuntimeError(
                f'H2H 动态库 API 版本为 {version}，Python 需要 {H2H_API_VERSION}。'
            )
        _LIBRARY_CACHE[resolved] = library
        return library


class H2HDistanceMatrix(ReadOnlyDistanceMatrix):
    """通过只读 mmap 原生索引实现 `distance['truck'][u][v]`。"""

    def __init__(
        self,
        index_path: str | Path,
        library_path: str | Path | None = None,
        node_count: int | None = None,
        graph_hash: str | None = None,
        stats_enabled: bool | None = None,
    ) -> None:
        """
        保存索引身份与动态库路径，但延迟到首次查询才打开原生句柄。

        输入：index.bin、可选 DLL/SO、节点数、图哈希和统计开关。
        输出：可 pickle 的只读距离矩阵。
        """
        self._initialize(index_path, library_path, node_count, graph_hash, stats_enabled)

    def _initialize(
        self,
        index_path: str | Path,
        library_path: str | Path | None,
        node_count: int | None,
        graph_hash: str | None,
        stats_enabled: bool | None,
    ) -> None:
        """初始化可序列化字段和进程本地运行状态，并校验索引头身份。"""
        self.index_path = Path(index_path).expanduser().resolve()
        default_library = native_artifact_paths()[1]
        self.library_path = Path(library_path or default_library).expanduser().resolve()
        header = _read_index_header(self.index_path)
        header_node_count = int(header[5])
        if node_count is not None and int(node_count) != header_node_count:
            raise ValueError('传入 node_count 与 index.bin 头部不一致。')
        header_hash = header[20].hex()
        if graph_hash is not None and graph_hash != header_hash:
            raise ValueError('传入 graph_hash 与 index.bin 头部不一致。')
        self.graph_hash = graph_hash or header_hash
        self.backend_version = H2H_API_VERSION
        self.stats_enabled = H2H_QUERY_STATS if stats_enabled is None else bool(stats_enabled)
        super().__init__(range(header_node_count))
        # 以下字段只属于当前进程，绝不能进入 pickle state。
        self._library: ctypes.CDLL | None = None
        self._handle: int | None = None
        self._open_lock = threading.Lock()
        self._query_count = 0
        self._query_seconds = 0.0
        self._load_seconds = 0.0

    def _ensure_open(self) -> None:
        """线程安全地加载动态库并 mmap 打开索引；重复调用不做工作。"""
        if self._handle is not None:
            return
        with self._open_lock:
            if self._handle is not None:
                return
            started_at = time.perf_counter()
            library = _load_native_library(self.library_path)
            error_buffer = ctypes.create_string_buffer(2048)
            handle = library.h2h_open(
                str(self.index_path).encode('utf-8'), error_buffer, len(error_buffer)
            )
            if not handle:
                message = error_buffer.value.decode('utf-8', errors='replace')
                raise RuntimeError(f'打开 H2H 索引失败：{message}')
            self._library = library
            self._handle = handle
            self._load_seconds += time.perf_counter() - started_at

    def close(self) -> None:
        """关闭当前进程的原生 mmap 句柄；路径和可 pickle 状态保持可复用。"""
        handle = self._handle
        library = self._library
        self._handle = None
        self._library = None
        if handle is not None and library is not None:
            library.h2h_close(handle)

    def __del__(self):
        """对象回收时尽力关闭原生句柄，不传播解释器退出阶段异常。"""
        try:
            self.close()
        except Exception:
            pass

    def query(self, source: Any, target: Any) -> float:
        """
        查询一个有向最短距离并返回 Python float。

        输入：Python/NumPy 整数源点和终点。
        输出：有限 float；原生错误或越界由明确异常报告。
        """
        source_id = self._validate_node(source, '源')
        target_id = self._validate_node(target, '目标')
        self._ensure_open()
        started_at = time.perf_counter() if self.stats_enabled else 0.0
        value = float(self._library.h2h_query(self._handle, source_id, target_id))
        if not math.isfinite(value):
            raise RuntimeError(f'H2H 原生查询 {source_id} -> {target_id} 失败。')
        if self.stats_enabled:
            self._query_count += 1
            self._query_seconds += time.perf_counter() - started_at
        return value

    def query_batch(self, sources, targets) -> list[float]:
        """
        通过一次 C ABI 调用查询等长有序节点对。

        输入：sources 和 targets 可迭代对象。
        输出：保持输入顺序的 Python float 列表。
        """
        source_ids = [self._validate_node(source, '源') for source in sources]
        target_ids = [self._validate_node(target, '目标') for target in targets]
        if len(source_ids) != len(target_ids):
            raise ValueError('sources 与 targets 长度必须一致。')
        self._ensure_open()
        count = len(source_ids)
        source_array = (ctypes.c_uint32 * count)(*source_ids)
        target_array = (ctypes.c_uint32 * count)(*target_ids)
        output_array = (ctypes.c_double * count)()
        started_at = time.perf_counter() if self.stats_enabled else 0.0
        status = self._library.h2h_query_batch(
            self._handle, source_array, target_array, count, output_array
        )
        if status != 0:
            raise RuntimeError(f'H2H 原生批量查询失败，状态码 {status}。')
        if self.stats_enabled:
            self._query_count += count
            self._query_seconds += time.perf_counter() - started_at
        return [float(value) for value in output_array]

    @property
    def statistics(self) -> dict[str, float | int]:
        """返回当前进程的加载和查询统计，不触发索引打开。"""
        return {
            'query_count': self._query_count,
            'query_seconds': self._query_seconds,
            'load_seconds': self._load_seconds,
        }

    def __getstate__(self) -> dict[str, Any]:
        """
        仅序列化索引/DLL 路径、版本、节点数、哈希和统计开关。

        输出中不包含 ctypes.CDLL、原生句柄、mmap 或线程锁。
        """
        return {
            'index_path': str(self.index_path),
            'library_path': str(self.library_path),
            'node_count': self.node_count,
            'graph_hash': self.graph_hash,
            'backend_version': self.backend_version,
            'stats_enabled': self.stats_enabled,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """在 worker 中恢复少量状态；首次查询时重新加载 DLL 并 mmap 索引。"""
        if state.get('backend_version') != H2H_API_VERSION:
            raise ValueError('pickle 中的 H2H 后端版本不受支持。')
        self._initialize(
            state['index_path'],
            state['library_path'],
            state['node_count'],
            state['graph_hash'],
            state['stats_enabled'],
        )
