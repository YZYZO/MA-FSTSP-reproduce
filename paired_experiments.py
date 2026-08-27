"""同实例、多方法、可续跑的第一阶段配对实验运行器。"""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

from experiment_results import _save_npz, _solve_model_with_process_data
from experiment_store import PairedExperimentStore, stable_instance_hash, to_jsonable
from problem import sample_multiagent_instances
from src.fstsp import MultiAgentFlyingSidekickTSP
from src.partition import prepare_set_gtds, solve_prepared_set_gtds


GTDS_VARIANTS = {
    # 每个变体显式声明三个实验因素，避免方法名与实际求解参数发生漂移。
    'directed_set_gtds': {
        'epsilon': 0.01, 'apply_budget': True,
        'active_depot_policy': 'all', 'drone_cost_mode': 'paper',
    },
    'set_gtds_no_budget': {
        'epsilon': 0.01, 'apply_budget': False,
        'active_depot_policy': 'all', 'drone_cost_mode': 'paper',
    },
    'gtds_free_eps01': {
        'epsilon': 0.01, 'apply_budget': True,
        'active_depot_policy': 'free', 'drone_cost_mode': 'paper',
    },
    'gtds_sqrt2': {
        'epsilon': 0.01, 'apply_budget': True,
        'active_depot_policy': 'all', 'drone_cost_mode': 'smst_compatible',
    },
    'gtds_all_eps000': {
        'epsilon': 0.0, 'apply_budget': True,
        'active_depot_policy': 'all', 'drone_cost_mode': 'paper',
    },
    'gtds_all_eps005': {
        'epsilon': 0.005, 'apply_budget': True,
        'active_depot_policy': 'all', 'drone_cost_mode': 'paper',
    },
    'gtds_all_eps010': {
        'epsilon': 0.01, 'apply_budget': True,
        'active_depot_policy': 'all', 'drone_cost_mode': 'paper',
    },
    'gtds_all_eps020': {
        'epsilon': 0.02, 'apply_budget': True,
        'active_depot_policy': 'all', 'drone_cost_mode': 'paper',
    },
    'gtds_all_eps050': {
        'epsilon': 0.05, 'apply_budget': True,
        'active_depot_policy': 'all', 'drone_cost_mode': 'paper',
    },
    # 旧方法名仅作为读取旧配置的兼容入口，新 V2 协议不再使用。
    'gtds_free_eps05': {
        'epsilon': 0.05, 'apply_budget': True,
        'active_depot_policy': 'free', 'drone_cost_mode': 'paper',
    },
    'gtds_free_eps10': {
        'epsilon': 0.10, 'apply_budget': True,
        'active_depot_policy': 'free', 'drone_cost_mode': 'paper',
    },
    'gtds_all_eps01': {
        'epsilon': 0.01, 'apply_budget': True,
        'active_depot_policy': 'all', 'drone_cost_mode': 'paper',
    },
}


def _file_sha256(path):
    """流式计算文件 SHA-256；输入为路径，输出十六进制摘要。"""

    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _graph_sha256(graph):
    """在图文件不可用的测试场景中，对内存图结构生成确定性 SHA-256。"""

    nodes = sorted(
        ((repr(node), to_jsonable(attributes)) for node, attributes in graph.nodes(data=True)),
        key=lambda item: item[0],
    )
    if graph.is_multigraph():
        raw_edges = graph.edges(keys=True, data=True)
        edges = [
            (repr(start), repr(end), repr(key), to_jsonable(attributes))
            for start, end, key, attributes in raw_edges
        ]
    else:
        edges = [
            (repr(start), repr(end), to_jsonable(attributes))
            for start, end, attributes in graph.edges(data=True)
        ]
    payload = {
        'directed': bool(graph.is_directed()),
        'multigraph': bool(graph.is_multigraph()),
        'nodes': nodes,
        'edges': sorted(edges, key=repr),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _environment_versions():
    """收集影响实验复现的解释器、平台与核心依赖版本。"""

    packages = ('numpy', 'scipy', 'networkx', 'elkai', 'gurobipy')
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': versions,
    }


def _implementation_fingerprint():
    """
    计算会影响配对实验语义的核心源码指纹。

    无输入；输出 SHA-256。该值写入协议签名，避免代码修改后把新结果静默续写进旧
    批次。需要继续新实现时应使用新的输出目录，而不是绕过 manifest 校验。
    """

    project_root = Path(__file__).resolve().parent
    source_paths = (
        project_root / 'config.py',
        project_root / 'paired_experiments.py',
        project_root / 'experiment_store.py',
        project_root / 'experiment_results.py',
        project_root / 'problem.py',
        project_root / 'utils.py',
        project_root / 'src' / 'partition.py',
        project_root / 'src' / 'set_tsp_solver.py',
        project_root / 'src' / 'fstsp.py',
    )
    digest = hashlib.sha256()
    for source_path in source_paths:
        digest.update(str(source_path.relative_to(project_root)).encode('utf-8'))
        digest.update(source_path.read_bytes())
    return digest.hexdigest()


def _protocol_signature(payload):
    """
    计算不包含运行时间戳的实验协议签名。

    输入为协议字段字典；输出 SHA-256。签名用于阻止不同方法、规模、seed 或求解器
    时限误用同一个续跑目录。
    """

    text = json.dumps(
        to_jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _balanced_method_order(methods, instance_index, seed):
    """按循环拉丁顺序平衡方法位置；输入固定时输出完全确定。"""

    ordered = list(methods)
    if not ordered:
        return []
    offset = (int(instance_index) + int(seed)) % len(ordered)
    return ordered[offset:] + ordered[:offset]


def _make_manifest(
    spec,
    customer_size,
    repetitions,
    methods,
    seed,
    time_limit,
    instance_time_limit,
    instances,
    graph=None,
):
    """
    构造包含完整采样输入的不可变配对实验清单。

    输入为数据集配置、规模、方法、随机种子、可选时限和采样实例；输出 manifest。
    每个实例同时记录共享 instance_id 与 input_hash，所有方法必须原样复制这两个值。
    """

    graph_path = Path(spec.graph_path)
    graph_hash = _file_sha256(graph_path) if graph_path.is_file() else _graph_sha256(graph)
    protocol = {
        'result_protocol_version': 2,
        'dataset_key': spec.result_stem,
        'dataset_label': spec.dataset_label,
        'graph_path': str(Path(spec.graph_path).resolve()),
        'graph_sha256': graph_hash,
        'customer_size': int(customer_size),
        'depot_count': int(spec.depot_count),
        'drone_count': int(spec.drone_count),
        'repetitions': int(repetitions),
        'methods': list(methods),
        'seed': int(seed),
        'set_tsp_time_limit': time_limit,
        'instance_time_limit': instance_time_limit,
        'method_order_policy': 'balanced_cyclic_v1',
        'environment': _environment_versions(),
        'implementation_fingerprint': _implementation_fingerprint(),
    }
    manifest_instances = []
    for instance_index, (depots, cities) in enumerate(instances):
        input_hash = stable_instance_hash(
            spec.result_stem,
            depots,
            cities,
            graph_hash=graph_hash,
        )
        manifest_instances.append({
            'instance_index': instance_index,
            'instance_id': (
                f'{spec.result_stem}-{int(customer_size)}-{instance_index:04d}'
            ),
            'input_hash': input_hash,
            'depots': to_jsonable(depots),
            'cities': to_jsonable(cities),
            'execution_order': _balanced_method_order(
                methods,
                instance_index,
                seed,
            ),
        })
    return {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_signature': _protocol_signature(protocol),
        'protocol': protocol,
        'instances': manifest_instances,
    }


def _variant_results(prepared, methods):
    """
    在一份 GTDS 公共准备数据上求解本轮需要的所有具名变体。

    输入为 ``PreparedSetGTDS`` 和待运行方法；输出 ``{method: SetGTDSResult}``。
    公共有向代价、巨路径与片段统计仅构造一次，epsilon 消融因此严格共享随机路径。
    """

    results = {}
    for method in methods:
        if method not in GTDS_VARIANTS:
            continue
        parameters = GTDS_VARIANTS[method]
        if parameters['drone_cost_mode'] != prepared.drone_cost_mode:
            raise ValueError(
                f'{method} 需要 {parameters["drone_cost_mode"]} 代价模式，'
                f'但公共准备结果为 {prepared.drone_cost_mode}。'
            )
        results[method] = solve_prepared_set_gtds(
            prepared,
            epsilon=parameters['epsilon'],
            apply_model_budget=parameters['apply_budget'],
            active_depot_policy=parameters['active_depot_policy'],
        )
    return results


def _write_paired_summary(store, methods):
    """
    将逐 JSON 检查点汇总为便于 NumPy/统计脚本读取的二维 NPZ。

    输入为存储对象和固定方法顺序；输出 ``paired_summary.npz`` 路径。缺失或失败值
    使用 NaN，状态矩阵保留 completed/incomplete/timeout/error/missing，便于计算完成率和 PAR-2。
    """

    instances = store.manifest['instances']
    shape = (len(instances), len(methods))
    costs = np.full(shape, np.nan, dtype=float)
    solve_seconds = np.full(shape, np.nan, dtype=float)
    phase1_seconds = np.full(shape, np.nan, dtype=float)
    phase2_build_seconds = np.full(shape, np.nan, dtype=float)
    phase2_optimize_seconds = np.full(shape, np.nan, dtype=float)
    max_phase2_build_seconds = np.full(shape, np.nan, dtype=float)
    max_phase2_optimize_seconds = np.full(shape, np.nan, dtype=float)
    total_phase2_seconds = np.full(shape, np.nan, dtype=float)
    max_phase2_seconds = np.full(shape, np.nan, dtype=float)
    phase3_seconds = np.full(shape, np.nan, dtype=float)
    max_q_bin = np.full(shape, np.nan, dtype=float)
    sum_q_bin = np.full(shape, np.nan, dtype=float)
    mean_q_bin = np.full(shape, np.nan, dtype=float)
    median_q_bin = np.full(shape, np.nan, dtype=float)
    cv_q_bin = np.full(shape, np.nan, dtype=float)
    active_depots = np.full(shape, np.nan, dtype=float)
    max_group_customers = np.full(shape, np.nan, dtype=float)
    completion = np.zeros(shape, dtype=bool)
    time_limit_reached = np.zeros(shape, dtype=bool)
    statuses = np.full(shape, 'missing', dtype='<U32')

    for instance_index, instance in enumerate(instances):
        for method_index, method in enumerate(methods):
            if not store.has_record(instance['instance_id'], method):
                continue
            record = store.read_record(instance['instance_id'], method)
            if record.get('input_hash') != instance['input_hash']:
                raise ValueError(
                    f"检查点 {instance['instance_id']}/{method} 的 input_hash 不匹配。"
                )
            statuses[instance_index, method_index] = record.get(
                'record_status', 'unknown'
            )
            if record.get('cost') is not None:
                costs[instance_index, method_index] = float(record['cost'])
            if record.get('solve_seconds') is not None:
                solve_seconds[instance_index, method_index] = float(
                    record['solve_seconds']
                )
            process = record.get('process_data') or {}
            phase1_seconds[instance_index, method_index] = float(
                process.get('partition_seconds', np.nan)
            )
            depots = process.get('depot_records') or []
            if depots:
                build_values = np.asarray([
                    float(item.get('set_tsp_build_seconds', 0.0))
                    for item in depots
                ])
                optimize_values = np.asarray([
                    float(item.get('set_tsp_optimize_seconds', 0.0))
                    for item in depots
                ])
                group_sizes = np.asarray([
                    len(item.get('customers') or [])
                    for item in depots
                ], dtype=float)
                active_mask = group_sizes > 0
                q_values = np.asarray([
                    float(item.get('estimated_q_bin', 0.0))
                    for item in depots
                ])[active_mask]
                phase2_build_seconds[instance_index, method_index] = build_values.sum()
                phase2_optimize_seconds[instance_index, method_index] = optimize_values.sum()
                max_phase2_build_seconds[instance_index, method_index] = build_values.max()
                max_phase2_optimize_seconds[instance_index, method_index] = optimize_values.max()
                depot_phase2 = build_values + optimize_values
                total_phase2_seconds[instance_index, method_index] = depot_phase2.sum()
                max_phase2_seconds[instance_index, method_index] = depot_phase2.max()
                active_depots[instance_index, method_index] = active_mask.sum()
                max_group_customers[instance_index, method_index] = group_sizes.max()
                sum_q_bin[instance_index, method_index] = q_values.sum()
                mean_q_bin[instance_index, method_index] = (
                    q_values.mean() if q_values.size else 0.0
                )
                median_q_bin[instance_index, method_index] = (
                    np.median(q_values) if q_values.size else 0.0
                )
                q_mean = q_values.mean() if q_values.size else 0.0
                cv_q_bin[instance_index, method_index] = (
                    q_values.std(ddof=0) / q_mean if q_mean > 0 else 0.0
                )
                phase3_seconds[instance_index, method_index] = sum(
                    float(item.get('local_search_seconds', 0.0))
                    for item in depots
                )
                time_limit_reached[instance_index, method_index] = any(
                    bool(item.get('time_limit_reached', False))
                    for item in depots
                ) or bool(process.get('instance_time_limit_reached', False))
            diagnostics = process.get('partition_diagnostics') or {}
            if diagnostics.get('max_q_bin') is not None:
                max_q_bin[instance_index, method_index] = float(
                    diagnostics['max_q_bin']
                )
            elif depots:
                # SMST/SNN 没有 GTDS 诊断，使用各仓库实际候选集合对应的估算 Q_bin。
                max_q_bin[instance_index, method_index] = max(
                    float(item.get('estimated_q_bin', 0.0))
                    for item in depots
                )
            completion[instance_index, method_index] = (
                record.get('record_status') == 'completed'
            )

    summary_path = store.root / 'paired_summary.npz'
    _save_npz(
        summary_path,
        result_schema_version=np.asarray(2, dtype=np.int64),
        protocol_signature=np.asarray(
            store.manifest['protocol_signature'], dtype=np.str_
        ),
        instance_time_limit=np.asarray(
            store.manifest['protocol'].get('instance_time_limit', np.nan)
            if store.manifest['protocol'].get('instance_time_limit') is not None
            else np.nan,
            dtype=float,
        ),
        methods=np.asarray(methods, dtype=np.str_),
        instance_ids=np.asarray(
            [item['instance_id'] for item in instances], dtype=np.str_
        ),
        input_hashes=np.asarray(
            [item['input_hash'] for item in instances], dtype=np.str_
        ),
        execution_orders_json=np.asarray([
            json.dumps(item.get('execution_order', methods), ensure_ascii=False)
            for item in instances
        ], dtype=np.str_),
        record_status=statuses,
        completed=completion,
        time_limit_reached=time_limit_reached,
        cost=costs,
        solve_time=solve_seconds,
        phase1_time=phase1_seconds,
        phase2_build_time=phase2_build_seconds,
        phase2_optimize_time=phase2_optimize_seconds,
        total_phase2_build_time=phase2_build_seconds,
        max_phase2_build_time=max_phase2_build_seconds,
        total_phase2_optimize_time=phase2_optimize_seconds,
        max_phase2_optimize_time=max_phase2_optimize_seconds,
        total_phase2_time=total_phase2_seconds,
        max_phase2_time=max_phase2_seconds,
        phase3_time=phase3_seconds,
        max_q_bin=max_q_bin,
        sum_q_bin=sum_q_bin,
        mean_q_bin=mean_q_bin,
        median_q_bin=median_q_bin,
        cv_q_bin=cv_q_bin,
        active_depots=active_depots,
        max_group_customers=max_group_customers,
    )
    return summary_path


def run_paired_road_experiment(
    spec,
    prepared_network,
    customer_size,
    repetitions,
    methods,
    output_root,
    seed=0,
    set_tsp_time_limit=None,
    instance_time_limit=None,
):
    """
    在同一批采样实例上依次运行全部第一阶段方法并持续写检查点。

    输入为路网配置、已准备的图/距离、客户规模、重复次数、方法顺序、输出目录、
    seed、可选单仓库 Set-TSP 时限和实例总时限；输出 ``paired_summary.npz``。
    任一方法失败会写 error 记录并继续其余配对单元。
    """

    methods = tuple(methods)
    unknown = [
        method
        for method in methods
        if method not in {'smst_original', 'snn'} | set(GTDS_VARIANTS)
    ]
    if unknown:
        raise ValueError(f'未知配对实验方法：{unknown}')
    graph, distance, _ = prepared_network
    sampled_depots, sampled_cities = sample_multiagent_instances(
        graph,
        repetitions,
        spec.depot_count,
        customer_size,
        seed=seed,
    )
    sampled_instances = list(zip(sampled_depots, sampled_cities))
    proposed_manifest = _make_manifest(
        spec,
        customer_size,
        repetitions,
        methods,
        seed,
        set_tsp_time_limit,
        instance_time_limit,
        sampled_instances,
        graph=graph,
    )
    store = PairedExperimentStore(output_root, proposed_manifest)

    # 续跑时以首次 manifest 中冻结的输入为准，不再依赖当前采样实现细节。
    manifest_instances = store.manifest['instances']
    for instance in manifest_instances:
        execution_order = instance.get(
            'execution_order',
            _balanced_method_order(methods, instance['instance_index'], seed),
        )
        pending_methods = [
            method
            for method in execution_order
            if not store.has_record(instance['instance_id'], method)
        ]
        if not pending_methods:
            continue
        depots = instance['depots']
        cities = instance['cities']

        candidate_start = time.perf_counter()
        candidate_model = MultiAgentFlyingSidekickTSP(
            graph,
            depots,
            cities,
            distance,
            spec.drone_count,
            theta=(0.5, 0.5),
        )
        candidate_sets = candidate_model.get_boundary_convex_sets(
            candidate_model.theta[0]
        )
        candidate_seconds = time.perf_counter() - candidate_start

        # 按清单顺序懒加载不同代价模式的公共准备结果，兼顾严格顺序与变体复用。
        prepared_by_cost_mode = {}
        preparation_errors = {}

        for method in pending_methods:
            precomputed_partition_result = None
            if method in GTDS_VARIANTS:
                parameters = GTDS_VARIANTS[method]
                drone_cost_mode = parameters['drone_cost_mode']
                try:
                    if drone_cost_mode in preparation_errors:
                        raise RuntimeError(preparation_errors[drone_cost_mode]['message'])
                    if drone_cost_mode not in prepared_by_cost_mode:
                        prepared_by_cost_mode[drone_cost_mode] = prepare_set_gtds(
                            depots=depots,
                            cities=cities,
                            candidate_sets=candidate_sets,
                            truck_distance=distance['truck'],
                            drone_distance=distance['drone'],
                            speed=candidate_model.speed,
                            drone_cost_mode=drone_cost_mode,
                        )
                    precomputed_partition_result = _variant_results(
                        prepared_by_cost_mode[drone_cost_mode],
                        [method],
                    )[method]
                except Exception as error:
                    failure = {
                        'type': type(error).__name__,
                        'message': str(error),
                        'traceback': traceback.format_exc(),
                    }
                    if drone_cost_mode not in prepared_by_cost_mode:
                        preparation_errors[drone_cost_mode] = failure
                    store.write_record(instance['instance_id'], method, {
                        'input_hash': instance['input_hash'],
                        'record_status': 'error',
                        'cost': None,
                        'solve_seconds': None,
                        'solution': None,
                        'process_data': None,
                        'error': failure,
                    })
                    continue

            try:
                model = MultiAgentFlyingSidekickTSP(
                    graph,
                    depots,
                    cities,
                    distance,
                    spec.drone_count,
                    theta=(0.5, 0.5),
                    partition_method=method,
                    set_tsp_time_limit=set_tsp_time_limit,
                    precomputed_partition_result=precomputed_partition_result,
                )
                solution, cost, process_data = _solve_model_with_process_data(
                    model,
                    candidate_sets=candidate_sets,
                    candidate_set_seconds=candidate_seconds,
                    instance_time_limit=instance_time_limit,
                )
                completed = process_data.get('instance_status') == 'complete'
                record = {
                    'input_hash': instance['input_hash'],
                    'record_status': (
                        'completed'
                        if completed
                        else process_data.get('instance_status', 'incomplete')
                    ),
                    'cost': float(cost) if np.isfinite(cost) else None,
                    'solve_seconds': process_data.get('solve_seconds'),
                    'solution': solution,
                    'process_data': process_data,
                    'error': None,
                }
            except Exception as error:
                record = {
                    'input_hash': instance['input_hash'],
                    'record_status': 'error',
                    'cost': None,
                    'solve_seconds': None,
                    'solution': None,
                    'process_data': None,
                    'error': {
                        'type': type(error).__name__,
                        'message': str(error),
                        'traceback': traceback.format_exc(),
                    },
                }
            store.write_record(instance['instance_id'], method, record)

    return _write_paired_summary(store, methods)
