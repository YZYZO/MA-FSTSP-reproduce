"""服务器阶段 A/B 标签采集与无求解结果缓存的配对复测。"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from config import MANHATTAN1k_GRAPH_PATH, PROJECT_ROOT
from problem import prepare_manhattan_road_network, sample_multiagent_instances
from src.fstsp import MultiAgentFlyingSidekickTSP
from .candidates import Candidate, generate_candidates, symmetric_mst
from .evaluator import evaluate_group, evaluate_partition, fixed_boundary, solve_with_records
from .features import FeatureContext
from .selector import METHODS, select_candidate
from .settings import RepairOptions, SolverOptions
from .storage import (RecordTable, context_fingerprint, file_fingerprint, fingerprint,
                      group_cache_key, read_json, runtime_metadata, save_json)


def build_parser(mode):
    """输入运行模式，返回统一参数解析器；采集配置与服务器硬件不绑定。"""
    parser = argparse.ArgumentParser(description='MA-FSTSP 分区修复采集与独立复测')
    parser.add_argument('--output', type=Path, required=True, help='本次运行独立结果目录')
    parser.add_argument('--graph', type=Path, default=MANHATTAN1k_GRAPH_PATH)
    parser.add_argument('--limit-instances', type=int, help='只执行清单前若干实例；去掉此参数可继续完整清单')
    if mode == 'evaluate':
        parser.add_argument('--manifest', type=Path, required=True, help='已采集目录中的 manifest.json；继承其求解与修复设置')
        parser.add_argument('--methods', choices=METHODS, nargs='+', default=list(METHODS))
        parser.add_argument('--repeats', type=int, default=1)
        return parser
    parser.add_argument('--stage', choices=('A', 'B'), default='B')
    parser.add_argument('--sizes', type=int, nargs='+', default=[50, 100, 150])
    parser.add_argument('--instances-per-size', type=int)
    parser.add_argument('--sample-seed', type=int, default=905020)
    parser.add_argument('--depots', type=int, default=5)
    parser.add_argument('--drones', type=int, default=3)
    parser.add_argument('--drone-limit', type=float, default=1.5)
    parser.add_argument('--drone-speed', type=float, default=1.6)
    parser.add_argument('--time-limit', type=float, default=30.0, help='每组 Set-TSP 优化预算，非整个阶段预算')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--solver-seed', type=int, default=0)
    parser.add_argument('--geometry-weight', type=float, default=1.0)
    return parser


def make_model(graph, distance, instance, configuration):
    """按清单构造一个全新模型，固定客户顺序和车辆参数，输出可独立求解的对象。"""
    return MultiAgentFlyingSidekickTSP(
        graph, np.asarray(instance['depots']), np.asarray(instance['cities']), distance,
        configuration['drones'], limit=configuration['drone_limit'],
        speed=configuration['drone_speed'], theta=(0.5, 0.5),
    )


def make_manifest(args, graph, runtime, graph_id):
    """输入采样配置和路网，生成按规模交替排列的新实例清单；阶段 A/B 使用独立种子区间。"""
    per_size = args.instances_per_size or (4 if args.stage == 'A' else 10)
    configuration = {
        'stage': args.stage, 'sizes': args.sizes, 'instances_per_size': per_size,
        'sample_seed': args.sample_seed, 'depots': args.depots, 'drones': args.drones,
        'drone_limit': args.drone_limit, 'drone_speed': args.drone_speed,
        'solver': SolverOptions(args.time_limit, args.threads, args.solver_seed).to_dict(),
        'repair': RepairOptions(geometry_weight=args.geometry_weight).to_dict(),
        'graph_fingerprint': graph_id, 'runtime': runtime,
    }
    instances = []
    for index in range(per_size):
        for size in args.sizes:
            seed = args.sample_seed + (10000000 if args.stage == 'A' else 20000000) + size * 10000 + index
            depots, cities = sample_multiagent_instances(graph, 1, args.depots, size, seed=seed)
            payload = {'depots': list(map(int, depots[0])), 'cities': sorted(map(int, cities[0]))}
            instances.append(dict(payload, id=f'{args.stage.lower()}-{size}-{index:03d}',
                                  size=size, seed=seed, family_id=fingerprint(payload), split='development'))
    return {'configuration': configuration, 'instances': instances,
            'created_utc': datetime.now(timezone.utc).isoformat()}


def ensure_configuration(path, payload):
    """保存或核对同一结果目录的固定设置，防止续跑把不同源码、机器或预算混成同一实验。"""
    path = Path(path)
    if path.exists():
        old = read_json(path)
        if fingerprint(old['configuration']) != fingerprint(payload['configuration']):
            raise ValueError(f'{path} 的配置与本次不同，请使用新的 --output 目录。')
        return old
    save_json(path, payload)
    return payload


def export_tables(tables):
    """将每张持久化表导出，正常结束和键盘中断时均可保留已完成的样本。"""
    for table in tables:
        table.export()


def collect(args):
    """按固定清单评价全部候选，复用相同组的离线观测，输出三张表和采集预算记录。"""
    start = time.perf_counter()
    graph, distance, distance_stats = prepare_manhattan_road_network(args.graph)
    preparation_seconds = time.perf_counter() - start
    runtime = runtime_metadata(PROJECT_ROOT)
    graph_id = file_fingerprint(args.graph)
    manifest = ensure_configuration(args.output / 'manifest.json', make_manifest(args, graph, runtime, graph_id))
    config = manifest['configuration']
    options, repair_options = SolverOptions(**config['solver']), RepairOptions(**config['repair'])
    instances = manifest['instances'][:args.limit_instances]
    instance_table = RecordTable(args.output, 'instances')
    group_table = RecordTable(args.output, 'group_runs')
    candidate_table = RecordTable(args.output, 'partition_candidates')
    tables = (instance_table, group_table, candidate_table)
    session_start = time.perf_counter()
    evaluated_groups = 0
    try:
        for number, instance in enumerate(instances, 1):
            existing = instance_table.records.get(instance['id'], {})
            if existing.get('complete'):
                print(f'[{number}/{len(instances)}] {instance["id"]} 已完成，跳过。', flush=True)
                continue
            print(f'[{number}/{len(instances)}] {instance["id"]} 准备实例。', flush=True)
            instance_start = time.perf_counter()
            model = make_model(graph, distance, instance, config)
            boundary_start = time.perf_counter()
            boundary = fixed_boundary(model)
            boundary_seconds = time.perf_counter() - boundary_start
            partition_start = time.perf_counter()
            baseline = symmetric_mst(model, boundary)
            partition_seconds = time.perf_counter() - partition_start
            feature_start = time.perf_counter()
            context = FeatureContext(model, boundary)
            feature_seconds = time.perf_counter() - feature_start
            repair_start = time.perf_counter()
            candidates = ([Candidate('stay', 'stay', 0.0, baseline)] if args.stage == 'A' else
                          generate_candidates(context, baseline, repair_options))
            repair_seconds = time.perf_counter() - repair_start
            feature_seconds += context.compute_seconds
            repair_seconds = max(0.0, repair_seconds - context.compute_seconds)
            feature_start = time.perf_counter()
            features = {c.name: context.difference(baseline, c.partition) for c in candidates}
            feature_seconds += time.perf_counter() - feature_start
            context_id = context_fingerprint(model, boundary, graph_id, runtime)
            instance_record = dict(instance, context_id=context_id, complete=False,
                                   initialization_seconds=model.initialization_seconds,
                                   boundary_seconds=boundary_seconds, partition_seconds=partition_seconds,
                                   feature_seconds=feature_seconds, repair_seconds=repair_seconds,
                                   candidate_count=len(candidates), baseline_partition=baseline,
                                   boundary_sizes={c: len(v) for c, v in boundary.items()})
            instance_table.put(instance_record)

            def provide_group(depot, customers):
                """按完整上下文查找离线组标签；缺失时真实求解并立即保存，输出带缓存来源的记录。"""
                nonlocal evaluated_groups
                key = group_cache_key(context_id, depot, customers, options)
                cached = group_table.records.get(key)
                if cached is not None and cached.get('complete'):
                    return dict(cached, cache_hit=True)
                group_start = time.perf_counter()
                print(f'  车组 {depot}: {len(customers)} 个客户，开始 Set-TSP → DP。', flush=True)
                try:
                    record = evaluate_group(model, depot, customers, boundary, options)
                except (KeyboardInterrupt, Exception) as error:
                    group_table.put({'id': key, 'instance_id': instance['id'], 'context_id': context_id,
                                     'depot_node': depot, 'customers': customers, 'complete': False,
                                     'error': type(error).__name__ + ': ' + str(error),
                                     'observed_seconds': time.perf_counter() - group_start})
                    raise
                record.update(id=key, instance_id=instance['id'], context_id=context_id,
                              solver_options=options.to_dict(), repeat=0, cache_hit=False,
                              acquisition_seconds=time.perf_counter() - group_start)
                group_table.put(record)
                evaluated_groups += 1
                return record

            # 候选顺序按实例随机化；基线标签最后统一关联，避免强制基线永远先运行。
            order = np.random.RandomState(instance['seed']).permutation(len(candidates))
            rows = []
            for index in order:
                candidate = candidates[int(index)]
                candidate_id = instance['id'] + '--' + candidate.name
                old = candidate_table.records.get(candidate_id)
                if old is not None and old.get('complete'):
                    rows.append(old)
                    continue
                print(f'  候选 {candidate.name}', flush=True)
                candidate_start = time.perf_counter()
                evaluation = evaluate_partition(model, candidate.partition, boundary, options, provide_group)
                group_records = evaluation.pop('depot_records')
                row = dict(evaluation, id=candidate_id, instance_id=instance['id'], size=instance['size'],
                           name=candidate.name, kind=candidate.kind, strength=candidate.strength,
                           partition=candidate.partition, baseline_partition=baseline,
                           features=features[candidate.name],
                           changed_depots=[d for d in context.depots if baseline[d] != candidate.partition[d]],
                           group_run_ids=[r['id'] for r in group_records],
                           reused_groups=sum(r.get('cache_hit', False) for r in group_records),
                           acquisition_seconds=time.perf_counter() - candidate_start,
                           timing_kind='offline_group_observations',
                           feature_seconds=feature_seconds, repair_seconds=repair_seconds)
                rows.append(candidate_table.put(row))
            base = next(row for row in rows if row['name'] == 'stay')
            for row in rows:
                if row['complete'] and base['complete']:
                    row['delta_cost'] = row['final_delivery_cost'] - base['final_delivery_cost']
                    row['saved_phase2_seconds'] = base['phase2_wall_seconds'] - row['phase2_wall_seconds']
                candidate_table.put(row)
            instance_record.update(complete=all(r['complete'] for r in rows),
                                   session_instance_seconds=time.perf_counter() - instance_start,
                                   group_evaluations=len({key for r in rows for key in r['group_run_ids']}))
            instance_table.put(instance_record)
            export_tables(tables)
    finally:
        export_tables(tables)
        elapsed = time.perf_counter() - session_start
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        save_json(args.output / 'sessions' / (stamp + '.json'), {
            'preparation_seconds': preparation_seconds, 'distance_stats': distance_stats,
            'collection_seconds': elapsed, 'new_group_evaluations': evaluated_groups,
            'seconds_per_new_group_with_collection_overhead': elapsed / evaluated_groups if evaluated_groups else None,
            'runtime': runtime,
        })
    print(f'采集完成：{args.output.resolve()}。下一步运行 analyze_partition_candidates.py。', flush=True)


def evaluate(args):
    """对清单中每个方法重新构造模型、选择并求解分区，输出配对的真实端到端记录。"""
    manifest = read_json(args.manifest)
    config = manifest['configuration']
    options, repair_options = SolverOptions(**config['solver']), RepairOptions(**config['repair'])
    # 求解设置继承清单，防止复测命令的默认值悄悄改变已固定实验。
    if file_fingerprint(args.graph) != config['graph_fingerprint']:
        raise ValueError('复测路网与实例清单不一致。')
    runtime = runtime_metadata(PROJECT_ROOT)
    for key in ('source_fingerprint', 'packages', 'distance_semantics', 'phase2_semantics', 'fallback'):
        if runtime[key] != config['runtime'][key]:
            raise ValueError(f'复测的 {key} 与采集清单不一致，请使用采集时的源码和依赖。')
    if 'symmetric_mst' not in args.methods:
        args.methods = ['symmetric_mst'] + args.methods
    evaluation_config = {'manifest': fingerprint(manifest), 'runtime': runtime,
                         'solver': options.to_dict(), 'repair': repair_options.to_dict(),
                         'methods': args.methods, 'repeats': args.repeats,
                         'expected_instances': [row['id'] for row in manifest['instances']]}
    ensure_configuration(args.output / 'evaluation_config.json', {'configuration': evaluation_config})
    preparation_start = time.perf_counter()
    graph, distance, distance_stats = prepare_manhattan_road_network(args.graph)
    preparation_seconds = time.perf_counter() - preparation_start
    table = RecordTable(args.output, 'evaluation_runs')
    instances = manifest['instances'][:args.limit_instances]
    try:
        for instance_index, instance in enumerate(instances):
            for repeat in range(args.repeats):
                # 方法顺序轮转并交替反向，所有方法使用同一清单、预算和求解种子。
                offset = (instance_index + repeat) % len(args.methods)
                methods = args.methods[offset:] + args.methods[:offset]
                if (instance_index + repeat) % 2:
                    methods = methods[::-1]
                for method in methods:
                    run_id = f'{instance["id"]}--{method}--{repeat}'
                    if table.records.get(run_id, {}).get('complete'):
                        continue
                    print(f'{instance["id"]} {method} repeat={repeat}', flush=True)
                    run_start = time.perf_counter()
                    try:
                        model = make_model(graph, distance, instance, config)
                        _, cost, process = solve_with_records(
                            model, partition_strategy=method, solver_options=options, repair_options=repair_options,
                        )
                        online_seconds = time.perf_counter() - run_start
                    except (KeyboardInterrupt, Exception) as error:
                        table.put({'id': run_id, 'instance_id': instance['id'], 'size': instance['size'],
                                   'method': method, 'repeat': repeat, 'complete': False,
                                   'observed_seconds': time.perf_counter() - run_start,
                                   'error': type(error).__name__ + ': ' + str(error)})
                        raise
                    records = process['depot_records']
                    table.put({
                        'id': run_id, 'instance_id': instance['id'], 'size': instance['size'],
                        'method': method, 'repeat': repeat, 'complete': True,
                        'final_delivery_cost': cost, 'phase2_wall_seconds': sum(r['phase2_wall_seconds'] for r in records),
                        'phase3_seconds': sum(r['phase3_seconds'] for r in records),
                        'online_seconds': online_seconds, 'process': process,
                        'timeout_groups': sum(r['timeout'] for r in records),
                        'fallback_groups': sum(r['fallback_used'] for r in records),
                        'timing_kind': 'fresh_wall_clock', 'solver_result_cache_enabled': False,
                        'distance_policy': 'shared_precomputed_map_distances',
                    })
                    table.export()
    finally:
        table.export()
        save_json(args.output / 'map_preparation.json', {
            'preparation_seconds': preparation_seconds, 'distance_stats': distance_stats,
            'policy': 'one_time_shared_map_preparation',
        })
    print(f'独立复测完成：{args.output.resolve()}。', flush=True)


def main(mode):
    """解析命令行并执行采集或复测；键盘中断保留完整记录，并以非零状态退出。"""
    args = build_parser(mode).parse_args()
    try:
        return collect(args) if mode == 'collect' else evaluate(args)
    except KeyboardInterrupt:
        print('运行已中断，已完成的记录已保存；使用相同命令继续。', flush=True)
        raise SystemExit(130)
