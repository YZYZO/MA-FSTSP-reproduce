"""求解前特征、真实差值标签与按完整实例隔离的学习数据。"""

from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from .reporting import load_table
from .storage import fingerprint, read_json


# 只列出求解前结构；真实成本、时间、gap 和实例身份不进入特征向量。
SUMMARY_FIELDS = ('customers_max', 'binary_sum', 'binary_max', 'boundary_total_max',
                  'geometry_sum', 'mst_length_sum', 'mst_max_edge_max',
                  'affinity_total_sum', 'direction_difference_sum', 'customers_variance')
CONTEXT_FIELDS = ('customer_count', 'depot_count', 'drone_count', 'drone_speed', 'drone_limit')
KINDS = ('stay', 'count', 'burden', 'cluster', 'relocate', 'swap')
FEATURE_NAMES = tuple(x for name in SUMMARY_FIELDS for x in ('log_before_' + name, 'relative_delta_' + name))
FEATURE_NAMES += CONTEXT_FIELDS + ('moved_fraction', 'changed_group_fraction', 'strength')
FEATURE_NAMES += tuple('kind_' + name for name in KINDS)
CANDIDATE_ORDER = ('stay', 'count_0.1', 'count_0.25', 'count_0.5', 'count_0.75', 'count_1',
                   'burden_0.1', 'burden_0.25', 'cluster_2', 'cluster_4', 'cluster_8',
                   'local_relocate', 'local_swap')


def feature_vector(record):
    """输入在线或离线候选的结构记录，按固定白名单输出数值向量，不读取任何标签。"""
    features = record['features']
    result = []
    for name in SUMMARY_FIELDS:
        before = features['before_' + name]
        result.extend((math.log1p(max(0.0, before)), features['delta_' + name] / max(abs(before), 1.0)))
    result.extend(features['before_' + name] for name in CONTEXT_FIELDS)
    result.extend((features['moved_customers'] / max(features['before_customer_count'], 1),
                   features['changed_groups'] / max(features['before_depot_count'], 1), record['strength']))
    result.extend(float(record['kind'] == kind) for kind in KINDS)
    return result


def instance_family(instance, configuration):
    """输入实例与地图配置，返回不依赖仓库排列和批次命名的物理实例身份。"""
    return fingerprint({'graph': configuration['graph_fingerprint'],
                        'depots': sorted(instance['depots']), 'cities': sorted(instance['cities'])})


def label_contract(configuration):
    """输入标签来源配置，输出可合并的求解语义与计时环境；源码来源另行逐份保留。"""
    runtime = configuration['runtime']
    return {**{k: configuration[k] for k in ('graph_fingerprint', 'solver', 'repair', 'drones',
                                            'drone_limit', 'drone_speed')},
            'runtime': {k: runtime.get(k) for k in ('packages', 'python', 'platform', 'machine',
                                                  'distance_semantics', 'phase2_semantics', 'fallback')}}


def load_dataset(directories):
    """读取一个或多个完整候选目录，输出带来源的实例列表；禁止重复身份和不同计时环境混入。"""
    instances, sources, seen = [], [], set()
    contract = None
    for directory in map(Path, directories):
        manifest = read_json(directory / 'manifest.json')
        config = manifest['configuration']
        if any(i.get('split') == 'test' for i in manifest['instances']):
            raise ValueError('最终测试实例不能作为训练或调参标签。')
        current = label_contract(config)
        if contract is not None and current != contract:
            raise ValueError('标签的地图、求解配置或计时环境不一致，请分别训练。')
        contract = current
        source_id = fingerprint(manifest)
        observed = {r['id']: r for r in load_table(directory, 'instances')}
        grouped = defaultdict(list)
        candidate_rows = load_table(directory, 'partition_candidates')
        sources.append({'id': source_id, 'directory': str(directory), 'configuration': config,
                        'label_fingerprint': fingerprint(sorted(candidate_rows, key=lambda r: r['id']))})
        for row in candidate_rows:
            grouped[row['instance_id']].append(row)
        for instance in manifest['instances']:
            rows = grouped[instance['id']]
            state = observed.get(instance['id'], {})
            if not state.get('complete') or len(rows) != state.get('candidate_count') or not all(r.get('complete') for r in rows):
                raise ValueError(f'{instance["id"]} 的候选标签未完整采集。')
            family = instance_family(instance, config)
            if family in seen:
                raise ValueError('发现重复物理实例，请勿重复计入样本或跨批次重复采集。')
            seen.add(family)
            rows.sort(key=lambda r: CANDIDATE_ORDER.index(r['name']))
            if rows[0]['name'] != 'stay':
                raise ValueError('候选集合缺少初始划分 stay。')
            base = rows[0]
            records = [dict(r, delta_cost=r['final_delivery_cost'] - base['final_delivery_cost'],
                            saved_phase2_seconds=base['phase2_wall_seconds'] - r['phase2_wall_seconds']) for r in rows]
            instances.append({'id': source_id + ':' + instance['id'], 'family_id': family,
                              'source_instance_id': instance['id'], 'source_id': source_id,
                              'size': instance['size'], 'candidates': records})
    return {'instances': instances, 'sources': sources, 'contract': contract,
            'feature_names': list(FEATURE_NAMES)}


def assert_disjoint(first, second):
    """输入两份实例列表，检查物理身份不重叠；用于训练、校准和留出集边界。"""
    if {i['family_id'] for i in first} & {i['family_id'] for i in second}:
        raise ValueError('训练与验证包含相同的完整实例。')


def grouped_folds(instances, folds=5, seed=0):
    """输入完整实例，按规模平衡地划分各折；返回训练/留出列表，同实例候选永不拆开。"""
    if folds < 2:
        raise ValueError('分组验证至少需要两折。')
    strata = defaultdict(list)
    for instance in sorted(instances, key=lambda i: i['family_id']):
        strata[instance['size']].append(instance)
    if not strata or min(map(len, strata.values())) < folds:
        raise ValueError('每种规模的完整实例数必须不少于折数。')
    rng = np.random.RandomState(seed)
    buckets = [[] for _ in range(folds)]
    for rows in strata.values():
        for index, position in enumerate(rng.permutation(len(rows))):
            buckets[index % folds].append(rows[int(position)])
    return [(sum((b for j, b in enumerate(buckets) if j != k), []), bucket)
            for k, bucket in enumerate(buckets)]
