"""阶段 C 数据来源、当前策略实验清单和分批独立采样。"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from config import PROJECT_ROOT
from problem import manhattan, sample_multiagent_instances

from .learning_dataset import instance_family, load_dataset
from .reporting import load_table
from .settings import EvaluationOptions
from .storage import file_fingerprint, fingerprint, read_json, runtime_metadata, save_json


def save_new_manifest(path, manifest):
    """将新实验清单写入未占用路径，避免改变既有实验的来源和身份。"""
    path = Path(path)
    if path.exists():
        raise ValueError('清单已经存在，请直接使用它或选择新输出路径。')
    save_json(path, manifest)
    return manifest


def derive_manifest(source_path, output, graph_path):
    """从既有实例派生当前源码的实验清单，保留原始来源与候选指纹，不伪造旧运行身份。"""
    source_path = Path(source_path)
    source = read_json(source_path)
    if file_fingerprint(graph_path) != source['configuration']['graph_fingerprint']:
        raise ValueError('指定地图与来源清单不一致。')
    manifest = deepcopy(source)
    manifest['configuration']['runtime'] = runtime_metadata(PROJECT_ROOT)
    manifest['configuration']['source_manifest_fingerprint'] = fingerprint(source)
    manifest['source_manifest'] = {'path': str(source_path), 'fingerprint': fingerprint(source),
                                   'configuration': source['configuration']}
    manifest['created_utc'] = datetime.now(timezone.utc).isoformat()
    candidates = load_table(source_path.parent, 'partition_candidates')
    if candidates:
        # 来源存在标签时只记录完整分区指纹，不将真实成本和时间嵌入在线清单。
        load_dataset([source_path.parent])
        manifest['expected_candidates'] = {
            i['id']: fingerprint({r['name']: r['partition'] for r in candidates if r['instance_id'] == i['id']})
            for i in source['instances']}
    return save_new_manifest(output, manifest)


def sample_manifest(template_path, output, graph_path, split, per_size=10, seed=906040, exclude_paths=()):
    """根据固定问题配置采样独立训练/验证/测试实例，排除来源和指定清单中的物理重复。"""
    template = read_json(template_path)
    config = deepcopy(template['configuration'])
    if file_fingerprint(graph_path) != config['graph_fingerprint']:
        raise ValueError('采样地图与模板清单不一致。')
    excluded, provenance = set(), []
    for path in [Path(template_path)] + list(map(Path, exclude_paths)):
        other = read_json(path)
        excluded.update(instance_family(i, other['configuration']) for i in other['instances'])
        provenance.append({'path': str(path), 'fingerprint': fingerprint(other)})
    graph = manhattan(graph_path)
    config.update(stage='C' if split != 'test' else 'D', split=split, sample_seed=seed,
                  instances_per_size=per_size, runtime=runtime_metadata(PROJECT_ROOT),
                  evaluation=EvaluationOptions().to_dict(),
                  excluded_manifest_fingerprints=[p['fingerprint'] for p in provenance])
    config.pop('source_manifest_fingerprint', None)
    instances = []
    offset = {'train': 30000000, 'validation': 40000000, 'test': 50000000}[split]
    for index in range(per_size):
        for size in config['sizes']:
            trial = seed + offset + size * 100000 + index
            while True:
                depots, cities = sample_multiagent_instances(graph, 1, config['depots'], size, seed=trial)
                payload = {'depots': list(map(int, depots[0])), 'cities': sorted(map(int, cities[0]))}
                family = instance_family(payload, config)
                if family not in excluded:
                    break
                trial += 100000000
            excluded.add(family)
            instances.append(dict(payload, id=f'{split}-{seed}-{size}-{index:03d}', size=size, seed=trial,
                                  family_id=family, split=split))
    manifest = {'configuration': config, 'instances': instances, 'exclusions': provenance,
                'created_utc': datetime.now(timezone.utc).isoformat()}
    return save_new_manifest(output, manifest)


def dataset_summary(directories, output):
    """检查已有标签并输出轻量索引，包含实例身份、来源与特征定义，不复制求解器结果缓存。"""
    dataset = load_dataset(directories)
    summary = {k: v for k, v in dataset.items() if k != 'instances'}
    summary['instances'] = [{k: v for k, v in i.items() if k != 'candidates'} |
                            {'candidate_count': len(i['candidates'])} for i in dataset['instances']]
    save_json(output, summary)
    return summary
