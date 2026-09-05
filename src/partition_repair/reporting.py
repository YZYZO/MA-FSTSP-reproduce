"""开发候选潜力和独立复测报告：总量之比、逐实例退化及分层配对区间。"""

from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .storage import read_json, save_json


def aggregate_pairs(pairs, bootstrap=1000, seed=0):
    """输入完整实例的配对结果，按总量之比汇总；分层重采样以实例为单位，返回统计字典。"""
    if not pairs:
        return {'instance_count': 0, 'passes_point_thresholds': False}
    base_cost = np.array([p['baseline_cost'] for p in pairs], dtype=float)
    cost = np.array([p['cost'] for p in pairs], dtype=float)
    base_time = np.array([p['baseline_time'] for p in pairs], dtype=float)
    elapsed = np.array([p['time'] for p in pairs], dtype=float)
    cost_changes = cost / np.maximum(base_cost, 1e-12) - 1
    time_changes = elapsed / np.maximum(base_time, 1e-12) - 1
    cost_ratio = float(cost.sum() / max(base_cost.sum(), 1e-12) - 1)
    saving = float(1 - elapsed.sum() / max(base_time.sum(), 1e-12))
    result = {
        'instance_count': len(pairs), 'cost_change': cost_ratio, 'phase2_saving': saving,
        'total_cost': float(cost.sum()), 'baseline_total_cost': float(base_cost.sum()),
        'total_phase2_seconds': float(elapsed.sum()), 'baseline_total_phase2_seconds': float(base_time.sum()),
        'mean_instance_cost_change': float(np.mean(cost_changes)),
        'median_instance_cost_change': float(np.median(cost_changes)),
        'p90_instance_cost_change': float(np.quantile(cost_changes, .9)),
        'cost_over_5_percent_count': int(np.sum(cost_changes > .05 + 1e-12)),
        'worst_cost_change': float(max(cost_changes)),
        'mean_instance_time_change': float(np.mean(time_changes)),
        'median_instance_time_change': float(np.median(time_changes)),
        'slower_instance_count': int(np.sum(time_changes > 0)),
        'worst_time_change': float(max(time_changes)),
        'passes_point_thresholds': cost_ratio <= .05 + 1e-12 and saving >= .2 - 1e-12,
    }
    if all('online_seconds' in p for p in pairs):
        total_online = sum(p['online_seconds'] for p in pairs)
        base_online = sum(p['baseline_online_seconds'] for p in pairs)
        result.update(online_saving=1 - total_online / max(base_online, 1e-12),
                      total_online_seconds=total_online, baseline_total_online_seconds=base_online,
                      total_phase3_seconds=sum(p['phase3_seconds'] for p in pairs),
                      baseline_total_phase3_seconds=sum(p['baseline_phase3_seconds'] for p in pairs))
    if bootstrap:
        strata = defaultdict(list)
        for index, pair in enumerate(pairs):
            strata[pair['size']].append(index)
        rng = np.random.RandomState(seed)
        samples = []
        for _ in range(bootstrap):
            indices = np.concatenate([rng.choice(values, len(values), replace=True) for values in strata.values()])
            samples.append((cost[indices].sum() / max(base_cost[indices].sum(), 1e-12) - 1,
                            1 - elapsed[indices].sum() / max(base_time[indices].sum(), 1e-12)))
        intervals = np.quantile(samples, [.025, .975], axis=0)
        result['cost_change_ci95'] = list(map(float, intervals[:, 0]))
        result['phase2_saving_ci95'] = list(map(float, intervals[:, 1]))
        result['thresholds_supported_by_ci95'] = bool(intervals[1, 0] <= .05 and intervals[0, 1] >= .2)
    return result


def pair_row(baseline, candidate):
    """输入同一完整实例的基线和候选结果，返回统一比较行。"""
    row = {
        'instance_id': baseline['instance_id'], 'size': baseline['size'],
        'baseline_cost': baseline['final_delivery_cost'], 'cost': candidate['final_delivery_cost'],
        'baseline_time': baseline['phase2_wall_seconds'], 'time': candidate['phase2_wall_seconds'],
        'selection': candidate.get('name', candidate.get('method')),
        'timeout_groups': candidate.get('timeout_groups', 0),
        'fallback_groups': candidate.get('fallback_groups', 0),
    }
    if 'online_seconds' in candidate:
        row.update(online_seconds=candidate['online_seconds'], baseline_online_seconds=baseline['online_seconds'],
                   phase3_seconds=candidate['phase3_seconds'], baseline_phase3_seconds=baseline['phase3_seconds'])
    row['cost_change'] = row['cost'] / max(row['baseline_cost'], 1e-12) - 1
    row['time_change'] = row['time'] / max(row['baseline_time'], 1e-12) - 1
    return row


def oracle_select(grouped, epsilon=.05, per_instance=False):
    """输入每实例已知真实候选，求有限候选集内的事后最好选择；返回配对行，绝非在线策略。"""
    instances = sorted(grouped)
    bases = {key: next(row for row in grouped[key] if row['name'] == 'stay') for key in instances}
    if per_instance:
        return [pair_row(bases[key], min(
            (r for r in grouped[key] if r['final_delivery_cost'] <= (1 + epsilon) * bases[key]['final_delivery_cost'] + 1e-10),
            key=lambda r: r['phase2_wall_seconds'],
        )) for key in instances]
    if not instances:
        return []
    flat = [(key, row) for key in instances for row in grouped[key]]
    # 每实例恰选一个候选，最后一行约束整个集合的成本预算。
    constraints = lil_matrix((len(instances) + 1, len(flat)), dtype=float)
    indices = {key: index for index, key in enumerate(instances)}
    base_cost = sum(row['final_delivery_cost'] for row in bases.values())
    for column, (key, row) in enumerate(flat):
        constraints[indices[key], column] = 1.0
        constraints[-1, column] = row['final_delivery_cost'] / max(base_cost, 1e-12)
    lower = np.r_[np.ones(len(instances)), -np.inf]
    upper = np.r_[np.ones(len(instances)), 1 + epsilon]
    result = milp(
        c=np.array([row['phase2_wall_seconds'] for _, row in flat]),
        integrality=np.ones(len(flat)), bounds=Bounds(0, 1),
        constraints=LinearConstraint(constraints.tocsr(), lower, upper),
        options={'mip_rel_gap': 0.0},
    )
    if not result.success:
        raise RuntimeError(f'有限候选事后选择未完成：{result.message}')
    return [pair_row(bases[key], row) for value, (key, row) in zip(result.x, flat) if value > .5]


def write_csv(path, rows):
    """输入扁平记录列表，输出带表头的 UTF-8 CSV，空列表仍生成可识别文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_table(directory, name):
    """读取逐记录目录，优先使用最新原子记录；也支持仅复制过来的整表 JSON。"""
    directory = Path(directory)
    if (directory / name).is_dir():
        return [read_json(path) for path in sorted((directory / name).glob('*.json'))]
    path = directory / (name + '.json')
    return read_json(path) if path.exists() else []


def candidate_report(directory, output, bootstrap=1000):
    """分析阶段 A/B 候选表，输出潜力曲线、瓶颈、逐实例附表和未完成清单。"""
    directory, output = Path(directory), Path(output)
    manifest = read_json(directory / 'manifest.json')
    instance_rows = {row['id']: row for row in load_table(directory, 'instances')}
    candidates = load_table(directory, 'partition_candidates')
    expected = [row['id'] for row in manifest['instances']]
    incomplete = [key for key in expected if not instance_rows.get(key, {}).get('complete')]
    grouped = defaultdict(list)
    for row in candidates:
        if row.get('complete') and row['instance_id'] not in incomplete:
            grouped[row['instance_id']].append(row)
    report = {'kind': 'development_candidate_potential', 'expected_instances': len(expected),
              'complete_instances': len(grouped), 'incomplete_instances': incomplete,
              'complete': not incomplete, 'timing_kind': 'offline_group_observations',
              'not_an_online_speed_measurement': True, 'potential': {}, 'by_size': {},
              'scope': 'fixed_candidate_set_with_known_true_answers'}
    chosen_rows = []
    for epsilon in (0.0, .01, .03, .05):
        pairs = oracle_select(grouped, epsilon)
        stats = aggregate_pairs(pairs, bootstrap)
        stats['eligible_for_stage_gate'] = not incomplete and bool(grouped)
        stats['passes_point_thresholds'] = not incomplete and stats['passes_point_thresholds']
        # 区间仅描述固定事后选择的样本波动，不是可部署算法的泛化区间。
        stats['ci_scope'] = 'fixed_hindsight_choices_only'
        stats['thresholds_supported_by_ci95'] = not incomplete and stats.get('thresholds_supported_by_ci95', False)
        report['potential'][str(epsilon)] = stats
        if epsilon == .05:
            chosen_rows = pairs
            report['per_instance_5_percent_oracle'] = aggregate_pairs(oracle_select(grouped, epsilon, True), bootstrap)
    sizes = sorted({row['size'] for row in manifest['instances']})
    for size in sizes:
        subset = {key: rows for key, rows in grouped.items() if rows[0]['size'] == size}
        report['by_size'][str(size)] = aggregate_pairs(oracle_select(subset, .05), bootstrap)
    group_runs = [r for r in load_table(directory, 'group_runs') if r.get('complete')]
    components = ['input', 'distance', 'build', 'optimize', 'extract', 'fallback', 'other']
    report['measured_group_phase2_components_seconds'] = {
        name: sum(r.get(f'phase2_{name}_seconds', 0.0) for r in group_runs) for name in components
    }
    report['measured_group_phase3_seconds'] = sum(r['phase3_seconds'] for r in group_runs)
    report['unique_completed_group_runs'] = len(group_runs)
    report['unfinished_group_runs'] = sum(not r.get('complete') for r in load_table(directory, 'group_runs'))
    report['timeout_groups'] = sum(bool(r.get('timeout')) for r in group_runs)
    report['fallback_groups'] = sum(bool(r.get('fallback_used')) for r in group_runs)
    sessions = [read_json(p) for p in sorted((directory / 'sessions').glob('*.json'))]
    report['total_recorded_collection_seconds'] = sum(s['collection_seconds'] for s in sessions)
    report['mean_measured_group_seconds'] = (
        sum(r['acquisition_seconds'] for r in group_runs) / len(group_runs) if group_runs else None)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'candidate_report.json', report)
    write_csv(output / 'oracle_per_instance.csv', chosen_rows)
    curve_rows = []
    for rows in grouped.values():
        base = next(row for row in rows if row['name'] == 'stay')
        curve_rows += [dict(pair_row(base, row), kind=row['kind']) for row in rows]
    write_csv(output / 'candidate_curve.csv', curve_rows)
    write_candidate_plot(output / 'candidate_curve.png', curve_rows)
    write_report_markdown(output / 'candidate_report.md', report)
    return report


def write_candidate_plot(path, rows):
    """输入候选配对行，生成可导出的成本—第二阶段耗时散点图，使用无界面的绘图后端。"""
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for kind in sorted({row['kind'] for row in rows}):
        values = [row for row in rows if row['kind'] == kind]
        ax.scatter([100 * row['cost_change'] for row in values],
                   [-100 * row['time_change'] for row in values], label=kind, alpha=.6, s=22)
    ax.axvline(5, color='gray', linestyle='--', linewidth=1)
    ax.axhline(20, color='gray', linestyle='--', linewidth=1)
    ax.set(xlabel='Final delivery cost change (%)', ylabel='Complete Phase 2 time saving (%)',
           title='Development candidates: recorded downstream observations')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluation_report(directory, output, bootstrap=1000):
    """分析独立复测表，先按实例平均重复运行，再计算方法比较；不把重复作为新增实例。"""
    directory, output = Path(directory), Path(output)
    config = read_json(directory / 'evaluation_config.json')['configuration']
    rows = load_table(directory, 'evaluation_runs')
    expected = config['expected_instances']
    grouped = defaultdict(list)
    for row in rows:
        if row.get('complete'):
            grouped[row['instance_id'], row['method']].append(row)
    reduced = {}
    fields = ('final_delivery_cost', 'phase2_wall_seconds', 'phase3_seconds', 'online_seconds',
              'timeout_groups', 'fallback_groups')
    for key, observations in grouped.items():
        if len(observations) != config['repeats']:
            continue
        reduced[key] = dict(observations[0], **{field: float(np.mean([r[field] for r in observations])) for field in fields})
    report = {'kind': 'fresh_development_evaluation', 'repeats_per_instance': config['repeats'],
              'expected_instances': len(expected), 'methods': {},
              'repeat_aggregation': 'arithmetic_mean_within_complete_instance'}
    preparation = read_json(directory / 'map_preparation.json') if (directory / 'map_preparation.json').exists() else {}
    report['map_preparation'] = preparation
    all_pairs = []
    for method in config['methods']:
        missing = [key for key in expected if (key, method) not in reduced or (key, 'symmetric_mst') not in reduced]
        pairs = [dict(pair_row(reduced[key, 'symmetric_mst'], reduced[key, method]), method=method)
                 for key in expected if key not in missing]
        stats = aggregate_pairs(pairs, bootstrap)
        stats.update(complete=not missing, incomplete_instances=missing)
        stats['passes_point_thresholds'] = not missing and stats['passes_point_thresholds']
        stats['thresholds_supported_by_ci95'] = not missing and stats.get('thresholds_supported_by_ci95', False)
        stats['by_size'] = {str(size): aggregate_pairs([p for p in pairs if p['size'] == size], bootstrap)
                            for size in sorted({p['size'] for p in pairs})}
        stats['timeout_group_observations'] = sum(r.get('timeout_groups', 0) for r in rows if r.get('complete') and r['method'] == method)
        stats['fallback_group_observations'] = sum(r.get('fallback_groups', 0) for r in rows if r.get('complete') and r['method'] == method)
        method_runs = [r for r in rows if r.get('complete') and r['method'] == method]
        stats['mean_overhead_seconds'] = {
            field: float(np.mean([r.get('process', {}).get(field, 0.0) for r in method_runs])) if method_runs else None
            for field in ('instance_initialization_seconds', 'boundary_convex_sets_seconds',
                          'mst_partition_seconds', 'feature_seconds', 'repair_seconds', 'selection_seconds')
        }
        if pairs:
            stats['cold_batch_seconds'] = stats['total_online_seconds'] + preparation.get('preparation_seconds', 0.0)
            stats['baseline_cold_batch_seconds'] = stats['baseline_total_online_seconds'] + preparation.get('preparation_seconds', 0.0)
        report['methods'][method] = stats
        all_pairs.extend(pairs)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'evaluation_report.json', report)
    write_csv(output / 'evaluation_per_instance.csv', all_pairs)
    write_report_markdown(output / 'evaluation_report.md', report)
    return report


def write_report_markdown(path, report):
    """输入候选或实测报告，输出直接可阅读的结果摘要与缺失情况。"""
    lines = ['# MA-FSTSP 客户划分实验报告', '']
    if report['kind'] == 'development_candidate_potential':
        lines += [f'完成实例：{report["complete_instances"]}/{report["expected_instances"]}。', '',
                  '以下为已知候选真实结果后的事后选择，使用离线组观测；不代表在线加速。', '',
                  '| 总体成本预算 | 成本变化 | 第二阶段节时 |', '|---|---:|---:|']
        for epsilon, stats in report['potential'].items():
            if stats['instance_count']:
                lines.append(f'| {float(epsilon):.0%} | {stats["cost_change"]:+.2%} | {stats["phase2_saving"]:.2%} |')
        lines += ['', f'未完成实例：{", ".join(report["incomplete_instances"]) or "无"}。', '',
                  '5% 是总体成本预算；逐实例恶化见 oracle_per_instance.csv。',
                  '候选潜力接近或超过 30% 可为学习误差留出余量；进入学习前应查看独立复测中的手工选择表现。']
    else:
        lines += ['每个方法均独立求解；重复运行先在完整实例内取均值。', '',
                  '| 方法 | 完成数 | 成本变化 | 第二阶段节时 | 端到端节时 |', '|---|---:|---:|---:|---:|']
        for method, stats in report['methods'].items():
            if stats['instance_count']:
                lines.append(f'| {method} | {stats["instance_count"]}/{report["expected_instances"]} | '
                             f'{stats["cost_change"]:+.2%} | {stats["phase2_saving"]:.2%} | {stats["online_saving"]:.2%} |')
            else:
                lines.append(f'| {method} | 0/{report["expected_instances"]} | — | — | — |')
        lines += ['', '未完成实例会阻止通过判定。总体成本增加 ≤5%、第二阶段节时 ≥20% 是点估计门槛；',
                  '配对区间、逐规模统计和逐实例退化见 JSON 与 CSV。开发集表现不能替代独立测试验收。']
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
