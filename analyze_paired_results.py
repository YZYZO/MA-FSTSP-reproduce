"""配对实验的完成率、置信区间、检验、PAR-2 与机制分析。"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from experiment_store import atomic_write_json


def bootstrap_mean_difference(first, second, repetitions=10000, seed=0):
    """
    计算配对均值差 ``first-second`` 的 percentile bootstrap 95% CI。

    输入为等长配对样本、重采样次数和 seed；输出样本均值与上下界字典。重采样以
    “实例对”为单位而不是分别抽样两种方法，保持实验配对结构。
    """

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError('bootstrap 输入必须是一维等长配对样本。')
    if len(first) == 0:
        return {'mean_difference': None, 'ci95_low': None, 'ci95_high': None}
    differences = first - second
    random_state = np.random.default_rng(seed)
    indices = random_state.integers(
        0,
        len(differences),
        size=(int(repetitions), len(differences)),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    return {
        'mean_difference': float(differences.mean()),
        'ci95_low': float(np.percentile(bootstrap_means, 2.5)),
        'ci95_high': float(np.percentile(bootstrap_means, 97.5)),
    }


def bootstrap_relative_difference(method, baseline, repetitions=10000, seed=0):
    """计算配对相对差 ``(method-baseline)/baseline`` 的均值和中位数 CI。"""

    method = np.asarray(method, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if method.shape != baseline.shape or method.ndim != 1:
        raise ValueError('相对差 bootstrap 输入必须是一维等长配对样本。')
    if len(method) == 0:
        return {
            'mean_relative_difference': None,
            'mean_ci95_low': None,
            'mean_ci95_high': None,
            'median_relative_difference': None,
            'median_ci95_low': None,
            'median_ci95_high': None,
        }
    if np.any(baseline == 0):
        raise ValueError('相对差的基准样本不能为零。')
    relative = (method - baseline) / baseline
    random_state = np.random.default_rng(seed)
    indices = random_state.integers(
        0,
        len(relative),
        size=(int(repetitions), len(relative)),
    )
    samples = relative[indices]
    bootstrap_means = samples.mean(axis=1)
    bootstrap_medians = np.median(samples, axis=1)
    return {
        'mean_relative_difference': float(relative.mean()),
        'mean_ci95_low': float(np.percentile(bootstrap_means, 2.5)),
        'mean_ci95_high': float(np.percentile(bootstrap_means, 97.5)),
        'median_relative_difference': float(np.median(relative)),
        'median_ci95_low': float(np.percentile(bootstrap_medians, 2.5)),
        'median_ci95_high': float(np.percentile(bootstrap_medians, 97.5)),
    }


def _spearman_summary(first, second, mask):
    """在给定严格掩码上计算 Spearman 相关，并把常量输入的 NaN 转成 ``None``。"""

    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 2:
        return {'sample_count': int(mask.sum()), 'rho': None, 'p_value': None}
    first_values = np.asarray(first, dtype=float)[mask]
    second_values = np.asarray(second, dtype=float)[mask]
    if np.all(first_values == first_values[0]) or np.all(second_values == second_values[0]):
        return {'sample_count': int(mask.sum()), 'rho': None, 'p_value': None}
    result = spearmanr(first_values, second_values)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    return {
        'sample_count': int(mask.sum()),
        'rho': rho if np.isfinite(rho) else None,
        'p_value': p_value if np.isfinite(p_value) else None,
    }


def paired_wilcoxon(first, second):
    """
    对两种方法的等长实例结果执行双侧 Wilcoxon signed-rank 检验。

    输入为配对的一维样本；输出统计量和 p 值。所有差值为零时返回 p=1，空样本
    返回 ``None``，避免 SciPy 边界行为中断批量分析。
    """

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if len(first) == 0:
        return {'statistic': None, 'p_value': None}
    difference = first - second
    if np.allclose(difference, 0.0):
        return {'statistic': 0.0, 'p_value': 1.0}
    result = wilcoxon(first, second, alternative='two-sided', zero_method='zsplit')
    return {'statistic': float(result.statistic), 'p_value': float(result.pvalue)}


def par2(times, completed, cutoff):
    """
    计算给定截止时间下的 PAR-2。

    输入为逐实例时间、完成标记和正截止秒数；输出完成实例真实时间与失败实例
    ``2*cutoff`` 的平均值。未提供截止时间时上层不调用本函数，避免臆造处罚口径。
    """

    if cutoff is None or float(cutoff) <= 0:
        raise ValueError('PAR-2 cutoff 必须是正数。')
    times = np.asarray(times, dtype=float)
    completed = np.asarray(completed, dtype=bool)
    penalized = np.where(completed & np.isfinite(times), times, 2.0 * float(cutoff))
    return float(penalized.mean())


def _finite_pair_mask(first, second, first_complete, second_complete):
    """
    构造两方法均完成且指标有限的严格配对掩码。

    输入为两列指标和完成标记；输出布尔数组，用于所有成对统计以避免样本错配。
    """

    return (
        np.asarray(first_complete, dtype=bool)
        & np.asarray(second_complete, dtype=bool)
        & np.isfinite(first)
        & np.isfinite(second)
    )


def analyze_summary(summary_path, baseline='smst_original', cutoff=None, seed=0):
    """
    分析一个 ``paired_summary.npz`` 并返回严格 JSON 兼容报告。

    输入为汇总路径、基准方法、可选 PAR-2 cutoff 和 bootstrap seed；输出包含方法
    完成率/超时率、基准配对相对差、Wilcoxon、bootstrap CI 及两组 Q_bin-Time
    Spearman 机制验证的字典。未显式传 cutoff 时优先读取 V2 summary 的实例时限。
    """

    with np.load(summary_path, allow_pickle=False) as data:
        methods = [str(item) for item in np.asarray(data['methods']).tolist()]
        if baseline not in methods:
            raise ValueError(f'基准方法 {baseline!r} 不在结果中。')
        completed = np.asarray(data['completed'], dtype=bool)
        timed_out = np.asarray(data['time_limit_reached'], dtype=bool)
        costs = np.asarray(data['cost'], dtype=float)
        solve_times = np.asarray(data['solve_time'], dtype=float)
        # V2 分别保存最大单组时间与总时间；旧文件按已有字段降级读取。
        total_phase2_times = np.asarray(
            data['total_phase2_time']
            if 'total_phase2_time' in data.files
            else data['phase2_optimize_time'],
            dtype=float,
        )
        max_phase2_times = np.asarray(
            data['max_phase2_time']
            if 'max_phase2_time' in data.files
            else data['phase2_optimize_time'],
            dtype=float,
        )
        max_q_bin = np.asarray(data['max_q_bin'], dtype=float)
        sum_q_bin = np.asarray(
            data['sum_q_bin'] if 'sum_q_bin' in data.files else data['max_q_bin'],
            dtype=float,
        )
        optional_metrics = {
            name: np.asarray(data[name], dtype=float)
            if name in data.files
            else np.full(costs.shape, np.nan, dtype=float)
            for name in (
                'active_depots', 'max_group_customers', 'mean_q_bin',
                'median_q_bin', 'cv_q_bin',
            )
        }
        input_hashes = [str(item) for item in np.asarray(data['input_hashes']).tolist()]
        if cutoff is None and 'instance_time_limit' in data.files:
            stored_cutoff = float(np.asarray(data['instance_time_limit']).reshape(()))
            if np.isfinite(stored_cutoff) and stored_cutoff > 0:
                cutoff = stored_cutoff

    method_summaries = {}
    for method_index, method in enumerate(methods):
        complete_mask = completed[:, method_index]
        finite_costs = costs[complete_mask, method_index]
        finite_costs = finite_costs[np.isfinite(finite_costs)]
        finite_times = solve_times[complete_mask, method_index]
        finite_times = finite_times[np.isfinite(finite_times)]
        max_q_mask = (
            complete_mask
            & np.isfinite(max_q_bin[:, method_index])
            & np.isfinite(max_phase2_times[:, method_index])
        )
        sum_q_mask = (
            complete_mask
            & np.isfinite(sum_q_bin[:, method_index])
            & np.isfinite(total_phase2_times[:, method_index])
        )
        method_summaries[method] = {
            'instance_count': int(len(complete_mask)),
            'completed_count': int(complete_mask.sum()),
            'completion_rate': float(complete_mask.mean()),
            'timeout_count': int(timed_out[:, method_index].sum()),
            'timeout_rate': float(timed_out[:, method_index].mean()),
            'cost_mean_completed': float(finite_costs.mean()) if len(finite_costs) else None,
            'cost_median_completed': float(np.median(finite_costs)) if len(finite_costs) else None,
            'solve_time_mean_completed': float(finite_times.mean()) if len(finite_times) else None,
            'solve_time_median_completed': float(np.median(finite_times)) if len(finite_times) else None,
            'par2': par2(solve_times[:, method_index], complete_mask, cutoff)
            if cutoff is not None else None,
            'group_structure_completed': {
                metric: {
                    'mean': float(values.mean()) if len(values) else None,
                    'median': float(np.median(values)) if len(values) else None,
                }
                for metric, matrix in optional_metrics.items()
                for values in [
                    matrix[complete_mask, method_index][
                        np.isfinite(matrix[complete_mask, method_index])
                    ]
                ]
            },
            'max_q_bin_vs_max_group_phase2_time_spearman': _spearman_summary(
                max_q_bin[:, method_index],
                max_phase2_times[:, method_index],
                max_q_mask,
            ),
            'sum_q_bin_vs_total_phase2_time_spearman': _spearman_summary(
                sum_q_bin[:, method_index],
                total_phase2_times[:, method_index],
                sum_q_mask,
            ),
        }

    comparisons = {}
    baseline_index = methods.index(baseline)
    for method_index, method in enumerate(methods):
        if method == baseline:
            continue
        pair_mask = _finite_pair_mask(
            costs[:, method_index],
            costs[:, baseline_index],
            completed[:, method_index],
            completed[:, baseline_index],
        ) & (costs[:, baseline_index] != 0)
        first = costs[pair_mask, method_index]
        second = costs[pair_mask, baseline_index]
        time_pair_mask = _finite_pair_mask(
            solve_times[:, method_index],
            solve_times[:, baseline_index],
            completed[:, method_index],
            completed[:, baseline_index],
        ) & (solve_times[:, baseline_index] != 0)
        method_times = solve_times[time_pair_mask, method_index]
        baseline_times = solve_times[time_pair_mask, baseline_index]
        relative_cost = bootstrap_relative_difference(first, second, seed=seed)
        relative_time = bootstrap_relative_difference(
            method_times,
            baseline_times,
            seed=seed,
        )
        comparisons[method] = {
            'baseline': baseline,
            'paired_completed_count': int(pair_mask.sum()),
            'cost_difference_method_minus_baseline': bootstrap_mean_difference(
                first,
                second,
                seed=seed,
            ),
            'cost_wilcoxon': paired_wilcoxon(first, second),
            'cost_relative_difference_method_minus_baseline': relative_cost,
            'cost_noninferiority_margin': 0.01,
            'cost_noninferiority_pass': (
                relative_cost['mean_ci95_high'] is not None
                and relative_cost['mean_ci95_high'] <= 0.01
            ),
            'time_paired_completed_count': int(time_pair_mask.sum()),
            'solve_time_difference_method_minus_baseline': bootstrap_mean_difference(
                method_times,
                baseline_times,
                seed=seed,
            ),
            'solve_time_wilcoxon': paired_wilcoxon(
                method_times,
                baseline_times,
            ),
            'solve_time_relative_difference_method_minus_baseline': relative_time,
            'solve_time_improvement_pass': (
                relative_time['mean_ci95_high'] is not None
                and relative_time['mean_ci95_high'] < 0.0
            ),
            'solve_time_median_reduction_target': -0.25,
            'solve_time_median_reduction_target_pass': (
                relative_time['median_relative_difference'] is not None
                and relative_time['median_relative_difference'] <= -0.25
            ),
        }

    return {
        'analysis_schema_version': 2,
        'summary_path': str(Path(summary_path).resolve()),
        'baseline': baseline,
        'par2_cutoff_seconds': cutoff,
        'instance_count': len(input_hashes),
        'unique_input_hash_count': len(set(input_hashes)),
        'methods': method_summaries,
        'comparisons': comparisons,
    }


def main():
    """
    解析命令行并把分析报告打印或写入 JSON。

    输入来自 CLI；输出默认打印到终端，传 ``--output`` 时使用原子 JSON 写入。
    """

    parser = argparse.ArgumentParser(description='分析 MA-FSTSP 第一阶段配对实验。')
    parser.add_argument('summary', type=Path, help='paired_summary.npz 路径')
    parser.add_argument('--baseline', default='smst_original', help='配对基准方法')
    parser.add_argument('--cutoff', type=float, default=None, help='PAR-2 截止秒数')
    parser.add_argument('--seed', type=int, default=0, help='bootstrap 随机种子')
    parser.add_argument('--output', type=Path, default=None, help='可选 JSON 输出路径')
    args = parser.parse_args()
    report = analyze_summary(
        args.summary,
        baseline=args.baseline,
        cutoff=args.cutoff,
        seed=args.seed,
    )
    if args.output is None:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        atomic_write_json(args.output, report)
        print(args.output)


if __name__ == '__main__':
    main()
