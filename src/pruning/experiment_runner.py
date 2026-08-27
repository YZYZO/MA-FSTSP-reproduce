"""
组织 P7 的原始密集、稀疏未剪和稀疏支配三组配对实验。

该模块复用现有实例采样、三阶段求解和 NPZ 保存函数；新增比较指标写入独立目录，不改变旧结果格式。
"""

import csv
import gc
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
import traceback
from typing import Optional, Sequence

import numpy as np

from config import RESULTS_DIR
from experiment_results import _save_stsp_batch_result, _solve_model_with_process_data
from problem import prepare_manhattan_road_network, sample_multiagent_instances
from src.fstsp import MultiAgentFlyingSidekickTSP

from .fstsp_adapter import (
    PruningMultiAgentFlyingSidekickTSP,
    attach_pruning_process_data,
)
from .types import PruningOptions


ORIGINAL_DENSE = 'original_dense'
SPARSE_BASELINE = 'sparse_baseline'
SPARSE_P7 = 'sparse_p7'
COMPARISON_VARIANTS = (ORIGINAL_DENSE, SPARSE_BASELINE, SPARSE_P7)


@dataclass(frozen=True)
class P7ComparisonOptions:
    """
    保存一次 P7 配对实验不会随实例变化的固定配置。

    输入：实例数、客户规模、随机种子、P7 预算、证据开关和可选输出目录。
    输出：不可变实验配置对象。
    逻辑：默认是 1K 路网试验档所需的 5 个实例、20/50 客户；正式实验可覆盖这些字段。
    """

    instance_count: int = 5
    customer_sizes: tuple = (20, 50)
    instance_seed: int = 0
    gurobi_seed: int = 0
    dominance_epsilon: float = 1e-9
    comparison_block_size: int = 4096
    max_pair_checks: Optional[int] = None
    max_preprocessing_seconds: Optional[float] = 30.0
    record_evidence: bool = False
    objective_tolerance: float = 1e-6
    output_root: Optional[Path] = None

    def validate(self):
        """
        校验配对实验规模、预算与目标值容差。

        输入：当前实验配置。
        输出：无；非法字段抛出 `ValueError`。
        逻辑：空实验或负容差无法产生可信比较，因此在加载路网前尽早失败。
        """

        if self.instance_count <= 0:
            raise ValueError('P7 比较实验的 instance_count 必须为正整数。')
        if not self.customer_sizes or any(int(size) <= 0 for size in self.customer_sizes):
            raise ValueError('P7 比较实验至少需要一个正客户规模。')
        if self.objective_tolerance < 0:
            raise ValueError('objective_tolerance 不能为负数。')
        self.pruning_options(False).validate()
        self.pruning_options(True).validate()

    def pruning_options(self, enabled):
        """
        为稀疏基线或 P7 组生成共享除启用开关外全部参数的剪枝配置。

        输入：是否启用端点对支配。
        输出：`PruningOptions`。
        逻辑：B/C 仅允许 `endpoint_pair_dominance` 不同，以隔离 P7 的真实贡献。
        """

        return PruningOptions(
            endpoint_pair_dominance=bool(enabled),
            dominance_epsilon=self.dominance_epsilon,
            comparison_block_size=self.comparison_block_size,
            max_pair_checks=self.max_pair_checks,
            max_preprocessing_seconds=self.max_preprocessing_seconds,
            record_evidence=self.record_evidence,
            gurobi_seed=self.gurobi_seed,
            gurobi_output=False,
        )


def _json_default(value):
    """
    把 NumPy 标量、数组、元组和路径转换为 JSON 可写类型。

    输入：标准 `json` 无法直接编码的对象。
    输出：等价基本类型；未知类型抛出 `TypeError`。
    逻辑：实验节点通常是 NumPy 整数，显式转换可保证不启用 pickle。
    """

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f'无法 JSON 序列化对象：{type(value).__name__}。')


def _within_tolerance(left, right, tolerance):
    """
    按绝对与相对混合尺度比较两个目标值。

    输入：两个目标值和基础容差。
    输出：差值未超过尺度容差时为 `True`。
    逻辑：目标值增大时同步放宽绝对误差，但不会掩盖具有实际量级的差异。
    """

    scale = max(1.0, abs(float(left)), abs(float(right)))
    return abs(float(left) - float(right)) <= tolerance * scale


def _variant_order(instance_index):
    """
    轮换同一实例的三种模型执行顺序，降低固定冷热启动顺序偏差。

    输入：当前实例编号。
    输出：三种变体名称元组。
    逻辑：按实例编号循环移位，使每种变体都能出现在第一、第二和第三位置。
    """

    shift = instance_index % len(COMPARISON_VARIANTS)
    return COMPARISON_VARIANTS[shift:] + COMPARISON_VARIANTS[:shift]


def _build_model(variant, graph, depots, cities, distance, drones, options):
    """
    为给定实验变体创建一个全新 MA-FSTSP 模型。

    输入：变体名、共享实例数据、无人机数和比较配置。
    输出：原始模型或剪枝适配模型。
    逻辑：A 使用未修改原类；B/C 使用同一个适配类且只有 P7 开关不同。
    """

    common_arguments = (
        graph,
        depots,
        cities,
        distance,
        drones,
    )
    if variant == ORIGINAL_DENSE:
        return MultiAgentFlyingSidekickTSP(
            *common_arguments,
            theta=(0.5, 0.5),
        )
    if variant == SPARSE_BASELINE:
        pruning_options = options.pruning_options(False)
    elif variant == SPARSE_P7:
        pruning_options = options.pruning_options(True)
    else:
        raise ValueError(f'未知 P7 比较变体：{variant!r}。')
    return PruningMultiAgentFlyingSidekickTSP(
        *common_arguments,
        theta=(0.5, 0.5),
        pruning_options=pruning_options,
    )


def _run_variant(
    variant,
    execution_position,
    graph,
    depots,
    cities,
    distance,
    drones,
    options,
):
    """
    构造并完整求解一个实验变体，同时保留模型供等价顺序审计。

    输入：变体和顺序信息、共享实例数据、车队参数及比较配置。
    输出：包含模型、解、成本、过程数据和分层耗时的内部结果字典。
    逻辑：模型初始化时间与既有 `solve_seconds` 分开记录，避免候选区域初始化被误计为 P7 时间。
    """

    total_start = time.perf_counter()
    initialization_start = time.perf_counter()
    model = _build_model(
        variant,
        graph,
        depots,
        cities,
        distance,
        drones,
        options,
    )
    initialization_seconds = time.perf_counter() - initialization_start
    solution, cost, process_data = _solve_model_with_process_data(model)
    if variant != ORIGINAL_DENSE:
        attach_pruning_process_data(model, process_data)
    total_seconds = time.perf_counter() - total_start
    return {
        'variant': variant,
        'execution_position': execution_position,
        'model': model,
        'solution': solution,
        'cost': float(cost),
        'process_data': process_data,
        'initialization_seconds': initialization_seconds,
        'solve_seconds': float(process_data['solve_seconds']),
        'total_seconds': total_seconds,
    }


def _pruning_metrics_from_result(result):
    """
    提取一个变体全部非空仓库组的 P7/稀疏模型指标。

    输入：`_run_variant` 的结果字典。
    输出：每仓库指标字典列表；原始密集组返回空列表。
    逻辑：只读取已经附加到 `depot_records` 的报告，不依赖模型内部缓存。
    """

    metrics = []
    for record in result['process_data']['depot_records']:
        if 'pruning' in record:
            metrics.append(record['pruning']['metrics'])
    return metrics


def _sum_metric(metrics, key):
    """
    对多个仓库组的同名数值指标求和。

    输入：指标字典列表和字段名。
    输出：字段值之和；空列表返回 `None` 以区别真实零值。
    逻辑：布尔和字符串字段不通过本函数汇总。
    """

    if not metrics:
        return None
    return sum(item.get(key, 0) or 0 for item in metrics)


def _build_instance_row(dataset_label, customer_size, instance_index, result):
    """
    把一个变体的三阶段过程与 P7 指标展平为 CSV 行。

    输入：数据集标签、客户数、实例编号和变体结果。
    输出：只包含标量和短字符串的字典。
    逻辑：Phase 2/3 时间按仓库求和；原始密集组没有内部 Gurobi 指标时保留空值。
    """

    process_data = result['process_data']
    records = process_data['depot_records']
    metrics = _pruning_metrics_from_result(result)
    sequences = [record['set_tsp_sequence'] for record in records if record['set_tsp_sequence']]
    return {
        'dataset': dataset_label,
        'customer_size': int(customer_size),
        'instance_index': int(instance_index),
        'variant': result['variant'],
        'execution_position': int(result['execution_position']),
        'final_cost': result['cost'],
        'model_initialization_seconds': result['initialization_seconds'],
        'solve_seconds': result['solve_seconds'],
        'total_seconds': result['total_seconds'],
        'phase1_boundary_seconds': process_data['boundary_convex_sets_seconds'],
        'phase1_partition_seconds': process_data['mst_partition_seconds'],
        'phase2_seconds': sum(record['set_tsp_seconds'] for record in records),
        'phase3_seconds': sum(record['local_search_seconds'] for record in records),
        'nonempty_depot_groups': len(sequences),
        'set_tsp_objective': _sum_metric(metrics, 'set_tsp_objective'),
        'endpoint_pairs_before': _sum_metric(metrics, 'endpoint_pairs_before'),
        'endpoint_pairs_kept': _sum_metric(metrics, 'endpoint_pairs_kept'),
        'endpoint_pairs_dominated': _sum_metric(metrics, 'endpoint_pairs_dominated'),
        'endpoint_pair_checks': _sum_metric(metrics, 'endpoint_pair_checks'),
        'endpoint_dominance_seconds': _sum_metric(metrics, 'endpoint_dominance_seconds'),
        'model_build_seconds': _sum_metric(metrics, 'model_build_seconds'),
        'gurobi_seconds': _sum_metric(metrics, 'gurobi_seconds'),
        'model_variables': _sum_metric(metrics, 'model_variables'),
        'model_constraints': _sum_metric(metrics, 'model_constraints'),
        'external_variables': _sum_metric(metrics, 'external_variables'),
        'gurobi_node_count': _sum_metric(metrics, 'gurobi_node_count'),
        'budget_exhausted': any(item.get('budget_exhausted', False) for item in metrics),
        'safe_fallback': any(item.get('safe_fallback', False) for item in metrics),
    }


def _reports_by_depot(result):
    """
    将稀疏变体的 P7 报告按仓库节点建立索引。

    输入：变体结果。
    输出：仓库节点到报告字典的映射。
    逻辑：空客户仓库没有 Phase 2 模型，因此不会出现在映射中。
    """

    reports = {}
    for record in result['process_data']['depot_records']:
        if 'pruning' in record:
            reports[record['depot_node']] = record['pruning']
    return reports


def _audit_sparse_pair(baseline, p7, tolerance):
    """
    验证 B/C 的 Set-TSP 目标值，并在集合顺序不同时进行交叉固定顺序复核。

    输入：稀疏未剪结果、P7 结果和目标值容差。
    输出：实例级正确性状态、顺序差异数及审计目标值记录。
    逻辑：目标不一致或 P7 无法以基线顺序达到同一最优值立即抛错；最终成本差异仅在等价顺序时标记。
    """

    baseline_reports = _reports_by_depot(baseline)
    p7_reports = _reports_by_depot(p7)
    if set(baseline_reports) != set(p7_reports):
        raise AssertionError('稀疏基线与 P7 的非空仓库组不一致。')

    sequence_difference_count = 0
    audit_records = []
    for depot in baseline_reports:
        baseline_report = baseline_reports[depot]
        p7_report = p7_reports[depot]
        baseline_objective = baseline_report['metrics']['set_tsp_objective']
        p7_objective = p7_report['metrics']['set_tsp_objective']
        if not _within_tolerance(baseline_objective, p7_objective, tolerance):
            raise AssertionError(
                f'仓库 {depot!r} 的 P7 Set-TSP 目标值改变：'
                f'baseline={baseline_objective}, p7={p7_objective}。'
            )

        baseline_sequence = baseline_report['sequence']
        p7_sequence = p7_report['sequence']
        if baseline_sequence == p7_sequence:
            continue
        sequence_difference_count += 1
        audit_start = time.perf_counter()
        baseline_fixed_p7 = baseline['model'].evaluate_fixed_sequence(depot, p7_sequence)
        p7_fixed_baseline = p7['model'].evaluate_fixed_sequence(depot, baseline_sequence)
        audit_seconds = time.perf_counter() - audit_start
        if not _within_tolerance(baseline_fixed_p7, baseline_objective, tolerance):
            raise AssertionError(
                f'仓库 {depot!r} 的 P7 顺序在完整稀疏候选中不是等价最优。'
            )
        if not _within_tolerance(p7_fixed_baseline, p7_objective, tolerance):
            raise AssertionError(
                f'仓库 {depot!r} 的基线最优顺序被 P7 端点剪枝破坏。'
            )
        audit_records.append({
            'depot_node': depot,
            'baseline_sequence': baseline_sequence,
            'p7_sequence': p7_sequence,
            'baseline_objective': baseline_objective,
            'p7_objective': p7_objective,
            'baseline_fixed_p7_objective': baseline_fixed_p7,
            'p7_fixed_baseline_objective': p7_fixed_baseline,
            'audit_seconds': audit_seconds,
        })

    same_final_cost = _within_tolerance(baseline['cost'], p7['cost'], tolerance)
    if sequence_difference_count == 0 and not same_final_cost:
        raise AssertionError('B/C 集合顺序完全相同，但最终 Phase 3 成本不一致。')
    if sequence_difference_count == 0:
        status = 'identical_sequence_and_cost'
    elif same_final_cost:
        status = 'phase2_tie_same_final_cost'
    else:
        status = 'phase2_tie_divergence'
    return {
        'status': status,
        'sequence_difference_count': sequence_difference_count,
        'same_final_cost': same_final_cost,
        'fixed_sequence_audits': audit_records,
    }


def _build_comparison_row(dataset_label, customer_size, instance_index, results, sparse_audit):
    """
    构造一个实例的 A/B/C 配对比较行。

    输入：实例标识、三种变体结果和 B/C 正确性审计结果。
    输出：包含目标差异、加速比、剪枝率和兼容状态的字典。
    逻辑：P7 贡献以 C/B 为主，C/A 只报告相对当前原算法的综合变化。
    """

    original = results[ORIGINAL_DENSE]
    baseline = results[SPARSE_BASELINE]
    p7 = results[SPARSE_P7]
    baseline_metrics = _pruning_metrics_from_result(baseline)
    p7_metrics = _pruning_metrics_from_result(p7)
    before = _sum_metric(p7_metrics, 'endpoint_pairs_before') or 0
    dominated = _sum_metric(p7_metrics, 'endpoint_pairs_dominated') or 0
    return {
        'dataset': dataset_label,
        'customer_size': int(customer_size),
        'instance_index': int(instance_index),
        'sparse_audit_status': sparse_audit['status'],
        'sequence_difference_count': sparse_audit['sequence_difference_count'],
        'original_cost': original['cost'],
        'sparse_baseline_cost': baseline['cost'],
        'p7_cost': p7['cost'],
        'p7_minus_sparse_cost': p7['cost'] - baseline['cost'],
        'p7_minus_original_cost': p7['cost'] - original['cost'],
        'sparse_baseline_phase2_seconds': sum(
            record['set_tsp_seconds'] for record in baseline['process_data']['depot_records']
        ),
        'p7_phase2_seconds': sum(
            record['set_tsp_seconds'] for record in p7['process_data']['depot_records']
        ),
        'phase2_speedup_p7_over_sparse': (
            sum(record['set_tsp_seconds'] for record in baseline['process_data']['depot_records'])
            / max(1e-12, sum(record['set_tsp_seconds'] for record in p7['process_data']['depot_records']))
        ),
        'total_speedup_p7_over_sparse': baseline['total_seconds'] / max(1e-12, p7['total_seconds']),
        'total_speedup_p7_over_original': original['total_seconds'] / max(1e-12, p7['total_seconds']),
        'endpoint_pairs_before': before,
        'endpoint_pairs_dominated': dominated,
        'endpoint_pair_pruning_rate': dominated / before if before else 0.0,
        'sparse_external_variables': _sum_metric(baseline_metrics, 'external_variables'),
        'p7_external_variables': _sum_metric(p7_metrics, 'external_variables'),
        'p7_preprocessing_seconds': _sum_metric(p7_metrics, 'endpoint_dominance_seconds'),
        'p7_budget_exhausted': any(item.get('budget_exhausted', False) for item in p7_metrics),
        'p7_safe_fallback': any(item.get('safe_fallback', False) for item in p7_metrics),
    }


def _write_csv(path, rows):
    """
    使用统一字段顺序写出一组实验标量记录。

    输入：输出路径和非空字典列表。
    输出：写入后的路径。
    逻辑：先写同目录临时文件再原子替换，并使用 UTF-8 BOM 便于 Excel 直接打开。
    """

    if not rows:
        raise ValueError(f'不能向 {path} 写入空实验记录。')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=path.parent,
            prefix=f'.{path.name}.',
            suffix='.tmp',
            encoding='utf-8-sig',
            newline='',
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def _atomic_write_text(path, content, encoding='utf-8'):
    """
    原子写入一份文本实验记录。

    输入：目标路径、完整文本和字符编码。
    输出：写入后的目标路径。
    逻辑：临时文件完整落盘后再替换目标，避免进程中断破坏上一版记录。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=path.parent,
            prefix=f'.{path.name}.',
            suffix='.tmp',
            encoding=encoding,
            newline='',
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def _atomic_write_json(path, value):
    """
    将实验字典原子写为易读 JSON。

    输入：目标路径和可由 `_json_default` 补充编码的对象。
    输出：写入后的目标路径。
    逻辑：统一使用 Unicode 和缩进格式，供检查点、失败记录与汇总复用。
    """

    content = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=_json_default,
    )
    return _atomic_write_text(path, content + '\n')


def _atomic_write_jsonl(path, records):
    """
    原子写入逐行 JSON 审计记录。

    输入：目标路径和记录序列。
    输出：写入后的目标路径。
    逻辑：每条记录独占一行，同时保持整份文件的原子替换语义。
    """

    content = ''.join(
        json.dumps(record, ensure_ascii=False, default=_json_default) + '\n'
        for record in records
    )
    return _atomic_write_text(path, content)


def _aggregate_summary(
    comparison_rows,
    failure_rows=(),
    attempted_instance_count=None,
):
    """
    汇总正式报告最关心的剪枝率、速度比和正确性状态。

    输入：成功实例比较行、失败实例记录和已经尝试的实例数。
    输出：包含总实例数、均值、中位数和状态计数的字典。
    逻辑：速度比同时给出均值与中位数；全部失败时统计值使用 `None`。
    """

    speedups = [float(row['phase2_speedup_p7_over_sparse']) for row in comparison_rows]
    pruning_rates = [float(row['endpoint_pair_pruning_rate']) for row in comparison_rows]
    statuses = {}
    for row in comparison_rows:
        status = row['sparse_audit_status']
        statuses[status] = statuses.get(status, 0) + 1
    failure_stage_counts = {}
    for row in failure_rows:
        stage = row['failure_stage']
        failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + 1
    succeeded_count = len(comparison_rows)
    failed_count = len(failure_rows)
    if attempted_instance_count is None:
        attempted_instance_count = succeeded_count + failed_count
    return {
        # `instance_count` 保留旧含义，继续表示可进入正式配对统计的成功实例数。
        'instance_count': succeeded_count,
        'attempted_instance_count': int(attempted_instance_count),
        'succeeded_instance_count': succeeded_count,
        'failed_instance_count': failed_count,
        'skipped_instance_count': failed_count,
        'phase2_speedup_mean': statistics.fmean(speedups) if speedups else None,
        'phase2_speedup_median': statistics.median(speedups) if speedups else None,
        'endpoint_pair_pruning_rate_mean': (
            statistics.fmean(pruning_rates) if pruning_rates else None
        ),
        'endpoint_pair_pruning_rate_median': (
            statistics.median(pruning_rates) if pruning_rates else None
        ),
        'audit_status_counts': statuses,
        'failure_stage_counts': failure_stage_counts,
        'budget_exhausted_count': sum(bool(row['p7_budget_exhausted']) for row in comparison_rows),
        'safe_fallback_count': sum(bool(row['p7_safe_fallback']) for row in comparison_rows),
    }


def _build_summary_document(
    timestamp,
    options,
    comparison_rows,
    fixed_sequence_audits,
    failure_rows,
    attempted_instance_count,
):
    """
    构造当前 P7 运行进度的完整汇总文档。

    输入：运行标识、配置、成功比较行、顺序审计、失败记录和尝试数。
    输出：可直接写入 `summary.json` 的字典。
    逻辑：成功与失败实例分开统计，使中途检查和最终报告采用同一数据口径。
    """

    return {
        'created_at': timestamp,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'options': dict(options.__dict__),
        'summary': _aggregate_summary(
            comparison_rows,
            failure_rows=failure_rows,
            attempted_instance_count=attempted_instance_count,
        ),
        'fixed_sequence_audits': fixed_sequence_audits,
        'failures': failure_rows,
    }


def _write_progress_outputs(
    output_directory,
    timestamp,
    options,
    instance_rows,
    comparison_rows,
    fixed_sequence_audits,
    dominance_evidence,
    failure_rows,
    attempted_instance_count,
):
    """
    在每个实验单元结束后刷新可直接读取的进度文件。

    输入：输出目录、运行上下文以及当前全部成功和失败记录。
    输出：最新汇总文档。
    逻辑：非空 CSV、可选审计 JSONL 与汇总 JSON 均采用原子替换，旧记录不会被半写文件覆盖。
    """

    output_directory = Path(output_directory)
    if instance_rows:
        _write_csv(output_directory / 'instance_metrics.csv', instance_rows)
    if comparison_rows:
        _write_csv(output_directory / 'paired_comparisons.csv', comparison_rows)
    if failure_rows:
        _write_csv(output_directory / 'failures.csv', failure_rows)
    if options.record_evidence and dominance_evidence:
        _atomic_write_jsonl(
            output_directory / 'dominance_audit.jsonl',
            dominance_evidence,
        )
    summary = _build_summary_document(
        timestamp,
        options,
        comparison_rows,
        fixed_sequence_audits,
        failure_rows,
        attempted_instance_count,
    )
    _atomic_write_json(output_directory / 'summary.json', summary)
    return summary


def _save_instance_checkpoint(
    output_directory,
    spec,
    customer_size,
    instance_index,
    results,
    depots,
    cities,
    distance,
    distance_stats,
    instance_metric_rows,
    comparison_row,
    fixed_sequence_audits,
    dominance_evidence,
):
    """
    保存一个完整 A/B/C 配对实例并最后提交完成标记。

    输入：实例标识、三种变体结果、共享输入、指标、审计与可选支配证据。
    输出：该实例的检查点目录。
    逻辑：先保存三个单实例 NPZ，全部成功后才写 `complete.json`；缺少完成标记的目录视为未提交。
    """

    checkpoint_directory = (
        Path(output_directory)
        / 'checkpoints'
        / spec.result_stem
        / f'customers-{int(customer_size)}'
        / f'instance-{int(instance_index):04d}'
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    variant_files = {}
    for variant in COMPARISON_VARIANTS:
        result = results[variant]
        variant_path = checkpoint_directory / f'{variant}.npz'
        _save_stsp_batch_result(
            variant_path,
            [(
                result['cost'],
                result['solve_seconds'],
                result['solution'],
                result['process_data'],
            )],
            [depots],
            [cities],
            distance,
            spec.drone_count,
            {'hc': [], 'stsp': [], 'lp': []},
            {'hc': [], 'stsp': [], 'lp': []},
            distance_initialization_stats=distance_stats,
        )
        variant_files[variant] = variant_path.name

    # 完成标记包含本实例全部标量记录，可在批量进程意外结束后独立审计。
    checkpoint_metadata = {
        'checkpoint_schema_version': 1,
        'completed_at': datetime.now().isoformat(timespec='seconds'),
        'dataset': spec.dataset_label,
        'result_stem': spec.result_stem,
        'customer_size': int(customer_size),
        'instance_index': int(instance_index),
        'variant_files': variant_files,
        'instance_metrics': instance_metric_rows,
        'paired_comparison': comparison_row,
        'fixed_sequence_audits': fixed_sequence_audits,
        'dominance_evidence': dominance_evidence,
    }
    _atomic_write_json(checkpoint_directory / 'complete.json', checkpoint_metadata)
    return checkpoint_directory


def _save_failure_record(
    output_directory,
    spec,
    customer_size,
    instance_index,
    failure_stage,
    failed_variant,
    execution_position,
    error,
):
    """
    立即保存一个被跳过实验单元的异常信息。

    输入：实例标识、失败阶段/变体、执行位置和异常对象。
    输出：用于 CSV 与汇总的紧凑失败记录。
    逻辑：完整堆栈写入独立 JSON，汇总只保留短字段和相对路径。
    """

    failure_path = (
        Path(output_directory)
        / 'failures'
        / spec.result_stem
        / f'customers-{int(customer_size)}'
        / f'instance-{int(instance_index):04d}.json'
    )
    compact_record = {
        'dataset': spec.dataset_label,
        'result_stem': spec.result_stem,
        'customer_size': int(customer_size),
        'instance_index': int(instance_index),
        'failure_stage': failure_stage,
        'failed_variant': failed_variant or '',
        'execution_position': (
            '' if execution_position is None else int(execution_position)
        ),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'failed_at': datetime.now().isoformat(timespec='seconds'),
        'record_path': str(failure_path.relative_to(output_directory)),
    }
    detail_record = {
        **compact_record,
        'traceback': traceback.format_exc(),
    }
    _atomic_write_json(failure_path, detail_record)
    return compact_record


def _format_optional_metric(value):
    """
    格式化可能因零个成功实例而不存在的汇总指标。

    输入：浮点数或 `None`。
    输出：六位小数字符串，或不可用标记。
    逻辑：避免全部失败时 README 的数值格式化再次抛出异常。
    """

    return 'N/A' if value is None else f'{float(value):.6f}'


def run_p7_endpoint_dominance_comparison(
    road_specs: Sequence,
    options: Optional[P7ComparisonOptions] = None,
):
    """
    在指定路网配置上运行 A/B/C 三组 P7 配对实验并保存完整结果。

    输入：路网配置序列和可选比较配置。
    输出：本次时间戳结果目录。
    逻辑：每张地图只初始化一次距离；每个实例共享输入并轮换执行顺序，随后执行 B/C 目标一致性审计。
    """

    options = options or P7ComparisonOptions()
    options.validate()
    if not road_specs:
        raise ValueError('P7 比较实验至少需要一档路网配置。')

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_root = Path(options.output_root or (RESULTS_DIR / 'pruning' / 'p7'))
    output_directory = output_root / timestamp
    output_directory.mkdir(parents=True, exist_ok=True)

    instance_rows = []
    comparison_rows = []
    fixed_sequence_audits = []
    dominance_evidence = []
    failure_rows = []
    attempted_instance_count = 0

    # 先创建空进度汇总；即使首个实例前发生外部中断，本轮输出目录也有明确状态。
    summary = _write_progress_outputs(
        output_directory,
        timestamp,
        options,
        instance_rows,
        comparison_rows,
        fixed_sequence_audits,
        dominance_evidence,
        failure_rows,
        attempted_instance_count,
    )

    for spec in road_specs:
        graph = None
        distance = None
        saved_results = None
        try:
            print(f'Preparing P7 comparison road data for {spec.dataset_label}.')
            graph, distance, distance_stats = prepare_manhattan_road_network(spec.graph_path)
            print(
                f'{spec.dataset_label} distance initialization: '
                f'{distance_stats["distance_initialization_seconds"]:.3f}s.'
            )
            for customer_size in options.customer_sizes:
                depots_by_instance, cities_by_instance = sample_multiagent_instances(
                    graph,
                    options.instance_count,
                    spec.depot_count,
                    int(customer_size),
                    seed=options.instance_seed,
                )
                saved_results = {variant: [] for variant in COMPARISON_VARIANTS}
                saved_depots = []
                saved_cities = []
                successful_instance_indices = []
                for instance_index in range(options.instance_count):
                    attempted_instance_count += 1
                    execution_order = _variant_order(instance_index)
                    print(
                        f'P7 comparison {spec.dataset_label}, customers={customer_size}, '
                        f'instance={instance_index + 1}/{options.instance_count}, '
                        f'order={execution_order}.'
                    )
                    results = {}
                    failure_stage = 'variant_solve'
                    failed_variant = ''
                    execution_position = None
                    try:
                        try:
                            for execution_position, variant in enumerate(execution_order):
                                failed_variant = variant
                                results[variant] = _run_variant(
                                    variant,
                                    execution_position,
                                    graph,
                                    depots_by_instance[instance_index],
                                    cities_by_instance[instance_index],
                                    distance,
                                    spec.drone_count,
                                    options,
                                )

                            failure_stage = 'sparse_audit'
                            failed_variant = ''
                            execution_position = None
                            sparse_audit = _audit_sparse_pair(
                                results[SPARSE_BASELINE],
                                results[SPARSE_P7],
                                options.objective_tolerance,
                            )

                            failure_stage = 'metrics_build'
                            local_fixed_sequence_audits = [
                                {
                                    'dataset': spec.dataset_label,
                                    'customer_size': int(customer_size),
                                    'instance_index': int(instance_index),
                                    **audit,
                                }
                                for audit in sparse_audit['fixed_sequence_audits']
                            ]
                            local_instance_rows = [
                                _build_instance_row(
                                    spec.dataset_label,
                                    customer_size,
                                    instance_index,
                                    results[variant],
                                )
                                for variant in COMPARISON_VARIANTS
                            ]
                            local_comparison_row = _build_comparison_row(
                                spec.dataset_label,
                                customer_size,
                                instance_index,
                                results,
                                sparse_audit,
                            )
                            local_dominance_evidence = []
                            if options.record_evidence:
                                for record in results[SPARSE_P7]['process_data']['depot_records']:
                                    report = record.get('pruning')
                                    if report is None:
                                        continue
                                    for evidence in report['evidence']:
                                        local_dominance_evidence.append({
                                            'dataset': spec.dataset_label,
                                            'customer_size': int(customer_size),
                                            'instance_index': int(instance_index),
                                            'depot_node': record['depot_node'],
                                            **evidence,
                                        })
                        except Exception as error:
                            # 求解、审计或指标构造失败只淘汰当前配对实例，已提交检查点不受影响。
                            failure_row = _save_failure_record(
                                output_directory,
                                spec,
                                customer_size,
                                instance_index,
                                failure_stage,
                                failed_variant,
                                execution_position,
                                error,
                            )
                            failure_rows.append(failure_row)
                            summary = _write_progress_outputs(
                                output_directory,
                                timestamp,
                                options,
                                instance_rows,
                                comparison_rows,
                                fixed_sequence_audits,
                                dominance_evidence,
                                failure_rows,
                                attempted_instance_count,
                            )
                            print(
                                f'Skipping failed P7 pair {spec.dataset_label}, '
                                f'customers={customer_size}, instance={instance_index}: '
                                f'{type(error).__name__}: {error}'
                            )
                            continue

                        # 保存位于异常跳过块之外；持久化失败必须终止，不能伪装成普通实验失败。
                        checkpoint_directory = _save_instance_checkpoint(
                            output_directory,
                            spec,
                            customer_size,
                            instance_index,
                            results,
                            depots_by_instance[instance_index],
                            cities_by_instance[instance_index],
                            distance,
                            distance_stats,
                            local_instance_rows,
                            local_comparison_row,
                            local_fixed_sequence_audits,
                            local_dominance_evidence,
                        )
                        for variant in COMPARISON_VARIANTS:
                            result = results[variant]
                            saved_results[variant].append((
                                result['cost'],
                                result['solve_seconds'],
                                result['solution'],
                                result['process_data'],
                            ))
                        saved_depots.append(depots_by_instance[instance_index])
                        saved_cities.append(cities_by_instance[instance_index])
                        successful_instance_indices.append(int(instance_index))
                        instance_rows.extend(local_instance_rows)
                        comparison_rows.append(local_comparison_row)
                        fixed_sequence_audits.extend(local_fixed_sequence_audits)
                        dominance_evidence.extend(local_dominance_evidence)
                        summary = _write_progress_outputs(
                            output_directory,
                            timestamp,
                            options,
                            instance_rows,
                            comparison_rows,
                            fixed_sequence_audits,
                            dominance_evidence,
                            failure_rows,
                            attempted_instance_count,
                        )
                        print(f'Saved completed P7 pair checkpoint: {checkpoint_directory}')
                    finally:
                        # 无论成功、跳过还是外部中断，都及时释放已构造的三份模型。
                        results = None
                        gc.collect()

                if successful_instance_indices:
                    # 正常完成当前规模后生成兼容旧消费者的成功实例批量 NPZ。
                    for variant in COMPARISON_VARIANTS:
                        _save_stsp_batch_result(
                            output_directory
                            / f'{spec.result_stem}-{int(customer_size)}-{variant}.npz',
                            saved_results[variant],
                            saved_depots,
                            saved_cities,
                            distance,
                            spec.drone_count,
                            {'hc': [], 'stsp': [], 'lp': []},
                            {'hc': [], 'stsp': [], 'lp': []},
                            distance_initialization_stats=distance_stats,
                        )
                    _atomic_write_json(
                        output_directory
                        / f'{spec.result_stem}-{int(customer_size)}-manifest.json',
                        {
                            'dataset': spec.dataset_label,
                            'customer_size': int(customer_size),
                            'successful_source_instance_indices': successful_instance_indices,
                            'saved_instance_count': len(successful_instance_indices),
                        },
                    )
                else:
                    print(
                        f'No successful P7 pairs for {spec.dataset_label}, '
                        f'customers={customer_size}; batch NPZ files were not created.'
                    )
        finally:
            # 地图级数据始终释放，单实例异常不会把大距离结构滞留到下一张地图。
            graph = None
            distance = None
            saved_results = None
            gc.collect()

    summary = _write_progress_outputs(
        output_directory,
        timestamp,
        options,
        instance_rows,
        comparison_rows,
        fixed_sequence_audits,
        dominance_evidence,
        failure_rows,
        attempted_instance_count,
    )
    report = summary['summary']
    _atomic_write_text(
        output_directory / 'README.md',
        '# P7 外部端点对支配配对实验\n\n'
        '本目录同时保存原始密集模型、稀疏未剪模型和稀疏 P7 模型的配对结果。\n\n'
        f'- 已尝试实例数：{report["attempted_instance_count"]}\n'
        f'- 成功实例数：{report["succeeded_instance_count"]}\n'
        f'- 失败并跳过实例数：{report["failed_instance_count"]}\n'
        f'- 平均端点对剪枝率：'
        f'{_format_optional_metric(report["endpoint_pair_pruning_rate_mean"])}\n'
        f'- Phase 2 平均加速比（P7/稀疏基线）：'
        f'{_format_optional_metric(report["phase2_speedup_mean"])}\n'
        f'- 预算耗尽实例数：{report["budget_exhausted_count"]}\n'
        f'- 安全回退实例数：{report["safe_fallback_count"]}\n\n'
        '`checkpoints/` 保存逐个完成的配对实例；'
        '`failures/` 和 `failures.csv` 保存被跳过实例的错误信息；'
        '`paired_comparisons.csv` 是 P7 独立效果的主要比较表。\n',
    )

    print(f'P7 comparison completed. Results: {output_directory}')
    return output_directory
