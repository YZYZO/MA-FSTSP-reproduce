"""固定分区的统一第二、第三阶段评价及完整流程计时。"""

from collections import Counter
import math
import time

from .candidates import generate_candidates, symmetric_mst
from .features import FeatureContext, canonical_partition
from .selector import select_candidate
from .settings import RepairOptions, SolverOptions


def fixed_boundary(model):
    """输入完整模型，按全局客户生成一次边界并稳定排序；分组改变时不重建。"""
    boundary = model.get_boundary_convex_sets(model.theta[0])
    return {int(city): sorted(map(int, nodes)) for city, nodes in boundary.items()}


def evaluate_group(model, depot, customers, boundary, solver_options=None):
    """输入指定仓库组和固定全局边界，输出顺序、路线、最终成本与完整阶段时间；不修改模型。"""
    phase2_start = time.perf_counter()
    group = sorted(map(int, customers))
    local_sets = [[int(depot)]] + [boundary[city] for city in group]
    input_seconds = time.perf_counter() - phase2_start
    sequence, info = model.get_seq(
        depot, local_sets, cities=group, solver_options=solver_options,
        return_info=True,
    )
    extract_start = time.perf_counter()
    visit_route = [int(depot)] + [group[index - 1] for index in sequence[1:-1]] + [int(depot)]
    info['phase2_extract_seconds'] += time.perf_counter() - extract_start
    phase2_seconds = time.perf_counter() - phase2_start
    phase3_start = time.perf_counter()
    if group:
        raw_solution, contribution = model.local_search_multi_drone_appr(visit_route, depot)
    else:
        raw_solution, contribution = {'truck': [int(depot), int(depot)], 'drone': []}, 0.0
    phase3_seconds = time.perf_counter() - phase3_start
    complete = math.isfinite(contribution)
    record = dict(info)
    record.update({
        'depot_node': int(depot), 'customers': group,
        'convex_set_sizes': [len(nodes) for nodes in local_sets],
        'set_tsp_sequence': list(sequence), 'visit_route': visit_route,
        'phase2_input_seconds': input_seconds, 'phase2_wall_seconds': phase2_seconds,
        'phase3_seconds': phase3_seconds,
        'final_delivery_cost': float(contribution) if complete else None,
        'complete': complete,
        # 旧结果记录字段与新的完整计时指向同一次实际评价。
        'set_tsp_seconds': phase2_seconds, 'local_search_seconds': phase3_seconds,
        'objective_contribution': float(contribution) if complete else None,
        'solution': model.convert(raw_solution),
    })
    component_sum = sum(record[f'phase2_{part}_seconds'] for part in (
        'input', 'distance', 'build', 'optimize', 'extract', 'fallback',
    ))
    record['phase2_other_seconds'] = max(0.0, phase2_seconds - component_sum)
    return record


def summarize_groups(records):
    """输入串行组评价结果，汇总实际或离线复用的下游时间与成本，并保留未完成标志。"""
    complete = all(record.get('complete', False) for record in records)
    return {
        'complete': complete,
        'final_delivery_cost': sum(r['final_delivery_cost'] for r in records) if complete else None,
        'phase2_wall_seconds': sum(r.get('phase2_wall_seconds', 0.0) for r in records),
        'phase3_seconds': sum(r.get('phase3_seconds', 0.0) for r in records),
        'timeout_groups': sum(bool(r.get('timeout')) for r in records),
        'fallback_groups': sum(bool(r.get('fallback_used')) for r in records),
    }


def evaluate_partition(model, partition, boundary, solver_options=None, group_provider=None):
    """输入完整分区，评价全部仓库；可注入离线组缓存提供者，返回下游汇总与各组记录。"""
    groups = canonical_partition(partition, model.depots)
    if Counter(c for members in groups.values() for c in members) != Counter(map(int, model.cities)):
        raise ValueError('分区必须让每个客户恰好归属一次。')
    records = []
    for index, depot in enumerate(map(int, model.depots)):
        record = (group_provider(depot, groups[depot]) if group_provider is not None else
                  evaluate_group(model, depot, groups[depot], boundary, solver_options))
        records.append(dict(record, depot_index=index))
    return dict(summarize_groups(records), depot_records=records)


def solve_with_records(model, *, partition=None, partition_strategy='original_mst', solver_options=None, repair_options=None):
    """输入模型和显式分区/选择策略，输出路线、成本、过程记录；普通 solve 与实验包装共用。"""
    total_start = time.perf_counter()
    solver_options = solver_options or SolverOptions()
    repair_options = repair_options or RepairOptions()
    start = time.perf_counter()
    boundary = fixed_boundary(model)
    boundary_seconds = time.perf_counter() - start
    partition_seconds = feature_seconds = repair_seconds = selection_seconds = 0.0
    selected_name = 'explicit'
    if partition is None:
        start = time.perf_counter()
        if partition_strategy == 'original_mst':
            model.set_mst(boundary)
            partition = canonical_partition(model.groups, model.depots)
            selected_name = 'original_mst'
        else:
            partition = symmetric_mst(model, boundary)
            selected_name = 'stay'
        partition_seconds = time.perf_counter() - start
        if partition_strategy not in ('original_mst', 'symmetric_mst'):
            start = time.perf_counter()
            context = FeatureContext(model, boundary)
            feature_seconds = time.perf_counter() - start
            start = time.perf_counter()
            candidates = generate_candidates(context, partition, repair_options)
            repair_seconds = time.perf_counter() - start
            # 候选生成中按需计算的组特征单独归入特征时间，避免与修复时间重叠。
            feature_seconds += context.compute_seconds
            repair_seconds = max(0.0, repair_seconds - context.compute_seconds)
            start = time.perf_counter()
            computed_before_selection = context.compute_seconds
            selected = select_candidate(context, partition, candidates, partition_strategy,
                                        repair_options.geometry_weight, solver_options.seed)
            selection_seconds = time.perf_counter() - start
            feature_delta = context.compute_seconds - computed_before_selection
            feature_seconds += feature_delta
            selection_seconds = max(0.0, selection_seconds - feature_delta)
            partition, selected_name = selected.partition, selected.name
    groups = canonical_partition(partition, model.depots)
    evaluation = evaluate_partition(model, groups, boundary, solver_options)
    if not evaluation['complete']:
        raise RuntimeError('第三阶段未产生完整有限配送成本。')
    model.groups = groups
    model.solution = [record['solution'] for record in evaluation['depot_records']]
    model.cost = evaluation['final_delivery_cost']
    process = {
        'boundary_set_sizes': {city: len(nodes) for city, nodes in boundary.items()},
        'groups': [{'depot_index': i, 'depot_node': int(d), 'customers': groups[d]}
                   for i, d in enumerate(model.depots)],
        'depot_records': evaluation['depot_records'],
        'boundary_convex_sets_seconds': boundary_seconds,
        'mst_partition_seconds': partition_seconds,
        'feature_seconds': feature_seconds, 'repair_seconds': repair_seconds,
        'selection_seconds': selection_seconds, 'selected_candidate': selected_name,
        'partition_strategy': partition_strategy, 'solver_options': solver_options.to_dict(),
        'instance_initialization_seconds': getattr(model, 'initialization_seconds', 0.0),
        'solve_seconds': time.perf_counter() - total_start,
    }
    return model.solution, float(model.cost), process
