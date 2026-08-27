"""
本文件统一管理 MA-FSTSP 实验结果的采集、校验、序列化与恢复。

主要内容：
1. 按论文三阶段流程求解单个模型并记录紧凑过程数据。
2. 保存全部实例的最终路线与 Phase 1/2/3 摘要。
3. 为 best、median、worst 代表实例生成详细一致性校验。
4. 使用不依赖 pickle 的压缩 NPZ 格式写入和恢复实验结果。

本模块不导入 `experiments.py` 或 `plot.py`，也不依赖任何绘图库，确保实验子进程
可以轻量加载，并避免实验入口与绘图入口之间形成循环导入。
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.fstsp import InstanceTimeLimitExceeded
from src.partition import calculate_set_tsp_model_size, normalize_candidate_sets
from utils import ensure_dir


def _save_npz(path, **arrays):
    """
    将一批实验数组压缩保存到 NPZ 文件。

    输入：
    - path: 目标文件路径。
    - arrays: 需要保存的命名数组。

    输出：无；负责创建父目录并写入压缩结果文件。
    """
    ensure_dir(Path(path).parent)
    np.savez_compressed(path, **arrays)


def _jsonable(value):
    """
    将 NumPy 标量、数组和嵌套容器转换为 JSON 可序列化对象。

    输入：任意实验数据对象。
    输出：仅包含 Python 基础类型、列表和字典的等价对象。
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _json_text(value):
    """
    将实验记录编码为紧凑 JSON 文本。

    输入：可由 `_jsonable` 转换的实验记录。
    输出：不依赖 pickle 的 Unicode JSON 字符串。
    """
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(',', ':'),
        allow_nan=False,
    )


def _json_array(records):
    """
    将一组变长实验记录编码为可直接写入 NPZ 的字符串数组。

    输入：记录序列。
    输出：一维 NumPy Unicode 数组；加载时无需 `allow_pickle=True`。
    """
    return np.asarray([_json_text(record) for record in records], dtype=np.str_)


def _solve_model_with_process_data(
    model,
    candidate_sets=None,
    candidate_set_seconds=None,
    instance_time_limit=None,
):
    """
    按论文三阶段流程求解一个实例，并同步采集可复核的紧凑过程数据。

    输入：
    - model: 已构造的 `MultiAgentFlyingSidekickTSP` 模型。

    输出：
    - `(solution, cost, process_data)`：最终联合路线、目标值和三阶段记录。

    实现逻辑：
    1. 分别计时候选边界集合构造和配置的客户分组方法。
    2. 对每个仓库记录 Set-TSP 顺序、访问顺序和求解耗时。
    3. 记录 Phase 3 的目标贡献、耗时及最终卡车/无人机路线。

    这里只保存阶段输出与最终选中路径，不保留完整 DP 状态表，避免大实例结果膨胀。
    """
    total_start = time.perf_counter()
    model.solution = []
    model.cost = 0

    # Phase 1a：构造客户候选区域的边界点集合。
    if candidate_sets is None:
        start = time.perf_counter()
        raw_sets = model.get_boundary_convex_sets(model.theta[0])
        boundary_seconds = time.perf_counter() - start
    else:
        # 配对运行器只计算一次候选集；各方法仍记录相同公共耗时以保持总耗时口径。
        raw_sets = candidate_sets
        boundary_seconds = float(candidate_set_seconds or 0.0)
    # 第一阶段和第二阶段共用正规化集合，避免空边界造成 Set-TSP 空集合。
    convex_sets = normalize_candidate_sets(model.cities, raw_sets)

    # Phase 1b：按模型配置执行原 SMST 或 Directed Set-GTDS 客户划分。
    start = time.perf_counter()
    model.partition_customers(convex_sets)
    measured_partition_seconds = time.perf_counter() - start
    partition_seconds = (
        float(model.partition_result.phase1_time)
        if model.precomputed_partition_result is not None
        else measured_partition_seconds
    )
    partition_diagnostics = (
        model.partition_result.diagnostics()
        if model.partition_result is not None
        else {}
    )

    def remaining_instance_seconds(current_record=None):
        """返回实例逻辑剩余秒数；可计入尚未追加的当前仓库记录。"""

        if instance_time_limit is None:
            return None
        consumed = (
            boundary_seconds
            + partition_seconds
            + sum(record['set_tsp_seconds'] for record in depot_records)
            + sum(record['local_search_seconds'] for record in depot_records)
        )
        if current_record is not None:
            consumed += (
                current_record['set_tsp_seconds']
                + current_record['local_search_seconds']
            )
        return max(0.0, float(instance_time_limit) - consumed)

    # 边界集合只记录规模，不保存可能很大的完整节点列表。
    boundary_set_sizes = {
        city: len(convex_sets[city])
        for city in model.cities
    }
    group_records = [
        {
            'depot_index': depot_index,
            'depot_node': depot,
            'customers': list(model.groups[depot]),
        }
        for depot_index, depot in enumerate(model.depots)
    ]

    depot_records = []
    for depot_index, depot in enumerate(model.depots):
        # 当前仓库的客户及候选集合用于 Phase 2 和 Phase 3。
        group = list(model.groups[depot])
        local_convex_sets = [[depot]] + [convex_sets[city] for city in group]
        record = {
            'depot_index': depot_index,
            'depot_node': depot,
            'customers': group,
            'convex_set_sizes': [len(nodes) for nodes in local_convex_sets],
            'set_tsp_solver': 'none',
            'set_tsp_sequence': [],
            'visit_route': [depot, depot],
            'objective_contribution': 0.0,
            'set_tsp_seconds': 0.0,
            'local_search_seconds': 0.0,
            'estimated_q_bin': 0,
            'estimated_q_var': 0,
            'estimated_q_con': 0,
            'set_tsp_status': 'not_required',
            'set_tsp_build_seconds': 0.0,
            'set_tsp_optimize_seconds': 0.0,
            'actual_num_bin_vars': 0,
            'actual_num_vars': 0,
            'actual_num_constraints': 0,
            'gurobi_node_count': None,
            'gurobi_mip_gap': None,
            'gurobi_solution_count': 0,
            'time_limit_reached': False,
            'has_incumbent': True,
            'set_tsp_error': None,
            'set_tsp_objective': None,
            'phase3_completed': len(group) == 0,
            'phase3_skipped_reason': None,
            'instance_time_limit_reached': False,
        }

        if len(group) == 0:
            raw_solution = {'truck': [depot, depot], 'drone': []}
        else:
            remaining_seconds = remaining_instance_seconds(record)
            if remaining_seconds is not None and remaining_seconds <= 0:
                # 前序阶段已耗尽总预算；仍写入完整仓库记录，便于统计超时率。
                record.update({
                    'set_tsp_status': 'instance_time_limit',
                    'time_limit_reached': True,
                    'has_incumbent': False,
                    'set_tsp_error': '实例级时间预算在 Phase 2 前已耗尽',
                    'phase3_skipped_reason': 'instance_time_limit',
                    'instance_time_limit_reached': True,
                })
                raw_solution = {'truck': [depot, depot], 'drone': []}
                model.solution.append(model.convert(raw_solution))
                depot_records.append(record)
                continue

            model_sizes = calculate_set_tsp_model_size(
                [len(nodes) for nodes in local_convex_sets]
            )
            record['estimated_q_bin'] = model_sizes[0]
            record['estimated_q_var'] = model_sizes[1]
            record['estimated_q_con'] = model_sizes[2]
            # Phase 2：求集合 TSP 顺序，再转换为实际客户节点访问顺序。
            record['set_tsp_solver'] = 'LKH' if model.theta[1] == 0 else 'Set-TSP'
            phase2_time_limit = model.set_tsp_time_limit
            if remaining_seconds is not None:
                phase2_time_limit = (
                    remaining_seconds
                    if phase2_time_limit is None
                    else min(float(phase2_time_limit), remaining_seconds)
                )
            if phase2_time_limit is None:
                solve_result = model.get_seq_result(depot, local_convex_sets)
            else:
                solve_result = model.get_seq_result(
                    depot,
                    local_convex_sets,
                    time_limit=phase2_time_limit,
                )
            sequence = solve_result.sequence
            record.update({
                'set_tsp_status': solve_result.status,
                'set_tsp_build_seconds': solve_result.build_seconds,
                'set_tsp_optimize_seconds': solve_result.optimize_seconds,
                'set_tsp_seconds': (
                    solve_result.build_seconds + solve_result.optimize_seconds
                ),
                'actual_num_bin_vars': solve_result.num_bin_vars,
                'actual_num_vars': solve_result.num_vars,
                'actual_num_constraints': solve_result.num_constraints,
                'gurobi_node_count': solve_result.node_count,
                'gurobi_mip_gap': solve_result.mip_gap,
                'gurobi_solution_count': solve_result.solution_count,
                'time_limit_reached': solve_result.time_limit_reached,
                'has_incumbent': solve_result.has_incumbent,
                'set_tsp_error': solve_result.error_message,
                'set_tsp_objective': solve_result.objective,
            })
            if sequence is None:
                # 无 incumbent 时不能进入 Phase 3；保留占位路线并继续处理其他仓库，
                # 从而让整批实验能够计算失败率，而不是由单个实例终止。
                record['phase3_skipped_reason'] = 'set_tsp_without_incumbent'
                raw_solution = {'truck': [depot, depot], 'drone': []}
                converted_solution = model.convert(raw_solution)
                model.solution.append(converted_solution)
                depot_records.append(record)
                continue
            visit_route = [depot] + [group[index - 1] for index in sequence[1:-1]] + [depot]
            record['set_tsp_sequence'] = list(sequence)
            record['visit_route'] = visit_route

            # Phase 3：在固定访问顺序下运行 DP，并保留最终选中的联合路线。
            start = time.perf_counter()
            remaining_seconds = remaining_instance_seconds(record)
            if remaining_seconds is not None and remaining_seconds <= 0:
                record.update({
                    'phase3_skipped_reason': 'instance_time_limit',
                    'instance_time_limit_reached': True,
                })
                raw_solution = {'truck': [depot, depot], 'drone': []}
                model.solution.append(model.convert(raw_solution))
                depot_records.append(record)
                continue
            phase3_deadline = (
                None
                if remaining_seconds is None
                else time.perf_counter() + remaining_seconds
            )
            try:
                if phase3_deadline is None:
                    raw_solution, contribution = model.local_search_multi_drone_appr(
                        visit_route,
                        depot,
                    )
                else:
                    raw_solution, contribution = model.local_search_multi_drone_appr(
                        visit_route,
                        depot,
                        deadline=phase3_deadline,
                    )
            except InstanceTimeLimitExceeded:
                record['local_search_seconds'] = time.perf_counter() - start
                record.update({
                    'phase3_skipped_reason': 'instance_time_limit',
                    'instance_time_limit_reached': True,
                })
                raw_solution = {'truck': [depot, depot], 'drone': []}
                model.solution.append(model.convert(raw_solution))
                depot_records.append(record)
                continue
            record['local_search_seconds'] = time.perf_counter() - start
            record['objective_contribution'] = float(contribution)
            record['phase3_completed'] = True
            model.cost += contribution

        converted_solution = model.convert(raw_solution)
        model.solution.append(converted_solution)
        depot_records.append(record)

    phase2_failed_depots = [
        record['depot_index']
        for record in depot_records
        if record.get('customers') and not record.get('has_incumbent', False)
    ]
    phase3_failed_depots = [
        record['depot_index']
        for record in depot_records
        if record.get('customers')
        and record.get('has_incumbent', False)
        and not record.get('phase3_completed', False)
    ]
    failed_depots = [
        record['depot_index']
        for record in depot_records
        if record.get('customers') and (
            not record.get('has_incumbent', False)
            or not record.get('phase3_completed', False)
        )
    ]
    instance_time_limit_reached = any(
        record.get('instance_time_limit_reached', False)
        for record in depot_records
    )
    logical_solve_seconds = (
        boundary_seconds
        + partition_seconds
        + sum(record['set_tsp_seconds'] for record in depot_records)
        + sum(record['local_search_seconds'] for record in depot_records)
    )
    process_data = {
        'partition_method': model.partition_method,
        'partition_diagnostics': partition_diagnostics,
        'boundary_set_sizes': boundary_set_sizes,
        'groups': group_records,
        'depot_records': depot_records,
        'boundary_convex_sets_seconds': boundary_seconds,
        'partition_seconds': partition_seconds,
        # 保留旧键，避免已有绘图和结果读取代码失效。
        'mst_partition_seconds': partition_seconds,
        'solve_seconds': logical_solve_seconds,
        'wall_seconds': time.perf_counter() - total_start,
        'instance_status': (
            'timeout'
            if instance_time_limit_reached
            else ('incomplete' if failed_depots else 'complete')
        ),
        'instance_time_limit': instance_time_limit,
        'instance_time_limit_reached': instance_time_limit_reached,
        'failed_depot_indices': failed_depots,
        'failed_depot_count': len(failed_depots),
        'phase2_failure_count': len(phase2_failed_depots),
        'phase3_failure_count': len(phase3_failed_depots),
    }
    final_cost = float('nan') if failed_depots else float(model.cost)
    return model.solution, final_cost, process_data


def _iter_solution_sorties(route):
    """
    统一遍历一条仓库路线中的无人机任务。

    输入：包含 `truck` 和 `drone` 的统一路线字典。
    输出：依次产生 `(drone_slot, sortie)`；`sortie` 是起飞、客户、回收节点序列。
    """
    for drone_slot, drone_routes in enumerate(route.get('drone', [])):
        if len(drone_routes) == 0:
            continue
        first = drone_routes[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            for sortie in drone_routes:
                if len(sortie) >= 2:
                    yield drone_slot, list(sortie)
        elif len(drone_routes) >= 2:
            yield drone_slot, list(drone_routes)


def _select_representative_indices(costs):
    """
    根据目标函数选择 best、median 和 worst 三类代表实例。

    输入：每个实例的目标函数序列。
    输出：按 `best/median/worst` 排列的 `(角色, 实例编号)` 列表。
    """
    values = np.asarray(costs, dtype=float).reshape(-1)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if len(finite_indices) == 0:
        return []

    # 并列时选择编号更小的实例，使结果选择保持确定性。
    best_index = min(finite_indices, key=lambda index: (values[index], int(index)))
    worst_index = min(finite_indices, key=lambda index: (-values[index], int(index)))
    median_value = float(np.median(values[finite_indices]))
    median_index = min(
        finite_indices,
        key=lambda index: (abs(values[index] - median_value), int(index)),
    )
    return [
        ('best', int(best_index)),
        ('median', int(median_index)),
        ('worst', int(worst_index)),
    ]


def _distance_or_none(distance_table, start, end):
    """
    安全读取距离提供器中的单对距离。

    输入：距离表、起点和终点。
    输出：有限距离的浮点值；查询失败或不可达时返回 `None`。
    """
    try:
        value = float(distance_table[start][end])
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None
    return value if np.isfinite(value) else None


def _build_representative_trace(
    role,
    instance_index,
    result,
    depots,
    cities,
    distance,
    drone_limit,
    drone_speed,
):
    """
    为代表性实例构造路线距离、航程约束和阶段一致性校验摘要。

    输入：代表角色、实例编号、求解结果、实例节点、距离对象及无人机参数。
    输出：可 JSON 序列化的详细追踪字典。
    """
    cost, elapsed, solution, process_data = result
    expected_customers = list(cities)
    assigned_customers = [
        customer
        for group in process_data['groups']
        for customer in group['customers']
    ]
    phase1_partition_valid = Counter(assigned_customers) == Counter(expected_customers)

    route_checks = []
    all_drone_limits_valid = True
    all_truck_routes_reachable = True
    all_service_records_valid = True
    for depot_index, route in enumerate(solution):
        record = process_data['depot_records'][depot_index]
        truck_route = list(route.get('truck', []))
        truck_segments = []
        truck_distance = 0.0
        for start, end in zip(truck_route[:-1], truck_route[1:]):
            segment_distance = _distance_or_none(distance['truck'], start, end)
            reachable = segment_distance is not None
            all_truck_routes_reachable = all_truck_routes_reachable and reachable
            if reachable:
                truck_distance += segment_distance
            truck_segments.append({
                'from': start,
                'to': end,
                'distance': segment_distance,
                'reachable': reachable,
            })

        drone_sorties = []
        drone_customer_counts = Counter()
        for drone_slot, sortie in _iter_solution_sorties(route):
            segment_distances = [
                _distance_or_none(distance['drone'], start, end)
                for start, end in zip(sortie[:-1], sortie[1:])
            ]
            sortie_distance = (
                float(sum(segment_distances))
                if all(value is not None for value in segment_distances)
                else None
            )
            within_limit = sortie_distance is not None and sortie_distance <= drone_limit
            all_drone_limits_valid = all_drone_limits_valid and within_limit
            customer = sortie[1] if len(sortie) >= 3 else None
            if customer is not None:
                drone_customer_counts[customer] += 1
            drone_sorties.append({
                'drone_slot': drone_slot,
                'nodes': sortie,
                'distance': sortie_distance,
                'flight_time': (
                    sortie_distance / drone_speed
                    if sortie_distance is not None and drone_speed
                    else None
                ),
                'within_limit': within_limit,
            })

        # 未由无人机服务的分组客户应当显式出现在卡车关键节点路线中。
        assigned = list(record['customers'])
        expected_truck_customers = [
            customer for customer in assigned
            if drone_customer_counts[customer] == 0
        ]
        missing_truck_customers = [
            customer for customer in expected_truck_customers
            if customer not in truck_route
        ]
        unexpected_drone_customers = [
            customer for customer in drone_customer_counts
            if customer not in assigned
        ]
        duplicate_drone_customers = [
            customer for customer, count in drone_customer_counts.items()
            if count != 1
        ]
        service_record_valid = not (
            missing_truck_customers
            or unexpected_drone_customers
            or duplicate_drone_customers
        )
        all_service_records_valid = all_service_records_valid and service_record_valid

        route_checks.append({
            'depot_index': depot_index,
            'depot_node': depots[depot_index],
            'assigned_customers': assigned,
            'truck': {
                'key_route': truck_route,
                'distance': truck_distance,
                'segments': truck_segments,
            },
            'drone': {
                'sortie_count': len(drone_sorties),
                'sorties': drone_sorties,
            },
            'service_check': {
                'valid': service_record_valid,
                'expected_truck_customers': expected_truck_customers,
                'missing_truck_customers': missing_truck_customers,
                'unexpected_drone_customers': unexpected_drone_customers,
                'duplicate_drone_customers': duplicate_drone_customers,
            },
        })

    phase_cost_sum = float(sum(
        record['objective_contribution']
        for record in process_data['depot_records']
    ))
    objective_error = abs(phase_cost_sum - float(cost))
    return {
        'role': role,
        'instance_index': instance_index,
        'objective_value': float(cost),
        'solve_seconds': float(elapsed),
        'depots': list(depots),
        'customers': expected_customers,
        'phase1': {
            'partition_method': process_data.get('partition_method', 'smst_original'),
            'partition_diagnostics': process_data.get('partition_diagnostics', {}),
            'boundary_set_sizes': process_data['boundary_set_sizes'],
            'boundary_convex_sets_seconds': process_data['boundary_convex_sets_seconds'],
            'partition_seconds': process_data.get(
                'partition_seconds',
                process_data['mst_partition_seconds'],
            ),
            'mst_partition_seconds': process_data['mst_partition_seconds'],
            'partition_valid': phase1_partition_valid,
        },
        'phase2_and_phase3': process_data['depot_records'],
        'route_checks': route_checks,
        'validation': {
            'phase_cost_sum': phase_cost_sum,
            'objective_absolute_error': objective_error,
            'objective_consistent': objective_error <= 1e-9,
            'all_truck_routes_reachable': all_truck_routes_reachable,
            'all_drone_limits_valid': all_drone_limits_valid,
            'all_service_records_valid': all_service_records_valid,
        },
    }


def _build_stsp_result_arrays(
    results,
    depots,
    cities,
    distance,
    drones,
    drone_limit=1.5,
    drone_speed=1.6,
    theta=(0.5, 0.5),
    distance_initialization_stats=None,
):
    """
    将一批 STSP 求解结果转换为紧凑、可直接写入 NPZ 的数组集合。

    输入：全部实例结果、实例节点、距离对象、无人机参数，以及可选的批次级距离初始化耗时。
    输出：包含全部紧凑解及 best/median/worst 详细追踪的字段字典。
    """
    if not (len(results) == len(depots) == len(cities)):
        raise ValueError('results、depots 与 cities 的实例数量必须一致。')
    if len(results) == 0:
        raise ValueError('至少需要一个实例才能保存 STSP 批次结果。')

    costs = np.asarray([result[0] for result in results], dtype=float)
    elapsed = np.asarray([result[1] for result in results], dtype=float)
    solutions = [result[2] for result in results]
    processes = [result[3] for result in results]
    depot_count = len(depots[0])

    # 规则二维数组无需解析 JSON 就能进行阶段耗时和成本统计。
    phase2_seconds = np.zeros((len(results), depot_count), dtype=float)
    phase2_build_seconds = np.zeros((len(results), depot_count), dtype=float)
    phase2_optimize_seconds = np.zeros((len(results), depot_count), dtype=float)
    phase2_actual_bin_vars = np.zeros((len(results), depot_count), dtype=np.int64)
    phase2_actual_vars = np.zeros((len(results), depot_count), dtype=np.int64)
    phase2_actual_constraints = np.zeros((len(results), depot_count), dtype=np.int64)
    phase2_node_count = np.full((len(results), depot_count), np.nan, dtype=float)
    phase2_mip_gap = np.full((len(results), depot_count), np.nan, dtype=float)
    phase2_objective = np.full((len(results), depot_count), np.nan, dtype=float)
    phase2_solution_count = np.zeros((len(results), depot_count), dtype=np.int64)
    phase2_time_limit = np.zeros((len(results), depot_count), dtype=bool)
    phase2_has_incumbent = np.zeros((len(results), depot_count), dtype=bool)
    phase2_status = np.full((len(results), depot_count), 'not_required', dtype='<U32')
    phase3_seconds = np.zeros((len(results), depot_count), dtype=float)
    phase_costs = np.zeros((len(results), depot_count), dtype=float)
    phase1_boundary_seconds = np.zeros(len(results), dtype=float)
    phase1_partition_seconds = np.zeros(len(results), dtype=float)
    phase1_groups = []
    phase2_orders = []
    partition_methods = []
    phase1_diagnostics = []
    instance_statuses = []

    for instance_index, process_data in enumerate(processes):
        records = process_data['depot_records']
        if len(records) != depot_count:
            raise ValueError(
                f'实例 {instance_index} 的仓库过程记录数 {len(records)} 与预期 {depot_count} 不一致。'
            )
        phase1_boundary_seconds[instance_index] = process_data['boundary_convex_sets_seconds']
        phase1_partition_seconds[instance_index] = process_data.get(
            'partition_seconds',
            process_data['mst_partition_seconds'],
        )
        partition_methods.append(process_data.get('partition_method', 'smst_original'))
        phase1_diagnostics.append(process_data.get('partition_diagnostics', {}))
        instance_statuses.append(process_data.get('instance_status', 'complete'))
        phase1_groups.append(process_data['groups'])
        phase2_orders.append([
            {
                'depot_index': record['depot_index'],
                'depot_node': record['depot_node'],
                'customers': record['customers'],
                'convex_set_sizes': record['convex_set_sizes'],
                'set_tsp_solver': record['set_tsp_solver'],
                'set_tsp_sequence': record['set_tsp_sequence'],
                'visit_route': record['visit_route'],
                'estimated_q_bin': record.get('estimated_q_bin', 0),
                'estimated_q_var': record.get('estimated_q_var', 0),
                'estimated_q_con': record.get('estimated_q_con', 0),
                'set_tsp_status': record.get('set_tsp_status', 'unknown'),
                'set_tsp_build_seconds': record.get('set_tsp_build_seconds', 0.0),
                'set_tsp_optimize_seconds': record.get('set_tsp_optimize_seconds', 0.0),
                'actual_num_bin_vars': record.get('actual_num_bin_vars', 0),
                'actual_num_vars': record.get('actual_num_vars', 0),
                'actual_num_constraints': record.get('actual_num_constraints', 0),
                'gurobi_node_count': record.get('gurobi_node_count'),
                'gurobi_mip_gap': record.get('gurobi_mip_gap'),
                'gurobi_solution_count': record.get('gurobi_solution_count', 0),
                'time_limit_reached': record.get('time_limit_reached', False),
                'has_incumbent': record.get('has_incumbent', False),
                'set_tsp_error': record.get('set_tsp_error'),
                'set_tsp_objective': record.get('set_tsp_objective'),
                'phase3_completed': record.get('phase3_completed', False),
                'phase3_skipped_reason': record.get('phase3_skipped_reason'),
                'instance_time_limit_reached': record.get(
                    'instance_time_limit_reached', False
                ),
            }
            for record in records
        ])
        for depot_index, record in enumerate(records):
            phase2_seconds[instance_index, depot_index] = record['set_tsp_seconds']
            phase2_build_seconds[instance_index, depot_index] = record.get(
                'set_tsp_build_seconds', 0.0
            )
            phase2_optimize_seconds[instance_index, depot_index] = record.get(
                'set_tsp_optimize_seconds', 0.0
            )
            phase2_actual_bin_vars[instance_index, depot_index] = record.get(
                'actual_num_bin_vars', 0
            )
            phase2_actual_vars[instance_index, depot_index] = record.get(
                'actual_num_vars', 0
            )
            phase2_actual_constraints[instance_index, depot_index] = record.get(
                'actual_num_constraints', 0
            )
            if record.get('gurobi_node_count') is not None:
                phase2_node_count[instance_index, depot_index] = record['gurobi_node_count']
            if record.get('gurobi_mip_gap') is not None:
                phase2_mip_gap[instance_index, depot_index] = record['gurobi_mip_gap']
            if record.get('set_tsp_objective') is not None:
                phase2_objective[instance_index, depot_index] = record['set_tsp_objective']
            phase2_solution_count[instance_index, depot_index] = record.get(
                'gurobi_solution_count', 0
            )
            phase2_time_limit[instance_index, depot_index] = record.get(
                'time_limit_reached', False
            )
            phase2_has_incumbent[instance_index, depot_index] = record.get(
                'has_incumbent', False
            )
            phase2_status[instance_index, depot_index] = record.get(
                'set_tsp_status', 'unknown'
            )
            phase3_seconds[instance_index, depot_index] = record['local_search_seconds']
            phase_costs[instance_index, depot_index] = record['objective_contribution']

    representatives = _select_representative_indices(costs)
    representative_traces = [
        _build_representative_trace(
            role,
            instance_index,
            results[instance_index],
            depots[instance_index],
            cities[instance_index],
            distance,
            drone_limit,
            drone_speed,
        )
        for role, instance_index in representatives
    ]

    result_arrays = {
        'result_schema_version': np.asarray(4, dtype=np.int64),
        'instance_indices': np.arange(len(results), dtype=np.int64),
        'depots': np.asarray(depots),
        'cities': np.asarray(cities),
        'drones_per_truck': np.asarray(drones, dtype=np.int64),
        'drone_limit': np.asarray(drone_limit, dtype=float),
        'drone_speed': np.asarray(drone_speed, dtype=float),
        'theta': np.asarray(theta, dtype=float),
        'stsp_cost': costs,
        'stsp_time': elapsed,
        'solutions_json': _json_array(solutions),
        'phase1_groups_json': _json_array(phase1_groups),
        'partition_methods': np.asarray(partition_methods, dtype=np.str_),
        'instance_status': np.asarray(instance_statuses, dtype=np.str_),
        'phase1_diagnostics_json': _json_array(phase1_diagnostics),
        'phase2_orders_json': _json_array(phase2_orders),
        'phase1_boundary_time': phase1_boundary_seconds,
        'phase1_partition_time': phase1_partition_seconds,
        'phase2_time': phase2_seconds,
        'phase2_build_time': phase2_build_seconds,
        'phase2_optimize_time': phase2_optimize_seconds,
        'phase2_actual_num_bin_vars': phase2_actual_bin_vars,
        'phase2_actual_num_vars': phase2_actual_vars,
        'phase2_actual_num_constraints': phase2_actual_constraints,
        'phase2_node_count': phase2_node_count,
        'phase2_mip_gap': phase2_mip_gap,
        'phase2_objective': phase2_objective,
        'phase2_solution_count': phase2_solution_count,
        'phase2_time_limit_reached': phase2_time_limit,
        'phase2_has_incumbent': phase2_has_incumbent,
        'phase2_status': phase2_status,
        'phase3_time': phase3_seconds,
        'phase_costs': phase_costs,
        'representative_roles': np.asarray(
            [role for role, _ in representatives], dtype=np.str_
        ),
        'representative_indices': np.asarray(
            [index for _, index in representatives], dtype=np.int64
        ),
        'representative_traces_json': _json_array(representative_traces),
    }

    if distance_initialization_stats is not None:
        # 三个标量描述整批实例共享的一次距离准备，不能扩展成逐实例数组。
        required_fields = (
            'truck_apsp_seconds',
            'drone_pairwise_seconds',
            'distance_initialization_seconds',
        )
        missing_fields = [
            field for field in required_fields
            if field not in distance_initialization_stats
        ]
        if missing_fields:
            raise ValueError(f'距离初始化统计缺少字段：{missing_fields}')
        for field in required_fields:
            value = float(distance_initialization_stats[field])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'距离初始化统计 {field} 必须是非负有限数，收到 {value!r}。')
            result_arrays[field] = np.asarray(value, dtype=float)

    return result_arrays


def _save_stsp_batch_result(
    path,
    results,
    depots,
    cities,
    distance,
    drones,
    costs,
    times,
    drone_limit=1.5,
    drone_speed=1.6,
    theta=(0.5, 0.5),
    distance_initialization_stats=None,
):
    """
    保存兼容旧统计字段并包含新路线/过程字段的一批 STSP 结果。

    输入：输出路径、实例求解结果、实例输入、距离对象、参数、旧成本/耗时表，
    以及可选的批次级距离初始化耗时。
    输出：写入后的 `Path`，便于调用者或测试定位本批次文件。
    """
    detail_arrays = _build_stsp_result_arrays(
        results,
        depots,
        cities,
        distance,
        drones,
        drone_limit=drone_limit,
        drone_speed=drone_speed,
        theta=theta,
        distance_initialization_stats=distance_initialization_stats,
    )
    # HC 字段继续保留，保证旧统计消费者仍能识别当前结果文件。
    _save_npz(
        path,
        hc_cost=np.asarray(costs['hc']),
        hc_time=np.asarray(times['hc']),
        **detail_arrays,
    )
    return Path(path)


def _decode_saved_json(data, field, instance_index):
    """
    解码新格式 NPZ 中某个实例对应的 JSON 字符串字段。

    输入：已打开的 NPZ、字段名和实例编号。
    输出：解码后的 Python 对象；字段不存在或编号越界时返回 `None`。
    """
    if field not in data.files:
        return None
    values = np.asarray(data[field]).reshape(-1)
    if instance_index >= len(values):
        return None
    return json.loads(str(values[instance_index]))


def _default_saved_instance_index(data, costs):
    """
    为未显式指定编号的新格式批次选择中位目标值代表实例。

    输入：已打开的 NPZ 和一维目标函数数组。
    输出：代表性实例编号；优先采用保存时记录的 `median` 角色。
    """
    if 'representative_roles' in data.files and 'representative_indices' in data.files:
        roles = [str(role) for role in np.asarray(data['representative_roles']).reshape(-1)]
        indices = np.asarray(data['representative_indices'], dtype=int).reshape(-1)
        if 'median' in roles:
            return int(indices[roles.index('median')])
    finite_indices = np.flatnonzero(np.isfinite(costs))
    if len(finite_indices) == 0:
        return 0
    median_value = float(np.median(costs[finite_indices]))
    return int(min(
        finite_indices,
        key=lambda index: (abs(costs[index] - median_value), int(index)),
    ))


def _load_large_road_saved_result(result_file, instance_index=None):
    """
    读取一批路网实验，并返回指定实例或默认中位代表实例的保存结果。

    输入：NPZ 路径和可选实例编号。
    输出：`(saved, count)`；新格式额外包含实例输入、最终解和三阶段过程数据。
    """
    data = np.load(result_file, allow_pickle=False)
    try:
        if 'stsp_cost' not in data.files:
            print(f"Skipping {result_file}: missing 'stsp_cost'.")
            return None, 0
        costs = np.asarray(data['stsp_cost']).reshape(-1)
        if instance_index is None:
            instance_index = _default_saved_instance_index(data, costs)
        if instance_index >= len(costs):
            print(
                f'Skipping {result_file}: instance_index={instance_index} '
                f'is outside saved result count {len(costs)}.'
            )
            return None, len(costs)
        saved = {
            'instance_index': int(instance_index),
            'cost': float(costs[instance_index]),
        }
        if 'stsp_time' in data.files:
            times = np.asarray(data['stsp_time']).reshape(-1)
            if instance_index < len(times):
                saved['time'] = float(times[instance_index])

        # 新格式字段存在时直接恢复实验输入、最终路线和阶段记录，不再重新求解。
        if all(field in data.files for field in ('depots', 'cities', 'solutions_json')):
            saved['depots'] = np.asarray(data['depots'])[instance_index].tolist()
            saved['cities'] = np.asarray(data['cities'])[instance_index].tolist()
            saved['solution'] = _decode_saved_json(data, 'solutions_json', instance_index)
            saved['phase1_groups'] = _decode_saved_json(
                data, 'phase1_groups_json', instance_index
            )
            if 'partition_methods' in data.files:
                saved['partition_method'] = str(
                    np.asarray(data['partition_methods']).reshape(-1)[instance_index]
                )
            if 'instance_status' in data.files:
                saved['instance_status'] = str(
                    np.asarray(data['instance_status']).reshape(-1)[instance_index]
                )
            saved['phase1_diagnostics'] = _decode_saved_json(
                data,
                'phase1_diagnostics_json',
                instance_index,
            )
            saved['phase2_orders'] = _decode_saved_json(
                data, 'phase2_orders_json', instance_index
            )
            saved['drones_per_truck'] = int(np.asarray(data['drones_per_truck']).item())
            saved['limit'] = float(np.asarray(data['drone_limit']).item())
            saved['speed'] = float(np.asarray(data['drone_speed']).item())
            if 'theta' in data.files:
                saved['theta'] = np.asarray(data['theta'], dtype=float).tolist()
            saved['phase1_boundary_time'] = float(
                np.asarray(data['phase1_boundary_time']).reshape(-1)[instance_index]
            )
            saved['phase1_partition_time'] = float(
                np.asarray(data['phase1_partition_time']).reshape(-1)[instance_index]
            )
            saved['phase2_time'] = np.asarray(data['phase2_time'])[instance_index].tolist()
            # Schema v4 遥测字段按存在性读取，继续兼容 v3 及更早结果。
            phase2_optional_fields = (
                'phase2_build_time',
                'phase2_optimize_time',
                'phase2_actual_num_bin_vars',
                'phase2_actual_num_vars',
                'phase2_actual_num_constraints',
                'phase2_node_count',
                'phase2_mip_gap',
                'phase2_objective',
                'phase2_solution_count',
                'phase2_time_limit_reached',
                'phase2_has_incumbent',
                'phase2_status',
            )
            for field in phase2_optional_fields:
                if field in data.files:
                    saved[field] = np.asarray(data[field])[instance_index].tolist()
            saved['phase3_time'] = np.asarray(data['phase3_time'])[instance_index].tolist()
            saved['phase_costs'] = np.asarray(data['phase_costs'])[instance_index].tolist()
        return saved, len(costs)
    finally:
        data.close()
