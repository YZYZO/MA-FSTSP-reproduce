"""为 experiments.py 中的 MST 改进主流程提供求解与结果序列化辅助。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from experiment_results import _solve_model_with_process_data
from src.fstsp import MultiAgentFlyingSidekickTSP

from .model import ImprovedMSTMultiAgentFlyingSidekickTSP


def json_compatible(value):
    """
    将 NumPy 标量、数组和节点键递归转换为 JSON 兼容对象。

    输入：任意实验记录值。
    输出：仅包含 JSON 原生类型的等价对象。
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_compatible(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {
            str(json_compatible(key)): json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [json_compatible(item) for item in value]
    return value


def solve_variant(
    variant_name,
    graph,
    depots,
    cities,
    distance,
    drones,
    theta,
    improvement_config,
):
    """
    在一个实例上运行原始或改进的 MST 分区变体。

    输入：算法名称、完整问题实例、无人机参数和可选改进配置。
    输出：成本、耗时、客户分组、最终路线和遥测字典。

    每个调用都创建独立模型，避免可变状态跨算法污染。
    """
    if improvement_config is None:
        model = MultiAgentFlyingSidekickTSP(
            graph, depots, cities, distance, drones, theta=theta
        )
        solution, cost, process_data = _solve_model_with_process_data(model)
        telemetry = {
            'boundary_seconds': process_data['boundary_convex_sets_seconds'],
            'initial_partition_seconds': process_data['mst_partition_seconds'],
            'search': {
                'iterations': 0,
                'evaluated_candidates': 0,
                'accepted_relocates': 0,
                'accepted_swaps': 0,
                'elapsed_seconds': 0.0,
                'stop_reason': 'original_algorithm',
            },
            'total_seconds': process_data['solve_seconds'],
            'process_data': process_data,
        }
        # 原算法没有分区后局部改进，搜索前后成本均为最终下游目标值。
        telemetry['partition_cost_before_search'] = float(cost)
        telemetry['partition_cost_after_search'] = float(cost)
        telemetry['final_group_costs'] = {
            record['depot_node']: float(record['objective_contribution'])
            for record in process_data['depot_records']
        }
        elapsed_seconds = float(process_data['solve_seconds'])
    else:
        model = ImprovedMSTMultiAgentFlyingSidekickTSP(
            graph,
            depots,
            cities,
            distance,
            drones,
            theta=theta,
            improvement_config=improvement_config,
        )
        solution, cost = model.solve()
        telemetry = model.telemetry
        elapsed_seconds = float(telemetry['total_seconds'])

    group_sizes = [len(model.groups[depot]) for depot in depots]
    search = telemetry['search']
    group_costs = [
        float(telemetry['final_group_costs'].get(depot, 0.0))
        for depot in depots
    ]
    partition_cost_before = float(telemetry['partition_cost_before_search'])
    partition_cost_after = float(telemetry['partition_cost_after_search'])

    if improvement_config is None:
        process_data = telemetry['process_data']
        phase2_seconds = float(sum(
            record['set_tsp_seconds'] for record in process_data['depot_records']
        ))
        phase3_seconds = float(sum(
            record['local_search_seconds'] for record in process_data['depot_records']
        ))
        directed_cost_seconds = 0.0
    else:
        phase2_seconds = float(telemetry['phase2_evaluation_seconds'])
        phase3_seconds = float(telemetry['phase3_evaluation_seconds'])
        directed_cost_seconds = float(telemetry['directed_cost_seconds'])

    return {
        'variant': variant_name,
        'cost': float(cost),
        'elapsed_seconds': elapsed_seconds,
        'groups': [
            {'depot': depot, 'customers': list(model.groups[depot])}
            for depot in depots
        ],
        'group_sizes': group_sizes,
        'max_group_size': max(group_sizes, default=0),
        'group_size_std': float(np.std(group_sizes)) if group_sizes else 0.0,
        'group_costs': group_costs,
        'max_group_cost': max(group_costs, default=0.0),
        'group_cost_std': float(np.std(group_costs)) if group_costs else 0.0,
        'partition_cost_before_search': partition_cost_before,
        'partition_cost_after_search': partition_cost_after,
        'partition_cost_change_percent': (
            (partition_cost_after - partition_cost_before)
            / partition_cost_before
            * 100.0
            if partition_cost_before != 0 else 0.0
        ),
        'phase1_seconds': float(
            telemetry['boundary_seconds']
            + directed_cost_seconds
            + telemetry['initial_partition_seconds']
        ),
        'partition_search_seconds': float(search['elapsed_seconds']),
        'phase2_seconds': phase2_seconds,
        'phase3_seconds': phase3_seconds,
        'accepted_relocates': int(search['accepted_relocates']),
        'accepted_swaps': int(search['accepted_swaps']),
        'solution': solution,
        'telemetry': telemetry,
    }


def save_records(output_directory, settings, records, run_id=None):
    """
    将 MST 改进实验同时保存为可读 JSON 和压缩 NPZ。

    输入：输出目录、批次设置、逐实例记录和可选的稳定批次标识。
    输出：包含 json 与 npz 文件路径的字典。

    当传入相同 `run_id` 时覆盖同一组文件，因此主流程可在每个实例结束后
    原子更新检查点，而不会产生大量零散文件。
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    json_path = output_directory / f'{run_id}-mst-improvement.json'
    npz_path = output_directory / f'{run_id}-mst-improvement.npz'
    json_temporary_path = json_path.with_suffix('.json.tmp')
    npz_temporary_path = npz_path.with_suffix('.npz.tmp')

    payload = json_compatible({'settings': settings, 'records': records})
    json_temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    # 使用文件对象避免 NumPy 为 `.tmp` 路径自动追加额外的 `.npz` 后缀。
    with npz_temporary_path.open('wb') as npz_file:
        np.savez_compressed(
            npz_file,
            map_names=np.asarray(
                [record.get('map_name', '') for record in records], dtype=np.str_
            ),
            variants=np.asarray(
                [record['variant'] for record in records], dtype=np.str_
            ),
            customer_counts=np.asarray(
                [record['customer_count'] for record in records], dtype=np.int64
            ),
            instance_indices=np.asarray(
                [record['instance_index'] for record in records], dtype=np.int64
            ),
            costs=np.asarray([record['cost'] for record in records], dtype=float),
            elapsed_seconds=np.asarray(
                [record['elapsed_seconds'] for record in records], dtype=float
            ),
            relative_cost_change_percent=np.asarray(
                [record['relative_cost_change_percent'] for record in records], dtype=float
            ),
            partition_cost_change_percent=np.asarray(
                [record['partition_cost_change_percent'] for record in records], dtype=float
            ),
            phase1_seconds=np.asarray(
                [record['phase1_seconds'] for record in records], dtype=float
            ),
            partition_search_seconds=np.asarray(
                [record['partition_search_seconds'] for record in records], dtype=float
            ),
            phase2_seconds=np.asarray(
                [record['phase2_seconds'] for record in records], dtype=float
            ),
            phase3_seconds=np.asarray(
                [record['phase3_seconds'] for record in records], dtype=float
            ),
            accepted_relocates=np.asarray(
                [record['accepted_relocates'] for record in records], dtype=np.int64
            ),
            accepted_swaps=np.asarray(
                [record['accepted_swaps'] for record in records], dtype=np.int64
            ),
            max_group_sizes=np.asarray(
                [record['max_group_size'] for record in records], dtype=np.int64
            ),
            records_json=np.asarray(
                [
                    json.dumps(
                        json_compatible(record),
                        ensure_ascii=False,
                        separators=(',', ':'),
                    )
                    for record in records
                ],
                dtype=np.str_,
            ),
        )
    # 临时文件完整落盘后再替换正式文件，降低中断时留下半写文件的概率。
    json_temporary_path.replace(json_path)
    npz_temporary_path.replace(npz_path)
    return {'json': json_path, 'npz': npz_path}
