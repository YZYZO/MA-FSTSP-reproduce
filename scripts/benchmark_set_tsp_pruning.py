"""运行 C0、P1、P3、P7 和完整组合的统一 Set-TSP 消融实验。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence

# 允许通过 ``python scripts/benchmark_set_tsp_pruning.py`` 从仓库根目录直接运行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pruning import (  # noqa: E402
    PruningOptions,
    SetTSPSolverOptions,
    apply_pruning_pipeline,
    solve_set_tsp,
)


def _ablation_groups() -> Dict[str, PruningOptions]:
    """返回原始空间、三个单项剪枝和完整流水线的稳定实验分组。"""

    return {
        'C0': PruningOptions(),
        'P1': PruningOptions(structural_stsp=True),
        'P3': PruningOptions(assignment_bound=True),
        'P7': PruningOptions(endpoint_pair_dominance=True),
        'P1_P3_P7': PruningOptions(
            structural_stsp=True,
            assignment_bound=True,
            endpoint_pair_dominance=True,
        ),
    }


def run_ablation(
    set_sizes: Sequence[int],
    distance: Sequence,
    internal_cost: Sequence,
    repetitions: int = 3,
    solver_options: SetTSPSolverOptions | None = None,
    objective_tolerance: float = 1e-7,
) -> list[Dict[str, Any]]:
    """在相同成本张量和求解配置上运行全部剪枝消融组。

    输入：集合规模、两类成本、重复次数、Gurobi 配置和目标一致性容差。
    输出：每个实验组每次重复的一条扁平指标记录。

    逻辑：先运行 C0 得到每次重复的正确性基准；其余组只有目标通过门禁时
    才计算相对总时间加速比。所有组共用同一个稀疏模型构造器。
    """

    if repetitions <= 0:
        raise ValueError('消融实验 repetitions 必须为正整数。')
    if objective_tolerance < 0:
        raise ValueError('目标一致性容差必须非负。')

    selected_solver_options = solver_options or SetTSPSolverOptions()
    normalized_sizes = tuple(int(size) for size in set_sizes)
    records: list[Dict[str, Any]] = []
    baseline_by_repetition: Dict[int, Dict[str, float]] = {}

    for group, pruning_options in _ablation_groups().items():
        for repetition in range(repetitions):
            pipeline_started = time.perf_counter()
            space, pipeline_report = apply_pruning_pipeline(
                normalized_sizes,
                distance,
                internal_cost,
                pruning_options,
            )
            pipeline_seconds = time.perf_counter() - pipeline_started
            result = solve_set_tsp(
                space,
                distance,
                internal_cost,
                solver_options=selected_solver_options,
            )
            total_seconds = pipeline_seconds + result.total_seconds

            if group == 'C0':
                baseline_by_repetition[repetition] = {
                    'objective': result.objective,
                    'total_seconds': total_seconds,
                }
            baseline = baseline_by_repetition[repetition]
            objective_delta = abs(result.objective - baseline['objective'])
            objective_consistent = objective_delta <= objective_tolerance

            record: Dict[str, Any] = {
                'group': group,
                'repetition': repetition,
                'objective': result.objective,
                'objective_bound': result.objective_bound,
                'objective_delta_from_c0': objective_delta,
                'objective_consistent': objective_consistent,
                'phase2_variables': result.variable_count,
                'phase2_constraints': result.constraint_count,
                'phase2_gurobi_nodes': result.node_count,
                'phase2_model_build_seconds': result.model_build_seconds,
                'phase2_gurobi_seconds': result.runtime_seconds,
                'phase2_pruning_pipeline_seconds': pipeline_seconds,
                'phase2_total_seconds': total_seconds,
                'phase2_mip_gap': result.mip_gap,
                'phase2_solver_seed': result.solver_seed,
                'phase2_solver_mip_gap_target': result.solver_mip_gap_target,
                'phase2_solver_mip_gap_abs_target': result.solver_mip_gap_abs_target,
                'phase2_solver_optimality_tolerance': result.solver_optimality_tolerance,
                'phase2_solver_feasibility_tolerance': result.solver_feasibility_tolerance,
                'phase2_solver_integrality_tolerance': result.solver_integrality_tolerance,
                'phase2_gurobi_version': '.'.join(map(str, result.gurobi_version)),
                'speedup_vs_c0': (
                    baseline['total_seconds'] / total_seconds
                    if objective_consistent and total_seconds > 0.0
                    else None
                ),
            }
            for name, count in pipeline_report.initial_counts.items():
                record[f'initial_{name}'] = count
            for name, count in pipeline_report.final_counts.items():
                record[f'final_{name}'] = count
            record.update(pipeline_report.metrics())
            records.append(record)
    return records


def _build_random_costs(set_sizes: Sequence[int], seed: int):
    """生成可复现的非负非对称成本，供命令行快速冒烟实验使用。"""

    generator = random.Random(seed)
    internal_cost = [
        [
            [float(generator.randint(0, 12)) for _ in range(size)]
            for _ in range(size)
        ]
        for size in set_sizes
    ]
    distance = []
    for source, source_size in enumerate(set_sizes):
        targets = []
        for target, target_size in enumerate(set_sizes):
            targets.append(
                [
                    [
                        float(1000 if source == target else generator.randint(1, 40))
                        for _ in range(target_size)
                    ]
                    for _ in range(source_size)
                ]
            )
        distance.append(targets)
    return distance, internal_cost


def _write_records(records: list[Dict[str, Any]], output_path: Path) -> None:
    """把消融记录同时写为 JSON 和同名 CSV，便于复核与绘图。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    csv_path = output_path.with_suffix('.csv')
    fieldnames = sorted({key for record in records for key in record})
    with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """解析命令行参数并运行一个可重复的小型消融实验。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set-sizes', default='1,3,3,1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/pruning/set_tsp_ablation.json'),
    )
    arguments = parser.parse_args()
    set_sizes = tuple(int(value) for value in arguments.set_sizes.split(','))
    distance, internal_cost = _build_random_costs(set_sizes, arguments.seed)
    records = run_ablation(
        set_sizes,
        distance,
        internal_cost,
        repetitions=arguments.repetitions,
        solver_options=SetTSPSolverOptions(seed=arguments.seed),
    )
    _write_records(records, arguments.output)


if __name__ == '__main__':
    main()
