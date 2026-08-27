"""
本文件是论文实验的主入口脚本。

主要内容：
1. 组织 Manhattan、Cambridge 与两档 NYC 路网规模实验。
2. 让同一张地图的不同客户规模共享路网和全点对距离数据。
3. 调用论文主算法，并将成本、耗时、路线与阶段记录保存为 `.npz` 文件。
"""

import argparse
import gc
from datetime import datetime

from src.fstsp import MultiAgentFlyingSidekickTSP

from problem import (
    multiagent_instance_on_manhattan,
    multiagent_instance_on_cambridge,
    prepare_manhattan_road_network,
    sample_multiagent_instances,
)
from tqdm import tqdm
from config import (
    BOSTON_DATA_DIR,
    MANHATTAN_DATA_DIR,
    MANHATTAN_1K_EXPERIMENT,
    MANHATTAN_11K_EXPERIMENT,
    ROAD_SCALE_CUSTOMER_SIZES,
    ROAD_SCALE_EXPERIMENTS,
    REPEATION_TIMES,
    FORMAL_PROTOCOL,
    PILOT_PROTOCOL,
    EPSILON_SENSITIVITY_PROTOCOL,
    ACTIVE_DEPOT_ABLATION_PROTOCOL,
    COST_FACTOR_SENSITIVITY_PROTOCOL,
    RESULTS_DIR,
)
from experiment_results import (
    _save_stsp_batch_result,
    _solve_model_with_process_data,
)
from paired_experiments import run_paired_road_experiment

def _print_distance_initialization_stats(label, stats):
    '''
    输出一批路网实验的全点对距离初始化耗时。

    输入：数据集显示名称和批次级耗时字典。
    输出：无；打印卡车、无人机和总初始化耗时。
    '''
    print(
        '{} distance initialization: truck APSP={:.3f}s, '
        'drone pairwise={:.3f}s, total={:.3f}s.'.format(
            label,
            stats['truck_apsp_seconds'],
            stats['drone_pairwise_seconds'],
            stats['distance_initialization_seconds'],
        )
    )


def test_manhattan(num, size):
    """
    在 Manhattan 路网实例上顺序运行论文主算法。

    输入：
    - num: 随机实例数量。
    - size: 每个实例的客户数量。

    输出：
    - 无显式返回值；打印平均成本与耗时，并保存包含路线和三阶段记录的批次结果。

    实现逻辑：
    1. 一次性生成共享路网、距离数据和全部随机实例。
    2. 按实例索引在主进程中依次构造并求解模型。
    3. 汇总结果，同时保留现有 NPZ 结果结构。
    """
    graph, depots, cities, distance = multiagent_instance_on_manhattan(num, 5, size)
    costs = {'hc': [], 'stsp': [], 'lp': []}
    times = {'hc': [], 'stsp': [], 'lp': []}
    stsp_results = []

    print('Running Manhattan experiment serially.')

    for i in tqdm(range(num), desc='Manhattan-STSP'):
        depot, city = depots[i], cities[i]
        model = MultiAgentFlyingSidekickTSP(
            graph, depot, city, distance, 3, theta=(0.5, 0.5)
        )
        solution, cost, process_data = _solve_model_with_process_data(model)
        elapsed = process_data['solve_seconds']
        stsp_results.append((cost, elapsed, solution, process_data))
        costs['stsp'].append(cost)
        times['stsp'].append(elapsed)

    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')
    _save_stsp_batch_result(
        MANHATTAN_DATA_DIR / f'road-size-{size}.npz',
        stsp_results,
        depots,
        cities,
        distance,
        3,
        costs,
        times,
    )


def test_cambridge(num, size):
    """
    在 Cambridge 路网实例上顺序运行论文主算法。

    输入：
    - num: 随机实例数量。
    - size: 每个实例的客户数量。

    输出：
    - 无显式返回值；打印平均成本与耗时，并保存包含路线和三阶段记录的批次结果。

    实现逻辑：
    1. 一次性生成共享路网、距离数据和全部随机实例。
    2. 按实例索引在主进程中依次构造并求解模型。
    3. 汇总结果，同时保留现有 NPZ 结果结构。
    """
    graph, depots, cities, distance = multiagent_instance_on_cambridge(num, 10, size)
    costs = {'hc': [], 'stsp': [], 'lp': []}
    times = {'hc': [], 'stsp': [], 'lp': []}
    stsp_results = []

    print('Running Cambridge experiment serially.')

    for i in tqdm(range(num), desc='Cambridge-STSP'):
        depot, city = depots[i], cities[i]
        model = MultiAgentFlyingSidekickTSP(
            graph, depot, city, distance, 3, theta=(0.5, 0.5)
        )
        solution, cost, process_data = _solve_model_with_process_data(model)
        elapsed = process_data['solve_seconds']
        stsp_results.append((cost, elapsed, solution, process_data))
        costs['stsp'].append(cost)
        times['stsp'].append(elapsed)

    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')
    _save_stsp_batch_result(
        BOSTON_DATA_DIR / f'road-size-{size}.npz',
        stsp_results,
        depots,
        cities,
        distance,
        3,
        costs,
        times,
    )


def _run_prepared_road_experiment(
    num,
    size,
    spec,
    prepared_network,
    run_timestamp,
    partition_method='smst_original',
):
    """
    在一张已经准备完成的 NYC 路网上运行单个客户规模实验。

    输入：
    - num: 随机实例数量。
    - size: 每个实例的客户数量。
    - spec: 当前地图的固定实验配置。
    - prepared_network: 已准备的 `(graph, distance, distance_stats)`。
    - run_timestamp: 当前实验批次共享的时间戳。
    - partition_method: 第一阶段客户划分方法，默认保持原 SMST。

    输出：
    - 无显式返回值；顺序求解全部实例并保存 NPZ 结果。

    实现逻辑：
    1. 复用调用方准备的地图和距离矩阵，仅重新采样当前客户规模的实例。
    2. 使用配置中的车队参数顺序求解全部实例。
    3. 按配置中的结果目录和文件标识保存完整实验记录。
    """
    graph, distance, distance_stats = prepared_network
    depot_instances, city_instances = sample_multiagent_instances(
        graph,
        num,
        spec.depot_count,
        size,
    )

    # 当前分支只运行论文主算法，仍保留 HC 与 LP 空槽以兼容既有结果格式。
    costs = {'hc': [], 'stsp': [], 'lp': []}
    times = {'hc': [], 'stsp': [], 'lp': []}
    stsp_results = []

    print(
        f'Running {spec.dataset_label} experiment serially with '
        f'{spec.depot_count} depots, {spec.drone_count} drones per truck, '
        f'{size} customers, and partition method {partition_method}.'
    )
    for index in tqdm(range(num), desc=spec.progress_label):
        # 仓库和客户通过相同实例索引配对，车队规模来自统一地图配置。
        depots = depot_instances[index]
        cities = city_instances[index]
        model = MultiAgentFlyingSidekickTSP(
            graph,
            depots,
            cities,
            distance,
            spec.drone_count,
            theta=(0.5, 0.5),
            partition_method=partition_method,
        )
        solution, cost, process_data = _solve_model_with_process_data(model)
        elapsed = process_data['solve_seconds']
        stsp_results.append((cost, elapsed, solution, process_data))
        costs['stsp'].append(cost)
        times['stsp'].append(elapsed)

    print(
        f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} '
        f'in {sum(times["stsp"]) / num}s'
    )
    # 新方法在文件名中显式记录分区名称，避免覆盖原 SMST 复现实验结果。
    result_stem = spec.result_stem
    if partition_method != 'smst_original':
        result_stem = f'{result_stem}-{partition_method}'
    _save_stsp_batch_result(
        spec.result_directory / f'{run_timestamp}-{result_stem}-{size}.npz',
        stsp_results,
        depot_instances,
        city_instances,
        distance,
        spec.drone_count,
        costs,
        times,
        distance_initialization_stats=distance_stats,
    )


def _run_standalone_road_experiment(num, size, spec):
    """
    为一个独立调用准备路网、运行单个客户规模并在结束后释放共享数据。

    输入：随机实例数量、客户数量和一档路网实验配置。
    输出：无；保存该客户规模的 NPZ 结果并释放地图级距离数据。
    逻辑：生成时间戳，准备一次地图，调用已准备路网执行器，最后执行垃圾回收。
    """
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    prepared_network = None
    print(f'Preparing shared road-network data for {spec.dataset_label}.')
    try:
        prepared_network = prepare_manhattan_road_network(spec.graph_path)
        _print_distance_initialization_stats(spec.dataset_label, prepared_network[2])
        _run_prepared_road_experiment(
            num,
            size,
            spec,
            prepared_network,
            run_timestamp,
        )
    finally:
        # 独立入口与批量入口采用相同的地图级资源释放语义。
        prepared_network = None
        gc.collect()
        print(f'Released shared road-network distance data for {spec.dataset_label}.')


def test_manhattan_1k(num, size):
    """
    在 1,024 节点 NYC 路网上使用 5 仓库、每车 3 架无人机运行主算法。

    输入：实例数量和客户数量。
    输出：无；结果保存为 `manhattan_1k` NPZ 文件。
    逻辑：使用统一的 1K 配置调用独立路网实验执行器。
    """
    _run_standalone_road_experiment(
        num,
        size,
        MANHATTAN_1K_EXPERIMENT,
    )


def test_manhattan_11k(num, size):
    """
    在 11,000 节点 NYC 路网上使用 10 仓库、每车 4 架无人机运行主算法。

    输入：实例数量和客户数量。
    输出：无；结果以 `nyc_11k_proxy` 命名，不冒充论文 Boston 数据。
    逻辑：使用统一的 11K 配置调用独立路网实验执行器。
    """
    _run_standalone_road_experiment(
        num,
        size,
        MANHATTAN_11K_EXPERIMENT,
    )


def run_full_experiments(
    num=REPEATION_TIMES,
    customer_sizes=ROAD_SCALE_CUSTOMER_SIZES,
):
    """
    按已启用地图配置运行旧版路网规模实验。

    输入：
    - num: 每个地图与客户规模需要求解的实例数量，默认使用论文完整实验的 100。
    - customer_sizes: 每张地图依次运行的客户规模，默认是 50、100、150。

    输出：
    - 无显式返回值，实验结果以终端输出和 `.npz` 文件形式保存。

    实现逻辑：
    1. 按统一配置顺序准备每张地图，并在当前地图上运行全部客户规模。
    2. 同一地图只初始化一次距离数据，切换地图前显式释放相关引用。
    3. 各地图的路径、结果命名和车队参数全部读取自统一配置对象。
    """
    sizes = tuple(customer_sizes)

    # 整轮规模实验共享时间戳，使同一批次的结果文件具有一致标识。
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    for spec in ROAD_SCALE_EXPERIMENTS:
        prepared_network = None
        print(f'Preparing shared road-network data for {spec.dataset_label}.')
        try:
            prepared_network = prepare_manhattan_road_network(spec.graph_path)
            _print_distance_initialization_stats(spec.dataset_label, prepared_network[2])
            for size in sizes:
                _run_prepared_road_experiment(
                    num,
                    size,
                    spec,
                    prepared_network,
                    run_timestamp,
                )
        finally:
            # 当前地图的所有客户规模结束后再释放共享距离数据。
            prepared_network = None
            gc.collect()
            print(f'Released shared road-network distance data for {spec.dataset_label}.')


def run_set_gtds_experiments(
    num=REPEATION_TIMES,
    customer_sizes=ROAD_SCALE_CUSTOMER_SIZES,
):
    """
    使用 Directed Set-GTDS 第一阶段运行旧版路网规模实验。

    输入：
    - num: 每个地图与客户规模的实例数量。
    - customer_sizes: 每张地图依次运行的客户规模。

    输出：
    - 无显式返回值；结果保存为带 ``directed_set_gtds`` 标识的 NPZ 文件。

    实现逻辑：
    1. 与 ``run_full_experiments`` 使用相同地图、实例采样和距离复用方式；
    2. 仅将第一阶段显式切换为 Directed Set-GTDS；
    3. 每张地图完成全部客户规模后释放共享距离数据。
    """

    sizes = tuple(customer_sizes)
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    for spec in ROAD_SCALE_EXPERIMENTS:
        prepared_network = None
        print(f'Preparing shared road-network data for {spec.dataset_label}.')
        try:
            prepared_network = prepare_manhattan_road_network(spec.graph_path)
            _print_distance_initialization_stats(spec.dataset_label, prepared_network[2])
            for size in sizes:
                _run_prepared_road_experiment(
                    num,
                    size,
                    spec,
                    prepared_network,
                    run_timestamp,
                    partition_method='directed_set_gtds',
                )
        finally:
            prepared_network = None
            gc.collect()
            print(f'Released shared road-network distance data for {spec.dataset_label}.')


def run_paired_protocol(protocol=PILOT_PROTOCOL, seed=0):
    """
    执行一套统一的同实例配对实验协议。

    输入为 ``ExperimentProtocol`` 与采样 seed；输出所有规模的 summary 路径列表。
    每张地图只初始化一次距离数据，各规模在稳定目录中按实例×方法原子续跑；协议的
    ``set_tsp_time_limit`` 控制单仓库求解器，``instance_time_limit`` 控制三阶段总预算。
    """

    summary_paths = []
    for spec in protocol.datasets:
        prepared_network = None
        print(f'Preparing paired experiment data for {spec.dataset_label}.')
        try:
            prepared_network = prepare_manhattan_road_network(spec.graph_path)
            _print_distance_initialization_stats(spec.dataset_label, prepared_network[2])
            for customer_size in protocol.customer_sizes:
                output_root = (
                    RESULTS_DIR
                    / 'paired'
                    / protocol.name
                    / spec.result_stem
                    / str(customer_size)
                )
                summary_paths.append(run_paired_road_experiment(
                    spec=spec,
                    prepared_network=prepared_network,
                    customer_size=customer_size,
                    repetitions=protocol.repetitions,
                    methods=protocol.methods,
                    output_root=output_root,
                    seed=seed,
                    set_tsp_time_limit=protocol.set_tsp_time_limit,
                    instance_time_limit=protocol.instance_time_limit,
                ))
        finally:
            prepared_network = None
            gc.collect()
            print(f'Released paired road-network data for {spec.dataset_label}.')
    return summary_paths


def main(argv=None):
    """
    解析命令行并通过统一配对运行器启动 Pilot 或正式实验。

    输入为可选命令行参数序列；``None`` 表示读取当前进程参数。输出各数据规模的
    summary 路径列表。无参数时默认运行 Pilot，避免误启动 4200 个正式求解单元。
    """

    parser = argparse.ArgumentParser(
        description='运行 MA-FSTSP 第一阶段同实例配对实验。'
    )
    parser.add_argument(
        '--protocol',
        choices=('pilot', 'formal', 'epsilon', 'active-depot', 'cost-factor'),
        default='pilot',
        help='实验协议；默认 pilot，formal 为每个设置 100 个实例。',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='实例采样随机种子，默认 0。',
    )
    arguments = parser.parse_args(argv)
    # 命令行名称与不可变协议对象的唯一映射，避免入口内重复声明实验参数。
    protocols = {
        'pilot': PILOT_PROTOCOL,
        'formal': FORMAL_PROTOCOL,
        'epsilon': EPSILON_SENSITIVITY_PROTOCOL,
        'active-depot': ACTIVE_DEPOT_ABLATION_PROTOCOL,
        'cost-factor': COST_FACTOR_SENSITIVITY_PROTOCOL,
    }
    return run_paired_protocol(
        protocol=protocols[arguments.protocol],
        seed=arguments.seed,
    )


if __name__ == '__main__':
    main()
