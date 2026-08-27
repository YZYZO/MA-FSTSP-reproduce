"""
Central project configuration.

Change values here instead of setting shell environment variables before
running experiments or plotting scripts.
"""

from dataclasses import dataclass
from pathlib import Path


# The real project root is the directory containing this config file. Keeping
# paths anchored here prevents accidental writes to the outer MA-FSTSP-main
# wrapper directory when scripts are launched from an IDE or another cwd.
PROJECT_ROOT = Path(__file__).resolve().parent

# Data and output roots.
DATASETS_DIR = PROJECT_ROOT / 'datasets'
RESULTS_DIR = PROJECT_ROOT / 'results'

# 论文中型 Manhattan 场景使用的 1,024 节点 NYC 路网。
MANHATTAN1k_GRAPH_PATH = DATASETS_DIR / 'nyc_1024.graphml'
# 论文大型 Boston 场景的同规模替代数据：11,000 节点 NYC 路网。
MANHATTAN11k_GRAPH_PATH = DATASETS_DIR / 'nyc_11000.graphml'

# 路网实验的结果目录，与地图输入路径一起由本模块统一管理。
MANHATTAN_DATA_DIR = RESULTS_DIR / 'manhattan' / 'data'
BOSTON_DATA_DIR = RESULTS_DIR / 'boston' / 'data'
NYC_PROXY_DATA_DIR = RESULTS_DIR / 'nyc_proxy' / 'data'

# 两档路网共同使用的客户规模，保持论文实验入口的默认行为。
ROAD_SCALE_CUSTOMER_SIZES = (50, 100, 150)


@dataclass(frozen=True)
class RoadExperimentSpec:
    """
    集中保存一档路网实验不会随客户规模变化的固定配置。

    输入：地图路径、显示标签、进度标签、结果目录与文件标识、仓库数和无人机数。
    输出：不可变配置对象，供批量入口和单地图入口共享。
    逻辑：用具名字段替代长位置参数，避免地图配置在多层函数中重复声明。
    """

    graph_path: Path
    dataset_label: str
    progress_label: str
    result_directory: Path
    result_stem: str
    depot_count: int
    drone_count: int


@dataclass(frozen=True)
class ExperimentProtocol:
    """
    定义一套可复核的配对实验协议。

    输入字段包括协议名、数据集、客户规模、重复次数、方法列表，以及可选的
    Set-TSP 单仓库时限和实例总时限；输出是不变配置对象，供配对运行器生成清单并
    校验续跑。``None`` 表示不主动设置对应时限。
    """

    name: str
    datasets: tuple
    customer_sizes: tuple
    repetitions: int
    methods: tuple
    set_tsp_time_limit: float | None = None
    instance_time_limit: float | None = None


# 两档地图的路径、命名和车队参数只在此处定义一次，防止实验入口之间发生配置漂移。
MANHATTAN_1K_EXPERIMENT = RoadExperimentSpec(
    graph_path=MANHATTAN1k_GRAPH_PATH,
    dataset_label='Manhattan 1K',
    progress_label='Manhattan-1K-STSP',
    result_directory=MANHATTAN_DATA_DIR,
    result_stem='manhattan_1k',
    depot_count=5,
    drone_count=3,
)
MANHATTAN_11K_EXPERIMENT = RoadExperimentSpec(
    graph_path=MANHATTAN11k_GRAPH_PATH,
    # 当前仓库只有 NYC 11K 路网，不能把它表述成论文的 Boston 数据集。
    dataset_label='NYC 11K proxy',
    progress_label='NYC-11K-proxy-STSP',
    result_directory=NYC_PROXY_DATA_DIR,
    result_stem='nyc_11k_proxy',
    depot_count=10,
    drone_count=4,
)
#启用的地图
ROAD_SCALE_EXPERIMENTS = (
    MANHATTAN_1K_EXPERIMENT,
    #MANHATTAN_11K_EXPERIMENT,
)

#实例重复次数
# 保留旧拼写作为兼容别名，新代码统一使用 ``REPETITION_TIMES``。
REPETITION_TIMES = 10
REPEATION_TIMES = REPETITION_TIMES


# 主实验只比较具有明确角色的四种方法；GTDS 主方法统一启用全部可用仓库。
PHASE1_MAIN_METHODS = (
    'smst_original',
    'snn',
    'set_gtds_no_budget',
    'directed_set_gtds',
)
# 保留旧常量名，避免已有调用入口失效；其语义已更新为 V2 主比较集合。
PHASE1_COMPARISON_METHODS = PHASE1_MAIN_METHODS

# epsilon、活跃仓库和无人机代价分别运行，避免把多种研究问题混进主结果表。
PHASE1_EPSILON_METHODS = (
    'gtds_all_eps000',
    'gtds_all_eps005',
    'gtds_all_eps010',
    'gtds_all_eps020',
    'gtds_all_eps050',
)
PHASE1_ACTIVE_DEPOT_METHODS = ('directed_set_gtds', 'gtds_free_eps01')
PHASE1_COST_FACTOR_METHODS = ('directed_set_gtds', 'gtds_sqrt2')

# Pilot 只用于验证实现和估计运行量，不作为论文主实验结论。
PILOT_PROTOCOL = ExperimentProtocol(
    name='phase1_pilot_v2',
    datasets=(MANHATTAN_1K_EXPERIMENT,),
    customer_sizes=(50,),
    repetitions=10,
    methods=PHASE1_MAIN_METHODS,
    instance_time_limit=600.0,
)

# 正式协议使用每个设置 100 个配对实例。NYC 11K 明确标为代理数据，避免与
# 论文 Boston 路网混淆；获得真实 Boston 数据后只需替换对应 RoadExperimentSpec。
FORMAL_PROTOCOL = ExperimentProtocol(
    name='phase1_formal_v2',
    datasets=(MANHATTAN_1K_EXPERIMENT, MANHATTAN_11K_EXPERIMENT),
    customer_sizes=ROAD_SCALE_CUSTOMER_SIZES,
    repetitions=100,
    methods=PHASE1_MAIN_METHODS,
    instance_time_limit=7200.0,
)

EPSILON_SENSITIVITY_PROTOCOL = ExperimentProtocol(
    name='phase1_epsilon_v2',
    datasets=(MANHATTAN_1K_EXPERIMENT, MANHATTAN_11K_EXPERIMENT),
    customer_sizes=ROAD_SCALE_CUSTOMER_SIZES,
    repetitions=100,
    methods=PHASE1_EPSILON_METHODS,
    instance_time_limit=7200.0,
)

ACTIVE_DEPOT_ABLATION_PROTOCOL = ExperimentProtocol(
    name='phase1_active_depot_v2',
    datasets=(MANHATTAN_1K_EXPERIMENT,),
    customer_sizes=ROAD_SCALE_CUSTOMER_SIZES,
    repetitions=100,
    methods=PHASE1_ACTIVE_DEPOT_METHODS,
    instance_time_limit=7200.0,
)

COST_FACTOR_SENSITIVITY_PROTOCOL = ExperimentProtocol(
    name='phase1_cost_factor_v2',
    datasets=(MANHATTAN_1K_EXPERIMENT,),
    customer_sizes=ROAD_SCALE_CUSTOMER_SIZES,
    repetitions=100,
    methods=PHASE1_COST_FACTOR_METHODS,
    instance_time_limit=7200.0,
)
