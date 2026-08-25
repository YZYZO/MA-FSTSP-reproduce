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
# 扩展规模实验使用的完整 NYC 路网，原始文件包含约 55,000 个节点。
MANHATTAN55k_GRAPH_PATH = DATASETS_DIR / 'nyc.graphml'

# 路网实验的结果目录，与地图输入路径一起由本模块统一管理。
MANHATTAN_DATA_DIR = RESULTS_DIR / 'manhattan' / 'data'
BOSTON_DATA_DIR = RESULTS_DIR / 'boston' / 'data'

# 三档路网共同使用的客户规模，保持论文实验入口的默认行为。
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


# 三档地图的路径、命名和车队参数只在此处定义一次，防止实验入口之间发生配置漂移。
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
    dataset_label='Boston 11K',
    progress_label='Boston-11K-STSP',
    result_directory=BOSTON_DATA_DIR,
    result_stem='boston_11k',
    depot_count=10,
    drone_count=4,
)
MANHATTAN_55K_EXPERIMENT = RoadExperimentSpec(
    graph_path=MANHATTAN55k_GRAPH_PATH,
    dataset_label='Manhattan 55K',
    progress_label='Manhattan-55K-STSP',
    result_directory=MANHATTAN_DATA_DIR,
    result_stem='manhattan_55k',
    depot_count=10,
    drone_count=4,
)

#启用的地图
ROAD_SCALE_EXPERIMENTS = (
    MANHATTAN_1K_EXPERIMENT,
    MANHATTAN_11K_EXPERIMENT,
    MANHATTAN_55K_EXPERIMENT,
)

#实例重复次数
REPEATION_TIMES = 50
