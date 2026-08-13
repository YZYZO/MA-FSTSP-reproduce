"""
Central project configuration.


"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Data and output roots.
DATASETS_DIR = PROJECT_ROOT / 'datasets'
RESULTS_DIR = PROJECT_ROOT / 'results'

# 显式地图路径；默认不允许候选路径或合成地图静默替代目标数据。
MANHATTAN_GRAPH_PATH = DATASETS_DIR / 'nyc.graphml'
#MANHATTAN_GRAPH_PATH = DATASETS_DIR / 'manhatten.graphml'

# 本机小实例只使用已冻结的 4,333 节点基线图，不替代 55k 目标图。
MANHATTAN_BASELINE_GRAPH_PATH = DATASETS_DIR / 'manhatten.graphml'
BOSTON_GRAPH_PATH = DATASETS_DIR / 'boston.graphml'
MANHATTAN1k_GRAPH_PATH = DATASETS_DIR / 'nyc_1024.graphml'
MANHATTAN11k_GRAPH_PATH = DATASETS_DIR / 'nyc_11000.graphml'

ALLOW_GRAPH_PATH_FALLBACK = False
ALLOW_SYNTHETIC_GRAPH_FALLBACK = False


# 距离后端：`auto` 在小图使用 eager，在更大图上要求 H2H。
DISTANCE_BACKEND = 'H2H'

# eager 会物化卡车全对最短距离，因此只允许用于小图和明确的回归测试。
EAGER_DISTANCE_MAX_NODES = 1000

# 阶段 3 原生构建目录与本机首选 MinGW 编译器；Linux 可由命令行覆盖。
H2H_NATIVE_BUILD_DIR = PROJECT_ROOT / 'build' / 'h2h'
H2H_CXX = Path(r'D:\dev\mingw64\bin\g++.exe')

# H2H 版本化缓存、自动索引构建和本机大图保护。
H2H_INDEX_DIR = DATASETS_DIR / 'indexes'
H2H_AUTO_BUILD = True

H2H_ENABLE_55K = False

H2H_LARGE_GRAPH_MIN_NODES = 50000

# 同一图哈希只允许一个进程构建；等待超时不会破坏持锁进程的文件。
H2H_BUILD_LOCK_TIMEOUT_SECONDS = 900.0
H2H_BUILD_LOCK_POLL_SECONDS = 0.1

# 原生 builder 的可选资源保护；0 表示交由服务器资源监控决定。
H2H_BUILDER_MAX_STRUCTURAL_EDGES = 0
H2H_BUILDER_MAX_SHORTCUT_ARCS = 0
H2H_BUILDER_PROGRESS_INTERVAL = 1000

# 调试时可开启查询次数/时间统计；生产默认关闭以减小标量调用开销。
H2H_QUERY_STATS = False
