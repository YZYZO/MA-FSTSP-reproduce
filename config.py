"""
Central project configuration.

Change values here instead of setting shell environment variables before
running experiments or plotting scripts.
"""

from pathlib import Path


# The real project root is the directory containing this config file. Keeping
# paths anchored here prevents accidental writes to the outer MA-FSTSP-main
# wrapper directory when scripts are launched from an IDE or another cwd.
PROJECT_ROOT = Path(__file__).resolve().parent

# Data and output roots.
DATASETS_DIR = PROJECT_ROOT / 'datasets'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Experiment mode. Keep this False for quick local checks; set True for the
# paper-scale experiment suite, which can take a long time on older CPUs.
RUN_FULL_EXPERIMENTS = True

# 显式地图路径；默认不允许候选路径或合成地图静默替代目标数据。
#MANHATTAN_GRAPH_PATH = DATASETS_DIR / 'nyc.graphml'
MANHATTAN_GRAPH_PATH = DATASETS_DIR / 'manhatten.graphml'

# 本机小实例只使用已冻结的 4,333 节点基线图，不替代 55k 目标图。
MANHATTAN_BASELINE_GRAPH_PATH = DATASETS_DIR / 'manhatten.graphml'
BOSTON_GRAPH_PATH = DATASETS_DIR / 'boston.graphml'
ALLOW_GRAPH_PATH_FALLBACK = False
ALLOW_SYNTHETIC_GRAPH_FALLBACK = False

# OSMnx/Boston map download settings. 只有显式修改配置后才联网刷新。
ALLOW_OSM_DOWNLOAD = False
REFRESH_OSM = False
OSM_CENTER_POINT = (42.3601, -71.0589)
OSM_DIST_METERS = 1600
OSM_MAX_NODES = 11000
OSM_TIMEOUT = 300
OVERPASS_ENDPOINTS = (
    'https://overpass.kumi.systems/api',
    'https://overpass-api.de/api',
)

# Demo visualization settings.
DEMO_DRONE_LIMIT = 0.5

# 距离后端：`auto` 在小图使用 eager，在更大图上要求 H2H。
DISTANCE_BACKEND = 'H2H'

# eager 会物化卡车全对最短距离，因此只允许用于小图和明确的回归测试。
EAGER_DISTANCE_MAX_NODES = 1000

# Python H2H 仅作为逐步对照参考实现，硬性限制规模以防误用于真实大图。
H2H_REFERENCE_MAX_NODES = 200

# 阶段 3 原生构建目录与本机首选 MinGW 编译器；Linux 可由命令行覆盖。
H2H_NATIVE_BUILD_DIR = PROJECT_ROOT / 'build' / 'h2h'
H2H_CXX = Path(r'D:\dev\mingw64\bin\g++.exe')

# H2H 版本化缓存、自动索引构建和本机大图保护。
H2H_INDEX_DIR = DATASETS_DIR / 'indexes'
H2H_AUTO_BUILD = True
H2H_ENABLE_55K = True
H2H_LARGE_GRAPH_MIN_NODES = 50000
H2H_DISTANCE_DTYPE = 'float64'
H2H_ALLOW_LARGE_GRAPH_DIJKSTRA_FALLBACK = False

# 同一图哈希只允许一个进程构建；等待超时不会破坏持锁进程的文件。
H2H_BUILD_LOCK_TIMEOUT_SECONDS = 900.0
H2H_BUILD_LOCK_POLL_SECONDS = 0.1

# 原生 builder 的可选资源保护；0 表示交由服务器资源监控决定。
H2H_BUILDER_MAX_STRUCTURAL_EDGES = 0
H2H_BUILDER_MAX_SHORTCUT_ARCS = 0
H2H_BUILDER_PROGRESS_INTERVAL = 1000

# 调试时可开启查询次数/时间统计；生产默认关闭以减小标量调用开销。
H2H_QUERY_STATS = False


def result_path(*parts):
    return RESULTS_DIR.joinpath(*parts)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path
