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

# 论文中型 Manhattan 场景使用的 1,024 节点 NYC 路网。
MANHATTAN1k_GRAPH_PATH = DATASETS_DIR / 'nyc_1024.graphml'
# 论文大型 Boston 场景的同规模替代数据：11,000 节点 NYC 路网。
MANHATTAN11k_GRAPH_PATH = DATASETS_DIR / 'nyc_11000.graphml'



# OSMnx/Boston map download settings.
ALLOW_OSM_DOWNLOAD = True
REFRESH_OSM = True
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


def result_path(*parts):
    return RESULTS_DIR.joinpath(*parts)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path
