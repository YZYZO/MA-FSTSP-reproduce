"""采集分区候选的真实 Set-TSP→DP 标签，支持组级断点续跑。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.partition_repair.runner import main


if __name__ == '__main__':
    main('collect')
