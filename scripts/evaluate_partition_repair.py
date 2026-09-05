"""在相同清单上独立重跑分区策略，记录真实端到端墙钟时间。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.partition_repair.runner import main


if __name__ == '__main__':
    main('evaluate')
