"""学习型客户划分的轻量配置对象。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SetTSPSolverSettings:
    """
    保存生成 Set-TSP 标签时使用的可复现求解配置。

    输入：时间上限、线程数和随机种子。
    输出：不可变配置对象。
    逻辑：时间上限默认不设置，由后续探测实验决定；线程数固定为 1 便于比较运行时间。
    """

    time_limit_seconds: float | None = None
    threads: int = 1
    seed: int = 0

