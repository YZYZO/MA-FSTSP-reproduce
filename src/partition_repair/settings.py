"""客户划分实验的固定配置，全部计算使用 CPU。"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SolverOptions:
    """输入优化预算与随机设置；输出不可变配置，供所有分区共用。"""

    time_limit: float | None = 30.0
    threads: int = 1
    seed: int = 0
    mip_gap: float = 1e-4

    def to_dict(self):
        """将求解配置转成可序列化字典，用于记录和缓存键。"""
        return asdict(self)


@dataclass(frozen=True)
class RepairOptions:
    """输入候选预算和几何权重；输出固定候选配置，不使用求解后信息。"""

    max_candidates: int = 12
    # 每次修复只检查有限的客户和仓库，控制候选生成本身的计算量。
    customers_per_group: int = 8
    destination_count: int = 2
    cluster_seeds: int = 3
    geometry_weight: float = 1.0

    def to_dict(self):
        """将候选配置转成字典，保证续跑使用相同生成预算。"""
        return asdict(self)
