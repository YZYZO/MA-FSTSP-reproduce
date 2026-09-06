"""客户划分实验的固定配置，全部计算使用 CPU。"""

from dataclasses import asdict, dataclass, field


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


@dataclass(frozen=True)
class EvaluationOptions:
    """输入总体成本和节时门槛，输出报告配置；不施加单实例成本硬限制。"""

    cost_limit: float = 0.10
    min_phase2_saving: float = 0.20

    def to_dict(self):
        """返回实际验收门槛，用于报告、校准和实验清单。"""
        return asdict(self)


@dataclass(frozen=True)
class SelectionOptions:
    """输入冻结的选择参数，输出独立于候选生成与求解器的策略配置。"""

    geometry_weight: float = 1.0
    size_weights: dict = field(default_factory=dict)
    cost_weight: float = 1.0
    force_stay: bool = False
    seed: int = 0

    def to_dict(self):
        """返回策略参数；规模键使用字符串以便 JSON 往返一致。"""
        result = asdict(self)
        result['size_weights'] = {str(k): v for k, v in self.size_weights.items()}
        return result
