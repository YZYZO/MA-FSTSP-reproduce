"""
定义状态空间剪枝模块共享的配置、指标与审计证据结构。

本模块只保存轻量数据对象，不依赖 Gurobi，便于纯函数测试和实验结果序列化。
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


# 端点对使用“集合编号、源端点局部编号、目标端点局部编号”描述，避免依赖节点类型是否可比较。
EndpointPair = Tuple[int, int]
SetArc = Tuple[int, int]


@dataclass(frozen=True)
class PruningOptions:
    """
    保存 P7 端点对支配剪枝和稀疏 Set-TSP 的运行配置。

    输入：是否启用支配、数值容差、分块规模、可选计算预算和 Gurobi 随机种子。
    输出：不可变配置对象，供预处理器、模型适配器与实验入口共享。
    逻辑：默认关闭 P7；预算耗尽只停止继续剪枝，不影响已证明安全的删除结果。
    """

    endpoint_pair_dominance: bool = False
    dominance_epsilon: float = 1e-9
    comparison_block_size: int = 4096
    max_pair_checks: Optional[int] = None
    max_preprocessing_seconds: Optional[float] = None
    record_evidence: bool = False
    gurobi_seed: int = 0
    gurobi_output: bool = False

    def validate(self):
        """
        校验剪枝配置是否能被安全执行。

        输入：当前配置对象。
        输出：无；字段非法时抛出 `ValueError`。
        逻辑：禁止负容差、非正分块大小和非正预算，避免循环边界或比较语义异常。
        """

        if self.dominance_epsilon < 0:
            raise ValueError('dominance_epsilon 不能为负数。')
        if self.comparison_block_size <= 0:
            raise ValueError('comparison_block_size 必须为正整数。')
        if self.max_pair_checks is not None and self.max_pair_checks <= 0:
            raise ValueError('max_pair_checks 必须为正整数或 None。')
        if self.max_preprocessing_seconds is not None and self.max_preprocessing_seconds <= 0:
            raise ValueError('max_preprocessing_seconds 必须为正数或 None。')


@dataclass
class DominanceEvidence:
    """
    记录一个端点候选被删除时的可复核支配证据。

    输入：集合弧、被删除候选、直接支配者、三项差分、总裕量和支配类型。
    输出：可转换为 JSON 字典的审计记录。
    逻辑：`terminal_dominator` 在全部比较结束后指向证据链末端的保留候选。
    """

    set_arc: SetArc
    victim: EndpointPair
    dominator: EndpointPair
    source_internal_delta: float
    external_delta: float
    target_internal_delta: float
    margin: float
    dominance_kind: str
    terminal_dominator: Optional[EndpointPair] = None

    def to_dict(self):
        """
        将证据转换为仅包含 JSON 基本类型的字典。

        输入：当前证据对象。
        输出：可直接交给 `json.dump` 的字典。
        逻辑：元组显式转换为列表，避免不同序列化器产生不一致结果。
        """

        result = asdict(self)
        result['set_arc'] = list(self.set_arc)
        result['victim'] = list(self.victim)
        result['dominator'] = list(self.dominator)
        if self.terminal_dominator is not None:
            result['terminal_dominator'] = list(self.terminal_dominator)
        return result


@dataclass
class PruningMetrics:
    """
    汇总一个仓库组在 P7 预处理和稀疏 Set-TSP 求解中的全部指标。

    输入：候选计数、支配耗时、模型规模、Gurobi 指标和可选回退原因。
    输出：统一指标对象，可附加到现有 `process_data` 或写入比较结果。
    逻辑：预处理器先填写剪枝字段，稀疏求解器随后补充模型与求解字段。
    """

    endpoint_pair_dominance_enabled: bool = False
    endpoint_pairs_before: int = 0
    endpoint_pairs_kept: int = 0
    endpoint_pairs_strictly_dominated: int = 0
    endpoint_pairs_equivalent_dominated: int = 0
    endpoint_pair_checks: int = 0
    endpoint_dominance_seconds: float = 0.0
    set_arcs_before: int = 0
    set_arcs_after: int = 0
    set_arcs_emptied_by_dominance: int = 0
    set_arcs_restored_by_safety: int = 0
    budget_exhausted: bool = False
    safe_fallback: bool = False
    fallback_reason: str = ''
    model_build_seconds: float = 0.0
    gurobi_seconds: float = 0.0
    set_tsp_total_seconds: float = 0.0
    set_tsp_objective: Optional[float] = None
    model_status: Optional[int] = None
    model_variables: int = 0
    model_constraints: int = 0
    select_variables: int = 0
    flow_variables: int = 0
    internal_variables: int = 0
    external_variables: int = 0
    gurobi_node_count: float = 0.0
    gurobi_mip_gap: Optional[float] = None

    @property
    def endpoint_pairs_dominated(self):
        """
        返回严格支配和等价代表删除的候选总数。

        输入：当前指标对象。
        输出：被 P7 删除的端点候选总数。
        逻辑：两类删除互斥，因此可直接相加。
        """

        return self.endpoint_pairs_strictly_dominated + self.endpoint_pairs_equivalent_dominated

    def assert_conservation(self):
        """
        检查端点候选计数守恒关系。

        输入：当前指标对象。
        输出：无；发现重复计数或候选丢失时抛出 `AssertionError`。
        逻辑：要求剪枝前候选恰好等于保留候选与两类删除候选之和。
        """

        accounted = self.endpoint_pairs_kept + self.endpoint_pairs_dominated
        if self.endpoint_pairs_before != accounted:
            raise AssertionError(
                'P7 候选计数不守恒：'
                f'before={self.endpoint_pairs_before}, accounted={accounted}。'
            )

    def to_dict(self):
        """
        把指标转换为 JSON 友好的字典并补充派生总数。

        输入：当前指标对象。
        输出：包含原始字段和 `endpoint_pairs_dominated` 的字典。
        逻辑：在序列化前执行守恒检查，防止错误统计进入正式实验结果。
        """

        self.assert_conservation()
        result = asdict(self)
        result['endpoint_pairs_dominated'] = self.endpoint_pairs_dominated
        return result


@dataclass
class EndpointDominanceResult:
    """
    保存 P7 预处理器的候选索引、指标与可选证据。

    输入：每条集合弧保留的端点对、统一指标和证据列表。
    输出：供稀疏模型构造器直接消费的结果对象。
    逻辑：键只包含 `i != j` 的有向集合弧，端点编号均为各自集合内的局部编号。
    """

    kept_pairs_by_arc: Dict[SetArc, Tuple[EndpointPair, ...]]
    metrics: PruningMetrics
    evidence: List[DominanceEvidence] = field(default_factory=list)

