"""公开导出 MA-FSTSP 状态空间剪枝模块的稳定接口。"""

from .endpoint_pair_dominance import prune_endpoint_pairs
from .experiment_runner import (
    P7ComparisonOptions,
    run_p7_endpoint_dominance_comparison,
)
from .fstsp_adapter import (
    PruningMultiAgentFlyingSidekickTSP,
    attach_pruning_process_data,
    build_current_set_tsp_costs,
)
from .set_tsp_solver import SetTSPSolveResult, solve_sparse_set_tsp
from .types import (
    DominanceEvidence,
    EndpointDominanceResult,
    PruningMetrics,
    PruningOptions,
)

__all__ = [
    'DominanceEvidence',
    'EndpointDominanceResult',
    'PruningMetrics',
    'PruningMultiAgentFlyingSidekickTSP',
    'PruningOptions',
    'P7ComparisonOptions',
    'SetTSPSolveResult',
    'attach_pruning_process_data',
    'build_current_set_tsp_costs',
    'prune_endpoint_pairs',
    'run_p7_endpoint_dominance_comparison',
    'solve_sparse_set_tsp',
]
