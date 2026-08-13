"""Set-TSP 严格剪枝组件。

该包将 P1、P3、P7 分别实现为独立的候选空间变换，并通过统一流水线组合。
原始 ``src.fstsp`` 保持不变，便于进行论文消融和回归比较。
"""

from .pipeline import apply_pruning_pipeline
from .set_tsp_model import SetTSPResult, solve_set_tsp
from .types import (
    CandidateSpace,
    PruningOptions,
    PruningPipelineReport,
    SetTSPSolverOptions,
)

__all__ = [
    'CandidateSpace',
    'PruningOptions',
    'PruningPipelineReport',
    'SetTSPResult',
    'SetTSPSolverOptions',
    'apply_pruning_pipeline',
    'solve_set_tsp',
]
