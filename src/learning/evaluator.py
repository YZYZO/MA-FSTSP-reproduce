"""复用原 MA-FSTSP 模型的单客户组真实评估器。"""

from dataclasses import asdict, dataclass
import time

from src.learning.settings import SetTSPSolverSettings
from src.set_tsp_solver import SetTSPSolveResult, solve_set_tsp_with_telemetry


def _stable_node_key(node):
    """
    为客户节点生成稳定排序键。

    输入：节点编号。
    输出：类型名与文本表示组成的元组。
    逻辑：相同客户集合无论输入顺序如何，都生成相同的 Set-TSP 编号体系和缓存结果。
    """
    return type(node).__name__, repr(node)


@dataclass(frozen=True)
class GroupEvaluationResult:
    """
    保存一个仓库客户组的第二、第三阶段真实评估结果。

    输入：仓库、客户、顺序、最终成本、两阶段耗时及 Set-TSP 遥测。
    输出：不可变结果对象，可写入 SQLite 缓存。
    逻辑：客户组是后续代理模型的最小标签单元，分区总指标可由各组结果聚合。
    """

    depot: object
    customers: tuple
    visit_route: tuple
    final_cost: float
    set_tsp_wall_seconds: float
    phase3_seconds: float
    set_tsp_result: SetTSPSolveResult
    cache_hit: bool = False

    def to_dict(self):
        """
        将客户组评估结果转换为缓存字典。

        输入：当前结果对象。
        输出：包含普通列表和嵌套 Set-TSP 字典的对象。
        逻辑：不保存完整 DP 状态或路线解，避免标签缓存膨胀。
        """
        payload = asdict(self)
        payload["customers"] = list(self.customers)
        payload["visit_route"] = list(self.visit_route)
        payload["set_tsp_result"] = self.set_tsp_result.to_dict()
        payload["cache_hit"] = False
        return payload

    @classmethod
    def from_dict(cls, payload, cache_hit=False):
        """
        从 SQLite 结果字典恢复客户组评估对象。

        输入：缓存字典和是否命中缓存的标记。
        输出：新的 `GroupEvaluationResult`。
        逻辑：恢复元组和嵌套遥测对象，并覆盖运行期 `cache_hit` 字段。
        """
        values = dict(payload)
        values["customers"] = tuple(values["customers"])
        values["visit_route"] = tuple(values["visit_route"])
        values["set_tsp_result"] = SetTSPSolveResult.from_dict(
            values["set_tsp_result"]
        )
        values["cache_hit"] = cache_hit
        return cls(**values)


def build_set_tsp_inputs(model, depot, customers, convex_sets):
    """
    按原 `get_seq` 公式构造一个客户组的 Set-TSP 三类输入。

    输入：原模型、仓库、稳定排序后的客户和全局候选集合字典。
    输出：`(local_sets, distance, internal_distance)`。
    逻辑：完全复用原模型的距离截断与集合间卡车距离定义，保证标签含义一致。
    """
    local_sets = [[depot]] + [list(convex_sets[city]) for city in customers]
    entities = [depot] + list(customers)
    internal_distance = [
        [
            [
                max(
                    model.distance["truck"][end][start],
                    model.cut_off(
                        model.distance["drone"][start][entity]
                        + model.distance["drone"][entity][end],
                        model.limit,
                    ),
                )
                / model.speed
                for start in candidate_set
            ]
            for end in candidate_set
        ]
        for candidate_set, entity in zip(local_sets, entities)
    ]
    between_distance = [
        [
            [
                [model.distance["truck"][start][end] for end in end_set]
                for start in start_set
            ]
            for end_set in local_sets
        ]
        for start_set in local_sets
    ]
    return local_sets, between_distance, internal_distance


class GroupEvaluator:
    """
    在不改变原模型分组和累计成本的前提下评估单个客户组。

    输入：原 MA-FSTSP 模型、候选集合、地图标识、可选缓存和求解配置。
    输出：通过 `evaluate` 返回 `GroupEvaluationResult`。
    逻辑：只调用原模型的 LKH 和第三阶段 DP，Set-TSP 使用独立遥测求解器。
    """

    def __init__(
        self,
        model,
        convex_sets,
        map_id,
        cache=None,
        solver_settings=None,
    ):
        """
        初始化单组评估器并保存只读依赖。

        输入：模型、候选集合、地图标识、缓存和求解设置。
        输出：可重复调用的评估器对象。
        逻辑：不复制大型路网距离矩阵，只持有原对象引用。
        """
        self.model = model
        self.convex_sets = convex_sets
        self.map_id = map_id
        self.cache = cache
        self.solver_settings = solver_settings or SetTSPSolverSettings()

    def _cache_parameters(self):
        """
        构造会影响单组求解结果的缓存参数。

        输入：当前模型和求解配置。
        输出：稳定的普通字典。
        逻辑：客户集合之外的算法参数变化时必须形成不同缓存键。
        """
        return {
            "solver": "set_tsp_telemetry_v1",
            "time_limit_seconds": self.solver_settings.time_limit_seconds,
            "threads": self.solver_settings.threads,
            "seed": self.solver_settings.seed,
            "drone_count": self.model.drone,
            "drone_limit": self.model.limit,
            "drone_speed": self.model.speed,
            "theta": list(self.model.theta),
        }

    def evaluate(self, depot, customers):
        """
        求解一个仓库客户组的 Set-TSP 顺序和第三阶段成本。

        输入：仓库节点和属于该仓库的客户集合。
        输出：包含最终成本、时间和遥测的 `GroupEvaluationResult`。
        逻辑：先查询缓存；未命中时用 LKH 准备兜底顺序，再真实求解并运行原 DP。
        """
        ordered_customers = tuple(sorted(customers, key=_stable_node_key))
        cache_key = None
        input_json = None
        if self.cache is not None:
            cache_key, input_json = self.cache.make_key(
                self.map_id,
                depot,
                ordered_customers,
                self._cache_parameters(),
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return GroupEvaluationResult.from_dict(cached, cache_hit=True)

        local_sets, between_distance, internal_distance = build_set_tsp_inputs(
            self.model,
            depot,
            ordered_customers,
            self.convex_sets,
        )
        fallback_sequence = self.model.lkh(depot, list(ordered_customers))

        set_tsp_start = time.perf_counter()
        set_tsp_result = solve_set_tsp_with_telemetry(
            local_sets,
            between_distance,
            internal_distance,
            time_limit_seconds=self.solver_settings.time_limit_seconds,
            threads=self.solver_settings.threads,
            seed=self.solver_settings.seed,
            fallback_sequence=fallback_sequence,
        )
        set_tsp_wall_seconds = time.perf_counter() - set_tsp_start

        sequence = set_tsp_result.sequence
        visit_route = (
            depot,
            *[ordered_customers[index - 1] for index in sequence[1:-1]],
            depot,
        )

        if len(ordered_customers) == 0:
            final_cost = 0.0
            phase3_seconds = 0.0
        else:
            phase3_start = time.perf_counter()
            _, final_cost = self.model.local_search_multi_drone_appr(
                list(visit_route),
                depot,
            )
            phase3_seconds = time.perf_counter() - phase3_start

        result = GroupEvaluationResult(
            depot=depot,
            customers=ordered_customers,
            visit_route=visit_route,
            final_cost=float(final_cost),
            set_tsp_wall_seconds=float(set_tsp_wall_seconds),
            phase3_seconds=float(phase3_seconds),
            set_tsp_result=set_tsp_result,
            cache_hit=False,
        )
        if self.cache is not None:
            self.cache.set(cache_key, input_json, result.to_dict())
        return result
