"""构造第一阶段使用的完整有向集合转移代价。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DirectedCosts = dict[Any, dict[Any, float]]


def _boundary_penalties(model, city, convex_sets: Mapping) -> list[tuple[Any, float]]:
    """
    计算客户边界候选点及其无人机服务惩罚。

    输入：
    - model: 原论文模型或其子类，提供卡车/无人机距离、速度和式 (3) 系数。
    - city: 当前客户节点。
    - convex_sets: 客户到边界候选点列表的映射。

    输出：
    - `(边界节点, 无人机服务时间)` 列表。

    实现逻辑：
    - 沿用原 `set_mst()` 的 `sqrt(2) * drone_distance / speed` 近似项，
      只补齐原实现缺失的反向转移，不改变候选点定义。
    """
    penalties = []
    for node in convex_sets.get(city, ()):  # 边界集合可能为空，此时保留卡车直达方案。
        penalty = model.distance['drone'][city][node] / model.speed * model.const
        penalties.append((node, float(penalty)))
    return penalties


def build_directed_set_costs(model, convex_sets: Mapping) -> DirectedCosts:
    """
    构造仓库和客户之间的完整有向集合转移代价矩阵。

    输入：
    - model: `MultiAgentFlyingSidekickTSP` 或兼容子类。
    - convex_sets: Phase 1 使用的客户边界候选点集合。

    输出：
    - 二层字典 `costs[start][end]`，严格分别保存两个方向。

    实现逻辑：
    1. 对仓库到客户、客户到仓库、客户到客户分别计算有向代价。
    2. 每个方向均保留原实现中的“卡车直接访问客户”备选方案。
    3. 边界候选方案沿用论文代码式 (3) 的近似服务代价。

    注意：
    - 原代码只显式计算 `depot -> city`，本函数用镜像逻辑补齐 `city -> depot`。
    - 客户对仍沿用“中心到中心”或“边界到边界”两类方案，避免额外改变原模型语义。
    """
    depots = list(model.depots)
    cities = list(model.cities)
    nodes = depots + cities
    depot_set = set(depots)
    city_set = set(cities)
    penalties = {
        city: _boundary_penalties(model, city, convex_sets)
        for city in cities
    }

    costs: DirectedCosts = {start: {} for start in nodes}
    for start in nodes:
        for end in nodes:
            if start == end:
                costs[start][end] = 0.0
                continue

            # 仓库之间没有客户服务动作，直接使用有向卡车最短路。
            if start in depot_set and end in depot_set:
                costs[start][end] = float(model.distance['truck'][start][end])
                continue

            # 仓库到客户：保留卡车直达，并枚举目标客户的边界服务点。
            if start in depot_set and end in city_set:
                best = float(model.distance['truck'][start][end])
                for end_node, end_penalty in penalties[end]:
                    best = min(
                        best,
                        float(model.distance['truck'][start][end_node]) + end_penalty,
                    )
                costs[start][end] = best
                continue

            # 客户到仓库：这是原实现缺失的反向情形。
            if start in city_set and end in depot_set:
                best = float(model.distance['truck'][start][end])
                for start_node, start_penalty in penalties[start]:
                    best = min(
                        best,
                        start_penalty + float(model.distance['truck'][start_node][end]),
                    )
                costs[start][end] = best
                continue

            # 客户到客户：沿用原代码的中心直达或两个边界点之间转移。
            best = float(model.distance['truck'][start][end])
            for start_node, start_penalty in penalties[start]:
                for end_node, end_penalty in penalties[end]:
                    best = min(
                        best,
                        start_penalty
                        + float(model.distance['truck'][start_node][end_node])
                        + end_penalty,
                    )
            costs[start][end] = best

    validate_directed_costs(costs, nodes)
    return costs


def validate_directed_costs(costs: Mapping, nodes: Sequence) -> None:
    """
    校验有向代价矩阵是否完整、有限且非负。

    输入：
    - costs: 二层有向代价映射。
    - nodes: 期望覆盖的全部仓库和客户节点。

    输出：
    - 无；发现缺失、负数、NaN 或无穷值时抛出异常。
    """
    for start in nodes:
        if start not in costs:
            raise KeyError(f'有向集合代价缺少起点 {start!r}。')
        for end in nodes:
            if end not in costs[start]:
                raise KeyError(f'有向集合代价缺少 {start!r} -> {end!r}。')
            value = float(costs[start][end])
            if value < 0 or value != value or value == float('inf'):
                raise ValueError(
                    f'有向集合代价 {start!r} -> {end!r} 非法：{value!r}。'
                )

