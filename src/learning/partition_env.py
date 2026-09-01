"""基于代理模型的层次化客户分区局部修复环境。"""

from dataclasses import dataclass
import math

import numpy as np
import torch

from src.learning.dataset import PartitionAction, apply_partition_action
from src.learning.features import (
    GROUP_FEATURE_NAMES,
    extract_group_feature_vector,
)
from src.learning.surrogate import (
    MultiTaskSurrogate,
    SurrogateModelConfig,
    predict_with_uncertainty,
)


ACTION_FEATURE_NAMES = (
    "is_relocate",
    "is_swap",
    "is_stop",
    "affinity_delta_tanh",
    "cost_before",
    "cost_after",
    "cost_improvement",
    "time_risk_before",
    "time_risk_after",
    "time_risk_improvement",
    "timeout_before",
    "timeout_after",
    "timeout_improvement",
    "uncertainty_before",
    "uncertainty_after",
    "uncertainty_improvement",
    "source_fraction_before",
    "target_fraction_before",
    "source_fraction_after",
    "target_fraction_after",
    "complexity_before",
    "complexity_after",
    "complexity_improvement",
)


def _stable_node_key(node):
    """
    为仓库和客户节点生成稳定排序键。

    输入：任意可哈希节点编号。
    输出：由类型名和文本表示组成的元组。
    逻辑：保证策略候选顺序不依赖字典或集合的遍历顺序。
    """
    return type(node).__name__, repr(node)


def _group_key(depot, customers):
    """
    构造代理预测缓存使用的客户组键。

    输入：仓库节点和客户集合。
    输出：稳定的 `(depot, customers)` 元组。
    逻辑：同一客户集合即使输入顺序不同，也只执行一次代理前向计算。
    """
    return depot, tuple(sorted(customers, key=_stable_node_key))


@dataclass(frozen=True)
class PartitionEnvironmentConfig:
    """
    保存局部修复环境的动作规模、奖励和真实复核设置。

    输入：最大步数、候选数量、第二阶段预算、奖励权重及复核阈值。
    输出：不可变环境配置对象。
    逻辑：默认完全使用代理模型；将真实复核预算设为正数即可启用高风险动作复核。
    """

    max_steps: int = 6
    relocate_candidates_per_direction: int = 4
    swap_candidates_per_pair: int = 4
    phase2_budget_seconds: float = 20.0
    cost_weight: float = 1.0
    time_budget_weight: float = 0.35
    timeout_weight: float = 0.75
    uncertainty_weight: float = 0.05
    move_penalty: float = 0.01
    repeated_partition_penalty: float = 0.05
    minimum_improvement: float = 0.01
    mc_samples: int = 1
    real_verification_budget: int = 0
    uncertainty_verification_threshold: float = 2.0
    timeout_verification_threshold: float = 0.75


@dataclass(frozen=True)
class GroupPrediction:
    """
    保存一个客户组的代理或真实下游评估。

    输入：原始特征、成本/时间预测、超时概率、不确定性和结果来源。
    输出：不可变的客户组评价对象。
    逻辑：环境只依赖这一个统一结构，因此真实求解结果可以覆盖代理结果。
    """

    features: tuple
    cost_mean: float
    cost_std: float
    log_time_mean: float
    log_time_std: float
    wall_time_mean: float
    timeout_probability: float
    source: str = "surrogate"


@dataclass(frozen=True)
class PartitionSnapshot:
    """
    保存整个分区的预测目标及其组成。

    输入：成本、超预算风险、超时风险、不确定性和加权总分。
    输出：不可变快照；分数越小表示分区越好。
    逻辑：奖励由动作前后快照的分数下降量计算，避免把成本和时间混为一个标签。
    """

    cost_sum: float
    time_budget_risk: float
    timeout_risk: float
    uncertainty: float
    score: float


@dataclass(frozen=True)
class ActionPreview:
    """
    保存下层候选动作及其代理评估。

    输入：可选分区动作、固定维特征、预计奖励、动作后分组与预测。
    输出：供下层策略打分并供环境执行的不可变对象。
    逻辑：`action=None` 表示下层停止，不改变当前划分。
    """

    action: PartitionAction | None
    features: tuple
    predicted_reward: float
    groups_after: dict
    predictions_after: dict


class SurrogateGroupScorer:
    """
    加载里程碑 B 检查点并批量评价客户组。

    输入：路网、距离、候选集合、检查点和 MC Dropout 次数。
    输出：通过 `predict_many` 返回 `GroupPrediction` 字典。
    逻辑：客户组特征在运行时提取，相同组使用内存缓存避免重复推理。
    """

    def __init__(
        self,
        graph,
        distance,
        convex_sets,
        model,
        normalization,
        feature_names,
        mc_samples=1,
    ):
        """
        初始化代理评分器并保存共享只读数据。

        输入：路网数据、候选集合、模型、归一化参数、特征顺序和采样次数。
        输出：可重复调用的评分器对象。
        逻辑：缓存键仅由仓库和客户集合组成；单个评分器只对应一张地图和一组候选集合。
        """
        self.graph = graph
        self.distance = distance
        self.convex_sets = convex_sets
        self.model = model
        self.normalization = normalization
        self.feature_names = tuple(feature_names)
        self.mc_samples = mc_samples
        self.cache = {}
        self.surrogate_query_count = 0

    @classmethod
    def from_checkpoint(
        cls,
        graph,
        distance,
        convex_sets,
        checkpoint_path,
        mc_samples=1,
    ):
        """
        从里程碑 B 的 PyTorch 检查点创建评分器。

        输入：实例数据、检查点路径和 MC Dropout 次数。
        输出：已加载权重并切换到评估模式的 `SurrogateGroupScorer`。
        逻辑：复原网络配置和权重，同时核对固定特征列顺序。
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        config = SurrogateModelConfig(**checkpoint["model_config"])
        model = MultiTaskSurrogate(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        feature_names = tuple(checkpoint["feature_names"])
        if feature_names != tuple(GROUP_FEATURE_NAMES):
            raise ValueError("代理模型特征顺序与当前环境不一致。")
        return cls(
            graph,
            distance,
            convex_sets,
            model,
            checkpoint["normalization"],
            feature_names,
            mc_samples=mc_samples,
        )

    @property
    def group_observation_dim(self):
        """返回上层策略中每个客户组的观察维数。"""
        return len(self.feature_names) + 5

    def _extract_features(self, depot, customers):
        """
        提取一个客户组的固定顺序原始特征。

        输入：仓库和客户集合。
        输出：NumPy 一维特征向量。
        逻辑：直接复用里程碑 B 的特征定义，避免训练和决策阶段产生口径漂移。
        """
        return extract_group_feature_vector(
            self.graph,
            self.distance,
            depot,
            customers,
            self.convex_sets,
        )

    def predict_many(self, requests):
        """
        批量预测若干客户组并返回按稳定键索引的结果。

        输入：`[(depot, customers), ...]` 请求列表。
        输出：`{group_key: GroupPrediction}` 字典。
        逻辑：先收集缓存缺失项，一次前向完成预测，再合并已有真实或代理缓存。
        """
        ordered_requests = []
        seen = set()
        for depot, customers in requests:
            key = _group_key(depot, customers)
            if key not in seen:
                ordered_requests.append((key, depot, list(customers)))
                seen.add(key)

        missing = [item for item in ordered_requests if item[0] not in self.cache]
        if missing:
            feature_matrix = np.asarray(
                [self._extract_features(depot, customers) for _, depot, customers in missing],
                dtype=np.float32,
            )
            predictions = predict_with_uncertainty(
                self.model,
                feature_matrix,
                self.normalization,
                mc_samples=self.mc_samples,
            )
            for index, (key, _, _) in enumerate(missing):
                self.cache[key] = GroupPrediction(
                    features=tuple(float(value) for value in feature_matrix[index]),
                    cost_mean=float(predictions["cost_mean"][index]),
                    cost_std=float(predictions["cost_std"][index]),
                    log_time_mean=float(predictions["log_time_mean"][index]),
                    log_time_std=float(predictions["log_time_std"][index]),
                    wall_time_mean=max(float(predictions["wall_time_mean"][index]), 0.0),
                    timeout_probability=float(predictions["timeout_probability"][index]),
                    source="surrogate",
                )
            self.surrogate_query_count += len(missing)
        return {key: self.cache[key] for key, _, _ in ordered_requests}

    def set_real_prediction(self, depot, customers, evaluation):
        """
        用真实第二、第三阶段结果覆盖指定客户组的代理缓存。

        输入：仓库、客户集合和 `GroupEvaluator.evaluate` 返回值。
        输出：新建的真实 `GroupPrediction`。
        逻辑：真实耗时转为与训练标签一致的 `log1p`，真实项的不确定性记为零。
        """
        key = _group_key(depot, customers)
        old_prediction = self.predict_many([(depot, customers)])[key]
        prediction = GroupPrediction(
            features=old_prediction.features,
            cost_mean=float(evaluation.final_cost),
            cost_std=0.0,
            log_time_mean=math.log1p(float(evaluation.set_tsp_wall_seconds)),
            log_time_std=0.0,
            wall_time_mean=float(evaluation.set_tsp_wall_seconds),
            timeout_probability=float(evaluation.set_tsp_result.timed_out),
            source="real",
        )
        self.cache[key] = prediction
        return prediction

    def group_observation(self, prediction):
        """
        将客户组预测转换为上层策略使用的归一化观察。

        输入：`GroupPrediction`。
        输出：固定长度 NumPy 向量。
        逻辑：结构特征和两个回归均值按 B 阶段统计量标准化，不确定性保留相对尺度。
        """
        normalization = self.normalization
        normalized_features = (
            np.asarray(prediction.features, dtype=np.float32)
            - np.asarray(normalization["feature_mean"], dtype=np.float32)
        ) / np.asarray(normalization["feature_std"], dtype=np.float32)
        extra = np.asarray(
            [
                (prediction.cost_mean - normalization["cost_mean"])
                / normalization["cost_std"],
                (prediction.log_time_mean - normalization["time_mean"])
                / normalization["time_std"],
                prediction.timeout_probability,
                prediction.cost_std / normalization["cost_std"],
                prediction.log_time_std,
            ],
            dtype=np.float32,
        )
        return np.clip(np.concatenate((normalized_features, extra)), -10.0, 10.0)


class PartitionRepairEnvironment:
    """
    在 Set-MST 初始分区上执行层次化 relocate/swap 局部修复。

    输入：实例、初始分区、代理评分器、环境配置和可选真实单组评估器。
    输出：通过观察、预览和 `step` 接口提供完整训练轨迹。
    逻辑：上层选择仓库对，下层从有效动作列表选择移动、交换或停止。
    """

    def __init__(
        self,
        graph,
        distance,
        depots,
        convex_sets,
        initial_groups,
        scorer,
        config=None,
        group_evaluator=None,
    ):
        """
        初始化一个可反复重置的分区修复环境。

        输入：路网数据、仓库、候选集合、初始组、评分器、配置和真实评估器。
        输出：环境对象，初始化后立即处于初始分区。
        逻辑：真实求解调用预算在环境生命周期内累计，防止多轮训练反复触发昂贵求解。
        """
        self.graph = graph
        self.distance = distance
        self.depots = tuple(sorted(depots, key=_stable_node_key))
        self.convex_sets = convex_sets
        self.initial_groups = {
            depot: tuple(sorted(initial_groups[depot], key=_stable_node_key))
            for depot in self.depots
        }
        self.scorer = scorer
        self.config = config or PartitionEnvironmentConfig()
        self.group_evaluator = group_evaluator
        self.total_customer_count = sum(len(group) for group in self.initial_groups.values())
        self.cost_scale = max(float(self.scorer.normalization["cost_std"]), 1.0)
        self.total_real_solver_calls = 0
        self.reset()

    @property
    def group_observation_dim(self):
        """返回上层客户组观察的固定维数。"""
        return self.scorer.group_observation_dim

    @property
    def upper_pair_feature_dim(self):
        """返回上层单个仓库对特征的固定维数。"""
        return self.group_observation_dim * 3

    @property
    def lower_action_feature_dim(self):
        """返回下层单个动作特征的固定维数。"""
        return len(ACTION_FEATURE_NAMES)

    def _partition_signature(self, groups):
        """
        生成检测重复分区所需的稳定签名。

        输入：客户组字典。
        输出：按仓库顺序排列的客户元组。
        逻辑：用于惩罚立即撤销或循环移动，不参与代理模型输入。
        """
        return tuple(
            (depot, tuple(sorted(groups[depot], key=_stable_node_key)))
            for depot in self.depots
        )

    def _predict_groups(self, groups):
        """
        批量获得当前分区所有客户组的评价。

        输入：客户组字典。
        输出：以仓库为键的 `GroupPrediction` 字典。
        逻辑：将评分器的稳定组键转换成环境更方便使用的仓库键。
        """
        requests = [(depot, groups[depot]) for depot in self.depots]
        keyed_predictions = self.scorer.predict_many(requests)
        return {
            depot: keyed_predictions[_group_key(depot, groups[depot])]
            for depot in self.depots
        }

    def _snapshot(self, predictions):
        """
        聚合各客户组预测并计算分区加权目标。

        输入：按仓库索引的客户组预测。
        输出：`PartitionSnapshot`。
        逻辑：成本为主目标，时间只惩罚超过预算部分，同时计入超时概率和平均不确定性。
        """
        values = list(predictions.values())
        cost_sum = sum(item.cost_mean for item in values)
        time_budget_risk = sum(
            min(
                max(item.wall_time_mean - self.config.phase2_budget_seconds, 0.0)
                / self.config.phase2_budget_seconds,
                5.0,
            )
            for item in values
        )
        timeout_risk = sum(item.timeout_probability for item in values)
        uncertainty = sum(
            item.cost_std / self.cost_scale + item.log_time_std
            for item in values
        ) / max(len(values), 1)
        score = (
            self.config.cost_weight * cost_sum / self.cost_scale
            + self.config.time_budget_weight * time_budget_risk
            + self.config.timeout_weight * timeout_risk
            + self.config.uncertainty_weight * uncertainty
        )
        return PartitionSnapshot(
            cost_sum=float(cost_sum),
            time_budget_risk=float(time_budget_risk),
            timeout_risk=float(timeout_risk),
            uncertainty=float(uncertainty),
            score=float(score),
        )

    def reset(self):
        """
        将环境恢复到原 Set-MST 初始分区。

        输入：无。
        输出：当前分区快照。
        逻辑：清空本轮轨迹和重复分区集合，但保留代理缓存及环境级真实调用计数。
        """
        self.groups = {
            depot: list(customers)
            for depot, customers in self.initial_groups.items()
        }
        self.predictions = self._predict_groups(self.groups)
        self.current_snapshot = self._snapshot(self.predictions)
        self.initial_snapshot = self.current_snapshot
        self.steps = 0
        self.done = False
        self.stop_reason = None
        self.visited_signatures = {self._partition_signature(self.groups)}
        self.episode_real_solver_calls_start = self.total_real_solver_calls
        return self.current_snapshot

    def group_observation_matrix(self):
        """
        构造当前所有客户组的上层观察矩阵。

        输入：无。
        输出：形状为 `[仓库数, group_observation_dim]` 的数组。
        逻辑：仓库顺序稳定，便于上层网络共享编码器。
        """
        return np.asarray(
            [self.scorer.group_observation(self.predictions[depot]) for depot in self.depots],
            dtype=np.float32,
        )

    def upper_pairs(self):
        """
        枚举上层策略可选择的无序仓库对。

        输入：无。
        输出：稳定的 `(source_depot, target_depot)` 列表。
        逻辑：一个仓库对的下层候选同时包含两个 relocate 方向，因此无需重复有序对。
        """
        return [
            (source, target)
            for index, source in enumerate(self.depots)
            for target in self.depots[index + 1:]
        ]

    def upper_pair_feature_matrix(self):
        """
        构造每个有效仓库对的上层特征。

        输入：无。
        输出：`[仓库对数, 3 * group_observation_dim]` 数组。
        逻辑：拼接两个组的观察和绝对差异，使网络同时看到规模、难度及不平衡程度。
        """
        observations = self.group_observation_matrix()
        depot_indices = {depot: index for index, depot in enumerate(self.depots)}
        pair_rows = []
        for source, target in self.upper_pairs():
            source_row = observations[depot_indices[source]]
            target_row = observations[depot_indices[target]]
            pair_rows.append(
                np.concatenate((source_row, target_row, np.abs(source_row - target_row)))
            )
        return np.asarray(pair_rows, dtype=np.float32).reshape(
            -1,
            self.upper_pair_feature_dim,
        )

    def _affinity(self, customer, depot):
        """
        计算客户与仓库的双向平均道路亲和距离。

        输入：客户和仓库节点。
        输出：两个有向道路距离的平均值。
        逻辑：优先保留靠近两个仓库分界的客户，控制下层候选规模。
        """
        truck = self.distance["truck"]
        return float((truck[depot][customer] + truck[customer][depot]) / 2.0)

    def _relocate_candidates(self, source, target):
        """
        生成一个方向上最有希望的 relocate 候选。

        输入：源仓库和目标仓库。
        输出：按亲和力变化排序并截断的 `PartitionAction` 列表。
        逻辑：变化越小越接近边界；稳定排序用于保证复现实验。
        """
        candidates = []
        for customer in self.groups[source]:
            affinity_delta = self._affinity(customer, target) - self._affinity(customer, source)
            candidates.append(
                PartitionAction(
                    action_type="relocate",
                    source_depot=source,
                    target_depot=target,
                    source_customer=customer,
                    target_customer=None,
                    affinity_delta=float(affinity_delta),
                )
            )
        candidates.sort(
            key=lambda action: (
                action.affinity_delta,
                abs(action.affinity_delta),
                _stable_node_key(action.source_customer),
            )
        )
        return candidates[: self.config.relocate_candidates_per_direction]

    def _swap_candidates(self, source, target):
        """
        生成一个仓库对上最有希望的 swap 候选。

        输入：两个仓库。
        输出：按交换前后亲和力变化排序并截断的动作列表。
        逻辑：只枚举当前两个组的客户组合，保留预测上更可能改善空间归属的交换。
        """
        candidates = []
        for source_customer in self.groups[source]:
            for target_customer in self.groups[target]:
                before = (
                    self._affinity(source_customer, source)
                    + self._affinity(target_customer, target)
                )
                after = (
                    self._affinity(source_customer, target)
                    + self._affinity(target_customer, source)
                )
                candidates.append(
                    PartitionAction(
                        action_type="swap",
                        source_depot=source,
                        target_depot=target,
                        source_customer=source_customer,
                        target_customer=target_customer,
                        affinity_delta=float(after - before),
                    )
                )
        candidates.sort(
            key=lambda action: (
                action.affinity_delta,
                abs(action.affinity_delta),
                _stable_node_key(action.source_customer),
                _stable_node_key(action.target_customer),
            )
        )
        return candidates[: self.config.swap_candidates_per_pair]

    def _candidate_actions(self, pair):
        """
        构造一个仓库对下全部有效的下层动作。

        输入：上层选中的无序仓库对。
        输出：双向 relocate、swap 和最终停止标记。
        逻辑：只返回真实可执行动作，相当于对非法动作应用动态掩码。
        """
        source, target = pair
        return (
            self._relocate_candidates(source, target)
            + self._relocate_candidates(target, source)
            + self._swap_candidates(source, target)
            + [None]
        )

    def _pair_terms(self, predictions, pair):
        """
        计算下层动作特征中的仓库对风险项。

        输入：预测字典和仓库对。
        输出：成本、时间风险、超时风险、不确定性和标准化复杂度。
        逻辑：各项与环境总目标使用相同口径，但仅聚合被动作改变的两个组。
        """
        pair_predictions = [predictions[depot] for depot in pair]
        cost = sum(item.cost_mean for item in pair_predictions) / self.cost_scale
        time_risk = sum(
            min(
                max(item.wall_time_mean - self.config.phase2_budget_seconds, 0.0)
                / self.config.phase2_budget_seconds,
                5.0,
            )
            for item in pair_predictions
        )
        timeout = sum(item.timeout_probability for item in pair_predictions)
        uncertainty = sum(
            item.cost_std / self.cost_scale + item.log_time_std
            for item in pair_predictions
        ) / 2.0
        complexity_index = GROUP_FEATURE_NAMES.index("set_tsp_complexity_proxy")
        mean_value = self.scorer.normalization["feature_mean"][complexity_index]
        std_value = self.scorer.normalization["feature_std"][complexity_index]
        complexity = sum(
            (item.features[complexity_index] - mean_value) / std_value
            for item in pair_predictions
        ) / 2.0
        return cost, time_risk, timeout, uncertainty, complexity

    def _action_feature_vector(self, action, pair, groups_after, predictions_after):
        """
        将一个下层候选转换为固定维动作特征。

        输入：动作、仓库对、动作后分区及预测。
        输出：长度为 `len(ACTION_FEATURE_NAMES)` 的元组。
        逻辑：提供动作类型及目标各组成项的前后变化，不直接暴露最终加权奖励。
        """
        before = self._pair_terms(self.predictions, pair)
        after = self._pair_terms(predictions_after, pair)
        source, target = pair
        total = max(self.total_customer_count, 1)
        action_type = action.action_type if action is not None else "stop"
        affinity_delta = action.affinity_delta if action is not None else 0.0
        affinity_scale = max(
            np.mean(
                [
                    max(self.predictions[depot].features[7], 0.0)
                    for depot in pair
                ]
            ),
            1e-6,
        )
        values = (
            float(action_type == "relocate"),
            float(action_type == "swap"),
            float(action_type == "stop"),
            math.tanh(affinity_delta / affinity_scale),
            before[0],
            after[0],
            before[0] - after[0],
            before[1],
            after[1],
            before[1] - after[1],
            before[2],
            after[2],
            before[2] - after[2],
            before[3],
            after[3],
            before[3] - after[3],
            len(self.groups[source]) / total,
            len(self.groups[target]) / total,
            len(groups_after[source]) / total,
            len(groups_after[target]) / total,
            before[4],
            after[4],
            before[4] - after[4],
        )
        return tuple(float(np.clip(value, -10.0, 10.0)) for value in values)

    def preview_pair(self, pair_index):
        """
        批量预览一个上层仓库对的全部下层候选。

        输入：`upper_pairs()` 中的索引。
        输出：`ActionPreview` 列表，最后一项固定为停止。
        逻辑：先收集所有动作后新组，再一次批量代理推理，降低逐动作调用开销。
        """
        pair = self.upper_pairs()[pair_index]
        actions = self._candidate_actions(pair)
        candidate_groups = []
        requests = []
        for action in actions:
            groups_after = (
                apply_partition_action(self.groups, action)
                if action is not None
                else {depot: list(customers) for depot, customers in self.groups.items()}
            )
            candidate_groups.append(groups_after)
            if action is not None:
                requests.extend(
                    (depot, groups_after[depot])
                    for depot in (action.source_depot, action.target_depot)
                )
        keyed_predictions = self.scorer.predict_many(requests) if requests else {}

        previews = []
        for action, groups_after in zip(actions, candidate_groups):
            predictions_after = dict(self.predictions)
            if action is not None:
                for depot in (action.source_depot, action.target_depot):
                    predictions_after[depot] = keyed_predictions[
                        _group_key(depot, groups_after[depot])
                    ]
                snapshot_after = self._snapshot(predictions_after)
                reward = self.current_snapshot.score - snapshot_after.score - self.config.move_penalty
                if self._partition_signature(groups_after) in self.visited_signatures:
                    reward -= self.config.repeated_partition_penalty
            else:
                reward = 0.0
            previews.append(
                ActionPreview(
                    action=action,
                    features=self._action_feature_vector(
                        action,
                        pair,
                        groups_after,
                        predictions_after,
                    ),
                    predicted_reward=float(np.clip(reward, -10.0, 10.0)),
                    groups_after=groups_after,
                    predictions_after=predictions_after,
                )
            )
        return previews

    def best_predicted_action(self):
        """
        用一步代理局部搜索生成层次策略的模仿标签。

        输入：无。
        输出：`(上层索引, 下层索引, 最佳奖励)`；上层停止索引等于仓库对数量。
        逻辑：遍历有效候选，改善不足时选择全局停止，作为低成本启发式教师。
        """
        best_upper = len(self.upper_pairs())
        best_lower = None
        best_reward = 0.0
        for upper_index in range(len(self.upper_pairs())):
            previews = self.preview_pair(upper_index)
            for lower_index, preview in enumerate(previews):
                if preview.predicted_reward > best_reward:
                    best_upper = upper_index
                    best_lower = lower_index
                    best_reward = preview.predicted_reward
        if best_reward < self.config.minimum_improvement:
            return len(self.upper_pairs()), None, 0.0
        return best_upper, best_lower, best_reward

    def _needs_real_verification(self, prediction):
        """
        判断一个动作后客户组是否需要真实求解复核。

        输入：代理客户组预测。
        输出：布尔值。
        逻辑：相对不确定性或超时概率超过阈值时触发，且只复核代理来源结果。
        """
        uncertainty = prediction.cost_std / self.cost_scale + prediction.log_time_std
        return prediction.source == "surrogate" and (
            uncertainty >= self.config.uncertainty_verification_threshold
            or prediction.timeout_probability >= self.config.timeout_verification_threshold
        )

    def _verify_action_groups(self, preview):
        """
        在全局预算内用真实求解器复核高风险动作后的两个客户组。

        输入：即将执行的 `ActionPreview`。
        输出：可能被真实结果更新的预测字典。
        逻辑：按风险从高到低调用 `GroupEvaluator`，每个客户组调用计入环境生命周期预算。
        """
        predictions = dict(preview.predictions_after)
        if self.group_evaluator is None or preview.action is None:
            return predictions
        remaining_budget = (
            self.config.real_verification_budget - self.total_real_solver_calls
        )
        if remaining_budget <= 0:
            return predictions

        action = preview.action
        candidate_depots = [action.source_depot, action.target_depot]
        candidate_depots.sort(
            key=lambda depot: (
                predictions[depot].cost_std / self.cost_scale
                + predictions[depot].log_time_std
                + predictions[depot].timeout_probability
            ),
            reverse=True,
        )
        for depot in candidate_depots:
            if remaining_budget <= 0:
                break
            if not self._needs_real_verification(predictions[depot]):
                continue
            evaluation = self.group_evaluator.evaluate(
                depot,
                preview.groups_after[depot],
            )
            predictions[depot] = self.scorer.set_real_prediction(
                depot,
                preview.groups_after[depot],
                evaluation,
            )
            self.total_real_solver_calls += 1
            remaining_budget -= 1
        return predictions

    def step(self, upper_index, lower_index=None):
        """
        执行一次层次动作并返回奖励、结束标记和轨迹信息。

        输入：上层仓库对索引或全局停止索引，以及可选下层候选索引。
        输出：`(snapshot, reward, done, info)`。
        逻辑：停止动作立即结束；移动动作可经真实复核后更新分区，并在最大步数处截断。
        """
        if self.done:
            raise RuntimeError("当前 episode 已结束，请先调用 reset()。")
        pair_count = len(self.upper_pairs())
        if upper_index == pair_count:
            self.done = True
            self.stop_reason = "upper_stop"
            return self.current_snapshot, 0.0, True, {
                "action_type": "stop",
                "stop_reason": self.stop_reason,
                "real_solver_calls": 0,
            }

        previews = self.preview_pair(upper_index)
        preview = previews[lower_index]
        if preview.action is None:
            self.done = True
            self.stop_reason = "lower_stop"
            return self.current_snapshot, 0.0, True, {
                "action_type": "stop",
                "stop_reason": self.stop_reason,
                "real_solver_calls": 0,
            }

        calls_before = self.total_real_solver_calls
        predictions_after = self._verify_action_groups(preview)
        snapshot_after = self._snapshot(predictions_after)
        reward = self.current_snapshot.score - snapshot_after.score - self.config.move_penalty
        signature = self._partition_signature(preview.groups_after)
        if signature in self.visited_signatures:
            reward -= self.config.repeated_partition_penalty
        self.groups = {
            depot: list(customers)
            for depot, customers in preview.groups_after.items()
        }
        self.predictions = predictions_after
        self.current_snapshot = snapshot_after
        self.visited_signatures.add(signature)
        self.steps += 1
        if self.steps >= self.config.max_steps:
            self.done = True
            self.stop_reason = "max_steps"
        return self.current_snapshot, float(np.clip(reward, -10.0, 10.0)), self.done, {
            "action_type": preview.action.action_type,
            "action": preview.action.to_dict(),
            "predicted_reward": preview.predicted_reward,
            "stop_reason": self.stop_reason,
            "real_solver_calls": self.total_real_solver_calls - calls_before,
        }

