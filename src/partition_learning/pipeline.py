"""组织 Oracle、初始监督学习和模型引导主动扩充三轮本地实验。"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable

from .cache import GroupEvaluationCache
from .candidates import (
    PartitionCandidate,
    generate_active_pool,
    generate_candidates,
)
from .dataset import (
    ExperimentInstance,
    discover_result_files,
    load_instances,
    oracle_subset_indices,
    select_instance_indices,
)
from .evaluator import EVALUATOR_VERSION, evaluate_partition, is_unlimited_solver_time
from .features import action_delta_features
from .models import ModelEnsemble, train_model_ensemble
from .partition_algorithms import (
    generate_direct_partition_candidates,
    generate_partition_action_candidates,
)
from .reporting import (
    append_jsonl,
    deduplicate_candidate_records,
    deduplicate_partition_records,
    oracle_report,
    read_jsonl,
    write_json,
    write_jsonl,
    write_oracle_markdown,
    write_partition_algorithm_markdown,
)
from .road import (
    DroneDistanceMatrix,
    TerminalRoadMatrix,
    build_road_csr,
    build_spatial_sets,
    compare_boundary_sizes,
    load_road_graph,
    precompute_terminal_distances,
)


def _customer_count_from_result_path(path: Path) -> int:
    """
    从历史结果 NPZ 文件名提取客户规模。

    输入为形如 `...-boston_11k-100.npz` 的路径；输出末尾的客户数量整数。
    """
    return int(path.stem.rsplit("-", 1)[-1])


# 24候选的固定族配额：既覆盖直接分区算法，也保留多种局部动作强度。
FAMILY_STRATIFIED_QUOTAS: tuple[tuple[str, int], ...] = (
    ("action_bottleneck", 3),
    ("action_pair_reassign", 3),
    ("action_relocate", 3),
    ("action_swap", 2),
    ("anchored_balanced_graph", 2),
    ("capacitated_road_kmedoids", 2),
    ("constrained_multiroot_forest", 2),
    ("generalized_assignment", 2),
    ("legacy_balance", 1),
    ("legacy_burden", 1),
    ("legacy_cluster", 1),
    ("legacy_road", 1),
)


def _spread_items(items: list, count: int) -> list:
    """输入同一算法族的有序候选，输出覆盖首尾强度的等距子集。"""
    if count >= len(items):
        return list(items)
    return [items[index] for index in select_instance_indices(len(items), count)]


def _family_stratified_subset(items: list, limit: int, family_of, is_stay) -> list:
    """
    按算法族和动作强度选择固定预算候选。

    输入为候选列表、数量上限及族/基线访问器；输出确定性子集。默认24个时包含MST、
    全部主要算法族，并为直接分区和局部动作保留多个强度，避免简单截断造成族偏置。
    """
    if len(items) <= limit:
        return list(items)
    stay_items = [item for item in items if is_stay(item)]
    selected = stay_items[:1]
    used = {id(item) for item in selected}
    buckets: dict[str, list] = {}
    for item in items:
        if id(item) in used:
            continue
        buckets.setdefault(str(family_of(item)), []).append(item)

    if limit >= 24:
        for family, quota in FAMILY_STRATIFIED_QUOTAS:
            for item in _spread_items(buckets.get(family, []), quota):
                if len(selected) >= limit:
                    break
                selected.append(item)
                used.add(id(item))
    else:
        # 小预算仍先保证算法族覆盖，每轮从每族取一个尚未使用的候选。
        family_order = [family for family, _ in FAMILY_STRATIFIED_QUOTAS]
        while len(selected) < limit and any(buckets.get(family) for family in family_order):
            for family in family_order:
                remaining = [item for item in buckets.get(family, []) if id(item) not in used]
                if remaining and len(selected) < limit:
                    selected.append(remaining[0])
                    used.add(id(remaining[0]))

    # 某些实例可能因分区去重缺少某族；剩余额度按族轮询补齐。
    family_order = [family for family, _ in FAMILY_STRATIFIED_QUOTAS]
    family_order.extend(sorted(set(buckets) - set(family_order)))
    while len(selected) < limit:
        added = False
        for family in family_order:
            remaining = [item for item in buckets.get(family, []) if id(item) not in used]
            if remaining and len(selected) < limit:
                selected.append(remaining[0])
                used.add(id(remaining[0]))
                added = True
        if not added:
            break
    return selected


def _round_robin_sequences(sequences: list[list]) -> list:
    """输入按来源分组的实例列表，输出逐轮交错的稳定顺序。"""
    output = []
    max_length = max((len(sequence) for sequence in sequences), default=0)
    for position in range(max_length):
        for sequence in sequences:
            if position < len(sequence):
                output.append(sequence[position])
    return output


def _quota_indices(total: int, quota: int, preferred: Iterable[int] = ()) -> list[int]:
    """
    以既有实例优先填充分层配额。

    输入为文件实例总数、目标数和已存在下标；输出稳定下标。既有数量足够时只从其中等距取样，
    不足时再从全文件的等距网格补齐，从而避免重复求解已经覆盖充分的来源。
    """
    preferred_values = sorted({int(index) for index in preferred if 0 <= int(index) < total})
    if len(preferred_values) >= quota:
        positions = select_instance_indices(len(preferred_values), quota)
        return [preferred_values[position] for position in positions]
    selected = list(preferred_values)
    for index in select_instance_indices(total, min(total, quota)) + list(range(total)):
        if index not in selected:
            selected.append(index)
        if len(selected) >= min(total, quota):
            break
    return sorted(selected)


class ThreeRoundExperiment:
    """以可续跑缓存执行三轮候选评价、监督建模和主动扩充。"""

    def __init__(
        self,
        result_root: str | Path,
        output_dir: str | Path,
        *,
        supervised_instances_per_file: int = 10,
        candidates_per_instance: int = 12,
        active_pool_size: int = 64,
        active_selections_per_instance: int = 3,
        cost_limit: float = 0.10,
        solver_time_limit: float = 600.0,
        solver_threads: int = 1,
        solver_seed: int = 0,
        solver_mip_gap: float = 1e-4,
        max_binary_variables: int = 0,
        distance_batch_size: int = 128,
        random_seed: int = 260915,
        include_55k: bool = False,
        only_graph: str | None = None,
        customer_counts: tuple[int, ...] | None = None,
        evaluation_workers: int = 1,
        algorithm_instances_per_file: int = 10,
        algorithm_candidates_per_instance: int = 24,
        bootstrap_record_paths: tuple[str | Path, ...] = (),
    ):
        """
        保存输入、输出和统一实验预算。

        输入为 NPZ 根目录、输出目录及三轮规模参数；输出为可调用的实验对象。
        所有轮次共用组缓存和求解设置，保证同一仓库组不会因轮次不同而重复计时。
        """
        self.result_root = Path(result_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.result_files = discover_result_files(self.result_root)
        self.include_55k = bool(include_55k)
        self.only_graph = only_graph
        # 可选客户规模过滤用于分别研究 50/100/150 客户的可学习性。
        self.customer_counts = (
            tuple(sorted(set(map(int, customer_counts)))) if customer_counts else None
        )
        self.evaluation_workers = int(evaluation_workers)
        self.algorithm_instances_per_file = int(algorithm_instances_per_file)
        self.algorithm_candidates_per_instance = int(algorithm_candidates_per_instance)
        # 既有精确候选可作为新一轮的种子数据，避免重复求解已经完成的实例。
        self.bootstrap_record_paths = tuple(Path(path).resolve() for path in bootstrap_record_paths)
        self.execution_files = [
            path for path in self.result_files
            if (self.include_55k or "manhattan_55k" not in path.name)
            and (self.only_graph is None or self.only_graph in path.name)
            and (
                self.customer_counts is None
                or _customer_count_from_result_path(path) in self.customer_counts
            )
        ]
        if not self.execution_files:
            raise ValueError("当前本地保护与路网过滤条件没有留下可执行的 NPZ。")
        self.supervised_instances_per_file = supervised_instances_per_file
        self.candidates_per_instance = candidates_per_instance
        self.active_pool_size = active_pool_size
        self.active_selections_per_instance = active_selections_per_instance
        self.cost_limit = cost_limit
        self.solver_time_limit = solver_time_limit
        self.solver_threads = solver_threads
        self.solver_seed = solver_seed
        self.solver_mip_gap = solver_mip_gap
        self.max_binary_variables = max_binary_variables
        self.distance_batch_size = distance_batch_size
        self.random_seed = random_seed
        self.algorithm_run_configuration = self._build_algorithm_run_configuration()
        configuration_text = json.dumps(
            self.algorithm_run_configuration, sort_keys=True, ensure_ascii=False
        )
        configuration_digest = hashlib.sha256(configuration_text.encode("utf-8")).hexdigest()[:12]
        readable_limit = format(float(self.solver_time_limit), "g").replace(".", "p")
        # 运行身份隔离不同时间上限和样本配置，防止旧 completed_instances 错误跳过新实验。
        self.algorithm_run_id = f"tl{readable_limit}_{configuration_digest}"
        self.cache_path = self.output_dir / "group_evaluations.sqlite3"
        self._graph_name: str | None = None
        self._graph = None
        self._road_csr = None
        self._write_manifest()

    def _build_algorithm_run_configuration(self) -> dict[str, object]:
        """
        生成决定算法候选、真值标签和训练集合的完整配置。

        输入来自已初始化的实验字段；输出可稳定序列化的字典，用于目录隔离和断点续跑校验。
        """
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "solver_time_limit": float(self.solver_time_limit),
            "solver_time_unlimited": is_unlimited_solver_time(self.solver_time_limit),
            "solver_threads": int(self.solver_threads),
            "solver_seed": int(self.solver_seed),
            "solver_mip_gap": float(self.solver_mip_gap),
            "max_binary_variables": int(self.max_binary_variables),
            "evaluation_workers": 1,
            "algorithm_instances_per_file": int(self.algorithm_instances_per_file),
            "algorithm_candidates_per_instance": int(self.algorithm_candidates_per_instance),
            "candidate_policy_version": "family_stratified_v1",
            "instance_order": "source_round_robin_v1",
            "bootstrap_record_paths": [str(path) for path in self.bootstrap_record_paths],
            "customer_counts": list(self.customer_counts) if self.customer_counts else None,
            "include_55k": bool(self.include_55k),
            "only_graph": self.only_graph,
            "random_seed": int(self.random_seed),
        }

    def _write_manifest(self) -> None:
        """把输入文件、抽样和统一求解预算写入实验清单。"""
        files = []
        for path in self.result_files:
            if path not in self.execution_files:
                # 本次不执行的文件只记路径；特别地，本地不会打开 55K NPZ。
                files.append({
                    "path": str(path),
                    "instance_count": None,
                    "supervised_indices": "generated_on_server",
                    "oracle_indices": "generated_on_server",
                    "enabled_for_this_run": False,
                })
                continue
            with __import__("numpy").load(path, allow_pickle=True) as data:
                total = int(len(data["instance_indices"]))
            supervised = select_instance_indices(total, self.supervised_instances_per_file)
            files.append({
                "path": str(path),
                "instance_count": total,
                "supervised_indices": supervised,
                "oracle_indices": oracle_subset_indices(supervised),
                "enabled_for_this_run": path in self.execution_files,
            })
        write_json(self.output_dir / "manifest.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "result_root": str(self.result_root),
            "files": files,
            "configuration": {
                "supervised_instances_per_file": self.supervised_instances_per_file,
                "candidates_per_instance": self.candidates_per_instance,
                "active_pool_size": self.active_pool_size,
                "active_selections_per_instance": self.active_selections_per_instance,
                "cost_limit": self.cost_limit,
                "solver_time_limit": self.solver_time_limit,
                "solver_time_unlimited": is_unlimited_solver_time(self.solver_time_limit),
                "solver_threads": self.solver_threads,
                "solver_seed": self.solver_seed,
                "solver_mip_gap": self.solver_mip_gap,
                "max_binary_variables": self.max_binary_variables,
                "distance_batch_size": self.distance_batch_size,
                "random_seed": self.random_seed,
                "include_55k": self.include_55k,
                "only_graph": self.only_graph,
                "customer_counts": self.customer_counts,
                "evaluation_workers": self.evaluation_workers,
                "algorithm_instances_per_file": self.algorithm_instances_per_file,
                "algorithm_candidates_per_instance": self.algorithm_candidates_per_instance,
                "candidate_policy_version": "family_stratified_v1",
                "instance_order": "source_round_robin_v1",
                "bootstrap_record_paths": [str(path) for path in self.bootstrap_record_paths],
                "algorithm_run_id": self.algorithm_run_id,
                "evaluator_version": EVALUATOR_VERSION,
            },
            "local_safety": {
                "manhattan_55k_enabled": self.include_55k,
                "excluded_55k_files": [
                    str(path) for path in self.result_files
                    if "manhattan_55k" in path.name and path not in self.execution_files
                ],
                "note": "默认禁止在本地载入和求解 55K；两份 NPZ 留给服务器任务。",
            },
        })
        self._write_55k_server_manifest()

    def _write_55k_server_manifest(self) -> None:
        """
        为被本地保护排除的两份 55K NPZ 写出服务器续跑清单。

        输入来自当前实验配置；无显式输出。清单只记录文件名，
        不在本地打开 55K NPZ/GraphML，索引由服务器运行时再生成。
        """
        # 只利用文件发现阶段已获得的路径字符串，避免本地触碰 55K 数据内容。
        files = [
            {"path": str(path), "selection": "generated_on_server"}
            for path in self.result_files
            if "manhattan_55k" in path.name
        ]
        write_json(self.output_dir / "server_55k_manifest.json", {
            "kind": "deferred_55k_server_tasks",
            "files": files,
            "reason": "用户明确要求本地不运行 55K，避免内存不足。",
            "server_command": (
                "python scripts/run_partition_learning_rounds.py --round algorithms "
                "--include-55k --only-graph manhattan_55k "
                "--solver-time-limit 600 --evaluation-workers 1 "
                "--algorithm-candidates-per-instance 24 "
                "--output-dir results/partition_learning_260915_server_55k"
            ),
        })

    def _instances(self, oracle_only: bool) -> list[ExperimentInstance]:
        """按文件构造第一轮 24 个或第二轮 80 个历史实例对象。"""
        selected: list[ExperimentInstance] = []
        for path in self.execution_files:
            with __import__("numpy").load(path, allow_pickle=True) as data:
                total = int(len(data["instance_indices"]))
            indices = select_instance_indices(total, self.supervised_instances_per_file)
            if oracle_only:
                indices = oracle_subset_indices(indices)
            selected.extend(load_instances(path, indices))
        return selected

    def _ensure_graph(self, instance: ExperimentInstance) -> None:
        """
        确保当前实例路网及 CSR 已加载。

        输入为实例；无显式输出。切换路网时释放上一张图，再重建稀疏矩阵，控制本地峰值内存。
        """
        if self._graph_name == instance.graph_name:
            return
        self._graph = None
        self._road_csr = None
        gc.collect()
        print(f"[{datetime.now().isoformat(timespec='seconds')}] 加载路网 {instance.graph_name}", flush=True)
        self._graph = load_road_graph(instance.graph_path)
        self._road_csr = build_road_csr(self._graph)
        self._graph_name = instance.graph_name

    def _stable_seed(self, instance_id: str, offset: int = 0) -> int:
        """根据实例身份和全局种子生成跨进程稳定的 32 位随机种子。"""
        digest = hashlib.sha256(f"{self.random_seed}:{instance_id}:{offset}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little")

    def _prepare_instance(self, instance: ExperimentInstance):
        """
        重建空间集合并预计算本实例有限终端距离。

        输入为轻量实例；输出边界、区域、兼容距离对象、审计与预处理统计。
        预处理不计入 Phase 2 标签，与原始 NPZ 把全图距离初始化放在批次外的语义一致。
        """
        self._ensure_graph(instance)
        preparation_started_at = time.perf_counter()
        boundary, regions = build_spatial_sets(
            self._graph,
            instance.cities,
            instance.drone_limit * instance.theta[0],
            instance.drone_limit / 2.0,
        )
        boundary_audit = compare_boundary_sizes(boundary, instance.boundary_sizes)
        if not boundary_audit["exact"]:
            raise RuntimeError(
                f"{instance.instance_id} 本地边界与 NPZ 不一致：{boundary_audit['examples']}"
            )
        # 终端必须覆盖阶段 2 边界与阶段 3 全部服务区域；只在单实例生命周期内物化。
        terminals = set(instance.depots) | set(instance.cities)
        for nodes in boundary.values():
            terminals.update(nodes)
        for nodes in regions.values():
            terminals.update(nodes)
        terminal_data = precompute_terminal_distances(
            self._road_csr,
            terminals,
            batch_size=self.distance_batch_size,
        )
        truck = TerminalRoadMatrix(terminal_data)
        drone = DroneDistanceMatrix(self._graph)
        return boundary, regions, {"truck": truck, "drone": drone}, {
            "boundary_audit": boundary_audit,
            "terminal_count": len(terminal_data.terminals),
            "distance_preprocessing_seconds": terminal_data.preprocessing_seconds,
            "total_preparation_seconds": time.perf_counter() - preparation_started_at,
        }

    @staticmethod
    def _compact_evaluation(evaluation: dict) -> dict:
        """移除逐组大字段，输出适合监督数据 JSONL 的候选级真实评价。"""
        return {key: value for key, value in evaluation.items() if key != "group_records"}

    @staticmethod
    def _attach_instance_ranks(records: list[dict]) -> list[dict]:
        """
        为同一实例的精确候选追加二三阶段总时间、成本和 Pareto 排名标签。

        输入为已完成真实评价的候选记录；输出原顺序记录。超时或回退候选不参与真值排名，
        Pareto 排名同时最小化串行 Phase 2 + Phase 3 总时间和最终目标值。
        """
        for record in records:
            record["labels"].update({
                "time_rank": None,
                "phase2_time_rank": None,
                "phase3_time_rank": None,
                "cost_rank": None,
                "pareto_rank": None,
                "pareto_front": False,
            })
        exact_indices = [
            index for index, record in enumerate(records)
            if bool(record["labels"].get("candidate_exact", False))
        ]
        if not exact_indices:
            return records

        time_order = sorted(
            exact_indices,
            key=lambda index: (
                records[index]["labels"]["downstream_total_seconds"],
                records[index]["candidate_name"],
            ),
        )
        phase2_order = sorted(
            exact_indices,
            key=lambda index: (
                records[index]["labels"]["phase2_serial_effective_seconds"],
                records[index]["candidate_name"],
            ),
        )
        phase3_order = sorted(
            exact_indices,
            key=lambda index: (
                records[index]["labels"]["phase3_seconds"],
                records[index]["candidate_name"],
            ),
        )
        cost_order = sorted(
            exact_indices,
            key=lambda index: (records[index]["evaluation"]["final_cost"], records[index]["candidate_name"]),
        )
        for rank, index in enumerate(time_order, start=1):
            records[index]["labels"]["time_rank"] = rank
        for rank, index in enumerate(phase2_order, start=1):
            records[index]["labels"]["phase2_time_rank"] = rank
        for rank, index in enumerate(phase3_order, start=1):
            records[index]["labels"]["phase3_time_rank"] = rank
        for rank, index in enumerate(cost_order, start=1):
            records[index]["labels"]["cost_rank"] = rank

        remaining = set(exact_indices)
        pareto_rank = 1
        while remaining:
            front = []
            for index in remaining:
                time_value = records[index]["labels"]["downstream_total_seconds"]
                cost_value = records[index]["evaluation"]["final_cost"]
                dominated = any(
                    other != index
                    and records[other]["labels"]["downstream_total_seconds"] <= time_value
                    and records[other]["evaluation"]["final_cost"] <= cost_value
                    and (
                        records[other]["labels"]["downstream_total_seconds"] < time_value
                        or records[other]["evaluation"]["final_cost"] < cost_value
                    )
                    for other in remaining
                )
                if not dominated:
                    front.append(index)
            for index in front:
                records[index]["labels"]["pareto_rank"] = pareto_rank
                records[index]["labels"]["pareto_front"] = pareto_rank == 1
                remaining.remove(index)
            pareto_rank += 1
        return records

    def _candidate_record(
        self,
        instance: ExperimentInstance,
        candidate: PartitionCandidate,
        features: dict[str, float],
        evaluation: dict,
        baseline_evaluation: dict,
        preparation: dict,
        *,
        round_name: str,
        acquisition: dict | None = None,
    ) -> dict:
        """把候选、求解前特征、真实动作差值标签和审计信息组成一条记录。"""
        baseline_cost = float(baseline_evaluation["final_cost"])
        candidate_cost = float(evaluation["final_cost"])
        baseline_phase2 = max(float(baseline_evaluation["phase2_serial_effective_seconds"]), 1e-9)
        baseline_phase3 = max(float(baseline_evaluation["phase3_seconds"]), 1e-9)
        baseline_downstream = max(float(
            baseline_evaluation.get(
                "downstream_total_seconds",
                baseline_phase2 + baseline_phase3,
            )
        ), 1e-9)
        candidate_phase2 = float(evaluation["phase2_serial_effective_seconds"])
        candidate_phase3 = float(evaluation["phase3_seconds"])
        candidate_downstream = float(evaluation.get(
            "downstream_total_seconds", candidate_phase2 + candidate_phase3
        ))
        baseline_wall = max(float(baseline_evaluation["phase2_wall_seconds"]), 1e-9)
        candidate_exact = bool(evaluation["candidate_exact"])
        baseline_exact = bool(baseline_evaluation["candidate_exact"])
        labels = {
            # 主时间目标改为用户真正关心的二、三阶段串行总耗时；同时保留分阶段标签。
            "time_saving_ratio": (baseline_downstream - candidate_downstream) / baseline_downstream,
            "downstream_time_saving_ratio": (
                baseline_downstream - candidate_downstream
            ) / baseline_downstream,
            "phase2_time_saving_ratio": (baseline_phase2 - candidate_phase2) / baseline_phase2,
            "phase3_time_saving_ratio": (baseline_phase3 - candidate_phase3) / baseline_phase3,
            "wall_time_saving_ratio": (baseline_wall - float(evaluation["phase2_wall_seconds"])) / baseline_wall,
            "cost_change_ratio": (candidate_cost - baseline_cost) / max(abs(baseline_cost), 1e-9),
            "cost_delta": candidate_cost - baseline_cost,
            "final_cost": candidate_cost,
            "candidate_censored": bool(evaluation["right_censored_groups"]),
            "candidate_right_censored": bool(evaluation["right_censored_groups"]),
            "baseline_censored": bool(baseline_evaluation["right_censored_groups"]),
            "baseline_right_censored": bool(baseline_evaluation["right_censored_groups"]),
            "runtime_label_valid": bool(evaluation["runtime_label_valid"]),
            "objective_label_exact": bool(evaluation["objective_label_exact"]),
            "candidate_exact": candidate_exact,
            "baseline_exact": baseline_exact,
            "relative_labels_exact": candidate_exact and baseline_exact,
            "phase2_serial_effective_seconds": candidate_phase2,
            "phase2_serial_wall_seconds": float(evaluation["phase2_serial_wall_seconds"]),
            "phase3_seconds": candidate_phase3,
            "downstream_total_seconds": candidate_downstream,
            "phase2_max_group_effective_seconds": float(evaluation["phase2_max_group_effective_seconds"]),
            "phase2_build_seconds": float(evaluation["phase2_build_seconds"]),
            "phase2_optimize_seconds": float(evaluation["phase2_optimize_seconds"]),
            "phase2_distance_seconds": float(evaluation["phase2_distance_seconds"]),
            "solver_work_sum": float(evaluation["solver_work_sum"]),
            "phase2_objective_sum": evaluation.get("phase2_objective_sum"),
            "phase2_objective_complete": bool(evaluation.get("phase2_objective_complete", False)),
            "estimated_binary_variables_sum": int(evaluation["estimated_binary_variables_sum"]),
            "estimated_binary_variables_max": int(evaluation["estimated_binary_variables_max"]),
            "num_binary_variables_sum": int(evaluation["num_binary_variables_sum"]),
            "num_constraints_sum": int(evaluation["num_constraints_sum"]),
            "timeout_groups": int(evaluation["timeout_groups"]),
            "right_censored_groups": int(evaluation["right_censored_groups"]),
            "guard_groups": int(evaluation["guard_groups"]),
            "fallback_groups": int(evaluation["fallback_groups"]),
            "solve_outcome_counts": dict(evaluation["solve_outcome_counts"]),
            "sequence_source_counts": dict(evaluation["sequence_source_counts"]),
            "guard_reason_counts": dict(evaluation["guard_reason_counts"]),
            "mip_gap_max": evaluation["mip_gap_max"],
            "cost_feasible_0pct": (candidate_cost - baseline_cost) <= 0.0,
            "cost_feasible_5pct": (candidate_cost - baseline_cost) <= 0.05 * abs(baseline_cost),
            "cost_feasible_10pct": (candidate_cost - baseline_cost) <= 0.10 * abs(baseline_cost),
        }
        return {
            "round": round_name,
            "instance_id": instance.instance_id,
            "source_name": instance.source_name,
            "source_path": str(instance.source_path),
            "graph_name": instance.graph_name,
            "graph_nodes": int(self._graph.number_of_nodes()),
            "graph_edges": int(self._graph.number_of_edges()),
            "instance_index": instance.instance_index,
            "customer_count": len(instance.cities),
            "depot_count": len(instance.depots),
            "candidate_name": candidate.name,
            "candidate_kind": candidate.kind,
            "candidate_strength": candidate.strength,
            "moved_customers": candidate.moved_customers,
            "generator": {
                "family": candidate.generator_family or candidate.kind,
                "parameters": dict(candidate.generator_parameters),
                "generation_seconds": candidate.generation_seconds,
                "parent_name": candidate.parent_name,
                "action_trace": list(candidate.action_trace),
            },
            "partition": {str(depot): list(group) for depot, group in candidate.partition.items()},
            "features": features,
            "evaluation": self._compact_evaluation(evaluation),
            "labels": labels,
            "historical_baseline": {
                "final_cost": instance.historical_cost,
                "phase2_seconds": instance.historical_phase2_seconds,
            },
            "preparation": preparation,
            "timing_protocol": {
                "group_execution": "serial" if self.evaluation_workers == 1 else "parallel_sampling_sum_labels",
                "primary_target": "downstream_total_seconds",
                "targets": [
                    "phase2_serial_effective_seconds",
                    "phase3_seconds",
                    "downstream_total_seconds",
                    "solver_work_sum",
                    "phase2_objective_sum",
                    "final_cost",
                ],
                "regression_eligibility": {
                    "absolute_targets": "candidate_exact_only",
                    "relative_targets": "candidate_and_baseline_exact_only",
                },
                "solver_threads": self.solver_threads,
                "solver_time_limit": self.solver_time_limit,
                "solver_time_unlimited": is_unlimited_solver_time(self.solver_time_limit),
                "solver_seed": self.solver_seed,
                "solver_mip_gap": self.solver_mip_gap,
                "max_binary_variables": self.max_binary_variables,
            },
            "acquisition": acquisition,
        }

    def _evaluate_standard_instance(
        self,
        instance: ExperimentInstance,
        cache: GroupEvaluationCache,
        *,
        round_name: str,
    ) -> list[dict]:
        """生成并真实评价一个实例的固定十二候选，返回带动作差值标签的记录。"""
        boundary, regions, distance, preparation = self._prepare_instance(instance)
        truck, drone = distance["truck"], distance["drone"]
        affinity = lambda depot, city: 0.5 * (truck[depot][city] + truck[city][depot])
        symmetric_road = lambda first, second: 0.5 * (truck[first][second] + truck[second][first])
        candidates = generate_candidates(
            instance.partition,
            instance.cities,
            instance.boundary_sizes,
            affinity,
            symmetric_road,
            max_candidates=self.candidates_per_instance,
            random_seed=self._stable_seed(instance.instance_id),
        )
        evaluations = []
        features = []
        for candidate in candidates:
            features.append(action_delta_features(
                instance.partition,
                candidate,
                instance.boundary_sizes,
                truck.query,
                drone.query,
                graph_nodes=self._graph.number_of_nodes(),
                graph_edges=self._graph.number_of_edges(),
            ))
            evaluations.append(evaluate_partition(
                instance_id=instance.instance_id,
                partition=candidate.partition,
                depots=instance.depots,
                boundary=boundary,
                regions=regions,
                distance=distance,
                drones_per_truck=instance.drones_per_truck,
                drone_limit=instance.drone_limit,
                drone_speed=instance.drone_speed,
                cache=cache,
                time_limit=self.solver_time_limit,
                threads=self.solver_threads,
                seed=self.solver_seed,
                mip_gap=self.solver_mip_gap,
                max_binary_variables=self.max_binary_variables,
                evaluation_workers=self.evaluation_workers,
            ))
        baseline_position = next(index for index, item in enumerate(candidates) if item.name == "stay")
        baseline_evaluation = evaluations[baseline_position]
        records = [
            self._candidate_record(
                instance,
                candidate,
                feature_row,
                evaluation,
                baseline_evaluation,
                preparation,
                round_name=round_name,
            )
            for candidate, feature_row, evaluation in zip(candidates, features, evaluations)
        ]
        return self._attach_instance_ranks(records)

    def _algorithm_instances(
        self,
        preferred_by_source: dict[str, list[int]] | None = None,
    ) -> list[ExperimentInstance]:
        """
        按每个 NPZ 的配额构造新划分算法实验实例，并在不同来源间轮流执行。

        输入可指定已有记录的实例下标；输出优先覆盖这些下标，再用等距分位点补齐。
        每个来源先独立选样，最终按来源轮询，避免某一路网或规模连续占用数天。
        """
        preferred_by_source = preferred_by_source or {}
        selected_by_source: list[list[ExperimentInstance]] = []
        for path in self.execution_files:
            with __import__("numpy").load(path, allow_pickle=True) as data:
                total = int(len(data["instance_indices"]))
            indices = _quota_indices(
                total,
                self.algorithm_instances_per_file,
                preferred_by_source.get(path.stem, ()),
            )
            selected_by_source.append(load_instances(path, indices))
        return _round_robin_sequences(selected_by_source)

    def _load_bootstrap_algorithm_records(self) -> list[dict]:
        """
        读取并裁剪其他实验目录中的已完成候选记录。

        输入来自 ``bootstrap_record_paths`` 配置；输出按每实例当前候选预算分层选出的记录。
        复用记录保留原始真值和求解口径，同时增加来源字段，便于报告追溯。
        """
        imported: list[dict] = []
        for path in self.bootstrap_record_paths:
            for source_record in read_jsonl(path):
                record = dict(source_record)
                record["reuse_provenance"] = {
                    "kind": "bootstrap_candidate_record",
                    "source_path": str(path),
                }
                imported.append(record)

        by_instance: dict[str, list[dict]] = {}
        for record in deduplicate_candidate_records(imported):
            by_instance.setdefault(str(record["instance_id"]), []).append(record)
        selected: list[dict] = []
        for instance_records in by_instance.values():
            selected.extend(_family_stratified_subset(
                instance_records,
                self.algorithm_candidates_per_instance,
                family_of=lambda row: row.get("generator", {}).get(
                    "family", row.get("candidate_kind", "unknown")
                ),
                is_stay=lambda row: row.get("candidate_name") == "stay",
            ))
        return selected

    def _generate_algorithm_candidate_pool(
        self,
        instance: ExperimentInstance,
        affinity,
        symmetric_road,
    ) -> list[PartitionCandidate]:
        """
        将原有规则、四类直接分区算法和四类局部动作按族轮询合并。

        输入为实例和道路查询；输出不超过配置上限且不重复的候选。
        轮询取样避免任何一类算法因候选较多而挤掉其他类别。
        """
        legacy_started_at = time.perf_counter()
        legacy = generate_candidates(
            instance.partition,
            instance.cities,
            instance.boundary_sizes,
            affinity,
            symmetric_road,
            max_candidates=12,
            random_seed=self._stable_seed(instance.instance_id),
        )
        legacy_seconds = time.perf_counter() - legacy_started_at
        legacy = [
            replace(
                candidate,
                generator_family=f"legacy_{candidate.kind}",
                generator_parameters=(("source", "existing_candidate_pool"),),
                generation_seconds=legacy_seconds / max(len(legacy), 1),
                action_trace=(f"legacy:{candidate.name}",),
            )
            for candidate in legacy
        ]
        direct = generate_direct_partition_candidates(
            instance.partition,
            instance.cities,
            instance.boundary_sizes,
            affinity,
            symmetric_road,
            max_candidates=16,
        )
        actions = generate_partition_action_candidates(
            instance.partition,
            instance.boundary_sizes,
            affinity,
            max_per_action=4,
        )
        combined = legacy + direct + actions
        return _family_stratified_subset(
            combined,
            self.algorithm_candidates_per_instance,
            family_of=lambda candidate: candidate.generator_family or candidate.kind,
            is_stay=lambda candidate: candidate.name == "stay",
        )

    def _evaluate_algorithm_instance(
        self,
        instance: ExperimentInstance,
        cache: GroupEvaluationCache,
    ) -> list[dict]:
        """
        串行真实评价一个实例的多算法与多动作分区候选。

        输入为实例和组缓存；输出含结构特征、组级求解标签与排名的监督记录。
        """
        boundary, regions, distance, preparation = self._prepare_instance(instance)
        truck, drone = distance["truck"], distance["drone"]
        affinity = lambda depot, city: 0.5 * (truck[depot][city] + truck[city][depot])
        symmetric_road = lambda first, second: 0.5 * (truck[first][second] + truck[second][first])
        candidates = self._generate_algorithm_candidate_pool(instance, affinity, symmetric_road)
        feature_rows, evaluations = [], []
        for candidate in candidates:
            feature_rows.append(action_delta_features(
                instance.partition,
                candidate,
                instance.boundary_sizes,
                truck.query,
                drone.query,
                graph_nodes=self._graph.number_of_nodes(),
                graph_edges=self._graph.number_of_edges(),
            ))
            evaluations.append(evaluate_partition(
                instance_id=instance.instance_id,
                partition=candidate.partition,
                depots=instance.depots,
                boundary=boundary,
                regions=regions,
                distance=distance,
                drones_per_truck=instance.drones_per_truck,
                drone_limit=instance.drone_limit,
                drone_speed=instance.drone_speed,
                cache=cache,
                time_limit=self.solver_time_limit,
                threads=self.solver_threads,
                seed=self.solver_seed,
                mip_gap=self.solver_mip_gap,
                max_binary_variables=self.max_binary_variables,
                evaluation_workers=1,
            ))
        baseline_index = next(index for index, candidate in enumerate(candidates) if candidate.name == "stay")
        baseline_evaluation = evaluations[baseline_index]
        records = [
            self._candidate_record(
                instance,
                candidate,
                features,
                evaluation,
                baseline_evaluation,
                preparation,
                round_name="partition_algorithms",
            )
            for candidate, features, evaluation in zip(candidates, feature_rows, evaluations)
        ]
        return self._attach_instance_ranks(records)

    @staticmethod
    def _algorithm_report(records: list[dict], cost_limit: float) -> dict[str, object]:
        """
        按生成算法族汇总精确二三阶段时间、成本、状态和 Pareto 表现。

        输入为去重候选记录与成本上限；输出算法族对比和全局 Oracle 报告。
        超时、复杂度保护或其他启发式回退只计入状态统计，不参与真值性能均值与 Oracle。
        """
        families = sorted({record["generator"]["family"] for record in records})
        by_family = {}
        for family in families:
            rows = [record for record in records if record["generator"]["family"] == family]
            exact_rows = [record for record in rows if record["labels"].get("candidate_exact", False)]
            exact_count = len(exact_rows)
            comparison_rows = [
                record for record in rows
                if record["labels"].get("relative_labels_exact", False)
            ]
            comparison_count = len(comparison_rows)
            by_family[family] = {
                "row_count": len(rows),
                "exact_row_count": exact_count,
                "relative_exact_row_count": comparison_count,
                "instance_count": len({row["instance_id"] for row in rows}),
                "exact_instance_count": len({row["instance_id"] for row in exact_rows}),
                "mean_phase2_serial_seconds": (
                    sum(row["labels"]["phase2_serial_effective_seconds"] for row in exact_rows) / exact_count
                    if exact_count else None
                ),
                "mean_phase3_seconds": (
                    sum(row["labels"]["phase3_seconds"] for row in exact_rows) / exact_count
                    if exact_count else None
                ),
                "mean_downstream_total_seconds": (
                    sum(row["labels"]["downstream_total_seconds"] for row in exact_rows) / exact_count
                    if exact_count else None
                ),
                "mean_time_saving_ratio": (
                    sum(row["labels"]["time_saving_ratio"] for row in comparison_rows) / comparison_count
                    if comparison_count else None
                ),
                "mean_cost_change_ratio": (
                    sum(row["labels"]["cost_change_ratio"] for row in comparison_rows) / comparison_count
                    if comparison_count else None
                ),
                "cost_feasible_fraction": sum(
                    row["labels"]["cost_change_ratio"] <= cost_limit for row in comparison_rows
                ) / comparison_count if comparison_count else None,
                "positive_feasible_fraction": sum(
                    row["labels"]["cost_change_ratio"] <= cost_limit
                    and row["labels"]["time_saving_ratio"] > 0
                    for row in comparison_rows
                ) / comparison_count if comparison_count else None,
                "censored_fraction": sum(bool(row["labels"]["candidate_censored"]) for row in rows) / len(rows),
                "guard_fraction": sum(bool(row["labels"]["guard_groups"]) for row in rows) / len(rows),
                "pareto_front_fraction": (
                    sum(bool(row["labels"]["pareto_front"]) for row in exact_rows) / exact_count
                    if exact_count else None
                ),
                "mean_generation_seconds": sum(
                    row["generator"]["generation_seconds"] for row in rows
                ) / len(rows),
            }
        exact_records = [record for record in records if record["labels"].get("candidate_exact", False)]
        comparison_records = [
            record for record in records
            if record["labels"].get("relative_labels_exact", False)
        ]
        return {
            "row_count": len(records),
            "exact_candidate_count": len(exact_records),
            "relative_exact_candidate_count": len(comparison_records),
            "excluded_regression_candidate_count": len(records) - len(exact_records),
            "instance_count": len({record["instance_id"] for record in records}),
            "customer_counts": sorted({int(record["customer_count"]) for record in records}),
            "cost_limit": cost_limit,
            "candidate_censored_fraction": sum(
                bool(record["labels"]["candidate_censored"]) for record in records
            ) / len(records),
            "guard_candidate_fraction": sum(
                bool(record["labels"]["guard_groups"]) for record in records
            ) / len(records),
            "timeout_group_total": sum(record["labels"]["timeout_groups"] for record in records),
            "guard_group_total": sum(record["labels"]["guard_groups"] for record in records),
            "fallback_group_total": sum(record["labels"]["fallback_groups"] for record in records),
            "by_family": by_family,
            "oracle": oracle_report(comparison_records, cost_limits=(0.0, 0.05, cost_limit)),
        }

    def run_partition_algorithms(self) -> dict[str, object]:
        """
        执行新分区算法与动作实验，并用新标签重训监督预测器。

        输入来自当前配置；输出算法报告和模型报告。
        每个实例完成后单独记录进度，可安全中断续跑。
        """
        directory = self.output_dir / "round4_partition_algorithms" / self.algorithm_run_id
        write_json(directory / "run_configuration.json", self.algorithm_run_configuration)
        print(f"本次实验输出目录：{directory}", flush=True)
        output_path = directory / "candidate_records.jsonl"
        progress_path = directory / "completed_instances.json"
        native_records = deduplicate_candidate_records(read_jsonl(output_path))
        bootstrap_records = self._load_bootstrap_algorithm_records()
        preferred_by_source: dict[str, list[int]] = {}
        for record in native_records + bootstrap_records:
            preferred_by_source.setdefault(str(record["source_name"]), []).append(
                int(record["instance_index"])
            )
        instances = self._algorithm_instances(preferred_by_source)
        selected_instance_ids = {instance.instance_id for instance in instances}
        bootstrap_records = [
            record for record in bootstrap_records
            if record["instance_id"] in selected_instance_ids
        ]
        records = deduplicate_candidate_records([
            record
            for record in native_records + bootstrap_records
            if record["instance_id"] in selected_instance_ids
        ])
        for record in records:
            # 兼容在 timing_protocol 字段加入前已完成的同口径串行记录。
            record.setdefault("timing_protocol", {
                "group_execution": "serial",
                "primary_target": "downstream_total_seconds",
                "targets": [
                    "phase2_serial_effective_seconds",
                    "phase3_seconds",
                    "downstream_total_seconds",
                    "solver_work_sum",
                    "phase2_objective_sum",
                    "final_cost",
                ],
                "regression_eligibility": {
                    "absolute_targets": "candidate_exact_only",
                    "relative_targets": "candidate_and_baseline_exact_only",
                },
                "solver_threads": self.solver_threads,
                "solver_time_limit": self.solver_time_limit,
                "solver_time_unlimited": is_unlimited_solver_time(self.solver_time_limit),
                "solver_seed": self.solver_seed,
                "solver_mip_gap": self.solver_mip_gap,
                "max_binary_variables": self.max_binary_variables,
            })
        records_per_instance: dict[str, int] = {}
        for record in records:
            instance_id = str(record["instance_id"])
            records_per_instance[instance_id] = records_per_instance.get(instance_id, 0) + 1
        if progress_path.is_file():
            progress_completed = set(json.loads(progress_path.read_text(encoding="utf-8")))
        else:
            progress_completed = set()
        # 历史记录只有在达到当前候选预算时才视为完整；当前目录的进度标志仍需至少有一条记录。
        completed = {
            instance_id
            for instance_id, count in records_per_instance.items()
            if count >= self.algorithm_candidates_per_instance
        }
        completed.update({
            instance_id
            for instance_id in progress_completed
            if records_per_instance.get(instance_id, 0) > 0
        })
        print(
            f"复用 {len(bootstrap_records)} 条历史候选；当前 {len(completed)}/{len(instances)} 个实例可跳过。",
            flush=True,
        )
        with GroupEvaluationCache(self.cache_path) as cache:
            for position, instance in enumerate(instances, start=1):
                if instance.instance_id in completed:
                    continue
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] partition_algorithms "
                    f"{position}/{len(instances)} {instance.instance_id}",
                    flush=True,
                )
                instance_records = self._evaluate_algorithm_instance(instance, cache)
                append_jsonl(output_path, instance_records)
                records.extend(instance_records)
                completed.add(instance.instance_id)
                write_json(progress_path, sorted(completed))
                gc.collect()
        records = deduplicate_candidate_records(records)
        write_jsonl(output_path, records)
        active_records = list(records)
        # 单独保存本轮筛选后的训练数据，避免累计文件中的历史客户规模造成误用。
        write_jsonl(directory / "candidate_records_active.jsonl", active_records)
        report = self._algorithm_report(active_records, self.cost_limit)
        write_json(directory / "algorithm_report.json", report)

        # 严格串行模型只使用本轮精确标签，避免旧的并行采样竞争时间混入主目标。
        serial_training_records = deduplicate_partition_records(active_records)
        _, serial_training_report = train_model_ensemble(
            serial_training_records,
            directory / "models_serial_only",
            random_seed=self.random_seed,
            cost_limit=self.cost_limit,
        )
        # 混合模型只作为利用旧数据的探索对照，不用于声称严格串行时间精度。
        previous_records = read_jsonl(self.output_dir / "round2_supervised" / "candidate_records.jsonl")
        if self.customer_counts is not None:
            previous_records = [
                record for record in previous_records
                if int(record["customer_count"]) in self.customer_counts
            ]
        combined_training_records = deduplicate_partition_records(previous_records + active_records)
        _, combined_training_report = train_model_ensemble(
            combined_training_records,
            directory / "models_mixed_exploratory",
            random_seed=self.random_seed,
            cost_limit=self.cost_limit,
        )
        result = {
            "run_id": self.algorithm_run_id,
            "run_directory": str(directory),
            "configuration": self.algorithm_run_configuration,
            "bootstrap_record_count": len(bootstrap_records),
            "algorithms": report,
            "training_serial_only": serial_training_report,
            "training_mixed_exploratory": combined_training_report,
        }
        write_json(directory / "experiment_summary.json", result)
        write_partition_algorithm_markdown(directory / "experiment_summary.md", result)
        return result

    def _run_standard_round(
        self,
        instances: Iterable[ExperimentInstance],
        output_path: Path,
        *,
        round_name: str,
    ) -> list[dict]:
        """逐实例执行固定候选轮次，并按完整实例粒度续跑。"""
        existing = read_jsonl(output_path)
        completed = {
            instance_id
            for instance_id in {row["instance_id"] for row in existing}
            if sum(row["instance_id"] == instance_id for row in existing) >= self.candidates_per_instance
        }
        instance_list = list(instances)
        with GroupEvaluationCache(self.cache_path) as cache:
            for position, instance in enumerate(instance_list, start=1):
                if instance.instance_id in completed:
                    continue
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] {round_name} "
                    f"{position}/{len(instance_list)} {instance.instance_id}",
                    flush=True,
                )
                records = self._evaluate_standard_instance(instance, cache, round_name=round_name)
                append_jsonl(output_path, records)
                existing.extend(records)
                gc.collect()
        return existing

    def run_oracle(self) -> dict[str, object]:
        """执行 8×3×12 第一轮，写出 0%、5%、10% 成本阈值下的事后最优收益。"""
        directory = self.output_dir / "round1_oracle"
        records = self._run_standard_round(
            self._instances(oracle_only=True),
            directory / "candidate_records.jsonl",
            round_name="oracle",
        )
        records = deduplicate_candidate_records(records)
        write_jsonl(directory / "candidate_records.jsonl", records)
        report = oracle_report(records, cost_limits=(0.0, 0.05, self.cost_limit))
        write_json(directory / "oracle_report.json", report)
        write_oracle_markdown(directory / "oracle_report.md", report)
        return report

    def run_supervised(self) -> tuple[ModelEnsemble, dict[str, object]]:
        """执行 8×10×12 第二轮并训练三类动作差值回归器。"""
        directory = self.output_dir / "round2_supervised"
        records = self._run_standard_round(
            self._instances(oracle_only=False),
            directory / "candidate_records.jsonl",
            round_name="supervised",
        )
        records = deduplicate_candidate_records(records)
        write_jsonl(directory / "candidate_records.jsonl", records)
        return train_model_ensemble(
            records,
            directory / "models_initial",
            random_seed=self.random_seed,
            cost_limit=self.cost_limit,
        )

    def _select_active_candidates(
        self,
        pool: list[PartitionCandidate],
        feature_rows: list[dict[str, float]],
        ensemble: ModelEnsemble,
    ) -> list[tuple[PartitionCandidate, dict[str, float], dict]]:
        """
        从未求解主动池中选择预测最优、最高不确定性和高改善 OOD 三类候选。

        输入为候选、特征和初始模型；输出不重复的三元组列表，尚不调用真实求解器。
        """
        scored = [
            {"candidate": candidate, "features": features, "prediction": ensemble.predict(features)}
            for candidate, features in zip(pool, feature_rows)
        ]
        selected = []
        used = set()

        def choose(category: str, eligible, score) -> None:
            """按给定过滤和分数加入一个尚未选择的候选。"""
            choices = [item for item in scored if id(item) not in used and eligible(item["prediction"])]
            if not choices:
                choices = [item for item in scored if id(item) not in used]
            item = max(choices, key=lambda candidate: score(candidate["prediction"]))
            used.add(id(item))
            selected.append((
                replace(item["candidate"], name=f"active_{category}"),
                item["features"],
                {"category": category, **item["prediction"]},
            ))

        choose(
            "predicted_best",
            lambda prediction: prediction["mean_cost_change_ratio"] <= self.cost_limit,
            lambda prediction: prediction["mean_time_saving_ratio"],
        )
        if self.active_selections_per_instance >= 2:
            choose(
                "uncertain",
                lambda prediction: True,
                lambda prediction: prediction["time_uncertainty"] + prediction["cost_uncertainty"],
            )
        if self.active_selections_per_instance >= 3:
            choose(
                "improving_ood",
                lambda prediction: prediction["mean_cost_change_ratio"] <= 1.5 * self.cost_limit,
                lambda prediction: prediction["mean_time_saving_ratio"] + 0.15 * prediction["ood_distance"],
            )
        return selected[: self.active_selections_per_instance]

    def _evaluate_active_instance(
        self,
        instance: ExperimentInstance,
        cache: GroupEvaluationCache,
        ensemble: ModelEnsemble,
    ) -> list[dict]:
        """为一个实例生成主动池、选择三类候选、真实求解并返回回流记录。"""
        boundary, regions, distance, preparation = self._prepare_instance(instance)
        truck, drone = distance["truck"], distance["drone"]
        affinity = lambda depot, city: 0.5 * (truck[depot][city] + truck[city][depot])
        pool = generate_active_pool(
            instance.partition,
            instance.boundary_sizes,
            affinity,
            pool_size=self.active_pool_size,
            random_seed=self._stable_seed(instance.instance_id, offset=3),
        )
        feature_rows = [
            action_delta_features(
                instance.partition,
                candidate,
                instance.boundary_sizes,
                truck.query,
                drone.query,
                graph_nodes=self._graph.number_of_nodes(),
                graph_edges=self._graph.number_of_edges(),
            )
            for candidate in pool
        ]
        selected = self._select_active_candidates(pool, feature_rows, ensemble)
        baseline_candidate = PartitionCandidate("stay", "stay", 0.0, instance.partition, 0)
        baseline_evaluation = evaluate_partition(
            instance_id=instance.instance_id,
            partition=instance.partition,
            depots=instance.depots,
            boundary=boundary,
            regions=regions,
            distance=distance,
            drones_per_truck=instance.drones_per_truck,
            drone_limit=instance.drone_limit,
            drone_speed=instance.drone_speed,
            cache=cache,
            time_limit=self.solver_time_limit,
            threads=self.solver_threads,
            seed=self.solver_seed,
            mip_gap=self.solver_mip_gap,
            max_binary_variables=self.max_binary_variables,
            evaluation_workers=self.evaluation_workers,
        )
        records = []
        for candidate, features, acquisition in selected:
            evaluation = evaluate_partition(
                instance_id=instance.instance_id,
                partition=candidate.partition,
                depots=instance.depots,
                boundary=boundary,
                regions=regions,
                distance=distance,
                drones_per_truck=instance.drones_per_truck,
                drone_limit=instance.drone_limit,
                drone_speed=instance.drone_speed,
                cache=cache,
                time_limit=self.solver_time_limit,
                threads=self.solver_threads,
                seed=self.solver_seed,
                mip_gap=self.solver_mip_gap,
                max_binary_variables=self.max_binary_variables,
                evaluation_workers=self.evaluation_workers,
            )
            records.append(self._candidate_record(
                instance,
                candidate,
                features,
                evaluation,
                baseline_evaluation,
                preparation,
                round_name="active",
                acquisition=acquisition,
            ))
        return records

    def run_active(self, ensemble: ModelEnsemble | None = None) -> tuple[ModelEnsemble, dict[str, object]]:
        """执行第三轮模型引导主动扩充，真实回流后重训三类模型。"""
        supervised_path = self.output_dir / "round2_supervised" / "candidate_records.jsonl"
        supervised_records = read_jsonl(supervised_path)
        if not supervised_records:
            raise RuntimeError("第二轮监督数据不存在，请先运行 run_supervised。")
        if ensemble is None:
            ensemble, _ = train_model_ensemble(
                supervised_records,
                self.output_dir / "round2_supervised" / "models_initial",
                random_seed=self.random_seed,
                cost_limit=self.cost_limit,
            )
        directory = self.output_dir / "round3_active"
        output_path = directory / "candidate_records.jsonl"
        active_records = read_jsonl(output_path)
        active_records = deduplicate_candidate_records(active_records)
        write_jsonl(output_path, active_records)
        completed = {
            instance_id
            for instance_id in {row["instance_id"] for row in active_records}
            if sum(row["instance_id"] == instance_id for row in active_records)
            >= self.active_selections_per_instance
        }
        instances = self._instances(oracle_only=False)
        with GroupEvaluationCache(self.cache_path) as cache:
            for position, instance in enumerate(instances, start=1):
                if instance.instance_id in completed:
                    continue
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] active "
                    f"{position}/{len(instances)} {instance.instance_id}",
                    flush=True,
                )
                records = self._evaluate_active_instance(instance, cache, ensemble)
                append_jsonl(output_path, records)
                active_records.extend(records)
                gc.collect()
        acquisition_summary = {
            "active_row_count": len(active_records),
            "active_instance_count": len({row["instance_id"] for row in active_records}),
            "by_category": {
                category: sum(row["acquisition"]["category"] == category for row in active_records)
                for category in ("predicted_best", "uncertain", "improving_ood")
            },
            "mean_true_time_saving_ratio": sum(row["labels"]["time_saving_ratio"] for row in active_records)
            / len(active_records),
            "mean_true_cost_change_ratio": sum(row["labels"]["cost_change_ratio"] for row in active_records)
            / len(active_records),
        }
        write_json(directory / "acquisition_report.json", acquisition_summary)
        updated_ensemble, training_report = train_model_ensemble(
            supervised_records + active_records,
            directory / "models_after_active",
            random_seed=self.random_seed,
            cost_limit=self.cost_limit,
        )
        return updated_ensemble, {"acquisition": acquisition_summary, "training": training_report}

    def run_all(self) -> dict[str, object]:
        """依次执行三轮，并写出最终机器可读汇总。"""
        oracle = self.run_oracle()
        ensemble, supervised = self.run_supervised()
        _, active = self.run_active(ensemble)
        summary = {
            "oracle": oracle,
            "supervised": supervised,
            "active": active,
        }
        write_json(self.output_dir / "final_summary.json", summary)
        return summary
