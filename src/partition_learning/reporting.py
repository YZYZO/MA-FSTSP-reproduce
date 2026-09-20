"""汇总 Oracle、监督模型和主动扩充三轮实验结果。"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable


def write_json(path: str | Path, payload) -> None:
    """输入路径和可序列化对象，以 UTF-8 缩进 JSON 写出。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    """输入路径和记录序列，以一行一个 JSON 的形式追加并立即落盘。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    """输入路径和记录序列，覆盖写出规范 JSONL，供去重后的无状态产物使用。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def deduplicate_candidate_records(records: Iterable[dict]) -> list[dict]:
    """
    按实例、候选名和完整分区去除续跑并发产生的精确重复记录。

    输入为候选记录；输出保留首次出现顺序的唯一记录，真实标签不做平均或覆盖。
    """
    unique = []
    seen = set()
    for record in records:
        key = (
            record["instance_id"],
            record["candidate_name"],
            json.dumps(record.get("partition", record.get("features", {})), sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def deduplicate_partition_records(records: Iterable[dict]) -> list[dict]:
    """
    按实例和最终分区去除由不同算法产生的结构重复样本。

    输入为候选记录；输出保留首次出现的训练记录。
    算法对比原始数据不调用此函数，因此仍可看到“多种算法生成同一分区”。
    """
    unique = []
    seen = set()
    for record in records:
        key = (
            record["instance_id"],
            json.dumps(record.get("partition", {}), sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def read_jsonl(path: str | Path) -> list[dict]:
    """输入 JSONL 路径，输出记录列表；文件不存在时返回空列表。"""
    source = Path(path)
    if not source.is_file():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def oracle_report(records: Iterable[dict], cost_limits=(0.0, 0.05, 0.10)) -> dict[str, object]:
    """
    计算不同成本阈值下精确候选集的事后最优二三阶段时间节省。

    输入为候选真值记录；输出逐实例选择和各阈值汇总。
    选择只在精确且真实成本变化不超过阈值的候选中最大化二三阶段总时间节省率。
    """
    by_instance: dict[str, list[dict]] = {}
    for record in records:
        labels = record["labels"]
        legacy_exact = (
            not bool(labels.get("candidate_censored", False))
            and int(labels.get("fallback_groups", 0)) == 0
        )
        candidate_exact = bool(labels.get("candidate_exact", legacy_exact))
        baseline_exact = bool(labels.get("baseline_exact", True))
        if not bool(labels.get("relative_labels_exact", candidate_exact and baseline_exact)):
            continue
        by_instance.setdefault(record["instance_id"], []).append(record)
    thresholds: dict[str, object] = {}
    for limit in cost_limits:
        selections = []
        for instance_id, candidates in sorted(by_instance.items()):
            feasible = [
                candidate for candidate in candidates
                if candidate["evaluation"]["complete"]
                and candidate["labels"]["cost_change_ratio"] <= limit
            ]
            if not feasible:
                continue
            selected = max(feasible, key=lambda candidate: candidate["labels"]["time_saving_ratio"])
            selections.append({
                "instance_id": instance_id,
                "candidate_name": selected["candidate_name"],
                "time_saving_ratio": selected["labels"]["time_saving_ratio"],
                "wall_time_saving_ratio": selected["labels"]["wall_time_saving_ratio"],
                "cost_change_ratio": selected["labels"]["cost_change_ratio"],
                "censored_groups": selected["evaluation"]["censored_groups"],
            })
        savings = [item["time_saving_ratio"] for item in selections]
        cost_changes = [item["cost_change_ratio"] for item in selections]
        thresholds[f"{limit:.2f}"] = {
            "cost_limit": limit,
            "instance_count": len(selections),
            "mean_time_saving_ratio": mean(savings) if savings else 0.0,
            "median_time_saving_ratio": median(savings) if savings else 0.0,
            "positive_saving_fraction": (
                sum(value > 0 for value in savings) / len(savings) if savings else 0.0
            ),
            "mean_cost_change_ratio": mean(cost_changes) if cost_changes else 0.0,
            "selections": selections,
        }
    return {"kind": "candidate_oracle", "thresholds": thresholds}


def write_oracle_markdown(path: str | Path, report: dict[str, object]) -> None:
    """输入 Oracle 报告，写出便于直接阅读的 Markdown 摘要。"""
    lines = [
        "# 第一轮 Oracle 可行性试验",
        "",
        "时间指标为相同本机预算下的 Phase 2 有效时间；超时或复杂度保护按删失下界计入。",
        "",
        "| 成本上限 | 实例数 | 平均时间节省 | 中位时间节省 | 正收益比例 | 平均成本变化 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["thresholds"].values():
        lines.append(
            "| {cost_limit:.0%} | {instance_count} | {mean_time_saving_ratio:.2%} | "
            "{median_time_saving_ratio:.2%} | {positive_saving_fraction:.2%} | {mean_cost_change_ratio:.2%} |".format(
                **item
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_feasible_saving(records: Iterable[dict], cost_limit: float) -> dict[str, float]:
    """
    计算每个实例在成本约束下的最大时间节省。

    输入为候选记录与成本变化上限；输出平均、中位数和正收益比例。
    将保持原分区视为 0 收益后备，避免所有新候选均不可行时被迫选择负收益。
    """
    by_instance: dict[str, list[dict]] = {}
    for record in records:
        by_instance.setdefault(record["instance_id"], []).append(record)
    savings = [
        max(
            [0.0]
            + [
                float(record["labels"]["time_saving_ratio"])
                for record in candidates
                if float(record["labels"]["cost_change_ratio"]) <= cost_limit
            ]
        )
        for candidates in by_instance.values()
    ]
    return {
        "instance_count": len(savings),
        "mean_time_saving_ratio": mean(savings),
        "median_time_saving_ratio": median(savings),
        "positive_saving_fraction": sum(value > 0 for value in savings) / len(savings),
    }


def _active_category_summary(records: Iterable[dict], cost_limit: float) -> dict[str, dict[str, float]]:
    """
    按主动采样类别汇总真实回流结果。

    输入为第三轮记录与成本上限；输出每类的样本数、收益、可行率、
    删失率及集成预测 MAE。
    """
    categories = sorted({record["acquisition"]["category"] for record in records})
    summary: dict[str, dict[str, float]] = {}
    for category in categories:
        rows = [record for record in records if record["acquisition"]["category"] == category]
        summary[category] = {
            "row_count": len(rows),
            "mean_time_saving_ratio": mean(record["labels"]["time_saving_ratio"] for record in rows),
            "mean_cost_change_ratio": mean(record["labels"]["cost_change_ratio"] for record in rows),
            "cost_feasible_fraction": sum(
                record["labels"]["cost_change_ratio"] <= cost_limit for record in rows
            ) / len(rows),
            "positive_feasible_fraction": sum(
                record["labels"]["cost_change_ratio"] <= cost_limit
                and record["labels"]["time_saving_ratio"] > 0
                for record in rows
            ) / len(rows),
            "candidate_censored_fraction": sum(
                bool(record["labels"]["candidate_censored"]) for record in rows
            ) / len(rows),
            "prediction_time_mae": mean(
                abs(
                    record["acquisition"]["mean_time_saving_ratio"]
                    - record["labels"]["time_saving_ratio"]
                )
                for record in rows
            ),
            "prediction_cost_mae": mean(
                abs(
                    record["acquisition"]["mean_cost_change_ratio"]
                    - record["labels"]["cost_change_ratio"]
                )
                for record in rows
            ),
        }
    return summary


def build_local_final_report(output_dir: str | Path, cost_limit: float = 0.10) -> dict[str, object]:
    """
    对已完成的三轮本地实验构建无需路网的最终审计报告。

    输入为实验输出目录与成本上限；输出样本结构、Oracle、模型、
    主动回流和 55K 延后状态。该函数只读 JSON/JSONL，不读 NPZ 或 GraphML。
    """
    root = Path(output_dir)
    oracle_rows = deduplicate_candidate_records(read_jsonl(root / "round1_oracle" / "candidate_records.jsonl"))
    supervised_rows = deduplicate_candidate_records(
        read_jsonl(root / "round2_supervised" / "candidate_records.jsonl")
    )
    active_rows = deduplicate_candidate_records(read_jsonl(root / "round3_active" / "candidate_records.jsonl"))

    def load_json(path: Path) -> dict:
        """输入 JSON 路径，输出字典；文件不存在时返回空字典。"""
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    initial_model = load_json(root / "round2_supervised" / "models_initial" / "training_report.json")
    updated_model = load_json(root / "round3_active" / "models_after_active" / "training_report.json")
    manifest = load_json(root / "manifest.json")
    server_manifest = load_json(root / "server_55k_manifest.json")

    # 监督数据与主动数据共享实例，候选行数不等于独立实例数。
    instance_ids = sorted({record["instance_id"] for record in supervised_rows})
    supervised_best = {
        instance_id: max(
            [0.0]
            + [
                record["labels"]["time_saving_ratio"]
                for record in supervised_rows
                if record["instance_id"] == instance_id
                and record["labels"]["cost_change_ratio"] <= cost_limit
            ]
        )
        for instance_id in instance_ids
    }
    combined = supervised_rows + active_rows
    combined_best = {
        instance_id: max(
            [0.0]
            + [
                record["labels"]["time_saving_ratio"]
                for record in combined
                if record["instance_id"] == instance_id
                and record["labels"]["cost_change_ratio"] <= cost_limit
            ]
        )
        for instance_id in instance_ids
    }
    active_overall = {
        "mean_time_saving_ratio": mean(record["labels"]["time_saving_ratio"] for record in active_rows),
        "mean_cost_change_ratio": mean(record["labels"]["cost_change_ratio"] for record in active_rows),
        "cost_feasible_fraction": sum(
            record["labels"]["cost_change_ratio"] <= cost_limit for record in active_rows
        ) / len(active_rows),
        "candidate_censored_fraction": sum(
            bool(record["labels"]["candidate_censored"]) for record in active_rows
        ) / len(active_rows),
    }
    return {
        "scope": {
            "local_graphs": sorted({record["graph_name"] for record in supervised_rows}),
            "local_source_file_count": len({record["source_name"] for record in supervised_rows}),
            "deferred_55k_file_count": len(server_manifest.get("files", [])),
            "solver_time_limit": manifest.get("configuration", {}).get("solver_time_limit"),
            "solver_time_unlimited": manifest.get("configuration", {}).get(
                "solver_time_unlimited", False
            ),
            "note": "本地仅汇总 1K/11K；55K 未在本地读取或求解。",
        },
        "data": {
            "oracle_rows": len(oracle_rows),
            "oracle_instances": len({record["instance_id"] for record in oracle_rows}),
            "supervised_rows": len(supervised_rows),
            "active_rows": len(active_rows),
            "independent_instances": len(instance_ids),
            "combined_candidate_rows": len(supervised_rows) + len(active_rows),
            "supervised_censored_fraction": sum(
                bool(record["labels"]["candidate_censored"]) for record in supervised_rows
            ) / len(supervised_rows),
        },
        "candidate_oracle": {
            "cost_limit": cost_limit,
            "round1": oracle_report(oracle_rows, (cost_limit,))["thresholds"][f"{cost_limit:.2f}"],
            "round2_candidate_pool": _best_feasible_saving(supervised_rows, cost_limit),
        },
        "models": {"initial": initial_model, "after_active": updated_model},
        "active": {
            "overall": active_overall,
            "by_category": _active_category_summary(active_rows, cost_limit),
            "active_only_oracle": _best_feasible_saving(active_rows, cost_limit),
            "combined_oracle": _best_feasible_saving(combined, cost_limit),
            "instances_improved_beyond_round2": sum(
                combined_best[instance_id] > supervised_best[instance_id] + 1e-12
                for instance_id in instance_ids
            ),
            "mean_increment_beyond_round2": mean(
                combined_best[instance_id] - supervised_best[instance_id]
                for instance_id in instance_ids
            ),
        },
    }


def write_local_final_markdown(path: str | Path, report: dict[str, object]) -> None:
    """
    把本地三轮审计报告写成可直接阅读的 Markdown。

    输入为路径和 build_local_final_report 输出；无显式输出。
    """
    data = report["data"]
    scope = report["scope"]
    oracle = report["candidate_oracle"]
    active = report["active"]
    lines = [
        "# 客户分区学习三轮本地实验总结",
        "",
        f"本地只运行 {scope['local_source_file_count']} 个 1K/11K NPZ；"
        f"{scope['deferred_55k_file_count']} 个 55K NPZ 已延后到服务器。",
        "",
        "## 数据规模",
        "",
        f"- Round 1：{data['oracle_instances']} 个实例，{data['oracle_rows']} 条候选记录。",
        f"- Round 2：{data['independent_instances']} 个独立实例，{data['supervised_rows']} 条候选记录。",
        f"- Round 3：{data['active_rows']} 条主动候选；合计 {data['combined_candidate_rows']} 条训练候选。",
        f"- Phase 2 候选删失率：{data['supervised_censored_fraction']:.2%}；"
        + (
            "Gurobi 不设置时间上限。"
            if scope["solver_time_unlimited"]
            else f"求解时限 {scope['solver_time_limit']} 秒。"
        ),
        "",
        "## 关键结果",
        "",
        "| 评价 | 平均时间节省 | 中位时间节省 | 正收益比例 |",
        "|---|---:|---:|---:|",
        "| Round 1 Oracle（10% 成本上限） | "
        f"{oracle['round1']['mean_time_saving_ratio']:.2%} | "
        f"{oracle['round1']['median_time_saving_ratio']:.2%} | "
        f"{oracle['round1']['positive_saving_fraction']:.2%} |",
        "| Round 2 的 12 候选 Oracle | "
        f"{oracle['round2_candidate_pool']['mean_time_saving_ratio']:.2%} | "
        f"{oracle['round2_candidate_pool']['median_time_saving_ratio']:.2%} | "
        f"{oracle['round2_candidate_pool']['positive_saving_fraction']:.2%} |",
        "| Round 3 的 3 候选 Oracle | "
        f"{active['active_only_oracle']['mean_time_saving_ratio']:.2%} | "
        f"{active['active_only_oracle']['median_time_saving_ratio']:.2%} | "
        f"{active['active_only_oracle']['positive_saving_fraction']:.2%} |",
        "",
        f"主动挑选的原始平均结果为：时间节省 "
        f"{active['overall']['mean_time_saving_ratio']:.2%}，成本变化 "
        f"{active['overall']['mean_cost_change_ratio']:.2%}。只有 "
        f"{active['instances_improved_beyond_round2']} 个实例超过了 Round 2 的候选 Oracle，"
        f"平均额外改善 {active['mean_increment_beyond_round2']:.2%}。",
        "",
        "## 解读",
        "",
        "1. 规则候选中存在可学信号，因为事后 Oracle 能稳定找到节时且成本可控的分区。",
        "2. 现有特征与样本覆盖不足以支持对新分区的外推；目前不宜直接进入大规模强化学习。",
        "3. 候选行数为 900，但独立实例只有 60；NPZ 数量不是关键，路网、客户密度和分区动作的多样性才是。",
        "4. 性能回归应只使用精确完成的候选；达到时间上限的候选作为右删失状态样本单独处理。",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_partition_algorithm_markdown(path: str | Path, result: dict[str, object]) -> None:
    """
    将新分区算法、划分动作和严格串行监督模型结果写成 Markdown。

    输入为路径和 run_partition_algorithms 结果；无显式输出。
    表格展示算法族真实收益，以及 Phase 2、Phase 3、总时间和最终目标值的预测质量。
    """
    algorithms = result["algorithms"]
    oracle = algorithms["oracle"]["thresholds"][f"{algorithms['cost_limit']:.2f}"]
    serial_training = result["training_serial_only"]
    # 各模型共用同一实例级测试集，因此从首个模型读取状态比例即可。
    first_model_report = next(iter(serial_training["models"].values()))
    test_censored_fraction = first_model_report["right_censored"]["positive_fraction"]
    test_guard_fraction = first_model_report["guard"]["positive_fraction"]

    def percent(value) -> str:
        """把可能为空的比例转换成表格文本。"""
        return "—" if value is None else f"{value:.2%}"

    lines = [
        "# 直接客户分区算法与划分动作实验",
        "",
        "时间口径：各仓库组严格串行，主目标为 Phase 2 串行时间与 Phase 3 时间之和。",
        "性能回归只使用 Phase 2 精确完成、未超时且未回退的候选；其他候选只用于状态分类。",
        "",
        f"- 实例数：{algorithms['instance_count']}",
        f"- 客户规模：{', '.join(map(str, algorithms['customer_counts']))}",
        f"- 候选记录：{algorithms['row_count']}",
        f"- 可用于精确回归的候选：{algorithms['exact_candidate_count']}",
        f"- 候选与 MST 基线均精确、可用于相对改善回归的候选："
        f"{algorithms['relative_exact_candidate_count']}",
        f"- 从回归中排除的候选：{algorithms['excluded_regression_candidate_count']}",
        f"- 候选总体右删失比例：{algorithms['candidate_censored_fraction']:.2%}",
        f"- 候选复杂度保护比例：{algorithms['guard_candidate_fraction']:.2%}",
        f"- 超时组 / 保护组 / 回退组累计：{algorithms['timeout_group_total']} / "
        f"{algorithms['guard_group_total']} / {algorithms['fallback_group_total']}",
        f"- {algorithms['cost_limit']:.0%} 成本上限下 Oracle 平均时间节省：{oracle['mean_time_saving_ratio']:.2%}",
        f"- Oracle 正收益实例比例：{oracle['positive_saving_fraction']:.2%}",
        f"- Oracle 平均成本变化：{oracle['mean_cost_change_ratio']:.2%}",
        "",
        "## 算法族真实表现",
        "",
        "| 算法族 | 总样本 | 精确样本 | 精确相对标签 | 平均总时间节省 | 平均成本变化 | 成本可行率 | 正收益且可行 | Pareto前沿 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, metrics in sorted(
        algorithms["by_family"].items(),
        key=lambda item: (-(item[1]["positive_feasible_fraction"] or -1.0), item[0]),
    ):
        lines.append(
            f"| {family} | {metrics['row_count']} | {metrics['exact_row_count']} | "
            f"{metrics['relative_exact_row_count']} | "
            f"{percent(metrics['mean_time_saving_ratio'])} | {percent(metrics['mean_cost_change_ratio'])} | "
            f"{percent(metrics['cost_feasible_fraction'])} | "
            f"{percent(metrics['positive_feasible_fraction'])} | {percent(metrics['pareto_front_fraction'])} |"
        )
    lines.extend([
        "",
        "## 监督模型预测",
        "",
        f"去重后共有 {serial_training['row_count']} 条候选，其中 "
        f"{serial_training['exact_row_count']} 条用于绝对值回归，"
        f"{serial_training['relative_exact_row_count']} 条用于相对改善回归；按实例划分为 "
        f"{serial_training['train_instance_count']} 个训练实例和 "
        f"{serial_training['test_instance_count']} 个测试实例。",
        f"测试候选的右删失比例为 {test_censored_fraction:.2%}，复杂度保护比例为 "
        f"{test_guard_fraction:.2%}。",
        "",
        "| 模型 | 目标 | MAE | R² | Spearman | 实例内Spearman | 最优Top-3命中 | 选择后悔率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    target_labels = {
        "phase2_serial_seconds": "Phase 2 时间（秒）",
        "phase3_seconds": "Phase 3 时间（秒）",
        "downstream_total_seconds": "二三阶段总时间（秒）",
        "final_cost": "最终目标值",
    }
    for model_name, model_report in serial_training["models"].items():
        for target_name, target_label in target_labels.items():
            metrics = model_report[target_name]
            lines.append(
                f"| {model_name} | {target_label} | {metrics['mae']:.3f} | "
                f"{metrics['r2']:.3f} | {metrics['spearman']:.3f} | "
                f"{metrics['mean_within_instance_spearman']:.3f} | "
                f"{metrics['true_fastest_top3_hit_fraction']:.2%} | "
                f"{metrics['predicted_fastest_mean_regret_ratio']:.2%} |"
            )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
