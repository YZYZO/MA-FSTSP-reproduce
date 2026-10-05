"""生成公平搜索方法的真实可行率、时间节省与候选池后悔值报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.rl_true_comparison import (  # noqa: E402
    build_true_search_comparison,
)


def parse_arguments() -> argparse.Namespace:
    """解析代理公平搜索报告、服务器候选记录、成本上限和输出目录。"""
    parser = argparse.ArgumentParser(description="分析公平搜索真实复核结果")
    parser.add_argument("--search-report", type=Path, required=True)
    parser.add_argument("--candidate-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cost-limit", type=float, default=0.10)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    """输入服务器candidate_records.jsonl，输出候选记录列表。"""
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _markdown(report: dict) -> str:
    """输入真实比较报告，输出适合论文实验核对的Markdown表格。"""
    lines = [
        "# 公平搜索真实复核", "",
        f"成本硬约束：相对MST不超过 {report['cost_limit']:.0%}。", "",
        "真实后悔值只相对于本次服务器已复核候选池，不代表全局最优。", "",
        "|方法|真实可行率|真实可行改进率|假可行率|可行时平均时间节省|真实后悔值|平均查询|平均生成秒|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in report["aggregate"].items():
        percent = lambda value: "—" if value is None else f"{value:.2%}"
        lines.append(
            f"|{method}|{metrics['true_cost_feasible_fraction']:.2%}|"
            f"{metrics['true_feasible_improvement_fraction']:.2%}|"
            f"{metrics['proxy_false_feasible_fraction']:.2%}|"
            f"{percent(metrics['mean_time_saving_given_feasible'])}|"
            f"{percent(metrics['mean_true_feasible_regret_ratio'])}|"
            f"{metrics['mean_surrogate_query_count']:.1f}|"
            f"{metrics['mean_generation_seconds']:.2f}|"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """连接代理与真实结果，写出JSON和Markdown两种报告。"""
    arguments = parse_arguments()
    search = json.loads(arguments.search_report.read_text(encoding="utf-8"))
    report = build_true_search_comparison(
        search,
        _load_jsonl(arguments.candidate_records),
        cost_limit=arguments.cost_limit,
    )
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "true_search_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "true_search_comparison.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(f"真实搜索比较报告：{output_dir / 'true_search_comparison.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
