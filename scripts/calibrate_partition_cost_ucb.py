"""从实例级折外GNN集成预测校准成本UCB乘数κ。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.partition_learning.cost_ucb_calibration import (  # noqa: E402
    cross_fitted_cost_calibration,
    load_oof_cost_records,
)


def parse_arguments() -> argparse.Namespace:
    """解析交叉验证目录、模型、覆盖率、成本上限及输出文件。"""
    parser = argparse.ArgumentParser(description="校准相对MST成本变化UCB")
    parser.add_argument("--cross-validation-root", type=Path, required=True)
    parser.add_argument("--model-name", default="balanced")
    parser.add_argument("--baseline-candidate-name", default="stay")
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--cost-limit", type=float, default=0.10)
    parser.add_argument("--sigma-floor", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """加载折外预测、执行交叉拟合校准并写出可部署κ及安全性报告。"""
    arguments = parse_arguments()
    records = load_oof_cost_records(
        arguments.cross_validation_root.resolve(),
        model_name=arguments.model_name,
        baseline_candidate_name=arguments.baseline_candidate_name,
        sigma_floor=arguments.sigma_floor,
    )
    report = cross_fitted_cost_calibration(
        records,
        coverage=arguments.coverage,
        cost_limit=arguments.cost_limit,
    )
    report["configuration"] = {
        "cross_validation_root": str(arguments.cross_validation_root.resolve()),
        "model_name": arguments.model_name,
        "baseline_candidate_name": arguments.baseline_candidate_name,
        "sigma_floor": arguments.sigma_floor,
        "record_count": len(records),
    }
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cross_fitted = report["cross_fitted"]
    print(
        f"成本UCB校准完成：deployment_kappa={report['deployment_kappa']:.4f}，"
        f"交叉拟合假可行率="
        f"{cross_fitted['false_feasible_rate_given_predicted_feasible']:.2%}，"
        f"输出={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
