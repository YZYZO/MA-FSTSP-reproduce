"""绘制 V2 配对实验的两组 Phase 2 机制散点图。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def _correlation_label(x_values, y_values):
    """计算有限、非退化样本的 Spearman 标签；输入为两列数据，输出短文本。"""

    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]
    if len(x_values) < 2 or np.all(x_values == x_values[0]) or np.all(y_values == y_values[0]):
        return f'n={len(x_values)}, rho=NA'
    result = spearmanr(x_values, y_values)
    return f'n={len(x_values)}, rho={float(result.statistic):.3f}'


def plot_mechanism_scatter(summary_path, output_path):
    """读取 V2 summary 并输出最大/总 Q_bin 对 Phase 2 时间的双面板散点图。"""

    with np.load(summary_path, allow_pickle=False) as data:
        required = (
            'methods', 'completed', 'max_q_bin', 'sum_q_bin',
            'max_phase2_time', 'total_phase2_time',
        )
        missing = [field for field in required if field not in data.files]
        if missing:
            raise ValueError(f'机制图需要 V2 汇总字段，当前缺少：{missing}')
        methods = [str(item) for item in np.asarray(data['methods']).tolist()]
        completed = np.asarray(data['completed'], dtype=bool)
        max_q_bin = np.asarray(data['max_q_bin'], dtype=float)
        sum_q_bin = np.asarray(data['sum_q_bin'], dtype=float)
        max_phase2_time = np.asarray(data['max_phase2_time'], dtype=float)
        total_phase2_time = np.asarray(data['total_phase2_time'], dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    panels = (
        (max_q_bin, max_phase2_time, 'Maximum Q_bin', 'Maximum group Phase 2 time (s)'),
        (sum_q_bin, total_phase2_time, 'Sum of Q_bin', 'Total Phase 2 time (s)'),
    )
    for method_index, method in enumerate(methods):
        mask = (
            completed[:, method_index]
            & np.isfinite(max_q_bin[:, method_index])
            & np.isfinite(sum_q_bin[:, method_index])
        )
        for axis, (x_matrix, y_matrix, x_label, y_label) in zip(axes, panels):
            x_values = x_matrix[mask, method_index]
            y_values = y_matrix[mask, method_index]
            axis.scatter(
                x_values,
                y_values,
                s=24,
                alpha=0.65,
                label=f'{method} ({_correlation_label(x_values, y_values)})',
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.grid(alpha=0.25)

    for axis in axes:
        axis.legend(fontsize=8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main(argv=None):
    """解析 summary 和输出路径参数，生成可直接用于实验报告的 PNG。"""

    parser = argparse.ArgumentParser(description='绘制配对实验 Phase 2 机制散点图。')
    parser.add_argument('summary', type=Path, help='V2 paired_summary.npz 路径')
    parser.add_argument('--output', type=Path, default=None, help='PNG 输出路径')
    arguments = parser.parse_args(argv)
    output = arguments.output or arguments.summary.with_name('mechanism_scatter_v2.png')
    print(plot_mechanism_scatter(arguments.summary, output))


if __name__ == '__main__':
    main()
