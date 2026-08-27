# 多智能体飞行侧援旅行商问题（MA-FSTSP）中文说明

## 项目简介
本仓库是论文 **Optimization of Multi-Agent Flying Sidekick Traveling Salesman Problem over Road Networks** 的实验代码实现。项目研究的是这样一类问题：

- 有多辆卡车从不同仓库出发。
- 每辆卡车可以携带若干架无人机。
- 客户既可以由卡车直接送达，也可以由无人机从卡车途中起飞并完成配送。
- 路网对卡车可达性有限制，无人机则更接近直线飞行，但受续航和速度约束。

本仓库实现了论文主算法，也包含多个基线算法、实验脚本和绘图脚本。



## 如何运行实验
### 1. 选择并运行实验
无参数启动时默认执行 Pilot 配对协议：

```bash
python experiments.py
```

正式协议需要显式指定，避免误启动大批量求解：

```bash
python experiments.py --protocol formal
```

两种协议都可以通过 `--seed` 指定实例采样随机种子。


## 如何生成图表
运行：

```bash
python plot.py
```

当前脚本已经做了兼容处理：

- 如果某些 `.npy` 结果文件不存在，对应图表会自动跳过，而不会报错退出。
- 即使没有完整论文实验结果，`plot.py` 也仍然可以生成一部分示意图与 HTML 地图。

### 绘制最新的 1K/11K 路网解

假设结果中的客户数量为 100，在项目根目录运行：

```powershell
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" -c "import plot; plot.plot_large_road_experiment_results(customer_count=100)"
```

绘图程序会分别查找最新的 `*-manhattan_1k-100.npz` 与 `*-nyc_11k_proxy-100.npz`。
新格式结果已经保存最终路线，因此绘图不会重新运行优化器；未指定 `instance_index` 时，默认选择成本最接近中位数的实例。


## 第一阶段配对实验

`config.py` 把主实验与三个敏感性/消融实验拆成互不混用的 V2 协议：

- `PILOT_PROTOCOL`：Manhattan 1K、50 客户、10 个实例、每实例 600 秒。
- `FORMAL_PROTOCOL`：Manhattan 1K 与 **NYC 11K proxy**、50/100/150 客户、每个设置 100 个实例。仓库中的 11K 文件不是论文 Boston 路网，因此结果不会再标成 Boston 复现。
- `EPSILON_SENSITIVITY_PROTOCOL`：固定全部仓库，比较 `epsilon={0,0.005,0.01,0.02,0.05}`。
- `ACTIVE_DEPOT_ABLATION_PROTOCOL`：比较全部仓库与自由仓库策略。
- `COST_FACTOR_SENSITIVITY_PROTOCOL`：比较论文 `1/speed` 与旧 SMST 兼容的 `sqrt(2)/speed` 口径。

主实验运行 `smst_original`、`snn`、`set_gtds_no_budget` 和 `directed_set_gtds`。两个 GTDS 主方法都尽可能启用全部仓库并使用论文代价系数；自由仓库和 `sqrt(2)` 只出现在独立消融中。每个实例采用确定性的循环平衡方法顺序，降低固定先后顺序带来的缓存与热启动偏差。

运行 Pilot：

```powershell
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" experiments.py
```

运行 100 次正式协议：

```powershell
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" experiments.py --protocol formal
```

三类独立敏感性实验分别使用 `--protocol epsilon`、`--protocol active-depot` 和 `--protocol cost-factor`。

结果写入 `results/paired/<protocol_v2>/<dataset>/<size>/`。`manifest.json` 冻结图 SHA-256、环境版本、源码指纹、采样输入、执行顺序和协议签名；`records/<instance_id>/<method>.json` 是原子检查点。`paired_summary.npz` 保存严格配对的成本、分阶段时间、完成状态、超时、分组规模以及最大/总 `Q_bin`。

Pilot 和正式协议的实例总时限分别为 600 秒与 7200 秒。运行器把实例剩余时间传给每个 Set-TSP，并在 Phase 3 的主要 DP 层检查截止时间。单仓库达到 Gurobi 时限但已有 incumbent 时可继续；没有 incumbent 或 Phase 3 超时则实例记录为不完整成本。

分析配对结果：

```powershell
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" analyze_paired_results.py results\paired\phase1_pilot_v2\manhattan_1k\50\paired_summary.npz --baseline smst_original --cutoff 600 --output results\paired\phase1_pilot_v2\manhattan_1k\50\analysis.json
```

报告包含完成率、超时率、配对相对差的均值/中位数 bootstrap 95% CI、Wilcoxon、PAR-2，以及两组规定的 Spearman 机制相关性。成本 CI 上界不超过 1% 和时间 CI 上界小于 0 会输出独立判定标记。

生成机制散点图：

```powershell
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" plot_paired_mechanisms.py results\paired\phase1_pilot_v2\manhattan_1k\50\paired_summary.npz
```
