# MA-FSTSP 分区修复：服务器运行指南

本流程使用 CPU，在 NYC 1K 路网上检验客户划分对 Set-TSP 和最终配送的影响。阶段 A 采集 12 个诊断实例；阶段 B 采集 30 个开发实例及其候选，并独立复测手工选择器。所有命令在项目根目录执行。

## 运行环境

激活服务器已有的 MA-FSTSP Python 环境，保证 `python` 指向该环境。需要 Python 3.10 及以上，以及项目已有的 NumPy、NetworkX、SciPy、gurobipy、elkai、osmnx、matplotlib。Gurobi 需要可用于目标规模的许可证。该流程不需要 GPU，也不加载神经网络。

路网文件为 `datasets/nyc_1024.graphml`。第一步运行小图测试：

```bash
python -m unittest discover -s tests -p 'test_partition_repair_*.py' -v
```

本机对应命令：

```powershell
& 'D:\anaconda3\envs\MA-FSTSP\python.exe' -X utf8 -B -m unittest discover -s tests -p 'test_partition_repair_*.py' -v
```

## 第一步：阶段 A，12 个实例的瓶颈诊断

```bash
python -u scripts/collect_partition_candidates.py --stage A --output results/partition_repair/stage_a
python scripts/analyze_partition_candidates.py --input results/partition_repair/stage_a
```

配置为 50/100/150 客户各 4 个新实例、5 个仓库、每车 3 架无人机。这里只评价对称 Set-MST 的 `stay` 分区，核对完整下游运行和耗时分解。

查看 `results/partition_repair/stage_a/report/candidate_report.json`：

| 字段 | 含义 |
|---|---|
| `complete_instances`、`expected_instances` | 应为 12/12 |
| `measured_group_phase2_components_seconds` | 输入、距离、建模、优化、恢复、回退与其余开销 |
| `measured_group_phase3_seconds` | 实际第三阶段总耗时 |
| `timeout_groups`、`fallback_groups` | 优化限时及无解/空边界回退的组数量 |
| `mean_measured_group_seconds` | 真实组评价平均耗时，用于估算采集预算 |

阶段 A 的候选潜力为零是正常的，因为只有 `stay`。重点是瓶颈分解、超时情况和标签完整性。如果大部分组都被 30 秒优化预算截断，先在开发设置中统一调整预算，再用新目录重跑；同一轮所有方法始终使用相同预算。

## 第二步：阶段 B，先采集 3 个实例

```bash
python -u scripts/collect_partition_candidates.py --stage B --limit-instances 3 --output results/partition_repair/stage_b
python scripts/analyze_partition_candidates.py --input results/partition_repair/stage_b
```

这 3 个实例分别为 50、100、150 客户。完整的 30 实例清单已经固定，因此试运行报告会显示 3/30，并明确标记尚未完成。候选按完整分区去重，最多 12 个，含 `stay`、不同人数修复前缀、两档变量负担修复、小簇迁移和一个局部迁移/交换候选。

`results/partition_repair/stage_b/sessions/` 中保存每次采集的实际耗时与新求解组数。相同组的离线标签可以复用，未改变组无需重新求解；实际采集成本与复用的下游时间分别记录。

确认试运行时间后，去掉实例限制继续完整清单：

```bash
python -u scripts/collect_partition_candidates.py --stage B --output results/partition_repair/stage_b
python scripts/analyze_partition_candidates.py --input results/partition_repair/stage_b
```

## 第三步：在同一清单上独立复测

采集结束后运行：

```bash
python -u scripts/evaluate_partition_repair.py --manifest results/partition_repair/stage_b/manifest.json --output results/partition_repair/stage_b_evaluation
python scripts/analyze_partition_candidates.py --input results/partition_repair/stage_b_evaluation
```

默认比较以下方法：

| 方法 | 在线行为 |
|---|---|
| `symmetric_mst` | 直接使用对称 Set-MST |
| `handcrafted` | 同一候选集上，用相对变量负担下降减去几何增量选择一次 |
| `count_only` | 同一候选集上，选择人数平方和最小的划分 |
| `burden_only` | 同一候选集上，选择总二元变量估计最少的划分 |
| `random` | 在同一候选集中按固定种子选择 |
| `original_mst` | 原始集合距离构图和 MST，作为附加对照 |

候选选择方法共用生成规则和预算。直接 MST 方法无需生成修复候选。每次方法运行从模型构造前开始计时，重新生成所需分区，只真实求解选中的一个分区。程序不会通过求解其他候选的真实成本指导在线选择。

复测继承 `manifest.json` 中的求解、车辆和修复设置。它关闭求解结果缓存；完成记录的续跑跳过不构成新一次计时。默认每实例每方法运行一次，方法顺序轮转并交替反向。需要重复计时时，从开始就设置 `--repeats 3` 并使用一个新输出目录；报告先对同一实例的重复运行取均值，再以完整实例为单位计算统计与配对区间。

## 读取结果

日常先阅读两个摘要：

- `stage_b/report/candidate_report.md`：有限候选集具有多少改进空间。
- `stage_b_evaluation/report/evaluation_report.md`：可实际运行的选择方法是否获得收益。

详细文件：

| 文件 | 用途 |
|---|---|
| `candidate_report.json` | 总体成本预算为 0/1/3/5% 时的候选潜力、各规模结果和瓶颈 |
| `candidate_curve.csv`、`candidate_curve.png` | 所有完整实例的候选成本—耗时关系 |
| `oracle_per_instance.csv` | 事后最好选择下每个实例的变化，检查总体预算掩盖的个体退化 |
| `evaluation_report.json` | 总体指标、逐规模统计、配对区间、第一阶段开销、超时与回退 |
| `evaluation_per_instance.csv` | 真实复测的逐实例成本、第二阶段时间、第三阶段时间和端到端时间 |

判定使用总量之比：

```text
成本变化 = Σ新方法最终配送成本 / Σ对称MST最终配送成本 − 1
第二阶段节时 = 1 − Σ新方法完整第二阶段时间 / Σ对称MST完整第二阶段时间
```

目标是总体成本增加不超过 5%、完整第二阶段节时至少 20%。逐实例可以超过 5%，因此报告同时列出超标数量、最坏成本恶化和最坏变慢。`passes_point_thresholds` 表示点估计达标；`thresholds_supported_by_ci95` 表示配对区间也位于门槛内。未完成实例会阻止完整通过判定。

`candidate_report` 的事后选择使用全部候选的真实答案和总体成本预算，是离线潜力诊断，不是可部署的策略。其区间只描述固定事后选择的样本波动。候选潜力达到约 30% 可为模型误差留出余量；若连 20% 潜力都没有，应先调整候选。若潜力充分而手工选择较弱，再进入监督学习。阶段 B 属于开发集，不能代替后续独立测试集验收。

## 计时、回退与续跑规则

默认每组 Gurobi `TimeLimit=30`、`Threads=1`、`Seed=0`、`MIPGap=0.0001`。单线程是本轮固定的可比较配置。优化限时不包含 Python 输入、建模、恢复、回退和第三阶段，所以整个实例可能远超 30 秒。可在采集开始时用 `--threads` 或 `--time-limit` 设定另一轮统一配置，并使用新的结果目录。

有可行解就使用现有解；没有可行解或客户边界为空时，采用按有向卡车距离的确定性最近邻顺序，再执行同一个第三阶段。回退时间计入完整第二阶段。间隙无穷大时保存 `gap=null`、`gap_is_finite=false`，保留可行解与限时状态。

第三阶段执行原有 DP；长任务可用 Ctrl+C 中断。已经完成的组即时保存，当前中断组记录为未完成，不生成虚假的完整成本。相同命令重跑会继续未完成部分。源码、依赖、机器或求解配置变化后，采集应使用新结果目录；复测的算法源码和依赖须与采集一致。

地图和全点对距离按运行准备一次，所有方法共享相同的预计算距离。复测的 `online_seconds` 是模型构造至最终结果的实测墙钟；地图准备作为一次性成本单列。`cold_batch_seconds` 是按相同一次地图准备成本加到各方法整批在线时间上的冷批次口径，重复运行先在实例内取均值。

`feature_seconds` 包括共享特征以及候选生成时实际计算的组摘要；这些组摘要时间已从 `repair_seconds` 中扣除。`selection_seconds` 是一次选择的实际开销。每个组的 `phase2_other_seconds` 保留计时函数、对象释放等未落入具名步骤的实际开销。

三张数据表为 `instances`、`group_runs`、`partition_candidates`。每条记录先原子保存为独立 JSON 文件，再导出同名整表 JSON。完成后的 `manifest.json`、三张表及报告应一起保留。不要让两个进程同时写同一个输出目录；采集与正式计时复测顺序运行。

## 代码接口

```python
from src.partition_repair.settings import SolverOptions

# 已有 model 时，用指定分区求解；返回值仍是路线与最终配送成本。
solution, cost = model.solve(
    partition={depot: customers for depot, customers in specified_groups.items()},
    solver_options=SolverOptions(time_limit=30, threads=1, seed=0),
)

# 用求解前特征一次选择候选，再执行下游求解。
solution, cost = model.solve(partition_strategy='handcrafted')
```

`model.solve()` 默认分区方法为 `original_mst`；普通入口与实验记录入口共用相同的固定分区评价逻辑和默认求解配置。第三阶段与 Set-TSP 内部代价沿用当前实现，所有新标签使用同一数学语义。
