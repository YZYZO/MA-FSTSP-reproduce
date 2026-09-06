# MA-FSTSP 客户划分：阶段 C 服务器指南

目标为相对对称 Set-MST，总体配送成本增加不超过 **10%**、完整第二阶段节时至少 **20%**。允许个别实例超限，报告超限比例和最坏情况，同时报告整体在线耗时。

所有命令在服务器项目根目录、已有 MA-FSTSP 环境中运行。阶段 C 只使用 CPU；本机负责正确性测试。下面先完成“已有数据诊断”，读取结果后再决定是否扩充标签。

## 现在运行：已有 30 个实例的开发诊断

将本次项目代码同步到服务器，包括 src、scripts 和学习依赖文件。保留原阶段 A/B 结果目录。激活原环境后，只需额外安装 CPU 学习依赖：

```bash
conda activate MA-FSTSP
python -m pip install -r requirements-partition-learning.txt
```

阶段 C 使用 Python 3.11 及以上；固定的 scikit-learn 1.9.0 支持现有 NumPy 1.25.1、SciPy 1.11.2。依赖依据见 [PyPI 发布说明](https://pypi.org/project/scikit-learn/1.9.0/)。无需重新安装包含 GPU 组件的完整 requirements.txt。

设置服务器上已有结果目录。如果实际路径不同，只修改这两行：

```bash
STAGE_A="results/partition_repair/stage_a"
STAGE_B="results/partition_repair/stage_b"
```

STAGE_B 应包含 manifest.json、instances.json、partition_candidates.json，或者对应的逐记录目录。它是候选采集目录，不是 stage_b_evaluation。

```bash
python -u scripts/train_partition_selector.py --mode diagnose --input "$STAGE_B" --output results/partition_repair/stage_c_diagnosis --folds 5 --inner-folds 3 --seed 906030 --cost-limit 0.10 --min-phase2-saving 0.20
```

这一步读取已有 30 个实例、360 个候选标签，训练和比较轻量模型，不重新运行 Set-TSP 或第三阶段。默认外层 5 折：每折 24 个实例用于训练和校准，6 个实例留出评价；学习规则使用训练部分内部的 3 折预测校准权重。相同实例的全部候选始终在同一侧。

比较方法：

| 方法 | 含义 |
|---|---|
| symmetric_mst | 保持初始划分 |
| handcrafted | 固定选择权重 1 的手工规则 |
| handcrafted_calibrated | 在训练部分选择一个统一评分权重 |
| size_handcrafted | 在训练部分分别确定 50/100/150 客户的评分权重 |
| learned_ridge | 两个线性回归器预测成本差和第二阶段节时，再校准选择权重 |
| learned_tree | 两个浅层树回归器预测相同差值，再校准选择权重 |

完成后下载整个 results/partition_repair/stage_c_diagnosis 目录，重点阅读：

- learning_report.md：各方法的成本、第二阶段节时、超限比例和最坏情况。
- learning_report.json：外层实例隔离、各折参数、8%/9%/10% 校准曲线、预测诊断和各规模结果。
- learning_per_instance.csv：每个实例最终选择的候选及其真实标签表现。

这些是开发集离线诊断，不含新策略在线开销，也不是独立测试通过的证明。diagnose 不生成正式 policy.json。输出目录非空时程序拒绝覆盖；重复诊断使用新输出目录。

## 可选：检查数据索引或重新分析预算

```bash
python scripts/prepare_partition_learning.py dataset --input "$STAGE_B" --output results/partition_repair/stage_c_data_index.json
python scripts/analyze_partition_candidates.py --input "$STAGE_B" --output results/partition_repair/stage_b_budget10 --cost-limit 0.10
```

数据索引检查标签完整性、物理实例去重与计时环境，不训练模型。预算报告中的事后选择知道全部候选答案，只用于衡量候选潜力。

## 根据诊断扩充：新增 30 个训练实例和 30 个验证实例

训练标签达到 60 个完整实例：已有 B 的 30 个加本节新增 30 个；验证集另采样 30 个。各规模均为 10 个，采样使用独立种子并排除已知实例。清单生成只读取地图，不求解配送问题。

```bash
python scripts/prepare_partition_learning.py sample --template-manifest "$STAGE_B/manifest.json" --split train --instances-per-size 10 --seed 906041 --exclude-manifests "$STAGE_A/manifest.json" --output results/partition_repair/stage_c_manifests/train_30.json
python scripts/prepare_partition_learning.py sample --template-manifest "$STAGE_B/manifest.json" --split validation --instances-per-size 10 --seed 906042 --exclude-manifests "$STAGE_A/manifest.json" results/partition_repair/stage_c_manifests/train_30.json --output results/partition_repair/stage_c_manifests/validation_30.json
```

如果还有已使用的实例清单，将它们追加到 --exclude-manifests。新清单会保存当前源码、地图、车辆和求解配置，以及被排除清单的指纹。不要改动已生成清单来切换训练/验证/测试角色。

先采集三个训练实例估算时间，再以同一清单继续：

```bash
python -u scripts/collect_partition_candidates.py --manifest results/partition_repair/stage_c_manifests/train_30.json --limit-instances 3 --output results/partition_repair/stage_c_train_30
python -u scripts/collect_partition_candidates.py --manifest results/partition_repair/stage_c_manifests/train_30.json --output results/partition_repair/stage_c_train_30
python -u scripts/collect_partition_candidates.py --manifest results/partition_repair/stage_c_manifests/validation_30.json --output results/partition_repair/stage_c_validation_30
```

采集按相同实例内的未改变组复用离线标签，保存三张表及 sessions 计时记录。中断后重复相同命令可续跑。源代码和配置变化时使用新清单、新输出目录；不要让两个进程同时写同一目录。

## 训练并冻结可部署策略

```bash
python -u scripts/train_partition_selector.py --mode fit --input "$STAGE_B" results/partition_repair/stage_c_train_30 --validation results/partition_repair/stage_c_validation_30 --output results/partition_repair/stage_c_model --seed 906030 --cost-limit 0.10
```

两个回归器只拟合训练实例。统一权重、按规模权重和学习评分参数在验证实例上确定，默认选择总体成本不超过 10% 时节时最多的配置，另报 8%/9% 的校准结果。stay 始终可选。

输出包括：

| 文件 | 用途 |
|---|---|
| policy.json | 冻结的手工与学习选择参数、训练/校准身份、环境和模型指纹 |
| learned_ridge.joblib、learned_tree.joblib | CPU 成本差和时间差模型 |
| learned_ridge_metadata.json、learned_tree_metadata.json | 特征顺序、模型参数、依赖及标签来源 |
| learning_report.md/json、learning_per_instance.csv | 验证集校准表现，不是最终测试结果 |

模型与 policy.json 一起保留。训练和推理使用相同的学习依赖；不允许替换模型文件后继续使用旧策略指纹。

## 在验证集上实际复测

从已采集实例派生当前策略清单，保存原标签来源以及每个完整候选分区的指纹：

```bash
python scripts/prepare_partition_learning.py derive --source-manifest results/partition_repair/stage_c_validation_30/manifest.json --output results/partition_repair/stage_c_manifests/validation_evaluation.json
python -u scripts/evaluate_partition_repair.py --manifest results/partition_repair/stage_c_manifests/validation_evaluation.json --policy results/partition_repair/stage_c_model/policy.json --methods symmetric_mst handcrafted_calibrated size_handcrafted learned_ridge learned_tree random --output results/partition_repair/stage_c_validation_evaluation
python scripts/analyze_partition_candidates.py --input results/partition_repair/stage_c_validation_evaluation --output results/partition_repair/stage_c_validation_evaluation/report --cost-limit 0.10
```

复测重新生成候选，并核对与来源的完整分区一致；选择阶段不会读取真实成本或耗时。每个策略只求解选中的一个划分，实际计时不使用已求解结果缓存。

默认每实例每方法一次，方法顺序轮转并交替反向。需要重复计时时，使用新输出目录并在开始时指定 --repeats 3。随机选择种子依赖实验种子、实例和重复编号，求解器种子保持所有方法一致。再次执行相同命令跳过已完成记录，不算新的计时观测。

下载整个 stage_c_validation_evaluation 目录。报告先对同一实例的重复取均值，再计算总体总量之比、按规模统计和配对区间。验证集用于校准，因此这里仍称验证表现。

## 按证据扩充训练集

需要更多标签时，以新的 seed 和输出文件再次采样 train，追加所有已使用的训练、验证、开发清单作为排除项。将新增采集目录追加到 fit 的 --input，使用新的模型输出目录。

训练规模按 30 → 60 → 90 → 120 个完整实例逐批扩大，验证集保持既有 30 个。只有实例与标签完整、同一计时环境时才允许合并；源码来源和标签内容指纹逐份保留。采集成本和实际选择收益共同决定是否继续，不默认一次采集全部规模。

## 方法固定后：90 个全新测试实例

以下示例适用于训练来源为原 B 加 train_30；若已扩充训练集，将新增清单全部追加到排除项。策略文件必须已经冻结。

```bash
python scripts/prepare_partition_learning.py sample --template-manifest "$STAGE_B/manifest.json" --split test --instances-per-size 30 --seed 906090 --exclude-manifests "$STAGE_A/manifest.json" results/partition_repair/stage_c_manifests/train_30.json results/partition_repair/stage_c_manifests/validation_30.json --output results/partition_repair/stage_c_manifests/test_90.json
python -u scripts/evaluate_partition_repair.py --manifest results/partition_repair/stage_c_manifests/test_90.json --policy results/partition_repair/stage_c_model/policy.json --methods symmetric_mst handcrafted_calibrated size_handcrafted learned_ridge learned_tree random --output results/partition_repair/stage_c_test_90
python scripts/analyze_partition_candidates.py --input results/partition_repair/stage_c_test_90 --output results/partition_repair/stage_c_test_90/report --cost-limit 0.10
```

最终测试不采集全候选标签，不再调权，也不事后重新分配测试集的成本预算。入口检查测试实例与训练、校准实例不重叠。点估计达标与配对区间是否跨越门槛分别报告。

## 计时和结果口径

```text
成本变化 = Σ新方法最终配送成本 / Σ对称MST最终配送成本 − 1
第二阶段节时 = 1 − Σ新方法完整第二阶段时间 / Σ对称MST完整第二阶段时间
```

每组共同使用 Gurobi TimeLimit=30、Threads=1、Seed=0、MIPGap=0.0001，除非固定清单明确指定其他值。优化限时不包含输入、建模、顺序恢复、回退及第三阶段，不能当作整个实例的耗时上限。

完整第二阶段包含组输入、距离构造、模型构建、优化、顺序提取、回退及其他实际开销。特征计算从候选生成和选择计时中扣除后单独记录，避免重复计算。没有可行解时使用共同的确定性有向最近邻回退；第三阶段使用固定配送评价器。

online_seconds 是模型构造到最终配送结果的实测墙钟，包含所需特征、候选生成和推理。模型常驻，实际冷加载耗时在 map_preparation.json 的 model_load_seconds 中按方法单列；一次共享地图准备也单列。cold_batch_seconds 将这些一次性成本加到按实例汇总的在线时间上，不能与常驻时间混为同一口径。

原始记录和旧 5% 报告保留。分析脚本未指定 --output 时使用带门槛的 report_cost_0.1_time_0.2 目录；需要复查旧门槛可显式传入 --cost-limit 0.05。

## 本机正确性测试

```powershell
& 'D:\anaconda3\envs\MA-FSTSP\python.exe' -X utf8 -m unittest discover -s tests -p 'test_partition_repair_*.py' -v
```

测试使用合成标签与微型人工图，不替代服务器上的 30 实例开发诊断、实际验证和最终测试。
