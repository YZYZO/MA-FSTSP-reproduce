# 多智能体飞行侧援旅行商问题（MA-FSTSP）中文说明

## 项目简介
本仓库是论文 **Optimization of Multi-Agent Flying Sidekick Traveling Salesman Problem over Road Networks** 的实验代码实现。项目研究的是这样一类问题：

- 有多辆卡车从不同仓库出发。
- 每辆卡车可以携带若干架无人机。
- 客户既可以由卡车直接送达，也可以由无人机从卡车途中起飞并完成配送。
- 路网对卡车可达性有限制，无人机则更接近直线飞行，但受续航和速度约束。

本仓库实现了论文主算法，也包含多个基线算法、实验脚本和绘图脚本。

## 代码结构
### 根目录主要文件
- `experiments.py`
  论文实验主入口，负责运行小规模实验、真实/合成路网实验、消融实验与扩展性实验。
- `plot.py`
  读取实验结果并生成图表、示意图与 HTML 地图。
- `problem.py`
  负责读取/构造 Manhattan 与 Cambridge 路网，并生成随机实验实例。
- `experiment_results.py`
  负责采集求解阶段记录，并保存、校验和恢复新格式路网 NPZ。
- `utils.py`
  提供距离计算、最近邻搜索、MST 分组、ATSP 近似求解等通用工具函数。

### `src/` 目录主要文件
- `src/fstsp.py`
  论文主算法 `MultiAgentFlyingSidekickTSP`。
- `src/lrmp.py`
  LRMP 基线算法。
- `src/lp.py`
  线性规划基线算法。
- `src/hc_vns.py`
  爬山法 + 变邻域搜索基线算法。
- `src/baseline.py`
  所有算法共享的基类。

## 环境准备
原始 README 使用的是 Python 3.7，不过当前项目也已经在本地 `conda` 环境下完成了适配。推荐使用独立环境：

```bash
conda create -n MA-FSTSP python=3.11
conda activate MA-FSTSP
pip install -r requirements.txt
```

如果你更希望严格贴近论文原始环境，也可以改成：

```bash
conda create -n MA-FSTSP python=3.7
conda activate MA-FSTSP
pip install -r requirements.txt
```

## 数据说明
论文原始代码默认依赖以下真实路网数据：

- `nyc.graphml`
- `cambridge.graphml`（可选）

当前 1K/11K 复现实验显式使用：

- `datasets/nyc_1024.graphml`：1,024 节点 Manhattan 场景。
- `datasets/nyc_11000.graphml`：替代论文 Boston 11K 场景的同规模 NYC 路网；结果仍命名为 `boston_11k`。

但当前仓库已经做了离线兼容：

- 如果缺少 `nyc.graphml`，程序会自动构造一个 **Manhattan 风格网格图**。
- 如果缺少 `cambridge.graphml`，程序会自动构造一个 **Cambridge 风格网格图**。
- 若允许联网，也可以让 `osmnx` 自动下载 Cambridge/Boston 路网。

缓存文件会自动生成到 `datasets/` 目录下，例如：

- `datasets/manhattan.json`
- `datasets/cambridge_all_pair_road_distance.pkl`

## 如何运行实验
### 1. 选择并运行实验
当前执行范围由 `experiments.py` 中的 `run_full_experiments()` 明确控制。确认其中启用的函数和规模后运行：

```bash
python experiments.py
```

1K 实验入口为 `test_manhattan_1k(num, size)`，使用 5 个仓库、每车 3 架无人机；11K 实验入口为
`test_manhattan_11k(num, size)`，使用 10 个仓库、每车 4 架无人机。11K 的完整批次建议在服务器运行。

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

绘图程序会分别查找最新的 `*-manhattan_1k-100.npz` 与 `*-boston_11k-100.npz`。
新格式结果已经保存最终路线，因此绘图不会重新运行优化器；未指定 `instance_index` 时，默认选择成本最接近中位数的实例。

输出位于：

- `results/manhattan/maps/*-solution.html`
- `results/manhattan/maps/*-summary.json`
- `results/boston/maps/*-solution.html`
- `results/boston/maps/*-summary.json`

旧版 11K NPZ 不包含路线，默认不会在绘图阶段隐式重新求解。如确实需要兼容，可显式传入
`allow_legacy_11k_resolve=True`，但应提前确认运行环境能够承担 11K 距离构造和求解。

典型输出包括：

- `results/small/figures/overlap.pdf`
- `results/manhattan/maps/map.html`
- `results/manhattan/maps/tsp.html`
- `results/manhattan/maps/solution.html`
- `results/boston/maps/map.html`
- `results/boston/maps/tsp.html`
- `results/boston/maps/solution.html`

## 配置说明
项目级路径集中放在 `config.py`，不需要在命令行中手动设置环境变量。

常用项：

- `RESULTS_DIR`：实验与绘图输出目录，默认是 `results/`。
- `MANHATTAN1k_GRAPH_PATH`：1K GraphML 路径。
- `MANHATTAN11k_GRAPH_PATH`：11K GraphML 路径。

OSM 下载、刷新、半径、节点数和 Overpass 地址与地图加载实现放在 `problem.py`。下载与刷新默认都关闭，
只有同时启用 `ALLOW_OSM_DOWNLOAD` 和 `REFRESH_OSM` 才会联网刷新 Boston/Cambridge 路网。

## 结果文件说明
完整实验运行后，通常会生成以下 `.npy` 文件：

- `results/manhattan/data/r-time.npy`
- `results/manhattan/data/r-cost.npy`
- `results/manhattan/data/speed-time.npy`
- `results/manhattan/data/speed-cost.npy`
- `results/manhattan/data/k-cost.npy`
- `results/manhattan/data/city-time.npy`
- `results/manhattan/data/city-cost.npy`
- `results/manhattan/data/rates-time.npy`
- `results/manhattan/data/rates-cost.npy`
- `results/manhattan/data/depots-time.npy`
- `results/manhattan/data/depots-cost.npy`

路网对比实验会额外保存：

- `results/manhattan/data/road-size-*.npz`
- `results/boston/data/road-size-*.npz`
- `results/manhattan/data/<时间戳>-manhattan_1k-<客户数>.npz`
- `results/boston/data/<时间戳>-boston_11k-<客户数>.npz`

新格式路网 NPZ 使用 `result_schema_version=3`，同时保存：

- 全部实例的成本、耗时、仓库和客户节点；
- 最终卡车/无人机联合路线；
- Phase 1 客户分组、Phase 2 Set-TSP 顺序和 Phase 3 耗时/成本；
- `phase1_partition_method`，用于区分原版无向覆盖语义与修正版双向平均语义；
- best、median、worst 代表实例的路线约束与目标值一致性检查。

修正版 Phase 1 方法标识为 `bidirectional-mean-set-mst-v1`。旧版 schema 2 文件仍可读取，
但由于未保存该字段，会标记为 `legacy-or-unspecified`。比较 `master` 与修复分支时，应复用
完全相同且顺序一致的仓库、客户数组，并按该字段分开汇总，不能把两类结果混入同一批次。

变长结构使用 Unicode JSON 字符串数组，读取时不需要开启 pickle。结果不会保存全对距离矩阵、完整 DP 表或
卡车逐道路节点展开结果。保存与恢复逻辑集中在 `experiment_results.py`。

其中：

- `r-*` 对应飞行半径上限消融实验。
- `speed-*` 对应无人机速度消融实验。
- `k-cost.npy` 对应无人机数量实验。
- `city-*`、`rates-*`、`depots-*` 对应扩展性实验。

## 当前仓库额外做过的改动
为了让代码在本地缺少原始数据时也能顺利运行，当前版本还加入了以下增强：

- 自动生成合成 Manhattan / Cambridge 路网。
- `experiments.py` 通过 `run_full_experiments()` 显式选择当前运行的实验和规模。
- `plot.py` 在缺少实验结果文件时自动跳过对应图，而不是报错。
- `plot.py` 已切换为非交互式后端，适合在终端/服务器环境中运行。
- 项目路径保存在 `config.py`，OSM 参数与地图加载逻辑共同保存在 `problem.py`。
- 所有新实验输出已统一写入 `results/`，并按地图划分为 `manhattan/`、`boston/`、`small/`。
- 代码中已经补充了较完整的中文注释、模块说明和函数说明。

## 建议阅读顺序
如果你第一次阅读这个仓库，建议按下面顺序看：

1. `README.md` / `README_zh.md`
2. `problem.py`
3. `utils.py`
4. `src/baseline.py`
5. `src/hc_vns.py`
6. `src/lp.py`
7. `src/lrmp.py`
8. `src/fstsp.py`
9. `experiments.py`
10. `plot.py`

这样会更容易先理解数据流，再理解算法实现。

## 引用
如果你在研究或项目中使用了本仓库，建议引用原论文。
