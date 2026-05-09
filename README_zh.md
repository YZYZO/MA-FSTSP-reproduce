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

但当前仓库已经做了离线兼容：

- 如果缺少 `nyc.graphml`，程序会自动构造一个 **Manhattan 风格网格图**。
- 如果缺少 `cambridge.graphml`，程序会自动构造一个 **Cambridge 风格网格图**。
- 若允许联网，也可以让 `osmnx` 自动下载 Cambridge/Boston 路网。

缓存文件会自动生成到 `datasets/` 目录下，例如：

- `datasets/manhattan.json`
- `datasets/cambridge_all_pair_road_distance.pkl`

## 如何运行实验
### 1. 运行轻量级演示实验
如果你当前没有真实地图数据，直接运行：

```bash
python experiments.py
```

此时脚本会自动进入 **轻量 demo 模式**，运行几个小规模可复现的合成实例，确认代码能正常工作。

### 2. 运行论文全量实验
如果你已经准备好了真实数据，或者明确要跑完整实验，请打开 `config.py`，将：

```python
RUN_FULL_EXPERIMENTS = True
```

然后运行：

```bash
python experiments.py
```

## 如何生成图表
运行：

```bash
python plot.py
```

当前脚本已经做了兼容处理：

- 如果某些 `.npy` 结果文件不存在，对应图表会自动跳过，而不会报错退出。
- 即使没有完整论文实验结果，`plot.py` 也仍然可以生成一部分示意图与 HTML 地图。

典型输出包括：

- `results/small/figures/overlap.pdf`
- `results/manhattan/maps/map.html`
- `results/manhattan/maps/tsp.html`
- `results/manhattan/maps/solution.html`
- `results/boston/maps/map.html`
- `results/boston/maps/tsp.html`
- `results/boston/maps/solution.html`

## 配置说明
项目参数集中放在 `config.py`，不需要在命令行中手动设置 `$env`。

常用项：

- `RUN_FULL_EXPERIMENTS`：是否运行论文全量实验。
- `ALLOW_OSM_DOWNLOAD`：是否允许用 `osmnx` 下载 Boston/Cambridge 路网。
- `REFRESH_OSM`：是否强制重新下载 Boston 路网。
- `OSM_DIST_METERS`：下载中心点周围的半径。
- `OSM_MAX_NODES`：用于控制真实路网规模，避免全对最短路过慢。
- `RESULTS_DIR`：实验与绘图输出目录，默认是 `results/`。

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
- `results/manhattan/data/quick-road-subset.npz`
- `results/boston/data/quick-road-instance.npz`

其中：

- `r-*` 对应飞行半径上限消融实验。
- `speed-*` 对应无人机速度消融实验。
- `k-cost.npy` 对应无人机数量实验。
- `city-*`、`rates-*`、`depots-*` 对应扩展性实验。

## 当前仓库额外做过的改动
为了让代码在本地缺少原始数据时也能顺利运行，当前版本还加入了以下增强：

- 自动生成合成 Manhattan / Cambridge 路网。
- `experiments.py` 自动区分“论文全量实验”和“离线轻量 demo”。
- `plot.py` 在缺少实验结果文件时自动跳过对应图，而不是报错。
- `plot.py` 已切换为非交互式后端，适合在终端/服务器环境中运行。
- 项目参数已集中到 `config.py`。
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
