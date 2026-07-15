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

地图现在使用 `config.py` 中的显式路径：

- `MANHATTAN_GRAPH_PATH = datasets/nyc.graphml`：论文目标 55k 图；本机默认硬性禁止读取、构建和求解。
- `MANHATTAN_BASELINE_GRAPH_PATH = datasets/manhatten.graphml`：4,333 节点标准化基线，仅用于本机小实例。
- `BOSTON_GRAPH_PATH = datasets/boston.graphml`：标准化后为 8,313 节点。

缺失文件默认直接报出完整路径，不会静默换用另一张地图、联网下载或生成合成图；这些行为只能通过独立配置显式开启。旧 `manhattan.json` 和 `cambridge_all_pair_road_distance.pkl` 不再加载、写入或自动删除。

## H2H 原生距离后端

先执行一次 Release 编译：

```powershell
D:\Anaconda3\envs\MA-FSTSP\python.exe scripts\build_h2h_native.py --release
```

小图由 `DISTANCE_BACKEND=auto` 使用 eager 基线；超过 `EAGER_DISTANCE_MAX_NODES` 后使用 H2H。索引按完整图 SHA-256 缓存到 `datasets/indexes/`，包含 `graph.bin`、`index.bin`、`metadata.json`、`build.log` 和 `READY`。多个进程首次访问同一图时由跨进程锁保证只构建一次。

本机 `H2H_ENABLE_55K=False` 时，选择 `datasets/nyc.graphml` 会在读取文件和启动 builder 前停止。只有约 200 GB RAM 的服务器才应显式改为 `True`；Linux 服务器需用同一脚本重新编译 `.so`。

阶段 6 的本机慢速验收使用独立临时索引，不污染 `datasets/indexes/`：

```powershell
$env:H2H_RUN_LOCAL_ACCEPTANCE = "1"
D:\Anaconda3\envs\MA-FSTSP\python.exe -m unittest tests.test_h2h_phase6_acceptance -v
```

该测试会在 4,333 和 8,313 节点标准化图上各对照 100,000 个有序节点对，并检查三条查询路径吞吐、缓存重载和两个 spawn worker。普通 `unittest discover` 会跳过这三个显式慢速项。

55k 服务器验收不需要永久修改 `config.py`，而由专用脚本要求一次性的明确授权：

```bash
python scripts/run_h2h_server_acceptance.py \
  --confirm-server-55k \
  --compiler g++ \
  --worker-counts 1,4,8,16 \
  --customer-counts 20
```

脚本只允许 Linux，默认要求至少 150 GiB 物理内存；它会重编译 `.so`、构建/加载 55k 索引、完成 200 个源共 100,000 个 Dijkstra 对照、查询吞吐、worker 扩展和 5 仓库/20 客户/3 无人机端到端实例，并原子写入 `results/h2h-server-55k-acceptance.json`。首个实例通过后可用 `--customer-counts 50,100,150` 继续扩展。完整说明见 `docs/H2H_SERVER_ACCEPTANCE.md`。

## 如何运行实验
### 1. 运行轻量级演示实验
准备好 `MANHATTAN_BASELINE_GRAPH_PATH` 和 `BOSTON_GRAPH_PATH` 后，将 `RUN_FULL_EXPERIMENTS=False` 并运行：

```bash
python experiments.py
```

此时脚本进入轻量 demo 模式，只在真实基线图上截取小规模子图；缺少显式地图时会直接报错，不会悄悄改用合成地图。

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
- `OSM_MAX_NODES`：显式联网刷新 Boston 时允许保留的节点上限。
- `DISTANCE_BACKEND`：`auto` 在小图使用 eager，较大图使用 H2H。
- `H2H_ENABLE_55K`：服务器运行 55k NYC 的显式开关，本机必须保持 `False`。
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
当前版本还加入了以下增强：

- 使用显式地图路径，缺失或选错地图时快速失败。
- `experiments.py` 由配置区分论文全量实验和本机轻量 demo。
- 使用 H2H 按需卡车距离和按需无人机距离，避免真实图全对矩阵。
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
