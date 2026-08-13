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
- `experiment_results.py`
  负责采集三阶段过程数据、校验并保存新格式 NPZ，以及从 NPZ 恢复指定实例。
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
### 绘制默认 1K/11K 路网结果

`plot.py` 当前默认配置绘制 `manhattan_1k` 与 `boston_11k`。例如，实验结果中的客户数为 20 时，
在项目根目录运行：

```powershell
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" -c "import plot; plot.plot_large_road_experiment_results(customer_count=20, num_instances=1)"
```

绘图程序会分别在 `results/manhattan/data/` 与 `results/boston/data/` 中查找
“同地图、同客户数量”的最新时间戳 NPZ。新格式 NPZ 已经保存最终路线，因此绘图不会重新运行优化算法。

如果一批实验只有 1 个实例，使用 `num_instances=1`，该实例编号为 0。如果一批实验有 100 个实例，
可以省略 `instance_index`；程序默认选择目标函数最接近中位数的 median 代表实例。也可以显式指定任意实例：

```powershell
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" -c "import plot; plot.plot_large_road_experiment_results(customer_count=100, instance_index=37, num_instances=100)"
```

### 一次绘制四个路网结果

下面的 PowerShell 命令会同时绘制 `manhatten`、`manhattan_1k`、`boston` 和
`boston_11k`。示例假设客户数为 20，且每批只有 1 个实例：

```powershell
@'
from config import (
    MANHATTAN_BASELINE_GRAPH_PATH,
    MANHATTAN1k_GRAPH_PATH,
    MANHATTAN11k_GRAPH_PATH,
    BOSTON_GRAPH_PATH,
)
from plot import plot_large_road_experiment_results

# 四个结果文件共用同一个绘图入口，但分别指定结果文件地图名和实际 GraphML。
city_configs = {
    "manhatten": {
        "label": "Manhattan baseline",
        "result_subdir": "manhattan",
        "result_map_name": "manhatten",
        "graph_loader": "manhattan",
        "graph_path": MANHATTAN_BASELINE_GRAPH_PATH,
        "num_depots": 5,
        "drones_per_truck": 3,
    },
    "manhattan_1k": {
        "label": "Manhattan 1K",
        "result_subdir": "manhattan",
        "result_map_name": "manhattan_1k",
        "graph_loader": "manhattan",
        "graph_path": MANHATTAN1k_GRAPH_PATH,
        "num_depots": 5,
        "drones_per_truck": 3,
    },
    "boston": {
        "label": "Boston",
        "result_subdir": "boston",
        "result_map_name": "boston",
        "graph_loader": "boston",
        "graph_path": BOSTON_GRAPH_PATH,
        "num_depots": 10,
        "drones_per_truck": 3,
    },
    "boston_11k": {
        "label": "Boston 11K",
        "result_subdir": "boston",
        "result_map_name": "boston_11k",
        "graph_loader": "manhattan",
        "graph_path": MANHATTAN11k_GRAPH_PATH,
        "num_depots": 10,
        "drones_per_truck": 4,
    },
}

generated = plot_large_road_experiment_results(
    city_configs=city_configs,
    cities=tuple(city_configs),
    customer_count=20,
    instance_index=0,
    num_instances=1,
)

# 打印每张交互式地图及其 JSON 摘要的实际输出路径。
for name, files in generated.items():
    print(f"{name}:")
    print(f"  map: {files['map']}")
    print(f"  summary: {files['summary']}")
'@ | & "D:\Anaconda3\envs\MA-FSTSP\python.exe" -
```

若四个路网结果各包含 100 个实例，删除 `instance_index=0`，并把 `num_instances=1`
改为 `num_instances=100`，程序就会为每批结果自动选择 median 实例。

生成的交互式路线地图和摘要位于：

```text
results/manhattan/maps/*-solution.html
results/manhattan/maps/*-summary.json
results/boston/maps/*-solution.html
results/boston/maps/*-summary.json
```

用浏览器打开 `*-solution.html` 即可查看卡车和无人机路线；`*-summary.json` 包含阶段耗时、
客户分组、Set-TSP 顺序和路线统计。

### 运行绘图脚本默认入口

也可以先修改 `plot.py` 顶部的 `LARGE_ROAD_*` 默认配置，然后直接运行：

```powershell
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" plot.py
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
项目级参数放在 `config.py`；只服务于地图加载或绘图的局部参数分别放在 `problem.py` 和 `plot.py`。

常用项：

- `RUN_FULL_EXPERIMENTS`：是否运行论文全量实验。
- `problem.py / ALLOW_OSM_DOWNLOAD`：是否允许用 `osmnx` 下载 Boston/Cambridge 路网。
- `problem.py / REFRESH_OSM`：是否强制重新下载 Boston 路网；只有与下载授权同时开启才联网。
- `problem.py / OSM_DIST_METERS`：下载中心点周围的半径。
- `problem.py / OSM_MAX_NODES`：显式联网刷新 Boston 时允许保留的节点上限。
- `plot.py / DEMO_DRONE_LIMIT`：绘图示例使用的无人机航程上限。
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

路网对比实验会按“运行开始时刻-地图名-客户数量”保存压缩 NPZ，例如：

- `results/manhattan/data/20260812-201927-manhattan_1k-100.npz`
- `results/boston/data/20260812-202012-boston_11k-100.npz`
- `results/manhattan/data/quick-road-subset.npz`
- `results/boston/data/quick-road-instance.npz`

每个新格式路网 NPZ 同时保存全部实例的成本、耗时、仓库/客户节点、Phase 1
客户分组、Phase 2 访问顺序和最终卡车/无人机联合路线。变长结构使用 JSON 字符串数组，
加载时不需要开启 pickle。批次还会记录 best、median、worst 三个代表实例的路线距离、
航程约束和目标值一致性校验；不会保存完整 DP 表或卡车逐道路节点展开结果。

上述结果格式的采集、写入和恢复集中在 `experiment_results.py`。`experiments.py`
只负责构造并调度实验，`plot.py` 则把恢复后的实例转换成统计图和路线地图。

`plot.py` 默认选择 median 代表实例并直接读取保存路线。只有读取旧版
`road-size-*.npz`（其中没有路线）时，才会重新构造实例并求解。

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
10. `experiment_results.py`
11. `plot.py`

这样会更容易先理解数据流，再理解算法实现。

## 引用
如果你在研究或项目中使用了本仓库，建议引用原论文。
