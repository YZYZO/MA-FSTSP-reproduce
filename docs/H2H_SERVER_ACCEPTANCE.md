# H2H 55k Linux 服务器验收说明

## 安全前提

服务器脚本默认拒绝运行，只有同时满足以下条件才会读取 `nyc.graphml`：

1. 命令行包含 `--confirm-server-55k`；
2. 操作系统为 Linux；
3. `/proc/meminfo` 报告的物理内存不少于 150 GiB；
4. 显式 GraphML 文件存在。

`config.py` 的 `H2H_ENABLE_55K` 应继续保持 `False`。脚本仅在所有检查通过后，为当前 Python 进程临时放开保护；退出后不会改变配置文件。Windows `.exe/.dll` 不可复制使用，脚本默认调用统一构建入口重新生成 Linux builder 和 `libh2h_query.so`。

## 首次 20 客户验收

在项目根目录执行：

```bash
python scripts/run_h2h_server_acceptance.py \
  --confirm-server-55k \
  --compiler g++ \
  --worker-counts 1,4,8,16 \
  --customer-counts 20 \
  --depots 5 \
  --drones 3 \
  --gurobi-threads 1
```

默认行为包括：

- Release 编译与原生 smoke test；
- 版本化 55k 索引构建或安全缓存命中；
- treewidth、treeheight、fill-in、shortcut、标签、索引大小、构建时间和 builder 峰值 RSS；
- 200 个源节点、合计 100,000 个有序节点对的分组 Dijkstra 对照；
- batch、`ctypes` 标量和完整双下标各 100,000 次吞吐；
- 1、4、8、16 个 spawn worker 共享只读 mmap 的 1,000,000 查询吞吐；
- 5 仓库、20 客户、3 无人机的主算法成本、时间、内存和路线可行性。

结果原子写入：

```text
results/h2h-server-55k-acceptance.json
```

任一正确性、100,000 queries/s、worker 吞吐收益或端到端可行性条件失败，报告状态为 `failed`，进程返回非零。

## 后续规模

20 客户报告通过后，再依次运行：

```bash
python scripts/run_h2h_server_acceptance.py \
  --confirm-server-55k \
  --skip-native-build \
  --worker-counts 1,4,8,16 \
  --customer-counts 50,100,150 \
  --gurobi-threads 1 \
  --report results/h2h-server-55k-scale-acceptance.json
```

不要一开始把 worker 数设置为逻辑 CPU 数。只有 4 worker 相比 1 worker 有实测收益且内存安全，才继续接受 8、16；每个 Gurobi 进程默认只使用 1 线程，避免进程数与求解器线程相乘造成过度并行。

## 失败处理

- builder 资源上限或异常退出时，保留 `.building-*` 目录和 `build.log` 诊断；不要修改原始 GraphML。
- 已发布缓存校验失败时，Python 会把它原子隔离为 `.invalid-*`，再在同一图哈希下重建。
- 若 fill-in、treewidth 或峰值 RSS 异常，先调整服务器资源上限或评估消元顺序，不修改精确查询定义。
- 若 H2H 查询正确但端到端耗时集中在客户区域扫描或 Set-TSP，转入独立空间索引优化阶段，不继续改已验证的 H2H 核心。

实际服务器报告是阶段 7 的退出证据。运行完成后应保存 JSON，并把其中的 `cache.statistics`、`correctness`、`query_performance`、`worker_scaling` 和 `end_to_end` 纳入项目验收记录。
