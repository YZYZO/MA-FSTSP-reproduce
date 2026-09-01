# MA-FSTSP 服务器扩大实验运行指南

本文档从服务器运行检查开始，默认服务器已经具备可用的 Python 环境、项目依赖和 Gurobi 许可证。

实验范围：

- 正式扩大 Manhattan 1K 实验；
- Manhattan 11K 只进行小规模泛化测试；
- 不运行 Manhattan 55K。

## 一、检查服务器能否运行项目

进入项目并激活已有环境：

```bash
cd ~/MA-FSTSP-reproduce
conda activate MA-FSTSP
```

确认必要的路网和模型文件存在：

```bash
ls -lh datasets/nyc_1024.graphml
ls -lh datasets/nyc_11000.graphml
ls -lh results/learning/surrogate/
ls -lh results/learning/hrl/
ls -lh results/learning/single_level/
```

运行项目测试：

```bash
python -m unittest discover -s tests -v
```

然后运行一个最小真实实验：

```bash
python -u scripts/evaluate_partition_methods.py \
  --maps manhattan1k \
  --customer-sizes 10 \
  --instances 1 \
  --methods legacy_mst symmetric_mst balanced_mst solver_aware_greedy single_level_rl solver_aware_hrl \
  --time-limit 10 \
  --surrogate-checkpoint results/learning/surrogate/surrogate_20260901-210029-final.pt \
  --hrl-checkpoint results/learning/hrl/hrl_policy_20260901-milestone-c-accepted.pt \
  --single-level-checkpoint results/learning/single_level/single_level_policy_20260901-milestone-d.pt \
  --run-id linux-smoke
```

满足以下条件，即可开始扩大实验：

- 单元测试通过；
- 没有 Gurobi 许可证错误；
- `linux-smoke` 正常结束并生成 CSV、JSON 和 Markdown 报告；
- 运行过程中没有出现内存不足或进程被系统终止。

## 二、扩大 Manhattan 1K 学习数据集

正式扩大前，建议先把 `--instances-per-size` 设为 `2` 测量服务器耗时。确认可以接受后，再使用下面的推荐设置：

```bash
python -u scripts/generate_learning_dataset.py \
  --customer-sizes 50 100 150 \
  --instances-per-size 20 \
  --relocate-samples 6 \
  --swap-samples 2 \
  --time-limit 30 \
  --phase2-budget 20 \
  --downstream-budget 50 \
  --seed 0 \
  --run-id linux-data-s0
```

该设置预计生成约 1260 个客户组样本，并按照原始实例划分训练集、验证集和测试集。

注意：Set-TSP 有 30 秒时间限制，但第三阶段目前没有真正的硬超时机制。若某个客户组长时间没有进展，可以中断任务；重新运行相同命令时，已有的客户组求解结果可以从 SQLite 缓存复用。

## 三、训练代理模型

```bash
python -u scripts/train_surrogate.py \
  --dataset results/learning/dataset/learning_dataset_linux-data-s0.csv \
  --max-epochs 600 \
  --patience 60 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --mc-samples 30 \
  --seed 0 \
  --run-id linux-surrogate-s0
```

训练完成后，主要检查：

- 成本预测的 Spearman 排序相关性；
- Set-TSP 时间预测的 Spearman 排序相关性；
- 超时识别的 AUC 和 F1；
- 输出中的 `eligible_for_RL` 是否为 `true`。

## 四、训练层次化强化学习策略

```bash
python -u scripts/train_hrl.py \
  --surrogate-checkpoint results/learning/surrogate/surrogate_linux-surrogate-s0.pt \
  --customer-sizes 50 100 150 \
  --train-instances-per-size 12 \
  --validation-instances-per-size 4 \
  --imitation-epochs 20 \
  --reinforcement-episodes 600 \
  --max-steps 8 \
  --relocate-candidates 8 \
  --swap-candidates 8 \
  --real-verification-budget 0 \
  --phase2-budget 20 \
  --seed 0 \
  --run-id linux-hrl-s0
```

`--real-verification-budget` 暂时保持为 `0`，避免训练过程中频繁调用真实 Set-TSP 和第三阶段求解器。

## 五、训练单层强化学习基线

单层 RL 应采用与 HRL 相同的实例数量、训练轮数和候选动作数量，保证对比公平：

```bash
python -u scripts/train_single_level_rl.py \
  --surrogate-checkpoint results/learning/surrogate/surrogate_linux-surrogate-s0.pt \
  --customer-sizes 50 100 150 \
  --train-instances-per-size 12 \
  --validation-instances-per-size 4 \
  --imitation-epochs 20 \
  --reinforcement-episodes 600 \
  --max-steps 8 \
  --relocate-candidates 8 \
  --swap-candidates 8 \
  --phase2-budget 20 \
  --seed 0 \
  --run-id linux-single-s0
```

## 六、扩大 Manhattan 1K 真实评估

先用每个客户规模 3 个实例进行试运行：

```bash
python -u scripts/evaluate_partition_methods.py \
  --maps manhattan1k \
  --customer-sizes 50 100 150 \
  --instances 3 \
  --time-limit 30 \
  --cost-tolerance 0.01 \
  --seed 2026 \
  --surrogate-checkpoint results/learning/surrogate/surrogate_linux-surrogate-s0.pt \
  --hrl-checkpoint results/learning/hrl/hrl_policy_linux-hrl-s0.pt \
  --single-level-checkpoint results/learning/single_level/single_level_policy_linux-single-s0.pt \
  --run-id linux-eval-pilot
```

确认耗时和内存可以接受后，将：

```text
--instances 3
```

改成：

```text
--instances 10
```

正式报告至少汇总以下指标：

- 最终成本；
- Set-TSP P90 时间和最大时间；
- 下游总时间 P90 和最大时间；
- 超时率；
- 划分策略耗时；
- 真实求解器调用次数；
- 最大客户组规模；
- 相对 `symmetric_mst` 的成本变化。

联合验收条件保持为：最终成本相对 `symmetric_mst` 上升不超过 1%，并且 P90 时间或超时率得到改善。

## 七、Manhattan 11K 小规模泛化测试

当前 11K 只测试 10 个客户，不直接运行 50、100、150 客户：

```bash
python -u scripts/evaluate_partition_methods.py \
  --maps manhattan11k \
  --customer-sizes 10 \
  --instances 3 \
  --methods symmetric_mst solver_aware_greedy single_level_rl solver_aware_hrl \
  --distance-mode lazy \
  --lazy-truck-cache-rows 64 \
  --time-limit 10 \
  --surrogate-checkpoint results/learning/surrogate/surrogate_linux-surrogate-s0.pt \
  --hrl-checkpoint results/learning/hrl/hrl_policy_linux-hrl-s0.pt \
  --single-level-checkpoint results/learning/single_level/single_level_policy_linux-single-s0.pt \
  --run-id linux-11k-pilot
```

如果 10 客户实验稳定，可以再尝试 20 客户。当前不建议继续扩大，因为第三阶段没有硬超时，而且代理模型还没有使用 11K 标签训练。

## 八、后台运行与查看进度

推荐使用 `tmux`：

```bash
tmux new -s mafstsp
```

在 tmux 中启动实验后，使用 `Ctrl+B`，再按 `D`，即可退出界面但保持任务运行。重新进入：

```bash
tmux attach -t mafstsp
```

也可以使用 `nohup`：

```bash
mkdir -p logs

nohup python -u scripts/evaluate_partition_methods.py \
  --maps manhattan1k \
  --customer-sizes 50 100 150 \
  --instances 3 \
  --time-limit 30 \
  --run-id linux-eval-pilot \
  > logs/linux-eval-pilot.log 2>&1 &
```

查看进程和日志：

```bash
echo $!
tail -f logs/linux-eval-pilot.log
```

当前不建议并行启动多个实验进程，避免争用 CPU、内存、SQLite 缓存和 Gurobi 许可证。如果以后需要并行运行不同随机种子，每个进程应使用不同的 `--cache-path` 和 `--run-id`。

## 九、推荐执行顺序

1. 运行单元测试和 `linux-smoke`。
2. 用每个规模 2 个实例测试数据生成耗时。
3. 扩大到每个规模 20 个数据实例。
4. 训练代理模型，确认质量门槛通过。
5. 使用相同设置训练 HRL 和单层 RL。
6. 用每个规模 3 个实例进行 1K 评估试运行。
7. 扩大到每个规模 10 个独立测试实例。
8. 最后进行 11K、10 客户的小规模泛化测试。

