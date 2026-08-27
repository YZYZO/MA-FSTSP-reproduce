# MA-FSTSP 第一阶段替代方案：Directed Set-GTDS

## 1. 方法概述

本文将 MA-FSTSP 原三阶段框架中的第一阶段客户划分替换为：

> **Directed Set-GTDS：基于有向集合代价、仓库感知开放巨路径和下游 Set-TSP 模型规模约束的客户划分方法。**

算法流程为：

1. 统一构造每个客户的候选服务点集合；
2. 计算客户与仓库之间的有向集合代价；
3. 求解一条仓库感知的全局客户开放巨路径；
4. 在给定巨路径上，通过仓库子集动态规划联合决定客户片段边界和仓库分配；
5. 在路线代理成本最多增加 1% 的约束下，最小化最大 Set-TSP 子问题的二进制变量规模；
6. 将最终客户分组交给原有 Phase 2 Set-TSP 和 Phase 3 动态规划继续求解。

本方法只替换第一阶段。以下模块保持不变：

- `get_seq()`；
- `set_tsp()`；
- `single_solution()`；
- `local_search_multi_drone_appr()`；
- 最终解格式与成本统计方式。

原始 MST 实现保留为公开代码基线，不参与新方法内部计算。

---

## 2. 输入与输出

### 2.1 输入

设：

- 路网为有向图 \(G=(V,E)\)；
- 仓库集合为 \(P\)，仓库数量为 \(m=|P|\)；
- 客户集合为 \(C\)，客户数量为 \(n=|C|\)；
- 卡车最短路距离为 \(d^{tr}(u,v)\)；
- 无人机距离为 \(d^{dr}(u,v)\)；
- 无人机速度参数为 \(s_{dr}\)；
- 每个客户的边界候选点集合为 \(R_c\)。

### 2.2 输出

输出形式与当前代码一致：

```python
groups = {
    depot_1: [city_a, city_b, ...],
    depot_2: [city_c, city_d, ...],
    ...
}
```

要求：

- 每个客户恰好分配一次；
- 每个仓库至多对应巨路径上的一个连续客户片段；
- 算法接口允许部分仓库没有客户，但 V2 主实验在客户数不少于仓库数时要求全部仓库非空；
- 每个非空客户分组均满足 Set-TSP 模型规模约束。

---

## 3. 候选集合正规化

先调用现有代码生成边界候选点集合：

```python
raw_sets = self.get_boundary_convex_sets(self.theta[0])
```

对每个客户统一正规化：

\[
\bar R_c=
\begin{cases}
R_c, & R_c\neq\varnothing,\\
\{c\}, & R_c=\varnothing.
\end{cases}
\]

仓库候选集合固定为：

\[
\bar R_p=\{p\},\qquad p\in P.
\]

实现形式：

```python
def normalize_candidate_sets(cities, raw_sets):
    return {
        city: list(raw_sets[city]) if raw_sets[city] else [city]
        for city in cities
    }
```

第一阶段和第二阶段必须共同使用同一份正规化集合，避免第一阶段代价计算与 Phase 2 Set-TSP 输入不一致。

---

## 4. 有向集合代价矩阵

### 4.1 候选点访问附加代价

对任意客户或仓库 \(u\in C\cup P\)，定义：

\[
\psi_u(x)=
\begin{cases}
0, & u\in P,\\
\dfrac{1}{s_{dr}}d^{dr}(u,x), & u\in C.
\end{cases}
\]

该系数直接对应论文式（3）。公开 SMST 代码使用的
\(\sqrt 2/s_{dr}\) 只保留为 `gtds_sqrt2` 敏感性实验，不用于 V2 主结果。

### 4.2 有向集合代价

对任意不同节点 \(u,v\in C\cup P\)，定义：

\[
A_{uv}=
\min\left\{
 d^{tr}(u,v),
 \min_{\substack{x\in\bar R_u\\y\in\bar R_v}}
 \left[
 \psi_u(x)+d^{tr}(x,y)+\psi_v(y)
 \right]
\right\}.
\]

其中：

- \(A_{uv}\) 表示从集合 \(u\) 转移到集合 \(v\) 的有向代理代价；
- \(A_{uv}\) 与 \(A_{vu}\) 分别计算、分别存储；
- 对角线元素设置为 0。

推荐存储结构：

```python
cost = {
    (u, v): directed_cost_uv
    for u in nodes
    for v in nodes
}
```

或者使用按照固定节点顺序建立的二维 `numpy.ndarray`。

### 4.3 实现要求

```python
def build_directed_set_cost(
    depots,
    cities,
    candidate_sets,
    truck_distance,
    drone_distance,
    speed,
):
    ...
```

实现时应满足：

- 不创建 `networkx.Graph`；
- 不调用 `minimum_spanning_tree()`；
- 不调用 `mst_partition()`；
- 所有节点遍历顺序固定；
- 相同实例在改变客户输入排列后，按节点编号重新对齐的代价矩阵保持一致。

---

## 5. 仓库感知开放巨路径

### 5.1 虚拟节点

加入虚拟节点 \(q\)。客户之间的代价保持为：

\[
\widetilde A_{c,c'}=A_{c,c'}.
\]

虚拟节点到客户的代价定义为最优仓库出发代价：

\[
\widetilde A_{q,c}=\min_{p\in P}A_{p,c}.
\]

客户到虚拟节点的代价定义为最优仓库返回代价：

\[
\widetilde A_{c,q}=\min_{p\in P}A_{c,p}.
\]

设置：

\[
\widetilde A_{q,q}=0.
\]

### 5.2 整数缩放

`elkai.DistanceMatrix` 使用整数距离矩阵。对非对角元素进行统一缩放：

```python
scaled[u][v] = max(1, round(scale * cost[u][v]))
```

建议主实验固定：

```python
scale = 1000
```

对角线保持为 0。

### 5.3 求解与恢复路径

调用：

```python
route = elkai.DistanceMatrix(matrix).solve_tsp()
```

处理步骤：

1. 在返回环中定位虚拟节点 \(q\)；
2. 将环旋转到 \(q\) 为首节点；
3. 删除首尾虚拟节点；
4. 删除可能存在的重复起点；
5. 检查每个客户恰好出现一次。

最终得到客户开放巨路径：

\[
\pi=(\pi_1,\pi_2,\ldots,\pi_n).
\]

接口：

```python
def solve_depot_aware_giant_path(
    depots,
    cities,
    directed_cost,
    scale=1000,
):
    ...
```

该步骤使用 LKH-based 启发式生成巨路径，不声明巨路径全局最优。

---

## 6. Set-TSP 模型规模指标

对巨路径中的连续片段：

\[
B_{i:j}=(\pi_i,\pi_{i+1},\ldots,\pi_j),
\qquad 1\leq i\leq j\leq n,
\]

设片段包含客户数：

\[
g_{i:j}=j-i+1.
\]

连同一个仓库集合后，Set-TSP 的集合数量为：

\[
N_{i:j}=g_{i:j}+1.
\]

设仓库集合大小为 1，每个客户集合大小为 \(|\bar R_c|\)。定义：

\[
W_{i:j}=1+\sum_{t=i}^{j}|\bar R_{\pi_t}|,
\]

\[
S^{(2)}_{i:j}=1+\sum_{t=i}^{j}|\bar R_{\pi_t}|^2.
\]

根据当前 `set_tsp()` 的变量结构，片段对应的预求解模型规模为：

### 二进制变量数量

\[
Q_{bin}(i,j)
=N_{i:j}^2+S^{(2)}_{i:j}+W_{i:j}^2.
\]

其中：

- `select` 变量数为 \(N_{i:j}^2\)；
- `internal` 变量数为 \(S^{(2)}_{i:j}\)；
- `external` 变量数为 \(W_{i:j}^2\)。

### 连续变量数量

\[
Q_{cont}(i,j)=N_{i:j}^2.
\]

### 总变量数量

\[
Q_{var}(i,j)
=2N_{i:j}^2+S^{(2)}_{i:j}+W_{i:j}^2.
\]

### 约束数量

\[
Q_{con}(i,j)
=2N_{i:j}^2+6N_{i:j}+1+2W_{i:j}.
\]

主算法以 \(Q_{bin}\) 作为下游复杂度约束指标。

对以下序列建立前缀和：

```text
|R_pi[t]|
|R_pi[t]|^2
```

从而在 \(O(1)\) 时间计算任意片段的 \(W_{i:j}\)、\(S^{(2)}_{i:j}\) 和 \(Q_{bin}(i,j)\)。

接口：

```python
def build_segment_statistics(giant_path, candidate_sets):
    ...
```

---

## 7. 片段路线代理成本

若仓库 \(p\) 服务连续片段 \(B_{i:j}\)，定义代理成本：

\[
H(p,i,j)
=A_{p,\pi_i}
+\sum_{t=i}^{j-1}A_{\pi_t,\pi_{t+1}}
+A_{\pi_j,p}.
\]

该代价包括：

- 仓库到片段首客户；
- 片段内部的有向客户连接；
- 片段末客户返回仓库。

对巨路径内部边代价建立前缀和：

\[
L_k=\sum_{t=1}^{k-1}A_{\pi_t,\pi_{t+1}}.
\]

则：

\[
H(p,i,j)
=A_{p,\pi_i}+(L_j-L_i)+A_{\pi_j,p}.
\]

因此任意 \(H(p,i,j)\) 可以在 \(O(1)\) 时间计算。

---

## 8. 仓库子集 Split 动态规划

### 8.1 状态定义

定义：

\[
F[M,j]
\]

其中：

- \(M\subseteq P\) 表示已经使用的仓库集合；
- \(j\in\{0,1,\ldots,n\}\) 表示巨路径前 \(j\) 个客户已经完成分配；
- \(F[M,j]\) 表示对应的最小代理成本。

初始状态：

\[
F[\varnothing,0]=0.
\]

其他状态初始化为正无穷。

### 8.2 转移

从状态 \(F[M,j]\) 出发，选择尚未使用的仓库 \(p\notin M\)，并将下一个非空连续片段：

\[
B_{j+1:t}
\]

分配给该仓库。

若片段满足当前模型规模阈值：

\[
Q_{bin}(j+1,t)\leq \bar Q,
\]

则执行转移：

\[
F[M\cup\{p\},t]
=
\min\left\{
F[M\cup\{p\},t],
F[M,j]+H(p,j+1,t)
\right\}.
\]

不需要单独设置“仓库跳过”转移。最终没有出现在仓库子集 \(M\) 中的仓库自动对应空分组。

### 8.3 终止条件

最终代理成本为：

\[
H^*(\bar Q)
=
\min_{M\subseteq P}F[M,n].
\]

如果所有 \(F[M,n]\) 均为正无穷，则当前阈值不可行。

### 8.4 父指针

对每次最优转移记录：

```python
parent[new_mask][t] = (
    old_mask,
    j,
    depot,
)
```

最终从取得最小值的 \((M,n)\) 状态回溯，恢复每个仓库对应的客户片段。

接口：

```python
def run_subset_split_dp(
    depots,
    giant_path,
    directed_cost,
    segment_stats,
    q_budget=None,
    keep_parent=False,
):
    ...
```

其中：

- `q_budget=None` 表示不限制模型规模；
- `keep_parent=False` 用于预算搜索；
- `keep_parent=True` 用于最终回溯。

---

## 9. 模型规模预算选择

### 9.1 无预算最优代理成本

先在不限制 \(Q_{bin}\) 的情况下运行一次子集 DP：

\[
H^*=\min_{M\subseteq P}F[M,n].
\]

### 9.2 代理成本容忍度

主实验固定：

\[
\varepsilon=0.01.
\]

最终划分必须满足：

\[
H^*(\bar Q)\leq(1+\varepsilon)H^*.
\]

### 9.3 最小预算

收集所有连续片段的二进制变量规模：

\[
\mathcal B=
\left\{
Q_{bin}(i,j)
\mid
1\leq i\leq j\leq n
\right\}.
\]

对去重后的 \(\mathcal B\) 升序排列，并二分搜索最小预算 \(\bar Q\)，使：

1. 当前预算下存在可行划分；
2. 当前最优代理成本满足 1% 容忍约束。

最终目标等价于：

\[
\min \bar Q
\]

满足：

\[
H^*(\bar Q)\leq1.01H^*.
\]

接口：

```python
def find_minimum_model_budget(
    depots,
    giant_path,
    directed_cost,
    segment_stats,
    epsilon=0.01,
):
    ...
```

预算确定后，使用该 \(\bar Q\) 再运行一次带父指针的 DP，生成最终分组。

---

## 10. 完整算法伪代码

```text
Algorithm Directed-Set-GTDS
Input:
    depots P
    customers C
    boundary candidate sets R
    truck distance d_tr
    drone distance d_dr
    drone speed s_dr
    epsilon = 0.01

Output:
    customer groups for all depots

1. Normalize candidate sets:
       R_bar[c] = R[c] if R[c] is nonempty else {c}
       R_bar[p] = {p}

2. Build directed set-cost matrix A on P union C.

3. Add virtual node q:
       A_tilde[q,c] = min_p A[p,c]
       A_tilde[c,q] = min_p A[c,p]
       A_tilde[c,c'] = A[c,c']

4. Solve ATSP on C union {q} with elkai.
   Remove q and obtain giant path pi.

5. Build prefix sums for:
       candidate-set sizes
       squared candidate-set sizes
       internal giant-path arc costs

6. For every segment [i,j], compute:
       Q_bin(i,j)
       H(p,i,j) for every depot p

7. Run subset Split DP without Q_bin budget and obtain H_star.

8. Binary-search the smallest Q_bar satisfying:
       H_star(Q_bar) <= (1 + epsilon) * H_star

9. Run subset Split DP with Q_bar and parent pointers.

10. Backtrack and construct groups.

11. Initialize all unused depots with empty customer lists.

12. Return groups and diagnostic information.
```

---

## 11. 建议的软件结构

新增文件：

```text
src/partition.py
```

建议包含：

```python
from dataclasses import dataclass


@dataclass
class SetGTDSResult:
    groups: dict
    giant_path: list
    surrogate_cost_unbounded: float
    surrogate_cost_final: float
    q_budget: int
    max_q_bin: int
    phase1_time: float


def normalize_candidate_sets(...):
    ...


def build_directed_set_cost(...):
    ...


def solve_depot_aware_giant_path(...):
    ...


def build_segment_statistics(...):
    ...


def run_subset_split_dp(...):
    ...


def find_minimum_model_budget(...):
    ...


def set_gtds_partition(...):
    ...
```

`set_gtds_partition()` 作为统一入口：

```python
def set_gtds_partition(
    depots,
    cities,
    candidate_sets,
    truck_distance,
    drone_distance,
    speed,
    epsilon=0.01,
    scale=1000,
):
    ...
```

---

## 12. 与当前 `master` 的集成方式

在 `MultiAgentFlyingSidekickTSP` 中增加：

```python
def set_gtds(self, convex_sets):
    result = set_gtds_partition(
        depots=list(self.depots),
        cities=list(self.cities),
        candidate_sets=convex_sets,
        truck_distance=self.distance["truck"],
        drone_distance=self.distance["drone"],
        speed=self.speed,
        epsilon=0.01,
        scale=1000,
    )
    self.groups = result.groups
    self.partition_result = result
```

在 `solve()` 中：

```python
raw_sets = self.get_boundary_convex_sets(self.theta[0])
convex_sets = normalize_candidate_sets(self.cities, raw_sets)
self.set_gtds(convex_sets)

for depot in self.depots:
    convex_set = [[depot]] + [
        convex_sets[city]
        for city in self.groups[depot]
    ]
    solution = self.single_solution(depot, convex_set)
    self.solution.append(self.convert(solution))
```

在 `solve_multiple_drones()` 中采用相同处理。

原有以下代码保留不变，用于基线实验：

```python
set_mst()
mst_partition()
```

---

## 13. 复杂度

设：

- 客户数为 \(n\)；
- 仓库数为 \(m\)；
- 每个起点平均有 \(b\) 个满足当前预算的可行片段终点；
- 候选预算数量为 \(|\mathcal B|\leq n(n+1)/2\)。

一次固定预算的子集 DP 时间复杂度为：

\[
O(2^m\,m\,n\,b).
\]

空间复杂度为：

\[
O(2^m n).
\]

二分搜索预算后的总 DP 复杂度为：

\[
O\left(
\log|\mathcal B|\cdot2^m\,m\,n\,b
\right).
\]

本方法的主要实验范围固定为：

\[
m\leq10.
\]

对于 \(m=15\) 的扩展实验，需要使用以下至少一种优化：

- Numba 编译 DP 循环；
- C++ 扩展；
- 预先压缩可行片段集合；
- 使用滚动数组和连续内存；
- 对不可达到状态进行稀疏存储。

---

## 14. 单元测试

完整实验前必须通过以下测试。

### 14.1 候选集合测试

```text
1. 所有正规化后的客户集合均非空。
2. 原集合为空时，正规化结果严格等于 [city]。
3. Phase 1 和 Phase 2 使用同一份正规化集合。
```

### 14.2 有向代价测试

```text
1. A[u,v] 与 A[v,u] 分别保存。
2. 允许 A[u,v] != A[v,u]。
3. 改变 customers 输入顺序后，按节点编号对齐的矩阵不变。
4. 不创建 nx.Graph，不调用 MST 相关函数。
```

### 14.3 巨路径测试

```text
1. 删除虚拟节点后，巨路径长度等于客户数。
2. 每个客户恰好出现一次。
3. 路径中不包含仓库和虚拟节点。
4. 整数缩放后所有非对角元素均大于等于 1。
```

### 14.4 模型规模公式测试

随机生成若干客户分组，构造对应 `set_tsp()` 模型，验证：

```python
estimated_q_bin == model.NumBinVars
estimated_q_var == model.NumVars
estimated_q_con == model.NumConstrs
```

比较必须在 Gurobi presolve 前完成。

### 14.5 DP 正确性测试

对 \(n\leq8\)、\(m\leq3\) 的小实例：

```text
1. 枚举全部连续切分和仓库排列。
2. 计算枚举最优代理成本。
3. 验证 DP 最优值与枚举结果一致。
4. 验证父指针恢复的分组与最优值一致。
```

### 14.6 最终分组测试

```text
1. 每个客户恰好分配一次。
2. 不存在重复客户。
3. 不存在遗漏客户。
4. 每个仓库至多对应一个巨路径连续片段。
5. 每个非空分组满足 Q_bin <= q_budget。
6. 最终代理成本满足 H_final <= 1.01 * H_star。
7. 未使用仓库对应空列表。
```

---

## 15. 实验设计

### 15.1 对照方法

主实验使用：

1. `SMST-original + STSP`：公开代码原始第一阶段；
2. `SNN + STSP`：最近仓库类型的速度参考；
3. `Set-GTDS-noBudget + STSP`：相同有向代价与 Split、不使用模型规模预算；
4. `Directed Set-GTDS + STSP`：本文完整方法。

所有方法使用完全相同的 Phase 2、Phase 3、实例和随机种子，并按实例采用确定性的
循环平衡执行顺序。两个 GTDS 主方法固定使用全部可用仓库和论文 `1/speed` 代价。

### 15.2 数据规模

按照论文场景组织：

| 路网 | 仓库数 | 客户数 |
|---|---:|---:|
| Manhattan | 5 | 50、100、150 |
| NYC 11K proxy | 10 | 50、100、150 |

每个设置运行相同的 100 个实例。

NYC 11K proxy 只用于扩展规模验证，不表述为论文 Boston 数据；获得真实 Boston
图后必须使用新的数据集标识和图哈希另起实验目录。

### 15.3 主参数

```text
epsilon = 0.01
integer scale = 1000
ATSP solver = elkai
```

主结果固定使用上述参数，不根据单个测试场景单独调整。

附录报告：

\[
\varepsilon\in\{0,0.005,0.01,0.02,0.05\}.
\]

### 15.4 必须记录的指标

#### 最终效果

- 最终 `self.cost`；
- 完成实例数量；
- 超时实例数量；
- 不可行实例数量。

#### 阶段时间

- 第一阶段总时间；
- 有向集合代价构建时间；
- 巨路径求解时间；
- 预算搜索和 Split DP 时间；
- Phase 2 建模时间；
- Phase 2 求解时间；
- Phase 3 时间；
- 总运行时间。

#### 分组结构

- 每组客户数量；
- 每组候选点总量 \(W_p\)；
- 每组 \(Q_{bin}\)、\(Q_{var}\)、\(Q_{con}\)；
- 最大、平均和中位 \(Q_{bin}\)；
- \(Q_{bin}\) 的变异系数；
- 最大单组 Phase 2 时间。

#### Gurobi 指标

- `NumBinVars`；
- `NumVars`；
- `NumConstrs`；
- 分支定界节点数；
- `MIPGap`；
- `SolCount`；
- 单个仓库是否达到时间限制。

### 15.5 统计分析

所有方法在同一实例上进行成对比较。

报告：

- 中位数相对差异；
- 均值相对差异；
- bootstrap 95% 置信区间；
- Wilcoxon signed-rank test；
- 完成率；
- PAR-2 时间。

相对差异定义为：

\[
\Delta(x)=
\frac{x_{Set\text{-}GTDS}-x_{baseline}}
{x_{baseline}}.
\]

### 15.6 主判定标准

最终成本满足非劣性要求：

\[
\text{成本相对差异的 95\% CI 上界}\leq1\%.
\]

总运行时间应优于基线：

\[
\text{总时间相对差异的 95\% CI 上界}<0.
\]

预期工程目标为：

```text
总运行时间中位数降低至少 25%。
```

### 15.7 机制验证

必须绘制并分析：

\[
\max Q_{bin}
\quad\text{与}\quad
\text{最大单组 Set-TSP 时间}
\]

以及：

\[
\sum_p Q_{bin,p}
\quad\text{与}\quad
\text{Phase 2 总时间}.
\]

使用 Spearman 相关系数和散点图验证模型规模指标是否能够解释 Phase 2 时间变化。

### 15.8 消融实验

主表包含一个机制消融：

- `Set-GTDS-noBudget`：使用相同有向代价、相同巨路径和相同仓库子集 Split，但不限制 \(Q_{bin}\)；
- `Directed Set-GTDS`：使用完整的 1% 代理成本约束和最小模型规模预算。

该消融用于区分以下两类收益：

- 巨路径和仓库联合 Split 带来的收益；
- 下游模型规模约束带来的收益。

另外单独运行两组口径敏感性，不并入主表：

- `directed_set_gtds` 对 `gtds_free_eps01`：识别“全部仓库非空”约束的影响；
- `directed_set_gtds` 对 `gtds_sqrt2`：识别论文 `1/speed` 与公开代码
  `sqrt(2)/speed` 代价口径的影响。

---

## 16. 论文中的方法表述

### 16.1 方法定位

论文将本方法描述为：

> 面向有向路网 MA-FSTSP 的下游模型规模感知分解方法。该方法使用有向集合代价构造仓库感知的客户开放巨路径，并通过仓库子集 Split 动态规划，在有限路线代理成本损失下最小化最大 Set-TSP 子问题规模。

### 16.2 贡献表述

可写为以下三项：

1. 提出一种面向有向路网 MA-FSTSP 的仓库感知巨路径分解方法，在不对称集合代价下联合决定客户连续片段和独占仓库分配。
2. 从下游 Set-TSP 数学模型中推导分组对应的预求解二进制变量数量，并将其作为第一阶段的复杂度资源；在路线代理成本最多增加 \(\varepsilon\) 的条件下，最小化最大子问题模型规模。
3. 给出固定巨路径下的精确仓库子集动态规划和预算搜索方法，并通过阶段级时间、MILP 规模、最终成本和超时率验证分解效果。

### 16.3 理论命题

论文至少给出以下命题。

#### 命题 1：模型规模公式

对任意客户片段 \(B_{i:j}\)，当前 Set-TSP 模型在 presolve 前创建：

\[
Q_{bin}(i,j)
=N_{i:j}^2+S^{(2)}_{i:j}+W_{i:j}^2
\]

个二进制变量。

#### 命题 2：固定预算下的 DP 最优性

给定巨路径 \(\pi\) 和模型规模预算 \(\bar Q\)，仓库子集 DP 能够在以下约束下获得最小代理成本：

- 每个客户恰好被覆盖一次；
- 每个客户组是巨路径的连续片段；
- 每个仓库至多使用一次；
- 每个非空客户组满足 \(Q_{bin}\leq\bar Q\)。

#### 命题 3：最小预算性质

预算搜索得到的 \(\bar Q\) 是满足：

\[
H^*(\bar Q)\leq(1+\varepsilon)H^*
\]

的最小候选预算。

#### 命题 4：复杂度

预算搜索和仓库子集 DP 的总复杂度为：

\[
O\left(
\log|\mathcal B|\cdot2^m mnb
\right),
\]

空间复杂度为：

\[
O(2^m n).
\]

### 16.4 适用范围与限制

论文明确说明：

- ATSP 巨路径由启发式求解器生成；
- DP 的最优性仅针对给定巨路径和给定模型规模预算；
- 路线代理成本不是原始 MA-FSTSP 目标函数的上界或下界；
- 不对原始 MA-FSTSP 给出近似比；
- 方法主要面向少量仓库、多客户场景，主实验范围为 \(m\leq10\)；
- Phase 2 会重新优化客户顺序，巨路径用于第一阶段客户嵌入和连续分组。

---

## 17. 实施顺序

### 阶段 1：独立实现与验证

1. 新建 `src/partition.py`；
2. 实现候选集合正规化；
3. 实现有向集合代价矩阵；
4. 实现仓库感知巨路径；
5. 实现片段模型规模公式；
6. 实现无预算仓库子集 DP；
7. 实现模型规模预算搜索；
8. 完成小规模枚举对照测试。

### 阶段 2：接入主算法

1. 在 `fstsp.py` 增加 `set_gtds()`；
2. 接入 `solve()`；
3. 接入 `solve_multiple_drones()`；
4. 确认 Phase 2 和 Phase 3 未发生行为变化；
5. 增加阶段计时和模型规模日志。

### 阶段 3：试运行

先运行：

```text
Manhattan 1K：5 个仓库，50 个客户，10 个实例
```

当前仓库没有论文 Boston 路网；NYC 11K 仅作为单独标注的 proxy 数据，不能写成
Boston 复现实验。

检查：

- 正确性断言；
- Phase 1 耗时；
- DP 状态数量；
- `Q_bin` 估计与 Gurobi 实际值；
- Phase 2 时间变化；
- 最终成本变化；
- 是否出现超时或内存异常。

### 阶段 4：完整实验

试运行通过后，再执行：

```text
2 个路网
3 个客户规模
每个设置 100 个实例
4 个主比较方法
1 个机制消融
```

所有原始结果按实例保存，禁止只保留汇总均值。

---

## 18. 最终执行规范

最终采用以下固定配置：

```text
方法名称：Directed Set-GTDS
候选集合：boundary candidate sets，空集退化为客户自身
集合代价：当前代码对应的有向集合代理代价
巨路径：仓库感知开放 ATSP
ATSP 求解器：elkai
整数缩放：1000
下游复杂度指标：Set-TSP 预求解二进制变量数 Q_bin
代理成本容忍度：epsilon = 0.01
Split 方法：仓库子集动态规划
主实验活跃仓库策略：all（客户数不少于仓库数时全部非空）
主实验无人机代价系数：1 / speed
主实验仓库范围：m <= 10
Pilot 实例总时限：600 秒
正式实验实例总时限：7200 秒
Phase 2：保持不变
Phase 3：保持不变
原 MST：保留为 SMST-original 基线，不修改
```
