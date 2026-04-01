# 研究方向：基于双层优化的物理信息宽度学习系统（BO-PIBLS）

## 一、核心思想

### 1.1 一句话概括

**将 BLS 的伪逆求解嵌入可微分计算图，形成"内层伪逆（解析最优）+ 外层梯度下降（特征学习）"的双层优化框架，用于求解偏微分方程。**

### 1.2 数学表述

$$\min_\theta \quad \mathcal{L}_{\text{PDE}}\Big(H(\theta) \cdot \beta^*(\theta)\Big)$$

$$\text{s.t.} \quad \beta^*(\theta) = \arg\min_\beta \|A(\theta)\beta - b\|^2 = \big(A^\top A + \lambda I\big)^{-1} A^\top b$$

其中：
- $\theta = \{W_{map}, B_{map}, W_{enh}, B_{enh}\}$：BLS 内部权重（可学习）
- $\beta$：输出层权重（每轮通过伪逆解析求解）
- $H(\theta)$：BLS 特征矩阵，是 $\theta$ 的可微函数
- $A(\theta)$：PDE 系统矩阵，包含 $H$ 及其微分算子（$\nabla^2 H$ 等）

### 1.3 核心创新点

BLS 的伪逆求解不是被梯度下降替代，而是**作为可微分算子嵌入计算图**：

```
前向传播：
  θ → H(θ) → 组装系统 A(θ)β=b → 伪逆求 β*(θ) → 计算 PDE 残差 L

反向传播：
  ∂L/∂θ = ∂L/∂β* · ∂β*/∂θ    ← 梯度穿过伪逆（隐式微分）
```

---

## 二、与现有方法的本质区别

### 2.1 对比表

| 特性 | PIBLS（原始） | GD-PIBLS | PINN | **BO-PIBLS（本方案）** |
|------|-------------|----------|------|---------------------|
| β 更新方式 | 伪逆（一次） | 梯度下降 | 梯度下降 | **伪逆（每轮解析解）** |
| θ 更新方式 | 随机固定 | 梯度下降 | 梯度下降 | **梯度下降（穿过伪逆）** |
| β 是否最优 | 是（给定 θ） | 否（迭代中间态） | 否 | **是（每轮全局最优）** |
| 优化空间 | 无优化 | (θ,β) 联合非凸 | (θ,β) 联合非凸 | **仅 θ 空间** |
| BLS 特性 | 完整保留 | **丧失** | 无关 | **完整保留** |
| 网络深度 | 2 层 | 2 层 | 4-8 层 | **2 层** |

### 2.2 为什么不是"浅层 PINN"

BO-PIBLS 与任何 PINN 变体的**根本区别**在于：

1. **PINN**：$\min_{\theta, \beta} \mathcal{L}(\theta, \beta)$ — 联合优化，梯度下降同时更新所有参数
2. **BO-PIBLS**：$\min_\theta \mathcal{L}(\theta, \beta^*(\theta))$ — β 被消去，只优化 θ

这不是工程trick，而是**优化问题的结构性变化**：
- 联合优化的景观是高维非凸的（鞍点、局部极小值密集）
- 消去 β 后，景观维度减半，且内层凸性被利用，外层更光滑

### 2.3 为什么不是简单的 HybridPIBLS

之前 HybridPIBLS 的做法：固定 β 算 SPSA 梯度 → 更新 θ → 重新伪逆。

**关键差别**：SPSA 估计的是 $\partial \mathcal{L}/\partial \theta \big|_{\beta \text{ fixed}}$，而不是 $d\mathcal{L}/d\theta = \partial \mathcal{L}/\partial \theta + \partial \mathcal{L}/\partial \beta^* \cdot \partial \beta^*/\partial \theta$。

前者忽略了"θ 变化导致最优 β 变化"这一关键项。BO-PIBLS 通过可微分伪逆获得**精确的全微分**。

---

## 三、技术路线

### 3.1 算法流程

```
Algorithm: BO-PIBLS Training

Input:  PDE 定义, 配点集 X, 初始节点数 N
Output: 训练好的 BLS 参数 (θ*, β*)

1. Xavier 初始化 θ = {W_map, B_map, W_enh, B_enh}
2. For epoch = 1, ..., T:
   a. [前向-BLS特征] H = BLS_features(X; θ)           — 可微
   b. [前向-微分算子] H_xx, H_yy = ∂²H/∂x², ∂²H/∂y²  — autograd
   c. [组装线性系统] A·β = b（将 PDE 离散化为线性系统）
   d. [内层求解-伪逆] β* = solve(A^T·A + λI, A^T·b)    — torch.linalg.solve（可微）
   e. [计算PDE残差] u = H·β*, L_pde = ||PDE(u)||²
   f. [外层更新-梯度] θ ← θ - η·∂L/∂θ                  — 梯度穿过 solve
3. Return θ*, β*
```

### 3.2 关键技术点

#### (1) 可微分伪逆

PyTorch 的 `torch.linalg.solve` 支持反向传播。对于 $\beta^* = (A^\top A + \lambda I)^{-1} A^\top b$，其梯度为：

$$\frac{\partial \beta^*}{\partial \theta} = -(A^\top A + \lambda I)^{-1} \left[\frac{\partial A^\top A}{\partial \theta} \beta^* - \frac{\partial A^\top b}{\partial \theta}\right]$$

PyTorch 自动处理这一计算，无需手动推导。

#### (2) PDE 系统矩阵组装

对于线性 PDE（如 $-\Delta u = f$）：

$$A = \begin{bmatrix} H_{\text{int}} \\ \alpha \cdot H_{\text{bc}} \end{bmatrix}, \quad b = \begin{bmatrix} -\Delta_H \cdot \mathbf{1}_{\text{placeholder}} + f_{\text{int}} \\ u_{\text{bc}} \end{bmatrix}$$

其中 $H$ 和 $\Delta_H$（H 的拉普拉斯）都是 θ 的可微函数。

对于非线性 PDE（如 $-\Delta u + u^3 = f$）：
- 内层仍做线性化伪逆（Newton 步）
- 外层梯度穿过 Newton 步传到 θ

#### (3) 数值稳定性

| 风险 | 缓解措施 |
|------|---------|
| $A^\top A$ 接近奇异 | Tikhonov 正则化 $\lambda I$，$\lambda$ 可自适应 |
| 梯度爆炸 | 梯度裁剪 + 条件数监控 |
| 二阶导数精度 | 使用 `torch.autograd.grad` + `create_graph=True` |

#### (4) 与增量学习结合（可选扩展）

```
1. 初始 N_0 个节点训练至收敛
2. 分析残差频谱 → 确定新节点频率
3. 扩展 θ（新节点参数加入优化器）
4. 继续训练（旧参数 warm-start，新参数 Xavier 初始化）
```

每次增量后，伪逆自动在更大特征空间中求解最优 β——无需重训旧节点。

---

## 四、理论分析方向

### 4.1 收敛性分析

**命题（非正式）：** 设 $\mathcal{L}(\theta) = \mathcal{L}_{\text{PDE}}(H(\theta) \cdot \beta^*(\theta))$ 是外层目标函数，若：
1. $H(\theta)$ 关于 θ 是 Lipschitz 连续的
2. $A^\top A + \lambda I$ 的最小特征值有下界
3. 学习率 η 满足 $\eta < 2 / L$（L 为 $\nabla_\theta \mathcal{L}$ 的 Lipschitz 常数）

则梯度下降收敛到 $\nabla_\theta \mathcal{L} = 0$ 的驻点。

**与 PINN 收敛性的区别**：PINN 的收敛分析需要处理 (θ,β) 联合非凸景观；BO-PIBLS 只需分析 θ 空间的景观，且内层凸性提供了更强的结构。

### 4.2 逼近阶分析

基于随机特征（Random Feature）理论：
- $N$ 个随机特征的逼近误差为 $O(1/\sqrt{N})$（Rahimi & Recht, 2007）
- BO-PIBLS 的特征是学习到的（非随机），逼近阶应该更优
- **开放问题**：能否证明学习特征的逼近阶为 $O(1/N)$ 或更优？

### 4.3 优化景观分析

- **猜想**：消去 β 后的外层目标函数 $\mathcal{L}(\theta)$ 的 Hessian 条件数低于 PINN 的联合 Hessian
- **验证方法**：在小规模问题上数值计算两者的 Hessian 特征谱并比较

---

## 五、实验计划

### 5.1 验证性实验（第一阶段）

| 编号 | 问题 | 目的 |
|------|------|------|
| E1 | 2D Poisson $-\Delta u = f$（低频/高频） | 基准验证，对比 PIBLS/PINN |
| E2 | 非线性 $-\Delta u + u^3 = f$ | 验证 Newton + 可微伪逆 |
| E3 | 强非线性 $-\Delta u + \sin(u) = f$ | 此前 PIBLS 弱项，验证改进 |
| E4 | 多尺度 $-\Delta u + 100 u = f$（高反应项） | 验证刚性问题处理 |

### 5.2 标准 Benchmark（第二阶段）

| 编号 | 问题 | 意义 |
|------|------|------|
| B1 | 1D Burgers 方程 | PINN 经典 benchmark |
| B2 | 2D 对流扩散方程 | 多尺度动力学 |
| B3 | 2D Helmholtz 方程 | 高频振荡解 |
| B4 | Allen-Cahn 方程 | 刚性非线性 |

### 5.3 对比方法

| 方法 | 说明 |
|------|------|
| PIBLS（原始） | 随机特征 + 一次伪逆 |
| NL-PIBLS | Newton-伪逆（固定特征） |
| PINN (vanilla) | Adam + L-BFGS，4 层 128 宽 |
| PINN (改进) | RAR / NTK / Adaptive λ |
| FEM/FDM | 精度上限参考 |
| **BO-PIBLS（本方法）** | 双层优化 |

### 5.4 评价指标

- **精度**：$L^2$ 相对误差 $\|u - u_{\text{exact}}\|_2 / \|u_{\text{exact}}\|_2$
- **效率**：训练时间（GPU 秒）、收敛 epoch 数
- **稳定性**：5 次随机种子的均值 ± 标准差
- **可扩展性**：节点数 N vs 精度曲线
- **优化景观**：Hessian 条件数（小规模问题）

---

## 六、V1.0 实验结果（2026-04-01）

### 6.1 实现概述

**代码**：`bo_pibls.py`（算法实现）、`test_bo_pibls.py`（测试脚本）

**实际实现的架构**：

```
BOPIBLS 类
├── 映射层: n_map=50 个 Fourier 特征  φ_j(x) = sin(ω_j · x + b_j)
│   ├── ω_j 可学习（多尺度初始化：低频/中频/高频各 1/3）
│   └── b_j 可学习
├── 增强层: n_enh=50 个 tanh 节点  E_k = tanh(M · W_enh + b_enh)
│   ├── W_enh, b_enh 可学习
│   └── 输入为映射层输出 M
├── 输出层: u = [M|E] · β
│   └── β 每轮通过可微伪逆求解（非梯度更新）
└── 拉普拉斯: 全部解析计算（不用 autograd 逐列求导）
    ├── ΔM_j = -||ω_j||² · sin(ω_j·x + b_j)
    └── ΔE_k = Σ_d [tanh''(s)·(∂s/∂x_d)² + tanh'(s)·∂²s/∂x_d²]
```

**训练流程**：

```
Phase 1: Adam (300 epochs, lr=5e-3, cosine annealing)
  每步: θ → H(θ), ΔH(θ) → 组装 A(θ)β=b → torch.linalg.solve → L_PDE → backward → 更新 θ

Phase 2: L-BFGS (100 epochs, lr=0.5, strong_wolfe line search)
  同上流程，但用 L-BFGS 精调 θ
```

**非线性 PDE 处理**：Picard 迭代（外层循环 3 次），每轮内部做双层优化：
1. 用当前 $\beta_{\text{prev}}$ 计算 $u_{\text{prev}} = H \cdot \beta_{\text{prev}}$
2. 线性化：$[-\Delta H + \text{diag}(g'(u_{\text{prev}})) \cdot H] \cdot \beta = f - g(u_{\text{prev}}) + g'(u_{\text{prev}}) \cdot u_{\text{prev}}$
3. 可微伪逆求解 → 计算真实非线性残差 → 梯度反传更新 θ

### 6.2 实验配置

| 参数 | BO-PIBLS | PIBLS (Fixed-BLS) | PINN |
|------|----------|-------------------|------|
| 特征/节点数 | 100 (50 map + 50 enh) | 200 (100 map + 100 enh) | 4L-64W |
| 内部权重 | **可学习** | 随机固定 | 可学习 |
| 输出权重 | **伪逆**（每轮） | 伪逆（一次） | 梯度下降 |
| 激活函数 | Fourier(sin) + tanh | tanh | tanh |
| 优化器 | Adam(300) + L-BFGS(100) | 无 | Adam(3000) + L-BFGS(500) |
| 配点 (内部/边界) | 900 / 200 | 900 / 200 | 900 / 200 |
| 正则化 | λ=1e-6 | λ=1e-8 | 无 |
| 精度 | float64 | float64 | float64 |
| 总参数量 | ~300 (θ) + 100 (β) | 100 (β only) | ~4500 |

### 6.3 实验结果

#### 汇总表

| 问题 | BO-PIBLS RMSE | 时间 | PIBLS RMSE | 时间 | PINN RMSE | 时间 | vs PIBLS | vs PINN |
|------|--------------|------|-----------|------|----------|------|----------|---------|
| **P1** 低频 Poisson | **7.97e-6** | 5.0s | 1.30e-5 | 0.05s | 9.42e-5 | 74s | **+38.8%** | **+91.5%** |
| **P2** 高频 Poisson | **2.77e-4** | 24.1s | 5.55e-2 | 0.03s | 1.37e-3 | 151s | **+99.5%** | **+79.8%** |
| **P3** 非线性 u³ | 3.90e-5 | 12.6s | **2.45e-6** | 0.17s | 6.30e-5 | 70s | -1488% | **+38.2%** |
| **P4** 非线性 sin(u) | 2.15e-5 | 7.0s | **2.46e-6** | 0.18s | 4.91e-4 | 114s | -774% | **+95.6%** |

#### P1 训练过程

```
Adam [   0/300]  loss=1.0652e+02  L_pde=1.0623e+02  L_bc=2.8966e-03
Adam [  50/300]  loss=5.2874e-02  L_pde=5.2760e-02  L_bc=1.1373e-05
Adam [ 150/300]  loss=2.1541e-03  L_pde=2.1522e-03  L_bc=1.9230e-07
Adam [ 299/300]  loss=1.3684e-04  L_pde=1.3679e-04  L_bc=5.3150e-08
L-BFGS final:    loss=3.0399e-07  L_pde=3.0387e-07  L_bc=1.2128e-10
→ RMSE = 7.97e-6
```

L-BFGS 将 loss 从 1.37e-4 降到 3.04e-7（降低 450 倍），体现了二阶优化在精调阶段的价值。

#### P2 训练过程

```
Adam [   0/300]  loss=1.6504e+03  L_pde=1.5706e+03  L_bc=7.9773e-01
Adam [ 150/300]  loss=2.1697e-01  L_pde=2.1637e-01  L_bc=5.9884e-05
Adam [ 299/300]  loss=6.5831e-02  L_pde=6.5785e-02  L_bc=4.5937e-05
L-BFGS final:    loss=3.2795e-04  L_pde=3.2768e-04  L_bc=2.7213e-07
→ RMSE = 2.77e-4
```

高频问题展现了 Fourier 可学习频率的巨大优势：PIBLS 随机 tanh 的 5.55e-2 → BO-PIBLS 的 2.77e-4，提升 **200 倍**。

#### P3/P4 非线性训练过程（问题）

```
Picard iteration 1/3:
  Adam [   0/ 50]  loss=5.9745e-08  → 收敛良好

Picard iteration 2/3:
  Adam [   0/ 50]  loss=1.8696e+03  → 爆炸！

Picard iteration 3/3:
  Adam [   0/ 50]  loss=8.4668e+02  → 继续不稳定
```

**Picard 迭代不稳定**：第 1 轮 Adam 将 θ 优化到适应线性化的特征空间，但第 2 轮线性化点变了，旧特征空间不适合新线性化的 PDE → loss 爆涨。得益于 best_params 回滚机制，最终结果仍然可用。

### 6.4 关键发现

#### 发现 1：线性 PDE 上 BO-PIBLS 全面碾压

BO-PIBLS 用 **100 个特征**（PIBLS 用 200 个）在线性问题上取得压倒性优势：

- vs PIBLS（Node-for-node 效率）：+38.8%（P1）、+99.5%（P2）
- vs PINN：+91.5%（P1）、+79.8%（P2）

核心原因：
- **Fourier 可学习频率** → 梯度下降找到最匹配 PDE 解的频率组合
- **伪逆每轮最优** → 给定特征空间下系数始终最优
- **双层结构** → 外层只优化特征（θ 空间），景观更光滑

#### 发现 2：非线性 PDE 上 BO-PIBLS 优于 PINN 但不如 NL-PIBLS

| 方法 | P3 (u³) | P4 (sin(u)) |
|------|---------|-------------|
| **NL-PIBLS** | **2.45e-6** | **2.46e-6** |
| BO-PIBLS | 3.90e-5 (差 16×) | 2.15e-5 (差 8.7×) |
| PINN | 6.30e-5 | 4.91e-4 |

原因分析：

- NL-PIBLS 使用 200 个 tanh 节点 + Newton-伪逆（固定特征，直接迭代求解）→ 特征矩阵完全固定，Newton 在其上收敛极快
- BO-PIBLS 使用 100 个节点 + Picard 迭代（每轮重新优化 θ）→ Picard 第 2 轮特征空间剧变导致不稳定

**根本矛盾**：双层优化在线性 PDE 上"特征学习 + 伪逆求解"互相促进；在非线性 PDE 上，Picard 线性化点在重新优化 θ 后不再有效，导致学习不稳定。

#### 发现 3：Fourier 特征的频率学习效果显著

| 初始频率分布 | 学习后 | 改善 |
|-------------|--------|------|
| 低频 σ=1.0, 中频 σ=3.0, 高频 σ=6.0 | 梯度自适应调整 | P2 提升 200× vs 随机 tanh |

Fourier 特征 $\phi(x) = \sin(\omega \cdot x + b)$ 的可学习频率 ω 是 BO-PIBLS 最有价值的创新点——从数据/PDE 自动发现解的频率成分。

#### 发现 4：速度分析

| 对比对 | 速度比 | BO-PIBLS 时间 | 对手时间 |
|--------|--------|-------------|---------|
| BO vs PINN (P1) | **14.8×** | 5.0s | 74s |
| BO vs PINN (P2) | **6.3×** | 24.1s | 151s |
| BO vs PINN (P3) | **5.6×** | 12.6s | 70s |
| BO vs PINN (P4) | **16.3×** | 7.0s | 114s |
| BO vs PIBLS (P1) | 0.01× | 5.0s | 0.05s |

BO-PIBLS 比 PINN 快 **5.6-16.3 倍**（浅层架构 + 更少参数），但比 PIBLS 慢 100 倍（增加了梯度迭代）。

### 6.5 可行性评估

#### 达到一区的条件 vs 当前状态

| 条件 | 当前状态 | 评估 |
|------|---------|------|
| 线性 PDE 全面优于 baseline | ✅ 碾压 PIBLS 和 PINN | **达标** |
| 非线性 PDE 优于 baseline | ⚠️ 优于 PINN，但弱于 NL-PIBLS | **需改进** |
| 方法一致性地有效 | ⚠️ 线性强、非线性弱 | **需改进** |
| Benchmark 充分 | ❌ 只有 4 个 2D 问题 | **差距大** |
| 与 SOTA 对比 | ❌ 只比了 vanilla PINN | **差距大** |
| 理论/景观分析 | ❌ 未做 | **差距大** |

#### 核心优势（可投稿的叙事基础）

1. **"伪逆作为可微分算子"是原创的优化结构** — 不同于 PINN 的联合梯度下降，不同于 PIBLS 的一次性伪逆
2. **可学习 Fourier 频率 + 伪逆系数求解** — 频率-系数解耦优化有物理直觉
3. **线性 PDE 结果极强** — 100 节点 BO-PIBLS 碾压 200 节点 PIBLS 和 4500 参数 PINN
4. **速度优势显著** — 比 PINN 快 6-16 倍

#### 核心弱点（投稿前必须解决）

1. **非线性 PDE 的 Picard 不稳定** — 最关键的问题，需要更稳定的非线性处理策略
2. **节点数不公平** — BO-PIBLS 用 100 节点 vs PIBLS 200 节点，需要同等条件对比
3. **Benchmark 不足** — 需要 Burgers、Helmholtz、Allen-Cahn 等经典问题
4. **与 PINN+L-BFGS 的公平对比** — PINN 目前只用了 3000+500 epochs，加强后差距可能缩小

### 6.6 已知 Bug 与修复记录

| Bug | 原因 | 修复 |
|-----|------|------|
| `_compute_laplacian_H` 每步 100×2 次 autograd 导致训练极慢（120s/150epochs） | 逐列 autograd 调用 | 改为全解析公式计算拉普拉斯 |
| `torch.no_grad()` 下调用 `autograd.grad` 报错 | Picard 提取 beta 时在 no_grad 上下文中 | 去掉 no_grad 包装 |

### 6.7 下一步改进方向（优先级排序）

| 优先级 | 改进 | 预期效果 |
|--------|------|---------|
| 🔴 P0 | 修复非线性 Picard 不稳定：改用 Newton 线性化 + 双层优化内层嵌套 | 非线性精度提升 10-100× |
| 🟡 P1 | 增加 Benchmark：Burgers、Helmholtz、对流扩散、Allen-Cahn | 验证泛化能力 |
| 🟡 P1 | 公平对比：BO-PIBLS 200 节点 vs PIBLS 200 节点 vs PINN 调参至最优 | 消除审稿人质疑 |
| 🟢 P2 | Hessian 条件数数值实验 | 提供优化景观优势的实证 |
| 🟢 P2 | 与 PINN 改进变体对比（RAR、NTK-PINN） | 增强论文说服力 |
| 🔵 P3 | 收敛性理论分析（至少半形式化论证） | 从"工程改进"升级为"方法论贡献" |

---

