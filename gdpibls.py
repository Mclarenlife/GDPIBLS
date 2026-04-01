"""
GD-PIBLS: Gradient-Descent Physics-Informed Broad Learning System

============================================================================
核心创新：端到端可微分的浅层宽网络求解 PDE
============================================================================

现有方法的局限：
- 原始 PIBLS：内部权重随机固定，仅用伪逆求解输出权重 → 特征表达力受限于随机投影
- Deep PINN：端到端梯度优化所有参数 → 深层架构导致梯度病态（条件数 ∝ L²）
- NL-PIBLS：Newton-伪逆迭代 → 内部权重仍然固定，精度天花板来自随机特征

GD-PIBLS 的设计原理：
1. 保留 BLS 的浅层拓扑（仅 2 层）→ 梯度条件数 O(1)，无深层病态
2. 所有参数可训练（W_map, B_map, W_enhance, B_enhance, beta）→ 特征空间自适应 PDE
3. 用 PyTorch autograd 计算 PDE 导数 → 精确梯度，无需手推链式法则
4. 伪逆热启动 → 比随机初始化快 10-100 倍收敛
5. 支持 L-BFGS 二阶优化 → 利用浅层架构的良好 Hessian

架构图：
    Input (x,y)
        │
        ▼
    ┌─────────────────────────┐
    │   Mapping Layer (N1)     │  Z = [x,y] @ W_map + B_map
    │   Multi-Activation       │  M = [identity | tanh | ReLU | sine](Z)
    └─────────────────────────┘
        │
        ▼
    ┌─────────────────────────┐
    │   Enhancement Layer (N2) │  E = tanh(M @ W_enhance + B_enhance)
    └─────────────────────────┘
        │
        ▼
    ┌─────────────────────────┐
    │   Feature Concat [M|E]   │  H ∈ R^{N×(N1+N2)}
    └─────────────────────────┘
        │
        ▼
    ┌─────────────────────────┐
    │   Output: u = H @ beta   │  beta ∈ R^{(N1+N2)×1}
    └─────────────────────────┘

PDE 约束通过 torch.autograd.grad 在 u 上直接计算：
    u_x  = ∂u/∂x,   u_y  = ∂u/∂y
    u_xx = ∂²u/∂x²,  u_yy = ∂²u/∂y²
    R = PDE_residual(u, u_x, u_y, u_xx, u_yy, x, y) = 0

损失函数:
    L = L_pde + λ_bc · L_bc
    L_pde = (1/N_pde) Σ R_i²
    L_bc  = (1/N_bc) Σ (u_bc - g)²

训练流程:
    Phase 1: 伪逆热启动（numpy 侧，固定内部权重求最优 beta）
    Phase 2: Adam 预训练（所有参数，快速下降）
    Phase 3: L-BFGS 精细化（所有参数，二阶收敛）
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import pinv


class GDPIBLS(nn.Module):
    """Gradient-Descent Physics-Informed Broad Learning System

    Parameters
    ----------
    N1 : int
        映射层节点数
    N2 : int
        增强层节点数
    multi_activation : bool
        是否使用多激活集成 (identity/tanh/ReLU/sine)。
        注意：对二阶 PDE，identity/ReLU 的二阶导为 0，
        仅 tanh/sine 节点贡献 u_xx/u_yy。
        设为 True 时，建议增大 N1 以补偿。
    lambda_bc : float
        边界条件损失权重
    """

    def __init__(self, N1, N2, multi_activation=False, lambda_bc=10.0):
        super().__init__()
        self.N1 = N1
        self.N2 = N2
        self.multi_activation = multi_activation
        self.lambda_bc = lambda_bc

        # 映射层参数: Z = [x, y] @ W_map + B_map
        self.W_map = nn.Parameter(torch.empty(2, N1))
        self.B_map = nn.Parameter(torch.empty(N1))

        # 增强层参数: E = tanh(M @ W_enhance + B_enhance)
        self.W_enhance = nn.Parameter(torch.empty(N1, N2))
        self.B_enhance = nn.Parameter(torch.empty(N2))

        # 输出权重: u = H @ beta
        self.beta = nn.Parameter(torch.empty(N1 + N2, 1))

        # 多激活分组
        if multi_activation:
            indices = np.array_split(np.arange(N1), 4)
            self._g0 = list(indices[0])  # identity
            self._g1 = list(indices[1])  # tanh
            self._g2 = list(indices[2])  # ReLU
            self._g3 = list(indices[3])  # sine
            self._act_groups_np = [
                np.array(self._g0), np.array(self._g1),
                np.array(self._g2), np.array(self._g3),
            ]

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier 初始化所有参数"""
        nn.init.xavier_normal_(self.W_map)
        nn.init.zeros_(self.B_map)
        nn.init.xavier_normal_(self.W_enhance)
        nn.init.zeros_(self.B_enhance)
        nn.init.zeros_(self.beta)

    def _apply_mapping_activation(self, Z):
        """映射层激活（支持多激活集成）"""
        if not self.multi_activation:
            return torch.tanh(Z)

        M = torch.empty_like(Z)
        M[:, self._g0] = Z[:, self._g0]                # identity
        M[:, self._g1] = torch.tanh(Z[:, self._g1])    # tanh
        M[:, self._g2] = torch.relu(Z[:, self._g2])    # ReLU
        M[:, self._g3] = torch.sin(Z[:, self._g3])     # sine
        return M

    def forward(self, x, y):
        """前向传播：(x, y) → u

        Parameters
        ----------
        x, y : Tensor, shape (N,), requires_grad=True
            空间坐标

        Returns
        -------
        u : Tensor, shape (N,)
            近似解
        """
        inp = torch.stack([x, y], dim=1)  # (N, 2)

        # 映射层
        Z_map = inp @ self.W_map + self.B_map  # (N, N1)
        M = self._apply_mapping_activation(Z_map)  # (N, N1)

        # 增强层
        Z_enh = M @ self.W_enhance + self.B_enhance  # (N, N2)
        E = torch.tanh(Z_enh)  # (N, N2)

        # 特征拼接 + 线性输出
        H = torch.cat([M, E], dim=1)  # (N, N1+N2)
        u = (H @ self.beta).squeeze(-1)  # (N,)
        return u

    def compute_derivatives(self, x, y):
        """用 autograd 计算 u 的偏导数

        Returns
        -------
        u, u_x, u_y, u_xx, u_yy : Tensor, shape (N,)
        """
        u = self.forward(x, y)

        # 一阶导数
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        # 二阶导数
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y, grad_outputs=torch.ones_like(u_y),
            create_graph=True, retain_graph=True
        )[0]

        return u, u_x, u_y, u_xx, u_yy

    # ==================================================================
    # 伪逆热启动
    # ==================================================================

    @torch.no_grad()
    def pseudoinverse_warmstart(self, x_pde, y_pde, x_bc, y_bc,
                                 source_fn, bc_fn):
        """Phase 1: 固定内部权重，用 numpy 解析链式法则求 Laplacian(H)，
        然后伪逆求解最优 beta。

        数学推导：
          映射层:  Z = [x,y] @ W + b,  M = act(Z)
          增强层:  Z_e = M @ W_e + b_e,  E = tanh(Z_e)
          特征:    H = [M | E]

          对 tanh 节点:
            dM/dx = act'(Z) * W[0,:],   d²M/dx² = act''(Z) * W[0,:]²
          增强层 (tanh):
            dE/dx = tanh'(Z_e) * (dM/dx @ W_e)
            d²E/dx² = tanh''(Z_e) * (dM/dx @ W_e)² + tanh'(Z_e) * (d²M/dx² @ W_e)

          Laplacian(H[:,j]) = d²H_j/dx² + d²H_j/dy²
        """
        W = self.W_map.detach().numpy()    # (2, N1)
        B = self.B_map.detach().numpy()    # (N1,)
        We = self.W_enhance.detach().numpy()  # (N1, N2)
        Be = self.B_enhance.detach().numpy()  # (N2,)

        def _act_fwd(Z):
            if not self.multi_activation:
                return np.tanh(Z)
            H = np.empty_like(Z)
            g = self._act_groups_np
            H[:, g[0]] = Z[:, g[0]]
            H[:, g[1]] = np.tanh(Z[:, g[1]])
            H[:, g[2]] = np.maximum(0, Z[:, g[2]])
            H[:, g[3]] = np.sin(Z[:, g[3]])
            return H

        def _act_d1(Z):
            if not self.multi_activation:
                return 1 - np.tanh(Z) ** 2
            d = np.empty_like(Z)
            g = self._act_groups_np
            d[:, g[0]] = 1.0
            d[:, g[1]] = 1 - np.tanh(Z[:, g[1]]) ** 2
            d[:, g[2]] = (Z[:, g[2]] > 0).astype(float)
            d[:, g[3]] = np.cos(Z[:, g[3]])
            return d

        def _act_d2(Z):
            if not self.multi_activation:
                t = np.tanh(Z)
                return -2 * t * (1 - t ** 2)
            dd = np.zeros_like(Z)
            g = self._act_groups_np
            t = np.tanh(Z[:, g[1]])
            dd[:, g[1]] = -2 * t * (1 - t ** 2)
            dd[:, g[3]] = -np.sin(Z[:, g[3]])
            return dd

        # 对 PDE 点和 BC 点构建特征及 Laplacian
        def _build(x, y):
            inp = np.column_stack([x, y])  # (N, 2)
            Z_map = inp @ W + B  # (N, N1)
            M = _act_fwd(Z_map)
            Z_enh = M @ We + Be  # (N, N2)
            E = np.tanh(Z_enh)
            H = np.hstack([M, E])  # (N, N1+N2)

            # 映射层导数
            dM = _act_d1(Z_map)
            ddM = _act_d2(Z_map)
            dM_dx = dM * W[0, :]  # (N, N1)
            dM_dy = dM * W[1, :]
            d2M_dx2 = ddM * W[0, :] ** 2
            d2M_dy2 = ddM * W[1, :] ** 2

            # 增强层导数
            t_e = np.tanh(Z_enh)
            dE_act = 1 - t_e ** 2  # tanh'
            ddE_act = -2 * t_e * (1 - t_e ** 2)  # tanh''

            dE_dx = dE_act * (dM_dx @ We)
            dE_dy = dE_act * (dM_dy @ We)
            d2E_dx2 = ddE_act * (dM_dx @ We) ** 2 + dE_act * (d2M_dx2 @ We)
            d2E_dy2 = ddE_act * (dM_dy @ We) ** 2 + dE_act * (d2M_dy2 @ We)

            LapH = np.hstack([d2M_dx2 + d2M_dy2, d2E_dx2 + d2E_dy2])
            return H, LapH

        H_pde, LapH = _build(x_pde, y_pde)
        H_bc, _ = _build(x_bc, y_bc)

        # 组装线性系统: [LapH; sqrt(λ)·H_bc] @ β = [f; sqrt(λ)·g]
        lam = np.sqrt(self.lambda_bc)
        A_np = np.vstack([LapH, lam * H_bc])
        b_np = np.concatenate([source_fn(x_pde, y_pde), lam * bc_fn(x_bc, y_bc)])

        # 伪逆求解
        beta_np = pinv(A_np) @ b_np.reshape(-1, 1)

        # 写回 PyTorch 参数
        self.beta.data = torch.tensor(
            beta_np, dtype=self.beta.dtype
        ).reshape_as(self.beta)

    # ==================================================================
    # 训练
    # ==================================================================

    def train_model(self, x_pde, y_pde, x_bc, y_bc,
                    pde_residual_fn, bc_fn,
                    source_fn=None,
                    epochs_adam=2000,
                    epochs_lbfgs=1000,
                    lr_adam=1e-3,
                    lr_lbfgs=0.5,
                    warmstart=True,
                    verbose=True,
                    log_every=200):
        """三阶段训练

        Parameters
        ----------
        x_pde, y_pde : ndarray, PDE 配点
        x_bc, y_bc : ndarray, 边界点
        pde_residual_fn : callable(u, u_x, u_y, u_xx, u_yy, x, y) -> Tensor
            PDE 残差函数 R，满足时为 0。
            例：Poisson:  R = u_xx + u_yy - f(x,y)
            例：非线性:   R = u_xx + u_yy - u³ - f(x,y)
        bc_fn : callable(x, y) -> ndarray
            Dirichlet 边界条件值
        source_fn : callable(x, y) -> ndarray
            PDE 右端项（仅用于伪逆热启动，若 warmstart=False 可省略）
        epochs_adam : int
            Adam 阶段 epochs
        epochs_lbfgs : int
            L-BFGS 阶段 epochs（0 则跳过）
        lr_adam : float
            Adam 学习率
        lr_lbfgs : float
            L-BFGS 学习率
        warmstart : bool
            是否用伪逆初始化 beta
        verbose : bool
            是否打印训练过程
        log_every : int
            每隔多少 epoch 打印一次

        Returns
        -------
        history : dict with 'loss', 'loss_pde', 'loss_bc' lists
        """
        import time
        t_start = time.time()

        # Phase 0: 伪逆热启动
        if warmstart and source_fn is not None:
            if verbose:
                print("[GD-PIBLS] Phase 0: Pseudoinverse warmstart...")
            self.pseudoinverse_warmstart(
                x_pde, y_pde, x_bc, y_bc, source_fn, bc_fn
            )
            # 评估热启动效果
            with torch.no_grad():
                x_p_t = torch.tensor(x_pde, dtype=torch.float32).requires_grad_(True)
                y_p_t = torch.tensor(y_pde, dtype=torch.float32).requires_grad_(True)
                u0 = self.forward(x_p_t, y_p_t)
            if verbose:
                print(f"  Warmstart done. u range: [{u0.min():.4f}, {u0.max():.4f}]")

        # 准备训练数据
        x_pde_t = torch.tensor(x_pde, dtype=torch.float32).requires_grad_(True)
        y_pde_t = torch.tensor(y_pde, dtype=torch.float32).requires_grad_(True)
        x_bc_t = torch.tensor(x_bc, dtype=torch.float32)
        y_bc_t = torch.tensor(y_bc, dtype=torch.float32)
        u_bc_target = torch.tensor(bc_fn(x_bc, y_bc), dtype=torch.float32)

        history = {'loss': [], 'loss_pde': [], 'loss_bc': []}

        def compute_loss():
            """计算 PDE + BC 损失"""
            u, u_x, u_y, u_xx, u_yy = self.compute_derivatives(
                x_pde_t, y_pde_t
            )
            R = pde_residual_fn(u, u_x, u_y, u_xx, u_yy, x_pde_t, y_pde_t)
            loss_pde = torch.mean(R ** 2)

            u_bc = self.forward(x_bc_t, y_bc_t)
            loss_bc = torch.mean((u_bc - u_bc_target) ** 2)

            loss = loss_pde + self.lambda_bc * loss_bc
            return loss, loss_pde, loss_bc

        # Phase 1: Adam
        if epochs_adam > 0:
            if verbose:
                print(f"[GD-PIBLS] Phase 1: Adam ({epochs_adam} epochs, lr={lr_adam})")
            optimizer = torch.optim.Adam(self.parameters(), lr=lr_adam)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs_adam, eta_min=lr_adam * 0.01
            )

            for epoch in range(epochs_adam):
                optimizer.zero_grad()
                loss, loss_pde, loss_bc = compute_loss()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

                optimizer.step()
                scheduler.step()

                history['loss'].append(loss.item())
                history['loss_pde'].append(loss_pde.item())
                history['loss_bc'].append(loss_bc.item())

                if verbose and (epoch % log_every == 0 or epoch == epochs_adam - 1):
                    print(
                        f"  Epoch {epoch:>5d}: loss={loss.item():.4e}  "
                        f"pde={loss_pde.item():.4e}  bc={loss_bc.item():.4e}"
                    )

        # Phase 2: L-BFGS
        if epochs_lbfgs > 0:
            if verbose:
                print(f"[GD-PIBLS] Phase 2: L-BFGS ({epochs_lbfgs} epochs)")

            # L-BFGS 需要重新创建 requires_grad 的 tensor
            x_pde_t = torch.tensor(x_pde, dtype=torch.float32).requires_grad_(True)
            y_pde_t = torch.tensor(y_pde, dtype=torch.float32).requires_grad_(True)

            optimizer_lbfgs = torch.optim.LBFGS(
                self.parameters(),
                lr=lr_lbfgs,
                max_iter=20,
                history_size=50,
                tolerance_grad=1e-9,
                tolerance_change=1e-11,
                line_search_fn='strong_wolfe',
            )

            lbfgs_step = [0]

            def closure():
                optimizer_lbfgs.zero_grad()
                loss, loss_pde, loss_bc = compute_loss()
                loss.backward()

                history['loss'].append(loss.item())
                history['loss_pde'].append(loss_pde.item())
                history['loss_bc'].append(loss_bc.item())

                lbfgs_step[0] += 1
                if verbose and lbfgs_step[0] % max(1, log_every // 10) == 0:
                    print(
                        f"  L-BFGS step {lbfgs_step[0]:>5d}: "
                        f"loss={loss.item():.4e}  "
                        f"pde={loss_pde.item():.4e}  bc={loss_bc.item():.4e}"
                    )
                return loss

            for _ in range(epochs_lbfgs):
                optimizer_lbfgs.step(closure)
                if len(history['loss']) > 1 and history['loss'][-1] < 1e-14:
                    break

        t_end = time.time()
        if verbose:
            print(
                f"[GD-PIBLS] Training complete. "
                f"Final loss={history['loss'][-1]:.4e}  "
                f"Time={t_end - t_start:.2f}s"
            )

        return history

    # ==================================================================
    # 预测
    # ==================================================================

    @torch.no_grad()
    def predict(self, x, y):
        """预测 u(x, y)

        Parameters
        ----------
        x, y : ndarray

        Returns
        -------
        u : ndarray
        """
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        u = self.forward(x_t, y_t)
        return u.numpy()


# =====================================================================
# 便捷工厂函数
# =====================================================================

def make_grid_data(nx_pde=30, nx_bc=50):
    """生成 [0,1]² 上的 PDE 配点和边界点

    Returns
    -------
    x_pde, y_pde : ndarray, 内部均匀网格点
    x_bc, y_bc : ndarray, 四条边上均匀采样
    """
    # 内部配点
    x_lin = np.linspace(0, 1, nx_pde + 2)[1:-1]
    y_lin = np.linspace(0, 1, nx_pde + 2)[1:-1]
    xx, yy = np.meshgrid(x_lin, y_lin)
    x_pde = xx.ravel()
    y_pde = yy.ravel()

    # 边界点 (四条边)
    t_bc = np.linspace(0, 1, nx_bc)
    x_bc = np.concatenate([t_bc, t_bc, np.zeros(nx_bc), np.ones(nx_bc)])
    y_bc = np.concatenate([np.zeros(nx_bc), np.ones(nx_bc), t_bc, t_bc])

    return x_pde, y_pde, x_bc, y_bc
