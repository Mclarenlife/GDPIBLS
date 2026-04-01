"""
Bilevel-Optimized Physics-Informed Broad Learning System (BO-PIBLS)

核心创新：将 BLS 伪逆求解嵌入 PyTorch 可微分计算图，
        形成"内层伪逆（解析最优 β）+ 外层梯度下降（学习最优 θ）"的双层优化。

=== 与现有方法的本质区别 ===
PIBLS:    θ 随机固定 + β 伪逆一次        → 特征质量受限于随机初始化
PINN:     (θ,β) 联合梯度下降             → β 永远不是最优的
GD-PIBLS: (θ,β) 联合梯度下降             → 本质是 shallow PINN，丧失 BLS 特性
BO-PIBLS: θ 梯度下降 + β 每轮伪逆（可微） → β 始终最优，梯度穿过伪逆更新 θ

=== 数学框架 ===
  min_θ  L_PDE( H(θ) · β*(θ) )
  s.t.   β*(θ) = argmin_β ||A(θ)β - b||² = (A^T A + λI)^{-1} A^T b

梯度通过 torch.linalg.solve 的自动微分反传到 θ。

=== Fourier 特征 ===
映射节点使用可学习频率的 Fourier 特征：
  φ_i(x) = sin(ω_i · x + b_i)
ω_i 是可学习参数（θ 的一部分），梯度下降优化最适合 PDE 解的频率组合。
增强节点使用 tanh 提供非线性表达力。
"""

import torch
import numpy as np
import time


class BOPIBLS:
    """
    双层优化物理信息宽度学习系统

    求解: -Δu + g(u) = f  on Ω,  u = h on ∂Ω

    BLS 特性保留:
      - 浅层架构（映射层 + 增强层 + 线性输出）
      - 输出权重 β 通过可微伪逆求解（非梯度更新）
      - 支持增量扩展（添加节点后伪逆自动适配）

    创新:
      - θ=(ω, b_map, W_enh, b_enh) 通过梯度下降优化
      - 梯度穿过 torch.linalg.solve 的可微伪逆
      - Fourier 特征使频率可学习，精确匹配 PDE 解的模态
    """

    def __init__(self,
                 n_map=50, n_enh=50,
                 ridge=1e-6, bc_weight=10.0,
                 lr=1e-2, epochs=500,
                 lr_lbfgs=1.0, epochs_lbfgs=200,
                 seed=42, verbose=True,
                 freq_init_scale=1.0):
        """
        Parameters
        ----------
        n_map : int
            映射节点数（Fourier 特征）
        n_enh : int
            增强节点数（tanh 非线性变换）
        ridge : float
            Tikhonov 正则化参数 λ
        bc_weight : float
            边界条件权重
        lr : float
            Adam 学习率
        epochs : int
            Adam 训练轮数
        lr_lbfgs : float
            L-BFGS 学习率
        epochs_lbfgs : int
            L-BFGS 训练轮数
        seed : int
            随机种子
        verbose : bool
            是否打印训练过程
        freq_init_scale : float
            初始频率分布的尺度（=1 对应均匀分布约 [0, 2π]）
        """
        self.n_map = n_map
        self.n_enh = n_enh
        self.ridge = ridge
        self.bc_weight = bc_weight
        self.lr = lr
        self.epochs = epochs
        self.lr_lbfgs = lr_lbfgs
        self.epochs_lbfgs = epochs_lbfgs
        self.seed = seed
        self.verbose = verbose
        self.freq_init_scale = freq_init_scale
        self.history = []

        # 参数容器（fit 时初始化）
        self.omega = None      # Fourier 频率, (D, n_map)
        self.b_map = None      # Fourier 偏置, (n_map,)
        self.W_enh = None      # 增强层权重, (n_map, n_enh)
        self.b_enh = None      # 增强层偏置, (n_enh,)
        self._beta_warmstart = None  # Newton 热启动

    def _init_params(self, D):
        """Xavier-style 初始化可学习参数"""
        torch.manual_seed(self.seed)

        # Fourier 频率：多尺度初始化
        # 低频节点 + 中频节点 + 高频节点
        n_low = self.n_map // 3
        n_mid = self.n_map // 3
        n_high = self.n_map - n_low - n_mid

        omega_parts = []
        if n_low > 0:
            omega_parts.append(
                torch.randn(D, n_low) * self.freq_init_scale * 1.0
            )
        if n_mid > 0:
            omega_parts.append(
                torch.randn(D, n_mid) * self.freq_init_scale * 3.0
            )
        if n_high > 0:
            omega_parts.append(
                torch.randn(D, n_high) * self.freq_init_scale * 6.0
            )
        self.omega = torch.cat(omega_parts, dim=1).requires_grad_(True)

        self.b_map = (torch.rand(self.n_map) * 2 * np.pi
                      ).requires_grad_(True)

        # 增强层：映射到 tanh 非线性
        scale = 1.0 / np.sqrt(max(self.n_map, 1))
        self.W_enh = (torch.randn(self.n_map, self.n_enh) * scale
                      ).requires_grad_(True)
        self.b_enh = (torch.randn(self.n_enh) * 0.1
                      ).requires_grad_(True)

    def _get_params(self):
        """返回所有可学习参数（θ）"""
        return [self.omega, self.b_map, self.W_enh, self.b_enh]

    def _build_features_and_laplacian(self, X):
        """
        同时构建特征矩阵 H(θ) 和拉普拉斯 ΔH —— 全部用解析公式。

        解析公式基于 torch 运算符，因此仍然对 θ=(ω, b_map, W_enh, b_enh) 可微分，
        梯度可以穿过拉普拉斯计算反传到 θ。

        映射层 Fourier 特征:
          M_j = sin(ω_j · x + b_j)
          ΔM_j = -||ω_j||² · M_j

        增强层 tanh:
          s_k = Σ_i W_{ik}^enh · M_i + b_k^enh
          E_k = tanh(s_k)
          ΔE_k = Σ_d [ tanh''(s_k) · (∂s_k/∂x_d)² + tanh'(s_k) · ∂²s_k/∂x_d² ]

        Parameters
        ----------
        X : torch.Tensor, shape (N, D)

        Returns
        -------
        H : torch.Tensor, shape (N, n_map + n_enh)
        H_lap : torch.Tensor, shape (N, n_map + n_enh)
        """
        N, D = X.shape

        # ===== 映射层 =====
        Z_map = X @ self.omega + self.b_map     # (N, n_map)
        M = torch.sin(Z_map)                     # (N, n_map)
        cos_Z = torch.cos(Z_map)                 # (N, n_map)

        # ΔM_j = -||ω_j||² · sin(Z_j) = -||ω_j||² · M_j
        omega_sq_sum = torch.sum(self.omega ** 2, dim=0, keepdim=True)  # (1, n_map)
        M_lap = -omega_sq_sum * M                 # (N, n_map)

        # ===== 增强层 =====
        Z_enh = M @ self.W_enh + self.b_enh     # (N, n_enh)
        E = torch.tanh(Z_enh)                    # (N, n_enh)

        # tanh 导数: tanh'(s) = 1 - tanh²(s),  tanh''(s) = -2·tanh(s)·(1-tanh²(s))
        sech2 = 1.0 - E ** 2                     # tanh'(s), (N, n_enh)
        tanh_dd = -2.0 * E * sech2               # tanh''(s), (N, n_enh)

        # 增强层拉普拉斯: ΔE_k = Σ_d [tanh''·(ds/dx_d)² + tanh'·(d²s/dx_d²)]
        # ds/dx_d = (cos(Z) * ω[d,:]) @ W_enh
        # d²s/dx_d² = (-sin(Z) * ω[d,:]²) @ W_enh = (-M * ω[d,:]²) @ W_enh
        E_lap = torch.zeros(N, self.n_enh, dtype=X.dtype)
        for d in range(D):
            omega_d = self.omega[d:d+1, :]            # (1, n_map)
            ds_d = (cos_Z * omega_d) @ self.W_enh     # (N, n_enh)
            d2s_d = (-M * omega_d ** 2) @ self.W_enh  # (N, n_enh)
            E_lap = E_lap + tanh_dd * ds_d ** 2 + sech2 * d2s_d

        H = torch.cat([M, E], dim=1)
        H_lap = torch.cat([M_lap, E_lap], dim=1)
        return H, H_lap

    def _solve_beta(self, A, b):
        """
        可微分伪逆求解 β* = (A^T A + λI)^{-1} A^T b

        关键：torch.linalg.solve 支持自动微分，
        梯度可穿过此步骤反传到 A（即反传到 θ）。

        Parameters
        ----------
        A : torch.Tensor, shape (M, n_features)
        b : torch.Tensor, shape (M,)

        Returns
        -------
        beta : torch.Tensor, shape (n_features,)
        """
        n = A.shape[1]
        ATA = A.T @ A + self.ridge * torch.eye(n, dtype=A.dtype)
        ATb = A.T @ b
        beta = torch.linalg.solve(ATA, ATb)
        return beta

    def _compute_loss_linear(self, X_int, X_bc, f_vals, g_vals):
        """
        线性 PDE 的前向传播 + 伪逆 + 损失计算

        整个链路可微分：θ → H(θ) → ΔH(θ) → 组装 A(θ) → solve β*(θ) → L(θ)
        """
        X_all = torch.cat([X_int, X_bc], dim=0)
        N_int = X_int.shape[0]

        H, H_lap = self._build_features_and_laplacian(X_all)

        Hi = H[:N_int]        # 内部特征
        Hli = H_lap[:N_int]   # 内部拉普拉斯
        Hb = H[N_int:]        # 边界特征

        # 组装线性系统 A·β = b
        # PDE: -ΔH·β = f  =>  (-Hli)·β = f
        # BC:  H_bc·β = g
        A = torch.cat([-Hli, self.bc_weight * Hb], dim=0)
        b = torch.cat([f_vals, self.bc_weight * g_vals], dim=0)

        # 内层求解（可微伪逆）
        beta = self._solve_beta(A, b)

        # 计算 PDE 残差损失（外层目标）
        u_int = Hi @ beta
        lap_u = Hli @ beta
        pde_res = -lap_u - f_vals
        L_pde = torch.mean(pde_res ** 2)

        # 边界残差
        u_bc = Hb @ beta
        L_bc = torch.mean((u_bc - g_vals) ** 2)

        loss = L_pde + self.bc_weight * L_bc
        return loss, beta, L_pde.item(), L_bc.item()

    def _compute_loss_nonlinear(self, X_int, X_bc, g_fn, f_vals, g_vals,
                                 n_newton=5, damping=1.0):
        """
        非线性 PDE：Newton-in-the-loop 可微分求解 (V1.1b 热启动版)

        改进：
        - 使用 self._beta_warmstart 热启动（不从线性解重新开始）
        - 增加默认 Newton 步数到 5
        - Newton 线性化用 detach，但 solve(A,b) 中 A 依赖 θ → 梯度穿过
        - 最终残差计算保留完整计算图到 θ
        """
        X_all = torch.cat([X_int, X_bc], dim=0)
        N_int = X_int.shape[0]

        H, H_lap = self._build_features_and_laplacian(X_all)

        Hi = H[:N_int]
        Hli = H_lap[:N_int]
        Hb = H[N_int:]

        # 初始 beta：热启动或线性解
        if self._beta_warmstart is not None and \
           self._beta_warmstart.shape[0] == Hi.shape[1]:
            beta = self._beta_warmstart.detach().clone()
        else:
            # 线性初始猜测
            A_lin = torch.cat([-Hli, self.bc_weight * Hb], dim=0)
            b_lin = torch.cat([f_vals, self.bc_weight * g_vals], dim=0)
            with torch.no_grad():
                beta = self._solve_beta(A_lin, b_lin).detach()

        # Newton 迭代（fix β for linearization, solve through θ）
        for k in range(n_newton):
            # 当前解和非线性项（detached，仅用于线性化系数）
            u_cur = (Hi.detach() @ beta).detach()
            u_cur_rg = u_cur.requires_grad_(True)
            g_u_rg = g_fn(u_cur_rg)
            gp_u = torch.autograd.grad(g_u_rg.sum(), u_cur_rg)[0].detach()
            g_u = g_fn(u_cur).detach()

            # Newton 线性化系统（A 依赖 θ 因为 Hi, Hli 是 θ 的函数）
            A_int = -Hli + gp_u.unsqueeze(1) * Hi
            b_int = f_vals - g_u + gp_u * u_cur

            A_k = torch.cat([A_int, self.bc_weight * Hb], dim=0)
            b_k = torch.cat([b_int, self.bc_weight * g_vals], dim=0)

            # 可微伪逆求解
            beta_new = self._solve_beta(A_k, b_k)
            beta = (1 - damping) * beta.detach() + damping * beta_new

        # 保存热启动
        self._beta_warmstart = beta.detach().clone()

        # 真实非线性残差
        u_final = Hi @ beta
        lap_u = Hli @ beta
        pde_res = -lap_u + g_fn(u_final) - f_vals
        L_pde = torch.mean(pde_res ** 2)

        u_bc = Hb @ beta
        L_bc = torch.mean((u_bc - g_vals) ** 2)

        loss = L_pde + self.bc_weight * L_bc
        return loss, beta, L_pde.item(), L_bc.item()

    def fit_linear(self, X_int, X_bc, source_fn, bc_fn):
        """
        求解线性 PDE: -Δu = f, u|∂Ω = g

        训练流程:
        Phase 1: Adam 优化 θ（梯度穿过可微伪逆）
        Phase 2: L-BFGS 精调 θ
        """
        D = X_int.shape[1]
        self._init_params(D)

        X_int_t = torch.tensor(X_int, dtype=torch.float64)
        X_bc_t = torch.tensor(X_bc, dtype=torch.float64)
        f_vals = torch.tensor(source_fn(X_int), dtype=torch.float64)
        g_vals = torch.tensor(bc_fn(X_bc), dtype=torch.float64)

        # 转 float64 提高数值精度
        self.omega = self.omega.double().detach().requires_grad_(True)
        self.b_map = self.b_map.double().detach().requires_grad_(True)
        self.W_enh = self.W_enh.double().detach().requires_grad_(True)
        self.b_enh = self.b_enh.double().detach().requires_grad_(True)

        self._train_bilevel(
            compute_loss_fn=lambda: self._compute_loss_linear(
                X_int_t, X_bc_t, f_vals, g_vals
            )
        )
        return self

    def fit_nonlinear(self, X_int, X_bc, g_fn_torch, source_fn, bc_fn,
                      n_newton=3, damping=0.8):
        """
        求解非线性 PDE: -Δu + g(u) = f, u|∂Ω = h

        V1.1 改进：Newton-in-the-loop 策略。
        不再使用 Picard 外循环，而是在每个 Adam/L-BFGS 步内
        做 n_newton 次可微分 Newton 迭代。

        Parameters
        ----------
        g_fn_torch : callable
            非线性项 g(u)，必须兼容 torch 张量
        source_fn : callable
            源项 f(x)，numpy 输入
        bc_fn : callable
            边界条件 h(x)，numpy 输入
        n_newton : int
            每个 Adam 步内 Newton 迭代次数
        damping : float
            Newton 步阻尼系数 (0,1]
        """
        D = X_int.shape[1]
        self._init_params(D)

        X_int_t = torch.tensor(X_int, dtype=torch.float64)
        X_bc_t = torch.tensor(X_bc, dtype=torch.float64)
        f_vals = torch.tensor(source_fn(X_int), dtype=torch.float64)
        g_vals = torch.tensor(bc_fn(X_bc), dtype=torch.float64)

        self.omega = self.omega.double().detach().requires_grad_(True)
        self.b_map = self.b_map.double().detach().requires_grad_(True)
        self.W_enh = self.W_enh.double().detach().requires_grad_(True)
        self.b_enh = self.b_enh.double().detach().requires_grad_(True)

        g_fn = g_fn_torch
        self._beta_warmstart = None  # 重置热启动

        self._train_bilevel(
            compute_loss_fn=lambda: self._compute_loss_nonlinear(
                X_int_t, X_bc_t, g_fn, f_vals, g_vals,
                n_newton=n_newton, damping=damping
            )
        )
        return self

    def _train_bilevel(self, compute_loss_fn,
                       epochs_adam=None, epochs_lbfgs=None):
        """
        核心训练循环：Adam + L-BFGS

        注意：只优化 θ（内部权重），β 每次前向传播时由伪逆自动求解。
        """
        if epochs_adam is None:
            epochs_adam = self.epochs
        if epochs_lbfgs is None:
            epochs_lbfgs = self.epochs_lbfgs

        params = self._get_params()

        # ---------- Phase 1: Adam ----------
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_adam, eta_min=self.lr * 0.01
        )

        best_loss = float('inf')
        best_state = [p.detach().clone() for p in params]

        for epoch in range(epochs_adam):
            optimizer.zero_grad()

            loss, beta, lpde, lbc = compute_loss_fn()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)

            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = [p.detach().clone() for p in params]
                self._final_beta = beta.detach().clone()

            if self.verbose and (epoch % 50 == 0 or epoch == epochs_adam - 1):
                print(f"  Adam [{epoch:>4d}/{epochs_adam}]  "
                      f"loss={loss.item():.4e}  "
                      f"L_pde={lpde:.4e}  L_bc={lbc:.4e}")

            self.history.append({
                'phase': 'adam', 'epoch': epoch,
                'loss': loss.item(), 'lpde': lpde, 'lbc': lbc
            })

        # 恢复 Adam 阶段最优
        for p, s in zip(params, best_state):
            p.data.copy_(s)

        # ---------- Phase 2: L-BFGS ----------
        if epochs_lbfgs > 0:
            optimizer_lbfgs = torch.optim.LBFGS(
                params, lr=self.lr_lbfgs,
                max_iter=20, line_search_fn='strong_wolfe',
                history_size=50, tolerance_grad=1e-12, tolerance_change=1e-14
            )

            lbfgs_step = [0]

            def closure():
                optimizer_lbfgs.zero_grad()
                loss, beta, lpde, lbc = compute_loss_fn()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)

                if loss.item() < best_loss:
                    self._final_beta = beta.detach().clone()

                if self.verbose and lbfgs_step[0] % 20 == 0:
                    print(f"  L-BFGS [{lbfgs_step[0]:>4d}]  "
                          f"loss={loss.item():.4e}  "
                          f"L_pde={lpde:.4e}  L_bc={lbc:.4e}")
                lbfgs_step[0] += 1
                return loss

            for _ in range(epochs_lbfgs):
                optimizer_lbfgs.step(closure)

            # 读取最终 loss
            final_loss, final_beta, final_lpde, final_lbc = compute_loss_fn()
            self._final_beta = final_beta.detach().clone()

            if self.verbose:
                print(f"  L-BFGS final:  loss={final_loss.item():.4e}  "
                      f"L_pde={final_lpde:.4e}  L_bc={final_lbc:.4e}")

    def predict(self, X):
        """
        预测 u(X) = H(X; θ*) · β*

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        u : np.ndarray, shape (N,)
        """
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float64)
            H, _ = self._build_features_and_laplacian(X_t)
            u = H @ self._final_beta
            return u.numpy()

    def get_n_features(self):
        return self.n_map + self.n_enh
