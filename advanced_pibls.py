"""
高级物理信息宽度学习系统 (Advanced PIBLS)

实现两种创新方法：
1. HybridPIBLS  - 伪逆更新与梯度下降交替优化
2. NonlinearPIBLS - 基于Newton-伪逆迭代处理非线性方程
"""

import numpy as np
from scipy.linalg import pinv
from pibls_model import PIBLS


# =============================================================================
# 方法1: HybridPIBLS - 伪逆更新与梯度下降结合
# =============================================================================

class HybridPIBLS(PIBLS):
    """混合伪逆-梯度下降物理信息宽度学习系统 (GPIBLS)

    核心思想：交替执行两步优化——
      1. 伪逆步：固定内部权重，用伪逆一步求解最优输出权重 beta
      2. 梯度步：固定 beta，用 SPSA 梯度下降更新内部权重以改善特征表示

    关键改进（相比原始 PIBLS）：
    - 原始 BLS 内部权重 {W_map, B_map, W_enhance, B_enhance} 随机固定，
      表达力受限于随机投影的质量
    - HybridPIBLS 在每轮伪逆后微调内部权重，逐步优化特征空间
    - 采用保守更新策略：小学习率 + 梯度裁剪 + 回退机制，
      保证不破坏伪逆步已获得的解质量

    Parameters
    ----------
    N1, N2 : int
        映射层和增强层节点数
    lr : float
        内部权重梯度下降学习率（建议 1e-4 ~ 1e-2）
    max_iter : int
        外层交替优化迭代次数
    lambda_bc : float
        边界条件损失权重
    grad_method : str
        'spsa' (默认, 快速) 或 'fd' (精确但慢)
    n_grad_samples : int
        SPSA 采样数（越多越稳定，默认 15）
    """

    def __init__(
        self,
        N1,
        N2,
        map_func='tanh',
        enhance_func='sigmoid',
        source_fn=None,
        exact_solution_fn=None,
        lr=0.005,
        max_iter=50,
        lambda_bc=10.0,
        tol=1e-12,
        grad_method='spsa',
        multi_activation=False,
        n_grad_samples=15,
    ):
        super().__init__(N1, N2, map_func, enhance_func, source_fn, exact_solution_fn,
                         multi_activation=multi_activation)
        self.lr = lr
        self.max_iter = max_iter
        self.lambda_bc = lambda_bc
        self.tol = tol
        self.grad_method = grad_method
        self.n_grad_samples = n_grad_samples
        self.loss_history = []

    # ---- 损失计算 ----

    def _compute_loss(self, pde_data, bc_data):
        """计算总损失 = PDE残差 L2 + lambda * 边界条件 L2"""
        x_pde, y_pde = pde_data
        x_bc, y_bc = bc_data

        H_pde, z_pde = self._build_features(x_pde, y_pde)
        d2H_dx2, d2H_dy2 = self._compute_derivatives(x_pde, y_pde, z_pde)
        res_pde = ((d2H_dx2 + d2H_dy2) @ self.beta).flatten() - self.source_fn(x_pde, y_pde)
        L_pde = np.mean(res_pde ** 2)

        H_bc, _ = self._build_features(x_bc, y_bc)
        res_bc = (H_bc @ self.beta).flatten() - self.exact_solution_fn(x_bc, y_bc)
        L_bc = np.mean(res_bc ** 2)

        return L_pde + self.lambda_bc * L_bc

    # ---- 伪逆步后重新优化 beta ----

    def _pseudoinverse_step(self, pde_data, bc_data):
        """用当前内部权重重新构建系统并伪逆求解 beta"""
        A, b = self.build_system(pde_data, bc_data)
        self.beta = pinv(A) @ b.reshape(-1, 1)

    # ---- 梯度估计 ----

    def _spsa_gradient(self, param_name, pde_data, bc_data, eps=0.005):
        """SPSA 梯度估计，多次采样取平均"""
        param = getattr(self, param_name)
        grad_acc = np.zeros_like(param)

        for _ in range(self.n_grad_samples):
            delta = np.random.choice([-1.0, 1.0], size=param.shape)

            param += eps * delta
            # 每次扰动后重新求解 beta（关键：让梯度反映特征质量变化）
            self._pseudoinverse_step(pde_data, bc_data)
            lp = self._compute_loss(pde_data, bc_data)

            param -= 2.0 * eps * delta
            self._pseudoinverse_step(pde_data, bc_data)
            lm = self._compute_loss(pde_data, bc_data)

            param += eps * delta  # 还原
            grad_acc += (lp - lm) / (2.0 * eps * delta)

        return grad_acc / self.n_grad_samples

    def _fd_gradient(self, param_name, pde_data, bc_data, eps=1e-5):
        """有限差分梯度 (精确但慢)"""
        param = getattr(self, param_name)
        grad = np.zeros_like(param)

        for idx in np.ndindex(param.shape):
            old = param[idx]

            param[idx] = old + eps
            self._pseudoinverse_step(pde_data, bc_data)
            lp = self._compute_loss(pde_data, bc_data)

            param[idx] = old - eps
            self._pseudoinverse_step(pde_data, bc_data)
            lm = self._compute_loss(pde_data, bc_data)

            grad[idx] = (lp - lm) / (2.0 * eps)
            param[idx] = old

        return grad

    def _compute_grad(self, param_name, pde_data, bc_data):
        if self.grad_method == 'spsa':
            return self._spsa_gradient(param_name, pde_data, bc_data)
        return self._fd_gradient(param_name, pde_data, bc_data)

    # ---- 梯度下降步（带回退保护） ----

    def _gradient_step_safe(self, pde_data, bc_data, lr):
        """对内部权重执行一步梯度下降，如果 loss 恶化则回退"""
        # 保存当前状态
        saved = {}
        for name in ['W_map', 'B_map', 'W_enhance', 'B_enhance']:
            saved[name] = getattr(self, name).copy()
        saved_beta = self.beta.copy()
        loss_before = self._compute_loss(pde_data, bc_data)

        # 逐参数计算梯度并更新
        for name in ['W_map', 'B_map', 'W_enhance', 'B_enhance']:
            grad = self._compute_grad(name, pde_data, bc_data)

            gn = np.linalg.norm(grad)
            if gn > 1.0:
                grad = grad * (1.0 / gn)

            setattr(self, name, getattr(self, name) - lr * grad)

        # 更新后重新伪逆求解
        self._pseudoinverse_step(pde_data, bc_data)
        loss_after = self._compute_loss(pde_data, bc_data)

        # 如果 loss 恶化超过 5%，回退
        if loss_after > loss_before * 1.05:
            for name, val in saved.items():
                setattr(self, name, val)
            self.beta = saved_beta
            return loss_before, False  # 回退
        return loss_after, True  # 接受

    # ---- 训练 ----

    def fit(self, pde_data, bc_data):
        """交替伪逆更新和梯度下降训练"""
        self.loss_history = []

        # Step 0: 首次伪逆初始化
        self._pseudoinverse_step(pde_data, bc_data)
        loss = self._compute_loss(pde_data, bc_data)
        self.loss_history.append(loss)
        best_loss = loss
        best_state = {
            'beta': self.beta.copy(),
            'W_map': self.W_map.copy(),
            'B_map': self.B_map.copy(),
            'W_enhance': self.W_enhance.copy(),
            'B_enhance': self.B_enhance.copy(),
        }

        print(f"[HybridPIBLS] Init: loss = {loss:.6e}")

        for i in range(self.max_iter):
            # 学习率余弦衰减
            current_lr = self.lr * 0.5 * (1 + np.cos(np.pi * i / self.max_iter))

            # 梯度步 + 回退保护
            loss, accepted = self._gradient_step_safe(pde_data, bc_data, current_lr)
            self.loss_history.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_state = {
                    'beta': self.beta.copy(),
                    'W_map': self.W_map.copy(),
                    'B_map': self.B_map.copy(),
                    'W_enhance': self.W_enhance.copy(),
                    'B_enhance': self.B_enhance.copy(),
                }

            status = "accept" if accepted else "reject"
            if i % max(1, self.max_iter // 10) == 0:
                print(f"[HybridPIBLS] Iter {i:>4d}: loss = {loss:.6e}  [{status}]  lr = {current_lr:.2e}")

            if loss < self.tol:
                print(f"[HybridPIBLS] Converged at iter {i}")
                break

        # 恢复最优状态
        self.beta = best_state['beta']
        self.W_map = best_state['W_map']
        self.B_map = best_state['B_map']
        self.W_enhance = best_state['W_enhance']
        self.B_enhance = best_state['B_enhance']

        print(f"[HybridPIBLS] Best loss = {best_loss:.6e}")
        return self.beta


# =============================================================================
# 方法2: NonlinearPIBLS - 基于 Newton-伪逆迭代处理非线性方程
# =============================================================================

class NonlinearPIBLS(PIBLS):
    """基于 Newton-伪逆迭代的非线性 PDE 求解器 (NL-PIBLS)

    核心思想：将非线性 PDE 转化为迭代的线性最小二乘问题——
      1. 在当前近似解 u^(k) = H·β^(k) 处线性化 PDE 残差
      2. 构建 Jacobian 矩阵 J = ∂R/∂β
      3. Newton 更新: δβ = -pinv(J)·R
      4. 迭代至收敛

    数学推导：
      对非线性 PDE  N(u, u_x, u_y, u_xx, u_yy; x, y) = 0
      其中  u = H·β,  u_x = (∂H/∂x)·β,  ...

      Jacobian:
        J[i,j] = (∂N/∂u)·H[i,j] + (∂N/∂u_x)·(∂H/∂x)[i,j]
               + (∂N/∂u_y)·(∂H/∂y)[i,j] + (∂N/∂u_xx)·(∂²H/∂x²)[i,j]
               + (∂N/∂u_yy)·(∂²H/∂y²)[i,j]

    也支持 fit_hybrid(): Newton-伪逆 + 内部权重梯度下降

    Parameters
    ----------
    N1, N2 : int
        映射层和增强层节点数
    residual_fn : callable(u, u_x, u_y, u_xx, u_yy, x, y) -> array
        PDE 残差函数, 满足时为 0
    bc_fn : callable(x, y) -> array
        Dirichlet 边界条件值
    dR_du, dR_dux, dR_duy, dR_duxx, dR_duyy : callable or None
        残差对 u, u_x, u_y, u_xx, u_yy 的偏导数 (None 时自动用数值差分)
    lambda_bc : float
        边界条件权重
    """

    def __init__(
        self,
        N1,
        N2,
        map_func='tanh',
        enhance_func='sigmoid',
        residual_fn=None,
        bc_fn=None,
        dR_du=None,
        dR_dux=None,
        dR_duy=None,
        dR_duxx=None,
        dR_duyy=None,
        lambda_bc=10.0,
        multi_activation=False,
    ):
        super().__init__(N1, N2, map_func, enhance_func,
                         multi_activation=multi_activation)
        self.residual_fn = residual_fn
        self.bc_fn = bc_fn
        self.dR_du = dR_du
        self.dR_dux = dR_dux
        self.dR_duy = dR_duy
        self.dR_duxx = dR_duxx
        self.dR_duyy = dR_duyy
        self.lambda_bc = lambda_bc
        self.loss_history = []
        self._verbose = True

    def _compute_all_derivatives(self, x, y, z_values):
        """计算一阶偏导 (dH/dx, dH/dy) 和二阶偏导 (d²H/dx², d²H/dy²)"""
        Z_map, Z_enhance = z_values

        # 映射层
        if self.multi_activation:
            dH_map = self._multi_act_derivative(Z_map)
            ddH_map = self._multi_act_second_derivative(Z_map)
        else:
            dH_map = self.map_derivative(Z_map)
            ddH_map = self.map_second_derivative(Z_map)

        dH_dx_map = dH_map * self.W_map[0, :]
        dH_dy_map = dH_map * self.W_map[1, :]
        d2H_dx2_map = ddH_map * (self.W_map[0, :] ** 2)
        d2H_dy2_map = ddH_map * (self.W_map[1, :] ** 2)

        # 增强层
        dH_enh = self.enhance_derivative(Z_enhance)
        ddH_enh = self.enhance_second_derivative(Z_enhance)

        dH_dx_enh = dH_enh * (dH_dx_map @ self.W_enhance)
        dH_dy_enh = dH_enh * (dH_dy_map @ self.W_enhance)
        d2H_dx2_enh = (ddH_enh * (dH_dx_map @ self.W_enhance) ** 2
                       + dH_enh * (d2H_dx2_map @ self.W_enhance))
        d2H_dy2_enh = (ddH_enh * (dH_dy_map @ self.W_enhance) ** 2
                       + dH_enh * (d2H_dy2_map @ self.W_enhance))

        dH_dx = np.hstack([dH_dx_map, dH_dx_enh])
        dH_dy = np.hstack([dH_dy_map, dH_dy_enh])
        d2H_dx2 = np.hstack([d2H_dx2_map, d2H_dx2_enh])
        d2H_dy2 = np.hstack([d2H_dy2_map, d2H_dy2_enh])

        return dH_dx, dH_dy, d2H_dx2, d2H_dy2

    # ---- Jacobian 计算 ----

    def _jacobian_analytical(self, H, dH_dx, dH_dy, d2H_dx2, d2H_dy2,
                             u, u_x, u_y, u_xx, u_yy, x, y):
        """解析 Jacobian: J[i,j] = Σ (∂R/∂v_k) · (∂v_k/∂β_j)"""
        J = np.zeros_like(H)

        if self.dR_du is not None:
            J += self.dR_du(u, u_x, u_y, u_xx, u_yy, x, y).reshape(-1, 1) * H
        if self.dR_dux is not None:
            J += self.dR_dux(u, u_x, u_y, u_xx, u_yy, x, y).reshape(-1, 1) * dH_dx
        if self.dR_duy is not None:
            J += self.dR_duy(u, u_x, u_y, u_xx, u_yy, x, y).reshape(-1, 1) * dH_dy
        if self.dR_duxx is not None:
            J += self.dR_duxx(u, u_x, u_y, u_xx, u_yy, x, y).reshape(-1, 1) * d2H_dx2
        if self.dR_duyy is not None:
            J += self.dR_duyy(u, u_x, u_y, u_xx, u_yy, x, y).reshape(-1, 1) * d2H_dy2

        return J

    def _jacobian_numerical(self, H, dH_dx, dH_dy, d2H_dx2, d2H_dy2,
                            u, u_x, u_y, u_xx, u_yy, x, y, eps=1e-7):
        """数值 Jacobian (当解析偏导未提供时使用)

        利用 ∂R/∂β_j ≈ [R(β+εe_j) - R(β)] / ε
        其中扰动 β_j 等价于同时扰动 u, u_x, ...:
          Δu = ε·H[:,j],  Δu_x = ε·dH_dx[:,j], ...
        """
        R0 = self.residual_fn(u, u_x, u_y, u_xx, u_yy, x, y)
        M = H.shape[1]
        J = np.zeros((len(x), M))

        for j in range(M):
            R_pert = self.residual_fn(
                u + eps * H[:, j],
                u_x + eps * dH_dx[:, j],
                u_y + eps * dH_dy[:, j],
                u_xx + eps * d2H_dx2[:, j],
                u_yy + eps * d2H_dy2[:, j],
                x, y,
            )
            J[:, j] = (R_pert - R0) / eps

        return J

    # ---- 非线性损失 ----

    def _compute_nonlinear_loss(self, pde_data, bc_data):
        """计算非线性PDE残差总损失 (用于 fit_hybrid 梯度步)"""
        x_pde, y_pde = pde_data
        x_bc, y_bc = bc_data

        H_pde, z_pde = self._build_features(x_pde, y_pde)
        dH_dx, dH_dy, d2H_dx2, d2H_dy2 = self._compute_all_derivatives(
            x_pde, y_pde, z_pde
        )
        H_bc, _ = self._build_features(x_bc, y_bc)

        u = (H_pde @ self.beta).flatten()
        u_x = (dH_dx @ self.beta).flatten()
        u_y = (dH_dy @ self.beta).flatten()
        u_xx = (d2H_dx2 @ self.beta).flatten()
        u_yy = (d2H_dy2 @ self.beta).flatten()

        R_pde = self.residual_fn(u, u_x, u_y, u_xx, u_yy, x_pde, y_pde)
        R_bc = (H_bc @ self.beta).flatten() - self.bc_fn(x_bc, y_bc)

        return np.mean(R_pde ** 2) + self.lambda_bc * np.mean(R_bc ** 2)

    # ---- Newton-伪逆迭代 ----

    def fit(self, pde_data, bc_data, max_iter=50, tol=1e-10,
            damping=1.0, mu=0.0):
        """Newton-伪逆迭代求解非线性 PDE

        Parameters
        ----------
        pde_data : tuple (x_pde, y_pde)
            PDE 配点坐标
        bc_data : tuple (x_bc, y_bc)
            边界点坐标
        max_iter : int
            Newton 最大迭代次数
        tol : float
            收敛容差 (||δβ|| < tol)
        damping : float
            阻尼因子 (0, 1], 1 为标准 Newton
        mu : float
            Levenberg-Marquardt 正则化参数 (> 0 增加稳定性)

        Returns
        -------
        beta : ndarray
        """
        x_pde, y_pde = pde_data
        x_bc, y_bc = bc_data

        # 构建特征矩阵 (固定内部权重)
        H_pde, z_pde = self._build_features(x_pde, y_pde)
        dH_dx, dH_dy, d2H_dx2, d2H_dy2 = self._compute_all_derivatives(
            x_pde, y_pde, z_pde
        )
        H_bc, _ = self._build_features(x_bc, y_bc)
        b_bc = self.bc_fn(x_bc, y_bc)

        M = H_pde.shape[1]

        # 初始化 beta: 用边界条件最小二乘估计
        if self.beta is None:
            self.beta = pinv(H_bc) @ b_bc.reshape(-1, 1)

        self.loss_history = []

        has_analytical = any(
            fn is not None
            for fn in [self.dR_du, self.dR_dux, self.dR_duy,
                       self.dR_duxx, self.dR_duyy]
        )

        for k in range(max_iter):
            # 当前近似解及其偏导
            u = (H_pde @ self.beta).flatten()
            u_x = (dH_dx @ self.beta).flatten()
            u_y = (dH_dy @ self.beta).flatten()
            u_xx = (d2H_dx2 @ self.beta).flatten()
            u_yy = (d2H_dy2 @ self.beta).flatten()

            # PDE 残差
            R_pde = self.residual_fn(u, u_x, u_y, u_xx, u_yy, x_pde, y_pde)
            # BC 残差
            R_bc = (H_bc @ self.beta).flatten() - b_bc

            # Jacobian
            if has_analytical:
                J_pde = self._jacobian_analytical(
                    H_pde, dH_dx, dH_dy, d2H_dx2, d2H_dy2,
                    u, u_x, u_y, u_xx, u_yy, x_pde, y_pde,
                )
            else:
                J_pde = self._jacobian_numerical(
                    H_pde, dH_dx, dH_dy, d2H_dx2, d2H_dy2,
                    u, u_x, u_y, u_xx, u_yy, x_pde, y_pde,
                )

            J_bc = H_bc

            # 组装加权系统
            lam = self.lambda_bc
            J = np.vstack([J_pde, lam * J_bc])
            R = np.concatenate([R_pde, lam * R_bc])

            # Newton 更新 (可选 Levenberg-Marquardt 正则化)
            if mu > 0:
                JTJ = J.T @ J + mu * np.eye(M)
                delta = -np.linalg.solve(JTJ, J.T @ R.reshape(-1, 1))
            else:
                delta = -pinv(J) @ R.reshape(-1, 1)

            self.beta += damping * delta

            # 损失记录
            loss_pde = np.mean(R_pde ** 2)
            loss_bc = np.mean(R_bc ** 2)
            total_loss = loss_pde + lam * loss_bc
            self.loss_history.append(total_loss)

            delta_norm = np.linalg.norm(delta)
            if self._verbose and k % max(1, max_iter // 10) == 0:
                print(
                    f"[NL-PIBLS] Newton iter {k:>3d}: "
                    f"L_pde={loss_pde:.4e}  L_bc={loss_bc:.4e}  "
                    f"|δβ|={delta_norm:.4e}"
                )

            if delta_norm < tol:
                if self._verbose:
                    print(f"[NL-PIBLS] Newton converged at iter {k}")
                break

        return self.beta

    # ---- 混合模式: Newton-伪逆 + 特征学习 ----

    def fit_hybrid(self, pde_data, bc_data,
                   outer_iters=10, inner_iters=20,
                   lr=0.005, tol=1e-12, damping=0.8, mu=0.0,
                   grad_method='spsa', verbose=True):
        """混合模式: 外层特征学习 + 内层 Newton-伪逆"""
        self.loss_history = []
        best_beta = None
        best_loss = np.inf

        # 首轮初始化
        self.beta = None
        self._verbose = verbose
        self.fit(pde_data, bc_data, max_iter=inner_iters, damping=damping, mu=mu)
        loss = self._compute_nonlinear_loss(pde_data, bc_data)
        self.loss_history.append(loss)
        if verbose:
            print(f"[NL-Hybrid] Outer   0: loss = {loss:.6e}")

        # 保存最优
        best_loss = loss
        best_beta = self.beta.copy()
        best_W_map = self.W_map.copy()
        best_B_map = self.B_map.copy()
        best_W_enhance = self.W_enhance.copy()
        best_B_enhance = self.B_enhance.copy()

        for outer in range(1, outer_iters):
            # 梯度下降更新内部权重
            old_verbose = self._verbose
            self._verbose = False
            self._feature_gradient_step(
                pde_data, bc_data, lr=lr / (1 + 0.1 * outer),
                inner_iters=inner_iters, damping=damping, mu=mu,
            )
            self._verbose = old_verbose

            loss = self._compute_nonlinear_loss(pde_data, bc_data)
            self.loss_history.append(loss)

            if verbose:
                print(f"[NL-Hybrid] Outer {outer:>3d}: loss = {loss:.6e}")

            if loss < best_loss:
                best_loss = loss
                best_beta = self.beta.copy()
                best_W_map = self.W_map.copy()
                best_B_map = self.B_map.copy()
                best_W_enhance = self.W_enhance.copy()
                best_B_enhance = self.B_enhance.copy()

            if loss < tol:
                break
        best_loss = np.inf

        for outer in range(outer_iters):
            # === Newton-伪逆求解 beta (保留 warm start) ===
            if outer == 0:
                self.beta = None  # 首轮从 BC 初始化
            # 后续轮次保留上一轮 beta 作为热启动
            self.fit(
                pde_data, bc_data,
                max_iter=inner_iters, tol=tol,
                damping=damping, mu=mu,
            )

            loss = self._compute_nonlinear_loss(pde_data, bc_data)
            self.loss_history.append(loss)
            print(f"[NL-Hybrid] Outer {outer:>3d}: loss = {loss:.6e}")

            # 记录最优解
            if loss < best_loss:
                best_loss = loss
                best_beta = self.beta.copy()
                best_W_map = self.W_map.copy()
                best_B_map = self.B_map.copy()
                best_W_enhance = self.W_enhance.copy()
                best_B_enhance = self.B_enhance.copy()

            if loss < tol:
                print(f"[NL-Hybrid] Converged at outer iter {outer}")
                break

            # === 梯度下降更新内部权重 ===
            self._feature_gradient_step(pde_data, bc_data, lr, grad_method)

        # 恢复最优解
        self.beta = best_beta
        self.W_map = best_W_map
        self.B_map = best_B_map
        self.W_enhance = best_W_enhance
        self.B_enhance = best_B_enhance

        if verbose:
            print(f"[NL-Hybrid] Best loss = {best_loss:.6e}")
        return self.beta

    def _feature_gradient_step(self, pde_data, bc_data, lr, method='spsa',
                               eps_spsa=0.003, n_spsa=10, inner_iters=20,
                               damping=0.8, mu=0.0):
        """更新 W_map, B_map, W_enhance, B_enhance 以降低残差

        关键：每次扰动后做完整 Newton 求解 beta，
        这样梯度反映的是"改变特征后最优 beta 下的损失变化"
        """
        # 保存状态
        saved = {}
        for name in ['W_map', 'B_map', 'W_enhance', 'B_enhance']:
            saved[name] = getattr(self, name).copy()
        saved_beta = self.beta.copy()
        loss_before = self._compute_nonlinear_loss(pde_data, bc_data)

        for name in ['W_map', 'B_map', 'W_enhance', 'B_enhance']:
            param = getattr(self, name)
            grad = np.zeros_like(param)

            for _ in range(n_spsa):
                delta = np.random.choice([-1.0, 1.0], size=param.shape)

                param += eps_spsa * delta
                self.beta = None
                self.fit(pde_data, bc_data, max_iter=inner_iters,
                         damping=damping, mu=mu)
                lp = self._compute_nonlinear_loss(pde_data, bc_data)

                param -= 2.0 * eps_spsa * delta
                self.beta = None
                self.fit(pde_data, bc_data, max_iter=inner_iters,
                         damping=damping, mu=mu)
                lm = self._compute_nonlinear_loss(pde_data, bc_data)

                param += eps_spsa * delta  # 还原
                grad += (lp - lm) / (2.0 * eps_spsa * delta)

            grad /= n_spsa

            # 梯度裁剪
            gn = np.linalg.norm(grad)
            if gn > 1.0:
                grad = grad * (1.0 / gn)

            setattr(self, name, param - lr * grad)

        # 更新后 Newton 求解并判断是否回退
        self.beta = None
        self.fit(pde_data, bc_data, max_iter=inner_iters, damping=damping, mu=mu)
        loss_after = self._compute_nonlinear_loss(pde_data, bc_data)

        if loss_after > loss_before * 1.1:
            for name, val in saved.items():
                setattr(self, name, val)
            self.beta = saved_beta

    # ---- 预测 ----

    def predict(self, x, y):
        if self.beta is None:
            raise ValueError('Model not trained. Call fit() first.')
        H, _ = self._build_features(x, y)
        return (H @ self.beta).flatten()

    def predict_derivatives(self, x, y):
        """预测 u(x,y) 及其偏导数"""
        H, z = self._build_features(x, y)
        dH_dx, dH_dy, d2H_dx2, d2H_dy2 = self._compute_all_derivatives(x, y, z)

        u = (H @ self.beta).flatten()
        u_x = (dH_dx @ self.beta).flatten()
        u_y = (dH_dy @ self.beta).flatten()
        u_xx = (d2H_dx2 @ self.beta).flatten()
        u_yy = (d2H_dy2 @ self.beta).flatten()

        return u, u_x, u_y, u_xx, u_yy
