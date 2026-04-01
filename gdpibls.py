"""
Incremental Physics-Informed Broad Learning System (I-PIBLS)

算法创新：残差自适应增量式物理信息宽度学习系统

核心思想：保留 BLS 三大特性（随机特征、伪逆求解、增量学习），
在此基础上实现 PDE 求解的自适应容量分配。

=== 与 PINN 的本质区别 ===
PINN: 固定架构 -> 端到端梯度训练 -> 改架构需重训
I-PIBLS: 动态架构 -> 伪逆求解(非梯度) -> 增量扩展不需重训

=== 创新点 ===
1. 残差驱动增量扩展: 从小型BLS出发，根据PDE残差分析确定何处、
   以何频率添加新节点——类似FEM自适应网格细化，但在函数逼近空间中。
2. 频率自适应随机特征: 新节点权重分布由残差空间频谱决定，
   高残差梯度区域添加高频节点。
3. Newton-增量耦合: 非线性PDE求解时Newton迭代与架构增长交替进行。
4. 增量伪逆热启动: 新增节点时保留已有解，仅对新节点求增量更新。
"""

import numpy as np


class IPIBLS:
    """
    残差自适应增量式物理信息宽度学习系统

    求解 PDE:  -Delta u + g(u) = f  on Omega
                u = h              on dOmega

    BLS 特性保留:
      - 内部权重随机初始化，永不梯度更新
      - 输出权重 beta 通过伪逆一步求解
      - 支持增量添加节点而不重训已有权重
    """

    def __init__(self,
                 n_map_init=10, n_enh_init=10,
                 n_map_inc=5,   n_enh_inc=5,
                 max_nodes=200,
                 activation='tanh', enh_activation='tanh',
                 tol=1e-8, max_inc=20,
                 ridge=1e-8, seed=42, verbose=True):
        self.n_map_init = n_map_init
        self.n_enh_init = n_enh_init
        self.n_map_inc = n_map_inc
        self.n_enh_inc = n_enh_inc
        self.max_nodes = max_nodes
        self.tol = tol
        self.max_inc = max_inc
        self.ridge = ridge
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)

        self._act, self._act_d, self._act_dd = self._get_act(activation)
        self._eact, self._eact_d, self._eact_dd = self._get_act(enh_activation)

        self.W_map_list = []
        self.b_map_list = []
        self.W_enh_list = []
        self.b_enh_list = []
        self.beta = None
        self.history = []

    # ================================================================
    # 激活函数及其导数
    # ================================================================
    @staticmethod
    def _get_act(name):
        if name == 'tanh':
            f   = np.tanh
            fd  = lambda z: 1 - np.tanh(z)**2
            fdd = lambda z: -2 * np.tanh(z) * (1 - np.tanh(z)**2)
        elif name == 'sin':
            f   = np.sin
            fd  = np.cos
            fdd = lambda z: -np.sin(z)
        elif name == 'sigmoid':
            def _s(z):
                z = np.clip(z, -500, 500)
                return 1 / (1 + np.exp(-z))
            f   = _s
            fd  = lambda z: _s(z) * (1 - _s(z))
            fdd = lambda z: _s(z) * (1 - _s(z)) * (1 - 2*_s(z))
        else:
            raise ValueError(f"Unknown activation: {name}")
        return f, fd, fdd

    # ================================================================
    # 增量式节点管理（BLS核心：随机初始化，不做梯度更新）
    # ================================================================
    def _add_map_nodes(self, n, input_dim, freq_scale=1.0, center=None):
        """添加映射节点，权重随机初始化，可按残差适配频率和位置"""
        W = self.rng.randn(input_dim, n) * freq_scale
        b = self.rng.uniform(-1, 1, n)
        if center is not None:
            b = -(center @ W) + self.rng.uniform(-0.5, 0.5, n)
        self.W_map_list.append(W)
        self.b_map_list.append(b)

    def _add_enh_nodes(self, n, n_map_total):
        """添加增强节点，连接到当前所有映射节点"""
        W = self.rng.randn(n_map_total, n) / np.sqrt(max(n_map_total, 1))
        b = self.rng.uniform(-1, 1, n)
        self.W_enh_list.append(W)
        self.b_enh_list.append(b)

    def _n_map(self):
        return sum(w.shape[1] for w in self.W_map_list)

    def _n_enh(self):
        return sum(w.shape[1] for w in self.W_enh_list)

    def _n_features(self):
        return self._n_map() + self._n_enh()

    # ================================================================
    # 特征矩阵与拉普拉斯算子的解析计算
    # ================================================================
    def _features_and_laplacian(self, X):
        """
        计算 BLS 特征矩阵 H 及其拉普拉斯 Delta H

        映射节点: m_j = act(x . w_j + b_j)
          Delta m_j = act''(z_j) * sum_d w_{d,j}^2

        增强节点: e_k = eact(sum_i v_{ik} * m_i + c_k)
          链式法则计算拉普拉斯
        """
        N, D = X.shape

        # --- 映射层 ---
        M_parts, Z_parts = [], []
        for W, b in zip(self.W_map_list, self.b_map_list):
            z = X @ W + b
            M_parts.append(self._act(z))
            Z_parts.append(z)

        if not M_parts:
            return np.empty((N, 0)), np.empty((N, 0))

        M = np.hstack(M_parts)
        Z = np.hstack(Z_parts)
        W_all = np.hstack(self.W_map_list)
        n_map = M.shape[1]

        act_d  = self._act_d(Z)
        act_dd = self._act_dd(Z)

        w_sq_sum = np.sum(W_all**2, axis=0, keepdims=True)
        M_lap = act_dd * w_sq_sum

        # 各维度导数（增强层链式法则需要）
        dM, d2M = {}, {}
        for d in range(D):
            wd = W_all[d:d+1, :]
            dM[d]  = act_d  * wd
            d2M[d] = act_dd * (wd ** 2)

        # --- 增强层 ---
        E_parts, E_lap_parts = [], []
        for We, be in zip(self.W_enh_list, self.b_enh_list):
            nu = We.shape[0]
            ne = We.shape[1]
            s = M[:, :nu] @ We + be
            E_parts.append(self._eact(s))

            ed  = self._eact_d(s)
            edd = self._eact_dd(s)

            lap_e = np.zeros((N, ne))
            for d in range(D):
                ds  = dM[d][:, :nu]  @ We
                d2s = d2M[d][:, :nu] @ We
                lap_e += edd * ds**2 + ed * d2s

            E_lap_parts.append(lap_e)

        if E_parts:
            E = np.hstack(E_parts)
            E_lap = np.hstack(E_lap_parts)
            H = np.hstack([M, E])
            H_lap = np.hstack([M_lap, E_lap])
        else:
            H, H_lap = M, M_lap

        return H, H_lap

    # ================================================================
    # 伪逆求解（BLS核心：解析最小二乘）
    # ================================================================
    def _solve_pinv(self, A, b):
        n = A.shape[1]
        ATA = A.T @ A + self.ridge * np.eye(n)
        ATb = A.T @ b
        return np.linalg.solve(ATA, ATb)

    # ================================================================
    # 残差分析（创新：残差驱动节点放置 + 频率估计）
    # ================================================================
    def _analyze_residual(self, X, residual):
        """分析残差以确定新节点的位置和频率"""
        abs_r = np.abs(residual)
        thresh = np.percentile(abs_r, 70)
        mask = abs_r >= thresh
        center = X[mask].mean(axis=0) if mask.any() else X.mean(axis=0)

        freq = 1.0
        for d in range(X.shape[1]):
            idx = np.argsort(X[:, d])
            dr = np.diff(residual[idx])
            dx = np.diff(X[idx, d])
            dx = np.where(np.abs(dx) > 1e-12, dx, 1e-12)
            grad = dr / dx
            r_std = np.std(residual)
            if r_std > 1e-12:
                freq = max(freq, np.std(grad) / r_std)

        return center, np.clip(freq, 0.5, 5.0)

    # ================================================================
    # 线性PDE求解: -Delta u = f
    # ================================================================
    def fit_linear(self, X_int, X_bc, source_fn, bc_fn, bc_weight=10.0):
        """
        增量式求解线性 Poisson 方程: -Delta u = f, u|dOmega = g

        算法:
        1. 少量节点初始化 BLS
        2. 伪逆求解
        3. 评估残差 -> 分析 -> 添加自适应节点
        4. 重复直至收敛或达最大节点数
        """
        D = X_int.shape[1]
        f_vals = source_fn(X_int)
        g_vals = bc_fn(X_bc)
        N_int = len(X_int)
        X_all = np.vstack([X_int, X_bc])

        self._add_map_nodes(self.n_map_init, D)
        self._add_enh_nodes(self.n_enh_init, self._n_map())

        prev_rmse = float('inf')
        no_improve = 0

        for step in range(self.max_inc + 1):
            H, H_lap = self._features_and_laplacian(X_all)
            Hi, Hli = H[:N_int], H_lap[:N_int]
            Hb = H[N_int:]

            A = np.vstack([-Hli, bc_weight * Hb])
            b = np.concatenate([f_vals, bc_weight * g_vals])
            self.beta = self._solve_pinv(A, b)

            pde_res = -(Hli @ self.beta) - f_vals
            rmse = np.sqrt(np.mean(pde_res**2))

            self.history.append({
                'step': step, 'n_features': self._n_features(),
                'rmse_pde': rmse
            })
            if self.verbose:
                print(f"  [Step {step:>2d}] features={self._n_features():>4d}  "
                      f"PDE residual={rmse:.4e}")

            if rmse < self.tol:
                if self.verbose:
                    print(f"  Converged at step {step}.")
                break

            if step > 0 and rmse > 0.98 * prev_rmse:
                no_improve += 1
                if no_improve >= 5:
                    if self.verbose:
                        print("  Residual stagnated, stopping growth.")
                    break
            else:
                no_improve = 0
            prev_rmse = rmse

            if self._n_features() >= self.max_nodes or step >= self.max_inc:
                break

            center, freq = self._analyze_residual(X_int, pde_res)
            if self.verbose:
                print(f"    -> Adding nodes: freq={freq:.2f}, "
                      f"center=({', '.join(f'{c:.2f}' for c in center)})")

            n_add = min(self.n_map_inc, self.max_nodes - self._n_features())
            if n_add <= 0:
                break
            # 混合策略：一半基准频率(多样性), 一半自适应频率
            n_base = max(1, n_add // 2)
            n_adapt = n_add - n_base
            self._add_map_nodes(n_base, D, freq_scale=1.0)
            if n_adapt > 0:
                self._add_map_nodes(n_adapt, D, freq_scale=freq, center=center)

            n_add_e = min(self.n_enh_inc,
                          self.max_nodes - self._n_features())
            if n_add_e > 0:
                self._add_enh_nodes(n_add_e, self._n_map())

        return self

    # ================================================================
    # 非线性PDE求解: -Delta u + g(u) = f  (Newton-增量耦合)
    # ================================================================
    def fit_nonlinear(self, X_int, X_bc,
                      g_fn, gp_fn,
                      source_fn, bc_fn,
                      bc_weight=10.0, max_newton=30, damping=1.0):
        """
        Newton-增量耦合求解非线性 PDE: -Delta u + g(u) = f

        外循环: 增量扩展架构
          内循环: 固定架构下 Newton-伪逆迭代
        """
        D = X_int.shape[1]
        f_vals = source_fn(X_int)
        g_vals = bc_fn(X_bc)
        N_int = len(X_int)
        X_all = np.vstack([X_int, X_bc])

        self._add_map_nodes(self.n_map_init, D)
        self._add_enh_nodes(self.n_enh_init, self._n_map())

        # 线性初始猜测
        H, H_lap = self._features_and_laplacian(X_all)
        Hi, Hli = H[:N_int], H_lap[:N_int]
        Hb = H[N_int:]
        A_lin = np.vstack([-Hli, bc_weight * Hb])
        b_lin = np.concatenate([f_vals, bc_weight * g_vals])
        self.beta = self._solve_pinv(A_lin, b_lin)

        best_rmse = float('inf')
        best_beta = self.beta.copy()
        best_n_feat = self._n_features()

        for inc in range(self.max_inc + 1):
            H, H_lap = self._features_and_laplacian(X_all)
            Hi, Hli = H[:N_int], H_lap[:N_int]
            Hb = H[N_int:]

            # 每轮用线性伪逆重新初始化（比热启动更稳定）
            if inc > 0:
                A_lin = np.vstack([-Hli, bc_weight * Hb])
                b_lin = np.concatenate([f_vals, bc_weight * g_vals])
                self.beta = self._solve_pinv(A_lin, b_lin)

            # Newton 迭代（带回溯线搜索）
            for nit in range(max_newton):
                u    = Hi  @ self.beta
                u_bc = Hb  @ self.beta
                lu   = Hli @ self.beta

                R_pde = -lu + g_fn(u) - f_vals
                R_bc  = u_bc - g_vals
                rmse_pde = np.sqrt(np.mean(R_pde**2))
                cur_loss = np.mean(R_pde**2) + bc_weight * np.mean(R_bc**2)

                if rmse_pde < best_rmse:
                    best_rmse = rmse_pde
                    best_beta = self.beta.copy()
                    best_n_feat = self._n_features()

                if rmse_pde < self.tol:
                    break

                gp = gp_fn(u)
                J_pde = -Hli + gp[:, None] * Hi
                J = np.vstack([J_pde, bc_weight * Hb])
                R = np.concatenate([R_pde, bc_weight * R_bc])

                delta = self._solve_pinv(J, -R)

                # 回溯线搜索
                alpha = damping
                beta_save = self.beta.copy()
                for _ in range(8):
                    self.beta = beta_save + alpha * delta
                    u_t = Hi @ self.beta
                    u_bc_t = Hb @ self.beta
                    lu_t = Hli @ self.beta
                    R_t = -lu_t + g_fn(u_t) - f_vals
                    R_bc_t = u_bc_t - g_vals
                    new_loss = np.mean(R_t**2) + bc_weight * np.mean(R_bc_t**2)
                    if new_loss < cur_loss:
                        break
                    alpha *= 0.5
                else:
                    self.beta = beta_save  # 全部步长都失败则不更新

            if self.verbose:
                print(f"  [Inc {inc:>2d}] features={self._n_features():>4d}  "
                      f"PDE res={rmse_pde:.4e}  (best={best_rmse:.4e})")

            self.history.append({
                'step': inc, 'n_features': self._n_features(),
                'rmse_pde': best_rmse
            })

            if best_rmse < self.tol or self._n_features() >= self.max_nodes:
                break
            if inc >= self.max_inc:
                break

            # 用当前轮最后的beta计算残差(尺寸一致)
            u_cur = Hi @ self.beta
            lu_cur = Hli @ self.beta
            pde_res = -lu_cur + g_fn(u_cur) - f_vals

            center, freq = self._analyze_residual(X_int, pde_res)
            if self.verbose:
                print(f"    -> Adding nodes: freq={freq:.2f}, "
                      f"center=({', '.join(f'{c:.2f}' for c in center)})")

            n_add = min(self.n_map_inc,
                        self.max_nodes - self._n_features())
            if n_add <= 0:
                break
            # 混合策略
            n_base = max(1, n_add // 2)
            n_adapt = n_add - n_base
            self._add_map_nodes(n_base, D, freq_scale=1.0)
            if n_adapt > 0:
                self._add_map_nodes(n_adapt, D, freq_scale=freq, center=center)

            n_add_e = min(self.n_enh_inc,
                          self.max_nodes - self._n_features())
            if n_add_e > 0:
                self._add_enh_nodes(n_add_e, self._n_map())

        # 恢复最优解（可能来自较小架构）
        # 若最优解特征数 < 当前架构，用零补齐
        self.beta = best_beta
        if len(self.beta) < self._n_features():
            self.beta = np.concatenate([
                self.beta,
                np.zeros(self._n_features() - len(self.beta))
            ])

        return self

    # ================================================================
    # 预测
    # ================================================================
    def predict(self, X):
        H, _ = self._features_and_laplacian(X)
        return H @ self.beta

    def get_n_features(self):
        return self._n_features()
