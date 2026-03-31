import numpy as np
from scipy.linalg import pinv


class PIBLS:
    """物理信息宽度学习系统（扩散方程版本）"""

    def __init__(
        self,
        N1,
        N2,
        map_func='tanh',
        enhance_func='sigmoid',
        source_fn=None,
        exact_solution_fn=None,
        multi_activation=False,
    ):
        self.N1 = int(N1)
        self.N2 = int(N2)

        # 通过依赖注入避免模块耦合到具体 PDE。
        self.source_fn = source_fn
        self.exact_solution_fn = exact_solution_fn

        # 多激活集成 (identity, tanh, ReLU, sine)
        self.multi_activation = multi_activation
        if multi_activation:
            self._act_groups = np.array_split(np.arange(self.N1), 4)

        # 获取激活函数
        self.map_act_name, self.map_activation = self._get_activation(map_func)
        self.enhance_act_name, self.enhance_activation = self._get_activation(enhance_func)

        # 获取导数函数
        self.map_derivative = self._get_derivative(map_func)
        self.map_second_derivative = self._get_second_derivative(map_func)
        self.enhance_derivative = self._get_derivative(enhance_func)
        self.enhance_second_derivative = self._get_second_derivative(enhance_func)

        self.W_map = None
        self.B_map = np.random.randn(self.N1)
        self.W_enhance = np.random.randn(self.N1, self.N2)
        self.B_enhance = np.random.randn(self.N2)

        self.beta = None
        self.is_initialized = False

    def _get_activation(self, activation):
        """获取激活函数"""
        activations = {
            'relu': ('relu', lambda x: np.maximum(0, x)),
            'tanh': ('tanh', lambda x: np.tanh(x)),
            'sigmoid': ('sigmoid', lambda x: 1 / (1 + np.exp(-x))),
            'linear': ('linear', lambda x: x)
        }
        return activations.get(activation.lower(), activations['tanh'])

    def _get_derivative(self, activation):
        """获取一阶导数函数"""
        derivatives = {
            'tanh': lambda x: 1 - np.tanh(x) ** 2,
            'relu': lambda x: np.where(x > 0, 1, 0),
            'sigmoid': lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))),
            'linear': lambda x: np.ones_like(x)
        }
        return derivatives.get(activation.lower(), derivatives['tanh'])

    def _get_second_derivative(self, activation):
        """获取二阶导数函数"""
        second_derivatives = {
            'tanh': lambda x: -2 * np.tanh(x) * (1 - np.tanh(x) ** 2),
            'relu': lambda x: np.zeros_like(x),
            'sigmoid': lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))) * (1 - 2 / (1 + np.exp(-x))),
            'linear': lambda x: np.zeros_like(x)
        }
        return second_derivatives.get(activation.lower(), second_derivatives['tanh'])

    # ---- 多激活集成 (论文核心: identity/tanh/ReLU/sine) ----

    def _multi_act_forward(self, Z):
        """多激活前向: 不同节点组使用不同激活函数"""
        H = np.empty_like(Z)
        g = self._act_groups
        H[:, g[0]] = Z[:, g[0]]                        # identity: 线性漂移
        H[:, g[1]] = np.tanh(Z[:, g[1]])                # tanh: 平滑非线性
        H[:, g[2]] = np.maximum(0, Z[:, g[2]])          # ReLU: 稀疏特征
        H[:, g[3]] = np.sin(Z[:, g[3]])                 # sine: 周期模式
        return H

    def _multi_act_derivative(self, Z):
        """多激活一阶导数"""
        dH = np.empty_like(Z)
        g = self._act_groups
        dH[:, g[0]] = 1.0
        dH[:, g[1]] = 1 - np.tanh(Z[:, g[1]]) ** 2
        dH[:, g[2]] = (Z[:, g[2]] > 0).astype(float)
        dH[:, g[3]] = np.cos(Z[:, g[3]])
        return dH

    def _multi_act_second_derivative(self, Z):
        """多激活二阶导数"""
        ddH = np.zeros_like(Z)
        g = self._act_groups
        # identity'' = 0, ReLU'' = 0
        t = np.tanh(Z[:, g[1]])
        ddH[:, g[1]] = -2 * t * (1 - t ** 2)
        ddH[:, g[3]] = -np.sin(Z[:, g[3]])
        return ddH

    def _build_features(self, x, y):
        X_bias = np.column_stack([x, y, np.ones_like(x)])
        if not self.is_initialized:
            self._initialize_weights(X_bias)

        # 映射层
        Z_map = X_bias @ self.W_map + self.B_map
        if self.multi_activation:
            H_map = self._multi_act_forward(Z_map)
        else:
            H_map = self.map_activation(Z_map)

        # 增强层
        Z_enhance = H_map @ self.W_enhance + self.B_enhance
        H_enhance = self.enhance_activation(Z_enhance)

        return np.hstack([H_map, H_enhance]), (Z_map, Z_enhance)

    def _initialize_weights(self, X_bias):
        init_W = np.random.randn(3, self.N1)
        self.W_map = self.sparse_bls(X_bias, X_bias @ init_W)
        self.is_initialized = True

    def shrinkage(self, a, b):
        return np.sign(a) * np.maximum(np.abs(a) - b, 0)

    def sparse_bls(self, A, b):
        lam = 0.001
        itrs = 50
        AA = A.T.dot(A)
        m = A.shape[1]
        n = b.shape[1]
        x1 = np.zeros([m, n])
        wk = ok = uk = x1
        L1 = np.linalg.inv(AA + np.eye(m))
        L2 = L1.dot(A.T).dot(b)
        for _ in range(itrs):
            ck = L2 + L1.dot(ok - uk)
            ok = self.shrinkage(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk

    def _compute_derivatives(self, x, y, z_values):
        Z_map, Z_enhance = z_values

        # 映射层导数
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

        # 增强层导数
        dH_enhance = self.enhance_derivative(Z_enhance)
        ddH_enhance = self.enhance_second_derivative(Z_enhance)

        # 一阶导数 (链式法则)
        dH_dx_enhance = dH_enhance * (dH_dx_map @ self.W_enhance)
        dH_dy_enhance = dH_enhance * (dH_dy_map @ self.W_enhance)

        # 二阶导数 (链式法则)
        d2H_dx2_enhance = ddH_enhance * (dH_dx_map @ self.W_enhance) ** 2 + dH_enhance * (d2H_dx2_map @ self.W_enhance)

        d2H_dy2_enhance = ddH_enhance * (dH_dy_map @ self.W_enhance) ** 2 + dH_enhance * (d2H_dy2_map @ self.W_enhance)

        # 合并特征
        d2H_dx2 = np.hstack([d2H_dx2_map, d2H_dx2_enhance])
        d2H_dy2 = np.hstack([d2H_dy2_map, d2H_dy2_enhance])

        return d2H_dx2, d2H_dy2

    def build_system(self, pde_data, bc_data):
        if self.source_fn is None or self.exact_solution_fn is None:
            raise ValueError('Please provide source_fn and exact_solution_fn when creating PIBLS.')

        x_pde, y_pde = pde_data
        x_bc, y_bc = bc_data

        # 扩散方程：u_xx + u_yy = R
        H_pde, z_pde = self._build_features(x_pde, y_pde)
        d2H_dx2, d2H_dy2 = self._compute_derivatives(x_pde, y_pde, z_pde)

        A_pde = d2H_dx2 + d2H_dy2
        b_pde = self.source_fn(x_pde, y_pde)

        # 边界条件
        H_bc, _ = self._build_features(x_bc, y_bc)
        b_bc = self.exact_solution_fn(x_bc, y_bc)

        A_matrix = np.vstack([A_pde, H_bc])
        b_vector = np.concatenate([b_pde, b_bc])

        return A_matrix, b_vector

    def fit(self, pde_data, bc_data):
        A, b = self.build_system(pde_data, bc_data)
        self.beta = pinv(A) @ b.reshape(-1, 1)
        return self.beta

    def predict(self, x, y):
        if self.beta is None:
            raise ValueError('Model not trained. Call fit() first.')
        H, _ = self._build_features(x, y)
        return (H @ self.beta).flatten()
