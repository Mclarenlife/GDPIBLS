"""
对比实验：PIBLS 系列 vs Deep PINN

在多个 PDE 基准问题上系统对比：
  - 标准 PIBLS (单次伪逆)
  - HybridPIBLS (伪逆 + 梯度下降)
  - NonlinearPIBLS (Newton-伪逆迭代)
  - Deep PINN (2层/4层/6层, PyTorch)

评价指标：RMSE, 训练时间(s), 收敛曲线
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pibls_model import PIBLS
from advanced_pibls import HybridPIBLS, NonlinearPIBLS

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cpu')


# =====================================================================
#  Deep PINN  (PyTorch)
# =====================================================================

class DeepPINN(nn.Module):
    """标准 Deep Physics-Informed Neural Network

    Parameters
    ----------
    layers : list[int]
        网络结构, e.g. [2, 64, 64, 1] 表示 2层隐藏层各64节点
    activation : str
        激活函数: 'tanh' (默认, PINN 标准选择)
    """

    def __init__(self, layers, activation='tanh'):
        super().__init__()
        self.depth = len(layers) - 2  # 隐藏层数

        # 构建网络层
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # 非输出层加激活
                if activation == 'tanh':
                    net.append(nn.Tanh())
                elif activation == 'relu':
                    net.append(nn.ReLU())
                elif activation == 'sin':
                    net.append(SinActivation())
        self.net = nn.Sequential(*net)

        # Xavier 初始化
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        """前向传播, 输入为 (N,) 张量"""
        inp = torch.stack([x, y], dim=1)  # (N, 2)
        return self.net(inp).squeeze(-1)   # (N,)


class SinActivation(nn.Module):
    """Sine 激活函数"""
    def forward(self, x):
        return torch.sin(x)


class PINNSolver:
    """Deep PINN 求解器：封装训练、预测、计时

    Parameters
    ----------
    layers : list[int]
        网络结构
    pde_residual_fn : callable(model, x, y) -> residual tensor
        PDE 残差函数 (使用 torch.autograd 计算导数)
    bc_fn : callable(x_np, y_np) -> bc_values_np
        Dirichlet 边界条件 (numpy)
    lr : float
        Adam 学习率
    epochs : int
        训练轮数
    lambda_bc : float
        边界损失权重
    activation : str
        激活函数
    """

    def __init__(self, layers, pde_residual_fn, bc_fn,
                 lr=1e-3, epochs=5000, lambda_bc=10.0,
                 activation='tanh'):
        self.model = DeepPINN(layers, activation).to(DEVICE)
        self.pde_residual_fn = pde_residual_fn
        self.bc_fn = bc_fn
        self.lr = lr
        self.epochs = epochs
        self.lambda_bc = lambda_bc
        self.loss_history = []
        self.train_time = 0.0

    def train(self, pde_data, bc_data):
        """训练 PINN

        Parameters
        ----------
        pde_data : tuple (x_np, y_np) 内部配点
        bc_data  : tuple (x_np, y_np) 边界点
        """
        x_pde_np, y_pde_np = pde_data
        x_bc_np, y_bc_np = bc_data

        # 转张量 (requires_grad 用于自动微分)
        x_pde = torch.tensor(x_pde_np, dtype=torch.float64,
                             requires_grad=True, device=DEVICE)
        y_pde = torch.tensor(y_pde_np, dtype=torch.float64,
                             requires_grad=True, device=DEVICE)
        x_bc = torch.tensor(x_bc_np, dtype=torch.float64, device=DEVICE)
        y_bc = torch.tensor(y_bc_np, dtype=torch.float64, device=DEVICE)
        u_bc_target = torch.tensor(
            self.bc_fn(x_bc_np, y_bc_np),
            dtype=torch.float64, device=DEVICE,
        )

        # 使用 float64 以公平对比
        self.model.double()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01,
        )

        self.loss_history = []
        t0 = time.perf_counter()

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # PDE 残差
            R = self.pde_residual_fn(self.model, x_pde, y_pde)
            loss_pde = torch.mean(R ** 2)

            # 边界损失
            u_bc_pred = self.model(x_bc, y_bc)
            loss_bc = torch.mean((u_bc_pred - u_bc_target) ** 2)

            loss = loss_pde + self.lambda_bc * loss_bc
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step()

            self.loss_history.append(loss.item())

            if epoch % max(1, self.epochs // 10) == 0:
                print(f"  [PINN] Epoch {epoch:>5d}: "
                      f"L_pde={loss_pde.item():.4e}  "
                      f"L_bc={loss_bc.item():.4e}  "
                      f"total={loss.item():.4e}")

        self.train_time = time.perf_counter() - t0
        print(f"  [PINN] Training done: {self.train_time:.2f}s, "
              f"final loss={self.loss_history[-1]:.4e}")

    @torch.no_grad()
    def predict(self, x_np, y_np):
        """预测 (numpy in/out)"""
        self.model.eval()
        x = torch.tensor(x_np, dtype=torch.float64, device=DEVICE)
        y = torch.tensor(y_np, dtype=torch.float64, device=DEVICE)
        return self.model(x, y).cpu().numpy()

    def param_count(self):
        return sum(p.numel() for p in self.model.parameters())


# =====================================================================
#  PDE 残差函数 (用于 Deep PINN, 调用 torch.autograd)
# =====================================================================

def poisson_residual(model, x, y, source_fn_torch):
    """Poisson: Δu - f = 0"""
    u = model(x, y)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return u_xx + u_yy - source_fn_torch(x, y)


def nonlinear_residual(model, x, y, source_fn_torch):
    """-Lap(u) + u^3 - f = 0"""
    u = model(x, y)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return -u_xx - u_yy + u ** 3 - source_fn_torch(x, y)


def helmholtz_residual(model, x, y, k, source_fn_torch):
    """Helmholtz: Δu + k²u - f = 0"""
    u = model(x, y)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return u_xx + u_yy + k ** 2 * u - source_fn_torch(x, y)


# =====================================================================
#  数据生成工具
# =====================================================================

def generate_interior(n, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
    x = np.random.uniform(xmin, xmax, n)
    y = np.random.uniform(ymin, ymax, n)
    return x, y


def generate_boundary(n_per_side, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
    t = np.linspace(0, 1, n_per_side)
    x_bc = np.concatenate([
        xmin + (xmax - xmin) * t,  # bottom
        xmin + (xmax - xmin) * t,  # top
        np.full(n_per_side, xmin),  # left
        np.full(n_per_side, xmax),  # right
    ])
    y_bc = np.concatenate([
        np.full(n_per_side, ymin),
        np.full(n_per_side, ymax),
        ymin + (ymax - ymin) * t,
        ymin + (ymax - ymin) * t,
    ])
    return x_bc, y_bc


def generate_test_grid(n=50, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
    xx = np.linspace(xmin, xmax, n)
    yy = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(xx, yy)
    return X.flatten(), Y.flatten(), X, Y


def compute_rmse(u_pred, u_exact):
    return np.sqrt(np.mean((u_pred - u_exact) ** 2))


# =====================================================================
#  Problem 1: 低频 Poisson (线性, 简单)
# =====================================================================

def problem1_poisson_lowfreq():
    """Δu = f, u = sin(πx)sin(πy), f = -2π²sin(πx)sin(πy)"""
    print("\n" + "=" * 75)
    print("  Problem 1: 低频 Poisson 方程  Δu = f  (线性, 简单)")
    print("  精确解: u = sin(πx)·sin(πy)")
    print("=" * 75)

    exact = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    source = lambda x, y: -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    source_torch = lambda x, y: -2 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)

    n_pde, n_bc_side = 800, 60
    x_pde, y_pde = generate_interior(n_pde)
    x_bc, y_bc = generate_boundary(n_bc_side)
    pde_data = (x_pde, y_pde)
    bc_data = (x_bc, y_bc)
    x_test, y_test, X, Y = generate_test_grid(50)
    u_exact = exact(x_test, y_test)

    results = {}

    # ---- PIBLS (N=30) ----
    print("\n--- 标准 PIBLS (N1=N2=30, tanh) ---")
    np.random.seed(42)
    t0 = time.perf_counter()
    m = PIBLS(30, 30, 'tanh', 'sigmoid', source, exact)
    m.fit(pde_data, bc_data)
    t_pibls = time.perf_counter() - t0
    u_pred = m.predict(x_test, y_test)
    rmse = compute_rmse(u_pred, u_exact)
    results['PIBLS'] = {'rmse': rmse, 'time': t_pibls, 'params': 30+30}
    print(f"  RMSE = {rmse:.6e}, Time = {t_pibls:.3f}s")

    # ---- Deep PINN ----
    for n_layers, width, epochs in [(2, 64, 3000), (4, 64, 3000), (6, 64, 5000)]:
        layers = [2] + [width] * n_layers + [1]
        label = f'PINN-{n_layers}L-{width}W'
        print(f"\n--- {label} (epochs={epochs}) ---")
        torch.manual_seed(42)

        sf = source_torch  # 闭包捕获
        pinn = PINNSolver(
            layers,
            pde_residual_fn=lambda mdl, x, y, _sf=sf: poisson_residual(mdl, x, y, _sf),
            bc_fn=exact,
            lr=1e-3, epochs=epochs, lambda_bc=10.0,
        )
        pinn.train(pde_data, bc_data)
        u_pred_pinn = pinn.predict(x_test, y_test)
        rmse_pinn = compute_rmse(u_pred_pinn, u_exact)
        results[label] = {
            'rmse': rmse_pinn,
            'time': pinn.train_time,
            'params': pinn.param_count(),
            'loss_history': pinn.loss_history,
        }
        print(f"  RMSE = {rmse_pinn:.6e}, Params = {pinn.param_count()}")

    print_results_table("Problem 1: 低频 Poisson", results)
    plot_comparison(results, 'benchmark_p1_poisson_lowfreq.png',
                    'Problem 1: Low-freq Poisson')
    return results


# =====================================================================
#  Problem 2: 高频 Poisson (线性, 困难)
# =====================================================================

def problem2_poisson_highfreq():
    """高频: u = sin(3πx)sin(3πy) + 0.5sin(πx)sin(πy)"""
    print("\n" + "=" * 75)
    print("  Problem 2: 高频 Poisson 方程  Δu = f  (线性, 困难)")
    print("  精确解: u = sin(3πx)·sin(3πy) + 0.5·sin(πx)·sin(πy)")
    print("=" * 75)

    exact = lambda x, y: (np.sin(3*np.pi*x) * np.sin(3*np.pi*y)
                           + 0.5 * np.sin(np.pi*x) * np.sin(np.pi*y))
    source = lambda x, y: (-18*np.pi**2 * np.sin(3*np.pi*x) * np.sin(3*np.pi*y)
                            - np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y))
    source_torch = lambda x, y: (-18*np.pi**2 * torch.sin(3*np.pi*x) * torch.sin(3*np.pi*y)
                                  - np.pi**2 * torch.sin(np.pi*x) * torch.sin(np.pi*y))

    n_pde, n_bc_side = 1200, 80
    x_pde, y_pde = generate_interior(n_pde)
    x_bc, y_bc = generate_boundary(n_bc_side)
    pde_data = (x_pde, y_pde)
    bc_data = (x_bc, y_bc)
    x_test, y_test, X, Y = generate_test_grid(50)
    u_exact = exact(x_test, y_test)

    results = {}

    # ---- 标准 PIBLS (N=20, 较少节点暴露困难性) ----
    print("\n--- 标准 PIBLS (N1=N2=20, tanh) ---")
    np.random.seed(42)
    t0 = time.perf_counter()
    m = PIBLS(20, 20, 'tanh', 'sigmoid', source, exact)
    m.fit(pde_data, bc_data)
    t_pibls = time.perf_counter() - t0
    u_pred = m.predict(x_test, y_test)
    rmse = compute_rmse(u_pred, u_exact)
    results['PIBLS'] = {'rmse': rmse, 'time': t_pibls}
    print(f"  RMSE = {rmse:.6e}, Time = {t_pibls:.3f}s")

    # ---- HybridPIBLS ----
    print("\n--- HybridPIBLS (N1=N2=20, SPSA, 50 iters) ---")
    np.random.seed(42)
    t0 = time.perf_counter()
    m_hyb = HybridPIBLS(
        20, 20, 'tanh', 'sigmoid', source, exact,
        lr=0.01, max_iter=50, lambda_bc=10.0,
        grad_method='spsa', n_grad_samples=15,
    )
    m_hyb.fit(pde_data, bc_data)
    t_hyb = time.perf_counter() - t0
    u_pred_hyb = m_hyb.predict(x_test, y_test)
    rmse_hyb = compute_rmse(u_pred_hyb, u_exact)
    results['HybridPIBLS'] = {
        'rmse': rmse_hyb, 'time': t_hyb,
        'loss_history': m_hyb.loss_history,
    }
    print(f"  RMSE = {rmse_hyb:.6e}, Time = {t_hyb:.3f}s")

    # ---- Deep PINN (多种深度) ----
    for n_layers, width, epochs in [(2, 64, 3000), (4, 64, 5000), (6, 64, 5000)]:
        layers = [2] + [width] * n_layers + [1]
        label = f'PINN-{n_layers}L-{width}W'
        print(f"\n--- {label} (epochs={epochs}) ---")
        torch.manual_seed(42)

        sf = source_torch  # 闭包捕获
        pinn = PINNSolver(
            layers,
            pde_residual_fn=lambda mdl, x, y, _sf=sf: poisson_residual(mdl, x, y, _sf),
            bc_fn=exact,
            lr=1e-3, epochs=epochs, lambda_bc=10.0,
        )
        pinn.train(pde_data, bc_data)
        u_pred_pinn = pinn.predict(x_test, y_test)
        rmse_pinn = compute_rmse(u_pred_pinn, u_exact)
        results[label] = {
            'rmse': rmse_pinn,
            'time': pinn.train_time,
            'params': pinn.param_count(),
            'loss_history': pinn.loss_history,
        }
        print(f"  RMSE = {rmse_pinn:.6e}, Params = {pinn.param_count()}")

    print_results_table("Problem 2: 高频 Poisson", results)
    plot_comparison(results, 'benchmark_p2_poisson_highfreq.png',
                    'Problem 2: High-freq Poisson')
    return results


# =====================================================================
#  Problem 3: Nonlinear PDE  -Lap(u) + u^3 = f
# =====================================================================

def problem3_nonlinear():
    """-Lap(u) + u^3 = f, u = sin(pi*x)sin(pi*y)"""
    print("\n" + "=" * 75)
    print("  Problem 3: Nonlinear PDE  -Lap(u) + u^3 = f")
    print("  Exact: u = sin(pi*x)*sin(pi*y)")
    print("=" * 75)

    exact = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    source = lambda x, y: (2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
                            + np.sin(np.pi*x)**3 * np.sin(np.pi*y)**3)
    source_torch = lambda x, y: (2 * np.pi**2 * torch.sin(np.pi*x) * torch.sin(np.pi*y)
                                  + torch.sin(np.pi*x)**3 * torch.sin(np.pi*y)**3)

    def residual_fn(u, u_x, u_y, u_xx, u_yy, x, y):
        return -u_xx - u_yy + u**3 - source(x, y)

    def dR_du(u, u_x, u_y, u_xx, u_yy, x, y):
        return 3.0 * u**2

    def dR_duxx(u, u_x, u_y, u_xx, u_yy, x, y):
        return -np.ones_like(u)

    def dR_duyy(u, u_x, u_y, u_xx, u_yy, x, y):
        return -np.ones_like(u)

    n_pde, n_bc_side = 600, 50
    x_pde, y_pde = generate_interior(n_pde)
    x_bc, y_bc = generate_boundary(n_bc_side)
    pde_data = (x_pde, y_pde)
    bc_data = (x_bc, y_bc)
    x_test, y_test, X, Y = generate_test_grid(50)
    u_exact = exact(x_test, y_test)

    results = {}

    # ---- NonlinearPIBLS (Newton-伪逆) ----
    print("\n--- NonlinearPIBLS (Newton-伪逆, N=40) ---")
    np.random.seed(42)
    t0 = time.perf_counter()
    m_nl = NonlinearPIBLS(
        40, 40, 'tanh', 'sigmoid',
        residual_fn=residual_fn, bc_fn=exact,
        dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
        lambda_bc=20.0,
    )
    m_nl.fit(pde_data, bc_data, max_iter=50, damping=0.8)
    t_nl = time.perf_counter() - t0
    u_pred_nl = m_nl.predict(x_test, y_test)
    rmse_nl = compute_rmse(u_pred_nl, u_exact)
    results['NL-PIBLS'] = {
        'rmse': rmse_nl, 'time': t_nl,
        'loss_history': m_nl.loss_history,
    }
    print(f"  RMSE = {rmse_nl:.6e}, Time = {t_nl:.3f}s")

    # ---- Deep PINN (多种深度) ----
    for n_layers, width, epochs in [(2, 64, 5000), (4, 64, 5000), (6, 64, 5000)]:
        layers = [2] + [width] * n_layers + [1]
        label = f'PINN-{n_layers}L-{width}W'
        print(f"\n--- {label} (epochs={epochs}) ---")
        torch.manual_seed(42)

        sf = source_torch  # 闭包捕获
        pinn = PINNSolver(
            layers,
            pde_residual_fn=lambda mdl, x, y, _sf=sf: nonlinear_residual(mdl, x, y, _sf),
            bc_fn=exact,
            lr=1e-3, epochs=epochs, lambda_bc=20.0,
        )
        pinn.train(pde_data, bc_data)
        u_pred_pinn = pinn.predict(x_test, y_test)
        rmse_pinn = compute_rmse(u_pred_pinn, u_exact)
        results[label] = {
            'rmse': rmse_pinn,
            'time': pinn.train_time,
            'params': pinn.param_count(),
            'loss_history': pinn.loss_history,
        }
        print(f"  RMSE = {rmse_pinn:.6e}, Params = {pinn.param_count()}")

    print_results_table("Problem 3: Nonlinear -Lap(u)+u^3=f", results)
    plot_comparison(results, 'benchmark_p3_nonlinear.png',
                    'Problem 3: Nonlinear -Lap(u)+u^3=f')
    return results


# =====================================================================
#  Problem 4: 强非线性 PDE  -Δu + sin(u) = f  (更难)
# =====================================================================

def problem4_strong_nonlinear():
    """-Δu + sin(u) = f, u = sin(2πx)sin(2πy) (中频 + 强非线性交互)"""
    print("\n" + "=" * 75)
    print("  Problem 4: 强非线性 PDE  -Δu + sin(u) = f  (更难)")
    print("  精确解: u = sin(2πx)·sin(2πy)")
    print("=" * 75)

    exact = lambda x, y: np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    source = lambda x, y: (8 * np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
                            + np.sin(np.sin(2*np.pi*x) * np.sin(2*np.pi*y)))
    source_torch = lambda x, y: (8 * np.pi**2 * torch.sin(2*np.pi*x) * torch.sin(2*np.pi*y)
                                  + torch.sin(torch.sin(2*np.pi*x) * torch.sin(2*np.pi*y)))

    def residual_fn(u, u_x, u_y, u_xx, u_yy, x, y):
        return -u_xx - u_yy + np.sin(u) - source(x, y)

    def dR_du(u, u_x, u_y, u_xx, u_yy, x, y):
        return np.cos(u)

    def dR_duxx(u, u_x, u_y, u_xx, u_yy, x, y):
        return -np.ones_like(u)

    def dR_duyy(u, u_x, u_y, u_xx, u_yy, x, y):
        return -np.ones_like(u)

    n_pde, n_bc_side = 1000, 80
    x_pde, y_pde = generate_interior(n_pde)
    x_bc, y_bc = generate_boundary(n_bc_side)
    pde_data = (x_pde, y_pde)
    bc_data = (x_bc, y_bc)
    x_test, y_test, X, Y = generate_test_grid(50)
    u_exact = exact(x_test, y_test)

    results = {}

    # ---- NonlinearPIBLS ----
    print("\n--- NonlinearPIBLS (Newton-伪逆, N=50) ---")
    np.random.seed(42)
    t0 = time.perf_counter()
    m_nl = NonlinearPIBLS(
        50, 50, 'tanh', 'sigmoid',
        residual_fn=residual_fn, bc_fn=exact,
        dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
        lambda_bc=20.0,
    )
    m_nl.fit(pde_data, bc_data, max_iter=60, damping=0.6)
    t_nl = time.perf_counter() - t0
    u_pred_nl = m_nl.predict(x_test, y_test)
    rmse_nl = compute_rmse(u_pred_nl, u_exact)
    results['NL-PIBLS'] = {
        'rmse': rmse_nl, 'time': t_nl,
        'loss_history': m_nl.loss_history,
    }
    print(f"  RMSE = {rmse_nl:.6e}, Time = {t_nl:.3f}s")

    # ---- Deep PINN ----
    for n_layers, width, epochs in [(2, 64, 5000), (4, 64, 5000), (6, 64, 8000)]:
        layers = [2] + [width] * n_layers + [1]
        label = f'PINN-{n_layers}L-{width}W'
        print(f"\n--- {label} (epochs={epochs}) ---")
        torch.manual_seed(42)

        sf = source_torch  # 闭包捕获
        pinn = PINNSolver(
            layers,
            pde_residual_fn=lambda mdl, x, y, _sf=sf: nonlinear_residual_generic(
                mdl, x, y,
                nonlin_torch=lambda u: torch.sin(u),
                source_fn=_sf,
            ),
            bc_fn=exact,
            lr=1e-3, epochs=epochs, lambda_bc=20.0,
        )
        pinn.train(pde_data, bc_data)
        u_pred_pinn = pinn.predict(x_test, y_test)
        rmse_pinn = compute_rmse(u_pred_pinn, u_exact)
        results[label] = {
            'rmse': rmse_pinn,
            'time': pinn.train_time,
            'params': pinn.param_count(),
            'loss_history': pinn.loss_history,
        }
        print(f"  RMSE = {rmse_pinn:.6e}, Params = {pinn.param_count()}")

    print_results_table("Problem 4: 强非线性 -Δu+sin(u)=f", results)
    plot_comparison(results, 'benchmark_p4_strong_nonlinear.png',
                    'Problem 4: Strong Nonlinear $-\\Delta u + \\sin(u) = f$')
    return results


def nonlinear_residual_generic(model, x, y, nonlin_torch, source_fn):
    """-Δu + g(u) - f = 0, 通用非线性残差"""
    u = model(x, y)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return -u_xx - u_yy + nonlin_torch(u) - source_fn(x, y)


# =====================================================================
#  结果展示工具
# =====================================================================

def print_results_table(title, results):
    """打印格式化结果表"""
    print(f"\n{'─' * 75}")
    print(f"  {title} — 结果汇总")
    print(f"{'─' * 75}")
    print(f"  {'方法':<22s}  {'RMSE':>12s}  {'训练时间(s)':>12s}  {'参数量':>8s}")
    print(f"  {'─'*22}  {'─'*12}  {'─'*12}  {'─'*8}")

    for name, r in results.items():
        rmse_s = f"{r['rmse']:.4e}"
        time_s = f"{r['time']:.2f}" if 'time' in r else '—'
        params_s = str(r.get('params', '—'))
        print(f"  {name:<22s}  {rmse_s:>12s}  {time_s:>12s}  {params_s:>8s}")

    # 找最优
    best = min(results, key=lambda k: results[k]['rmse'])
    print(f"\n  ★ 最优: {best} (RMSE = {results[best]['rmse']:.4e})")
    print(f"{'─' * 75}")


def plot_comparison(results, filename, title):
    """绘制 RMSE 柱状图 + 收敛曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 柱状图
    names = list(results.keys())
    rmses = [results[n]['rmse'] for n in names]
    colors = []
    for n in names:
        if 'PIBLS' in n and 'PINN' not in n:
            colors.append('#FF6B35')  # 橙色 = 我们的方法
        elif 'NL-PIBLS' in n:
            colors.append('#FF6B35')
        else:
            colors.append('#4E89AE')  # 蓝色 = PINN 基线

    bars = axes[0].bar(range(len(names)), rmses, color=colors)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title(f'{title} — RMSE Comparison')
    axes[0].set_yscale('log')
    axes[0].grid(axis='y', alpha=0.3)

    # 在柱上标数值
    for bar, v in zip(bars, rmses):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v * 1.2,
                     f'{v:.2e}', ha='center', va='bottom', fontsize=8)

    # 收敛曲线
    has_curve = False
    for name, r in results.items():
        if 'loss_history' in r and r['loss_history']:
            axes[1].semilogy(r['loss_history'], label=name, linewidth=1.5)
            has_curve = True
    if has_curve:
        axes[1].set_xlabel('Iteration / Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Convergence')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No convergence data', ha='center', va='center',
                     transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  图表已保存: {filename}")


# =====================================================================
#  主程序
# =====================================================================

if __name__ == '__main__':
    print("=" * 75)
    print("  PIBLS 系列 vs Deep PINN  —  系统对比实验")
    print("=" * 75)

    all_results = {}

    # Problem 1: 低频 Poisson (线性, 简单)
    all_results['P1'] = problem1_poisson_lowfreq()

    # Problem 2: 高频 Poisson (线性, 困难)
    all_results['P2'] = problem2_poisson_highfreq()

    # Problem 3: Nonlinear -Lap(u) + u^3 = f
    all_results['P3'] = problem3_nonlinear()

    # Problem 4: 强非线性 -Δu + sin(u) = f
    all_results['P4'] = problem4_strong_nonlinear()

    # ---- 总结 ----
    print("\n" + "=" * 75)
    print("  全局总结")
    print("=" * 75)

    for pname, res in all_results.items():
        best = min(res, key=lambda k: res[k]['rmse'])
        pibls_methods = [k for k in res if 'PIBLS' in k]
        pinn_methods = [k for k in res if 'PINN' in k]

        best_pibls = min(pibls_methods, key=lambda k: res[k]['rmse']) if pibls_methods else None
        best_pinn = min(pinn_methods, key=lambda k: res[k]['rmse']) if pinn_methods else None

        print(f"\n  {pname}:")
        if best_pibls:
            print(f"    PIBLS 最优: {best_pibls} (RMSE={res[best_pibls]['rmse']:.4e})")
        if best_pinn:
            print(f"    PINN  最优: {best_pinn} (RMSE={res[best_pinn]['rmse']:.4e})")
        if best_pibls and best_pinn:
            ratio = res[best_pinn]['rmse'] / res[best_pibls]['rmse']
            if ratio > 1:
                print(f"    → PIBLS 优于 PINN {ratio:.1f}x")
            else:
                print(f"    → PINN 优于 PIBLS {1/ratio:.1f}x")

        # 训练时间对比
        if best_pibls and best_pinn and 'time' in res[best_pibls] and 'time' in res[best_pinn]:
            speedup = res[best_pinn]['time'] / max(res[best_pibls]['time'], 1e-6)
            print(f"    → 训练速度: PIBLS {speedup:.1f}x 快于 PINN")

    print("\n" + "=" * 75)
    print("  实验完成")
    print("=" * 75)
