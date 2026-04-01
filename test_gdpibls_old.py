"""
GD-PIBLS 对比实验：与 PIBLS、NL-PIBLS、Deep PINN 全面对比

测试问题：
  P1: 低频 Poisson (线性, 简单)     -Δu = 2π²sin(πx)sin(πy)
  P2: 高频 Poisson (线性, 困难)     -Δu = 18π²sin(3πx)sin(3πy) + ...
  P3: 非线性 u³                     -Δu + u³ = f
  P4: 强非线性 sin(u)               -Δu + sin(u) = f

对比方法：
  1. PIBLS (单次伪逆, numpy)
  2. NL-PIBLS (Newton-伪逆, numpy)
  3. GD-PIBLS (伪逆热启动 + Adam + L-BFGS, PyTorch)
  4. Deep PINN-4L-64W (Adam + L-BFGS, PyTorch)
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys

from gdpibls import GDPIBLS, make_grid_data
from pibls_model import PIBLS
from advanced_pibls import NonlinearPIBLS


# ========================================================
# 辅助函数
# ========================================================

def rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))


class DeepPINN(nn.Module):
    """标准 Deep PINN (对比用)"""

    def __init__(self, layers, activation='tanh'):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = torch.tanh if activation == 'tanh' else torch.relu

    def forward(self, x, y):
        h = torch.stack([x, y], dim=1)
        for i, layer in enumerate(self.layers[:-1]):
            h = self.activation(layer(h))
        return self.layers[-1](h).squeeze(-1)


def train_pinn(model, x_pde, y_pde, x_bc, y_bc,
               pde_residual_fn, bc_fn, lambda_bc=10.0,
               epochs_adam=3000, epochs_lbfgs=500,
               lr=1e-3, verbose=True):
    """训练 Deep PINN"""
    t0 = time.time()

    x_pde_t = torch.tensor(x_pde, dtype=torch.float32).requires_grad_(True)
    y_pde_t = torch.tensor(y_pde, dtype=torch.float32).requires_grad_(True)
    x_bc_t = torch.tensor(x_bc, dtype=torch.float32)
    y_bc_t = torch.tensor(y_bc, dtype=torch.float32)
    u_bc_target = torch.tensor(bc_fn(x_bc, y_bc), dtype=torch.float32)

    def compute_loss():
        u = model(x_pde_t, y_pde_t)
        u_x = torch.autograd.grad(u, x_pde_t, torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y_pde_t, torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde_t, torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_pde_t, torch.ones_like(u_y),
                                    create_graph=True, retain_graph=True)[0]
        R = pde_residual_fn(u, u_x, u_y, u_xx, u_yy, x_pde_t, y_pde_t)
        loss_pde = torch.mean(R ** 2)
        u_bc = model(x_bc_t, y_bc_t)
        loss_bc = torch.mean((u_bc - u_bc_target) ** 2)
        return loss_pde + lambda_bc * loss_bc, loss_pde, loss_bc

    # Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs_adam):
        optimizer.zero_grad()
        loss, lp, lb = compute_loss()
        loss.backward()
        optimizer.step()
        if verbose and epoch % 1000 == 0:
            print(f"  PINN Adam {epoch}: loss={loss.item():.4e}")

    # L-BFGS
    if epochs_lbfgs > 0:
        x_pde_t = torch.tensor(x_pde, dtype=torch.float32).requires_grad_(True)
        y_pde_t = torch.tensor(y_pde, dtype=torch.float32).requires_grad_(True)

        opt_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=0.5, max_iter=20,
            history_size=50, line_search_fn='strong_wolfe'
        )
        for _ in range(epochs_lbfgs):
            def closure():
                opt_lbfgs.zero_grad()
                loss, _, _ = compute_loss()
                loss.backward()
                return loss
            opt_lbfgs.step(closure)

    loss_final, _, _ = compute_loss()
    t1 = time.time()
    if verbose:
        print(f"  PINN final loss={loss_final.item():.4e}  time={t1-t0:.2f}s")

    return t1 - t0


def predict_pinn(model, x, y):
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        return model(x_t, y_t).numpy()


# ========================================================
# 测试问题定义
# ========================================================

def run_problem(problem_name, exact_fn, source_fn, bc_fn,
                pde_residual_fn_torch,
                residual_fn_numpy=None,
                dR_du=None, dR_duxx=None, dR_duyy=None,
                is_nonlinear=False,
                N1=40, N2=40,
                nx_pde=30, nx_bc=50):
    """运行一个问题的所有方法对比"""

    print(f"\n{'='*70}")
    print(f" {problem_name}")
    print(f"{'='*70}")

    # 生成数据
    x_pde, y_pde, x_bc, y_bc = make_grid_data(nx_pde, nx_bc)
    x_test, y_test, _, _ = make_grid_data(50, 10)
    u_true = exact_fn(x_test, y_test)

    results = {}

    # -------------------------------------------------------
    # 方法 1: PIBLS (单次伪逆, 仅线性)
    # -------------------------------------------------------
    if not is_nonlinear:
        print(f"\n--- PIBLS (N1={N1}, N2={N2}) ---")
        t0 = time.time()
        model_pibls = PIBLS(
            N1, N2, map_func='tanh', enhance_func='sigmoid',
            source_fn=source_fn, exact_solution_fn=bc_fn
        )
        model_pibls.fit((x_pde, y_pde), (x_bc, y_bc))
        t_pibls = time.time() - t0

        H_test, _ = model_pibls._build_features(x_test, y_test)
        u_pibls = (H_test @ model_pibls.beta).flatten()
        rmse_pibls = rmse(u_pibls, u_true)
        print(f"  RMSE = {rmse_pibls:.6e}  Time = {t_pibls:.3f}s")
        results['PIBLS'] = (rmse_pibls, t_pibls)

    # -------------------------------------------------------
    # 方法 2: NL-PIBLS (Newton-伪逆, 非线性)
    # -------------------------------------------------------
    if is_nonlinear and residual_fn_numpy is not None:
        print(f"\n--- NL-PIBLS (N1={N1}, N2={N2}) ---")
        t0 = time.time()
        model_nl = NonlinearPIBLS(
            N1, N2, map_func='tanh', enhance_func='sigmoid',
            residual_fn=residual_fn_numpy, bc_fn=bc_fn,
            dR_du=dR_du, dR_dux=None, dR_duy=None,
            dR_duxx=dR_duxx, dR_duyy=dR_duyy,
            lambda_bc=10.0
        )
        model_nl.fit((x_pde, y_pde), (x_bc, y_bc),
                      max_iter=50, damping=1.0)
        t_nl = time.time() - t0

        u_nl = model_nl.predict(x_test, y_test)
        rmse_nl = rmse(u_nl, u_true)
        print(f"  RMSE = {rmse_nl:.6e}  Time = {t_nl:.3f}s")
        results['NL-PIBLS'] = (rmse_nl, t_nl)

    # -------------------------------------------------------
    # 方法 3: GD-PIBLS (核心创新)
    # -------------------------------------------------------
    print(f"\n--- GD-PIBLS (N1={N1}, N2={N2}) ---")
    t0 = time.time()
    model_gd = GDPIBLS(N1, N2, multi_activation=False, lambda_bc=10.0)

    history = model_gd.train_model(
        x_pde, y_pde, x_bc, y_bc,
        pde_residual_fn=pde_residual_fn_torch,
        bc_fn=bc_fn,
        source_fn=None,
        epochs_adam=5000,
        epochs_lbfgs=500,
        lr_adam=1e-3,
        warmstart=False,
        verbose=True,
        log_every=500,
    )
    t_gd = time.time() - t0

    u_gd = model_gd.predict(x_test, y_test)
    rmse_gd = rmse(u_gd, u_true)
    print(f"  RMSE = {rmse_gd:.6e}  Time = {t_gd:.2f}s")
    results['GD-PIBLS'] = (rmse_gd, t_gd)

    # -------------------------------------------------------
    # 方法 4: Deep PINN-4L-64W
    # -------------------------------------------------------
    print(f"\n--- Deep PINN (4L-64W) ---")
    pinn = DeepPINN([2, 64, 64, 64, 64, 1])
    t_pinn = train_pinn(
        pinn, x_pde, y_pde, x_bc, y_bc,
        pde_residual_fn_torch, bc_fn,
        epochs_adam=3000, epochs_lbfgs=500,
    )
    u_pinn = predict_pinn(pinn, x_test, y_test)
    rmse_pinn = rmse(u_pinn, u_true)
    print(f"  RMSE = {rmse_pinn:.6e}  Time = {t_pinn:.2f}s")
    results['PINN-4L-64W'] = (rmse_pinn, t_pinn)

    # -------------------------------------------------------
    # 汇总
    # -------------------------------------------------------
    print(f"\n--- {problem_name} Results ---")
    print(f"{'Method':<20s} {'RMSE':<15s} {'Time(s)':<10s}")
    print("-" * 45)
    for name, (r, t) in sorted(results.items(), key=lambda x: x[1][0]):
        print(f"{name:<20s} {r:<15.6e} {t:<10.3f}")

    return results


# ========================================================
# Problem 1: 低频 Poisson
# ========================================================

def run_P1():
    exact = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    source = lambda x, y: -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    bc = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)

    def pde_res_torch(u, u_x, u_y, u_xx, u_yy, x, y):
        f = -2 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)
        return u_xx + u_yy - f  # Δu = f  →  Δu - f = 0

    return run_problem(
        "P1: Low-Freq Poisson (linear)",
        exact, source, bc, pde_res_torch,
        is_nonlinear=False, N1=40, N2=40,
    )


# ========================================================
# Problem 2: 高频 Poisson
# ========================================================

def run_P2():
    exact = lambda x, y: (np.sin(3*np.pi*x) * np.sin(3*np.pi*y)
                          + 0.5 * np.sin(np.pi*x) * np.sin(np.pi*y))
    source = lambda x, y: (-18*np.pi**2 * np.sin(3*np.pi*x) * np.sin(3*np.pi*y)
                           - np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y))
    bc = lambda x, y: exact(x, y)

    def pde_res_torch(u, u_x, u_y, u_xx, u_yy, x, y):
        f = (-18*np.pi**2 * torch.sin(3*np.pi*x) * torch.sin(3*np.pi*y)
             - np.pi**2 * torch.sin(np.pi*x) * torch.sin(np.pi*y))
        return u_xx + u_yy - f

    return run_problem(
        "P2: High-Freq Poisson (linear, hard)",
        exact, source, bc, pde_res_torch,
        is_nonlinear=False, N1=60, N2=60,
    )


# ========================================================
# Problem 3: 非线性 u^3
# ========================================================

def run_P3():
    exact = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    bc = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    source_np = lambda x, y: (2*np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
                               + (np.sin(np.pi*x) * np.sin(np.pi*y))**3)

    # numpy 残差: -Δu + u³ - f = 0  →  -u_xx - u_yy + u³ - f = 0
    def residual_np(u, u_x, u_y, u_xx, u_yy, x, y):
        return -u_xx - u_yy + u**3 - source_np(x, y)

    # torch 残差
    def pde_res_torch(u, u_x, u_y, u_xx, u_yy, x, y):
        f = (2*np.pi**2 * torch.sin(np.pi*x) * torch.sin(np.pi*y)
             + (torch.sin(np.pi*x) * torch.sin(np.pi*y))**3)
        return -u_xx - u_yy + u**3 - f

    # 解析 Jacobian 偏导
    dR_du = lambda u, ux, uy, uxx, uyy, x, y: 3*u**2
    dR_duxx = lambda u, ux, uy, uxx, uyy, x, y: -np.ones_like(u)
    dR_duyy = lambda u, ux, uy, uxx, uyy, x, y: -np.ones_like(u)

    return run_problem(
        "P3: Nonlinear -Laplacian + u^3 = f",
        exact, source_np, bc, pde_res_torch,
        residual_fn_numpy=residual_np,
        dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
        is_nonlinear=True, N1=50, N2=50,
    )


# ========================================================
# Problem 4: 强非线性 sin(u)
# ========================================================

def run_P4():
    exact = lambda x, y: np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    bc = lambda x, y: np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    source_np = lambda x, y: (8*np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
                               + np.sin(np.sin(2*np.pi*x) * np.sin(2*np.pi*y)))

    def residual_np(u, u_x, u_y, u_xx, u_yy, x, y):
        return -u_xx - u_yy + np.sin(u) - source_np(x, y)

    def pde_res_torch(u, u_x, u_y, u_xx, u_yy, x, y):
        f = (8*np.pi**2 * torch.sin(2*np.pi*x) * torch.sin(2*np.pi*y)
             + torch.sin(torch.sin(2*np.pi*x) * torch.sin(2*np.pi*y)))
        return -u_xx - u_yy + torch.sin(u) - f

    dR_du = lambda u, ux, uy, uxx, uyy, x, y: np.cos(u)
    dR_duxx = lambda u, ux, uy, uxx, uyy, x, y: -np.ones_like(u)
    dR_duyy = lambda u, ux, uy, uxx, uyy, x, y: -np.ones_like(u)

    return run_problem(
        "P4: Nonlinear -Laplacian + sin(u) = f",
        exact, source_np, bc, pde_res_torch,
        residual_fn_numpy=residual_np,
        dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
        is_nonlinear=True, N1=60, N2=60,
    )


# ========================================================
# 主函数
# ========================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    all_results = {}

    # 可通过命令行参数选择问题: python test_gdpibls.py 1 3
    args = sys.argv[1:]
    if args:
        problems = [int(a) for a in args]
    else:
        problems = [1, 2, 3, 4]

    runners = {1: run_P1, 2: run_P2, 3: run_P3, 4: run_P4}

    for p in problems:
        if p in runners:
            all_results[f'P{p}'] = runners[p]()

    # 最终汇总
    print(f"\n{'='*70}")
    print(" FINAL SUMMARY")
    print(f"{'='*70}")
    for pname, results in all_results.items():
        print(f"\n{pname}:")
        print(f"  {'Method':<20s} {'RMSE':<15s} {'Time(s)':<10s}")
        print(f"  {'-'*45}")
        for name, (r, t) in sorted(results.items(), key=lambda x: x[1][0]):
            print(f"  {name:<20s} {r:<15.6e} {t:<10.3f}")
