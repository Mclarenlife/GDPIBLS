"""
BO-PIBLS 测试脚本

测试问题:
  P1: 低频 Poisson (线性)     -Delta u = f,  u = sin(pi*x)*sin(pi*y)
  P2: 高频 Poisson (线性)     -Delta u = f,  u = sin(3pi*x)*sin(3pi*y) + 0.5*sin(pi*x)*sin(pi*y)
  P3: 非线性 u^3              -Delta u + u^3 = f
  P4: 强非线性 sin(u)         -Delta u + sin(u) = f

对比方法:
  BO-PIBLS:  双层优化（本方法）
  PIBLS:     标准 BLS（随机特征 + 伪逆, 来自 I-PIBLS 的 Fixed-BLS 模式）
  PINN:      引用已知结果
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import time
from bo_pibls import BOPIBLS
from gdpibls import IPIBLS

np.random.seed(42)

# ================================================================
# 配点与网格
# ================================================================
nx_int, ny_int = 30, 30
nx_bc = 50
nx_eval, ny_eval = 50, 50


def make_grid(nx, ny):
    x = np.linspace(0, 1, nx + 2)[1:-1]
    y = np.linspace(0, 1, ny + 2)[1:-1]
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def make_boundary(n_per_side):
    t = np.linspace(0, 1, n_per_side)
    sides = [
        np.column_stack([t, np.zeros_like(t)]),
        np.column_stack([t, np.ones_like(t)]),
        np.column_stack([np.zeros_like(t), t]),
        np.column_stack([np.ones_like(t), t]),
    ]
    return np.vstack(sides)


def make_eval_grid(nx, ny):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


X_int = make_grid(nx_int, ny_int)
X_bc = make_boundary(nx_bc)
X_eval = make_eval_grid(nx_eval, ny_eval)


def compute_rmse(pred, exact):
    return np.sqrt(np.mean((pred - exact) ** 2))


# ================================================================
# 问题定义
# ================================================================
problems = {}

# P1: 低频 Poisson
problems['P1'] = {
    'name': 'Low-freq Poisson',
    'type': 'linear',
    'exact': lambda X: np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]),
    'source': lambda X: 2 * np.pi ** 2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]),
    'bc': lambda X: np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]),
}

# P2: 高频 Poisson
problems['P2'] = {
    'name': 'High-freq Poisson',
    'type': 'linear',
    'exact': lambda X: (np.sin(3 * np.pi * X[:, 0]) * np.sin(3 * np.pi * X[:, 1])
                        + 0.5 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])),
    'source': lambda X: (18 * np.pi ** 2 * np.sin(3 * np.pi * X[:, 0]) * np.sin(3 * np.pi * X[:, 1])
                         + np.pi ** 2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])),
    'bc': lambda X: (np.sin(3 * np.pi * X[:, 0]) * np.sin(3 * np.pi * X[:, 1])
                     + 0.5 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])),
}

# P3: 非线性 u^3
problems['P3'] = {
    'name': 'Nonlinear u^3',
    'type': 'nonlinear',
    'exact': lambda X: np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]),
    'g_torch': lambda u: u ** 3,
    'source': lambda X: (2 * np.pi ** 2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
                         + (np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])) ** 3),
    'bc': lambda X: np.zeros(len(X)),
    # for IPIBLS
    'g_np': lambda u: u ** 3,
    'gp_np': lambda u: 3 * u ** 2,
}

# P4: 强非线性 sin(u)
problems['P4'] = {
    'name': 'Nonlinear sin(u)',
    'type': 'nonlinear',
    'exact': lambda X: np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]),
    'g_torch': lambda u: torch.sin(u),
    'source': lambda X: (2 * np.pi ** 2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
                         + np.sin(np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]))),
    'bc': lambda X: np.zeros(len(X)),
    # for IPIBLS
    'g_np': lambda u: np.sin(u),
    'gp_np': lambda u: np.cos(u),
}

# PINN 已知结果
pinn_results = {
    'P1': {'rmse': 9.42e-5, 'time': 74.0},
    'P2': {'rmse': 1.37e-3, 'time': 151.0},
    'P3': {'rmse': 6.30e-5, 'time': 70.0},
    'P4': {'rmse': 4.91e-4, 'time': 114.0},
}

# PINN+L-BFGS 加强版已知结果
pinn_strong_results = {
    'P3': {'rmse': 3.23e-5, 'time': 3298.0},
    'P4': {'rmse': 4.95e-5, 'time': 1324.0},
}

# ================================================================
# 运行实验
# ================================================================
results = {}

for pid, prob in problems.items():
    print(f"\n{'=' * 70}")
    print(f"Problem {pid}: {prob['name']}")
    print(f"{'=' * 70}")

    exact_eval = prob['exact'](X_eval)

    # --- BO-PIBLS ---
    print(f"\n>>> BO-PIBLS (Bilevel Optimization + Fourier features)")
    bo = BOPIBLS(
        n_map=50, n_enh=50,
        ridge=1e-6, bc_weight=10.0,
        lr=5e-3, epochs=300,
        lr_lbfgs=0.5, epochs_lbfgs=100,
        seed=42, verbose=True,
        freq_init_scale=1.0,
    )

    t0 = time.time()
    if prob['type'] == 'linear':
        bo.fit_linear(X_int, X_bc, prob['source'], prob['bc'])
    else:
        bo.fit_nonlinear(X_int, X_bc, prob['g_torch'],
                         prob['source'], prob['bc'],
                         n_picard=3)
    t_bo = time.time() - t0

    pred_bo = bo.predict(X_eval)
    rmse_bo = compute_rmse(pred_bo, exact_eval)
    print(f"  BO-PIBLS RMSE = {rmse_bo:.4e}  time = {t_bo:.2f}s  "
          f"features = {bo.get_n_features()}")

    # --- Fixed-BLS (PIBLS baseline) ---
    print(f"\n>>> Fixed-BLS (PIBLS baseline, 200 nodes, random features)")
    fix = IPIBLS(
        n_map_init=100, n_enh_init=100,
        max_nodes=200,
        activation='tanh', enh_activation='tanh',
        tol=1e-10, max_inc=0,
        ridge=1e-8, seed=42, verbose=False,
    )

    t0 = time.time()
    if prob['type'] == 'linear':
        fix.fit_linear(X_int, X_bc, prob['source'], prob['bc'])
    else:
        fix.fit_nonlinear(X_int, X_bc,
                          prob['g_np'], prob['gp_np'],
                          prob['source'], prob['bc'],
                          max_newton=30)
    t_fix = time.time() - t0

    pred_fix = fix.predict(X_eval)
    rmse_fix = compute_rmse(pred_fix, exact_eval)
    print(f"  Fixed-BLS RMSE = {rmse_fix:.4e}  time = {t_fix:.2f}s  "
          f"features = {fix.get_n_features()}")

    # --- 记录 ---
    pinn = pinn_results[pid]
    results[pid] = {
        'name': prob['name'],
        'bo_rmse': rmse_bo, 'bo_time': t_bo,
        'fix_rmse': rmse_fix, 'fix_time': t_fix,
        'pinn_rmse': pinn['rmse'], 'pinn_time': pinn['time'],
    }

    # 计算提升
    if rmse_fix > 1e-15:
        vs_fix = (rmse_fix - rmse_bo) / rmse_fix * 100
        print(f"  BO-PIBLS vs PIBLS: {vs_fix:+.1f}%")
    vs_pinn = (pinn['rmse'] - rmse_bo) / pinn['rmse'] * 100
    print(f"  BO-PIBLS vs PINN:  {vs_pinn:+.1f}%")
    speedup = pinn['time'] / max(t_bo, 0.01)
    print(f"  Speed vs PINN: {speedup:.1f}x")


# ================================================================
# 汇总
# ================================================================
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"{'Prob':>4s} | {'BO-PIBLS':>12s} {'time':>6s} | "
      f"{'PIBLS':>12s} {'time':>6s} | "
      f"{'PINN':>12s} {'time':>6s} | "
      f"{'BO vs PIBLS':>11s} | {'BO vs PINN':>10s}")
print("-" * 90)

for pid in ['P1', 'P2', 'P3', 'P4']:
    r = results[pid]
    vs_fix = (r['fix_rmse'] - r['bo_rmse']) / r['fix_rmse'] * 100 if r['fix_rmse'] > 1e-15 else 0
    vs_pinn = (r['pinn_rmse'] - r['bo_rmse']) / r['pinn_rmse'] * 100
    print(f"  {pid} | {r['bo_rmse']:>12.4e} {r['bo_time']:>5.1f}s | "
          f"{r['fix_rmse']:>12.4e} {r['fix_time']:>5.1f}s | "
          f"{r['pinn_rmse']:>12.4e} {r['pinn_time']:>5.1f}s | "
          f"{vs_fix:>+10.1f}% | {vs_pinn:>+9.1f}%")

print(f"\nNote: BO-PIBLS uses 100 features (50 map + 50 enh)")
print(f"      PIBLS uses 200 features (100 map + 100 enh)")
print(f"      PINN uses 4L-64W architecture")
