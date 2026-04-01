"""
I-PIBLS 对比实验脚本

测试问题:
  P1: 低频 Poisson (线性, 简单)     -Delta u = f,  u = sin(pi*x)sin(pi*y)
  P2: 高频 Poisson (线性, 困难)     -Delta u = f,  u = sin(3pi*x)sin(3pi*y)+0.5sin(pi*x)sin(pi*y)
  P3: 非线性 u^3                    -Delta u + u^3 = f
  P4: 强非线性 sin(u)               -Delta u + sin(u) = f

对比方法:
  I-PIBLS (增量式): 从20节点增长到200节点, 残差自适应
  Fixed-BLS (固定式): 直接200节点, 单次伪逆, 无增量 (消融对比)
  NL-PIBLS: 固定节点 Newton-伪逆 (从 advanced_pibls.py, 非线性问题)
  PINN: 引用已知结果 (不重跑)
"""

import numpy as np
import time
from gdpibls import IPIBLS

# ================================================================
# 公共配置
# ================================================================
np.random.seed(42)

# 配点
nx_int, ny_int = 30, 30
nx_bc = 50

# 评估网格
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
X_bc  = make_boundary(nx_bc)
X_eval = make_eval_grid(nx_eval, ny_eval)

# ================================================================
# 问题定义
# ================================================================
problems = {}

# P1: 低频 Poisson
problems['P1'] = {
    'name': 'Low-freq Poisson',
    'type': 'linear',
    'exact': lambda X: np.sin(np.pi*X[:,0]) * np.sin(np.pi*X[:,1]),
    'source': lambda X: 2 * np.pi**2 * np.sin(np.pi*X[:,0]) * np.sin(np.pi*X[:,1]),
    'bc': lambda X: np.sin(np.pi*X[:,0]) * np.sin(np.pi*X[:,1]),
}

# P2: 高频 Poisson
problems['P2'] = {
    'name': 'High-freq Poisson',
    'type': 'linear',
    'exact': lambda X: (np.sin(3*np.pi*X[:,0]) * np.sin(3*np.pi*X[:,1])
                        + 0.5*np.sin(np.pi*X[:,0]) * np.sin(np.pi*X[:,1])),
    'source': lambda X: (18*np.pi**2 * np.sin(3*np.pi*X[:,0]) * np.sin(3*np.pi*X[:,1])
                         + np.pi**2 * np.sin(np.pi*X[:,0]) * np.sin(np.pi*X[:,1])),
    'bc': lambda X: (np.sin(3*np.pi*X[:,0]) * np.sin(3*np.pi*X[:,1])
                     + 0.5*np.sin(np.pi*X[:,0]) * np.sin(np.pi*X[:,1])),
}

# P3: 非线性 u^3
_u3_exact = lambda X: np.sin(np.pi*X[:,0]) * np.sin(np.pi*X[:,1])
problems['P3'] = {
    'name': 'Nonlinear u^3',
    'type': 'nonlinear',
    'exact': _u3_exact,
    'g':  lambda u: u**3,
    'gp': lambda u: 3*u**2,
    'source': lambda X: (2*np.pi**2 * np.sin(np.pi*X[:,0]) * np.sin(np.pi*X[:,1])
                         + (np.sin(np.pi*X[:,0]) * np.sin(np.pi*X[:,1]))**3),
    'bc': lambda X: np.zeros(len(X)),
}

# P4: 强非线性 sin(u)
_su_exact = lambda X: np.sin(np.pi*X[:,0]) * np.sin(np.pi*X[:,1])
problems['P4'] = {
    'name': 'Nonlinear sin(u)',
    'type': 'nonlinear',
    'exact': _su_exact,
    'g':  lambda u: np.sin(u),
    'gp': lambda u: np.cos(u),
    'source': lambda X: (2*np.pi**2 * np.sin(np.pi*X[:,0]) * np.sin(np.pi*X[:,1])
                         + np.sin(np.sin(np.pi*X[:,0]) * np.sin(np.pi*X[:,1]))),
    'bc': lambda X: np.zeros(len(X)),
}

# ================================================================
# 已知 PINN 结果 (来自之前实验, 不重跑)
# ================================================================
pinn_results = {
    'P1': {'rmse': 9.42e-5,  'time': 74.0,  'config': 'PINN-4L-64W Adam+L-BFGS'},
    'P2': {'rmse': 1.37e-3,  'time': 151.0, 'config': 'PINN-4L-64W Adam+L-BFGS'},
    'P3': {'rmse': 6.30e-5,  'time': 70.0,  'config': 'PINN-4L-64W Adam+L-BFGS'},
    'P4': {'rmse': 4.91e-4,  'time': 114.0, 'config': 'PINN-4L-64W Adam+L-BFGS'},
}


def compute_rmse(pred, exact):
    return np.sqrt(np.mean((pred - exact)**2))


# ================================================================
# 运行实验
# ================================================================
results = {}

for pid, prob in problems.items():
    print(f"\n{'='*60}")
    print(f"Problem {pid}: {prob['name']}")
    print(f"{'='*60}")

    exact_eval = prob['exact'](X_eval)

    # --- 方法1: I-PIBLS (增量式) ---
    print(f"\n--- I-PIBLS (Incremental) ---")
    model_inc = IPIBLS(
        n_map_init=20, n_enh_init=20,
        n_map_inc=10,  n_enh_inc=10,
        max_nodes=200,
        activation='tanh', enh_activation='tanh',
        tol=1e-10, max_inc=20,
        ridge=1e-8, seed=42, verbose=True,
    )

    t0 = time.time()
    if prob['type'] == 'linear':
        model_inc.fit_linear(X_int, X_bc, prob['source'], prob['bc'])
    else:
        model_inc.fit_nonlinear(X_int, X_bc,
                                prob['g'], prob['gp'],
                                prob['source'], prob['bc'],
                                max_newton=30)
    t_inc = time.time() - t0

    pred_inc = model_inc.predict(X_eval)
    rmse_inc = compute_rmse(pred_inc, exact_eval)
    n_feat_inc = model_inc.get_n_features()
    print(f"  Result: RMSE={rmse_inc:.4e}, features={n_feat_inc}, time={t_inc:.2f}s")

    # --- 方法2: Fixed-BLS (固定200节点, 无增量) ---
    print(f"\n--- Fixed-BLS (No growth) ---")
    model_fix = IPIBLS(
        n_map_init=100, n_enh_init=100,
        max_nodes=200,
        activation='tanh', enh_activation='tanh',
        tol=1e-10, max_inc=0,   # 关键: 不增量
        ridge=1e-8, seed=42, verbose=True,
    )

    t0 = time.time()
    if prob['type'] == 'linear':
        model_fix.fit_linear(X_int, X_bc, prob['source'], prob['bc'])
    else:
        model_fix.fit_nonlinear(X_int, X_bc,
                                prob['g'], prob['gp'],
                                prob['source'], prob['bc'],
                                max_newton=30)
    t_fix = time.time() - t0

    pred_fix = model_fix.predict(X_eval)
    rmse_fix = compute_rmse(pred_fix, exact_eval)
    n_feat_fix = model_fix.get_n_features()
    print(f"  Result: RMSE={rmse_fix:.4e}, features={n_feat_fix}, time={t_fix:.2f}s")

    # --- 记录 ---
    pinn = pinn_results[pid]
    results[pid] = {
        'name': prob['name'],
        'ipibls_rmse': rmse_inc, 'ipibls_time': t_inc, 'ipibls_feat': n_feat_inc,
        'fixed_rmse': rmse_fix,  'fixed_time': t_fix,  'fixed_feat': n_feat_fix,
        'pinn_rmse': pinn['rmse'], 'pinn_time': pinn['time'],
        'inc_history': model_inc.history,
    }

    # 增量增长提升比
    if rmse_fix > 0:
        improve = (rmse_fix - rmse_inc) / rmse_fix * 100
        print(f"\n  Incremental vs Fixed: {improve:+.1f}% "
              f"({'I-PIBLS better' if improve > 0 else 'Fixed better'})")


# ================================================================
# 汇总结果
# ================================================================
print(f"\n\n{'='*80}")
print("                        SUMMARY TABLE")
print(f"{'='*80}")
print(f"{'Problem':<22s} | {'I-PIBLS':>12s} | {'Fixed-BLS':>12s} | "
      f"{'PINN':>12s} | {'Inc vs Fix':>10s} | {'Inc vs PINN':>11s}")
print(f"{'':<22s} | {'RMSE':>12s} | {'RMSE':>12s} | "
      f"{'RMSE':>12s} | {'Improve':>10s} | {'Speedup':>11s}")
print(f"{'-'*22}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*11}")

for pid in ['P1', 'P2', 'P3', 'P4']:
    r = results[pid]
    inc_vs_fix = (r['fixed_rmse'] - r['ipibls_rmse']) / r['fixed_rmse'] * 100
    speed = r['pinn_time'] / max(r['ipibls_time'], 0.001)
    print(f"{pid + ': ' + r['name']:<22s} | {r['ipibls_rmse']:>12.4e} | "
          f"{r['fixed_rmse']:>12.4e} | {r['pinn_rmse']:>12.4e} | "
          f"{inc_vs_fix:>+9.1f}% | {speed:>10.0f}x")

print(f"\n{'='*80}")
print("I-PIBLS Training Time:")
for pid in ['P1', 'P2', 'P3', 'P4']:
    r = results[pid]
    print(f"  {pid}: I-PIBLS={r['ipibls_time']:.2f}s  "
          f"Fixed={r['fixed_time']:.2f}s  "
          f"PINN={r['pinn_time']:.0f}s")

print(f"\n{'='*80}")
print("Incremental Growth Trace (I-PIBLS):")
for pid in ['P1', 'P2', 'P3', 'P4']:
    r = results[pid]
    hist = r['inc_history']
    steps_str = ' -> '.join(
        f"{h['n_features']}({h['rmse_pde']:.2e})"
        for h in hist
    )
    print(f"  {pid}: {steps_str}")

print("\nDone.")
