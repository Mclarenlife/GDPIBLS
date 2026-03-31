"""
测试脚本：验证 HybridPIBLS 和 NonlinearPIBLS

包含三组实验：
  1. 标准 PIBLS vs HybridPIBLS  (线性 Poisson 方程)
  2. NonlinearPIBLS 求解非线性方程 (-Δu + u³ = f)
  3. NonlinearPIBLS 混合模式 (Newton-伪逆 + 特征学习)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pibls_model import PIBLS
from advanced_pibls import HybridPIBLS, NonlinearPIBLS

np.random.seed(42)


# =====================================================================
# 数据生成工具
# =====================================================================

def generate_interior_points(n, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
    """均匀随机生成内部配点"""
    x = np.random.uniform(xmin, xmax, n)
    y = np.random.uniform(ymin, ymax, n)
    return x, y


def generate_boundary_points(n_per_side, xmin=0.0, xmax=1.0,
                              ymin=0.0, ymax=1.0):
    """四条边均匀采样边界点"""
    t = np.linspace(0, 1, n_per_side)
    # 下边
    x_b = xmin + (xmax - xmin) * t
    y_b = np.full_like(t, ymin)
    # 上边
    x_t = xmin + (xmax - xmin) * t
    y_t = np.full_like(t, ymax)
    # 左边
    x_l = np.full_like(t, xmin)
    y_l = ymin + (ymax - ymin) * t
    # 右边
    x_r = np.full_like(t, xmax)
    y_r = ymin + (ymax - ymin) * t

    x_bc = np.concatenate([x_b, x_t, x_l, x_r])
    y_bc = np.concatenate([y_b, y_t, y_l, y_r])
    return x_bc, y_bc


def generate_test_grid(n=50, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
    """生成用于评估的均匀网格"""
    xx = np.linspace(xmin, xmax, n)
    yy = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(xx, yy)
    return X.flatten(), Y.flatten(), X, Y


# =====================================================================
# 实验 1: 标准 PIBLS vs HybridPIBLS  (Poisson 方程)
# =====================================================================

def test_poisson():
    """Poisson 方程: Δu = f  on [0,1]²
    高频精确解: u(x,y) = sin(3πx)·sin(3πy) + 0.5·sin(πx)·sin(πy)
    源项:   f(x,y) = Δu = -18π²·sin(3πx)·sin(3πy) - π²·sin(πx)·sin(πy)

    选择高频多模态解：随机特征更难一次性捕获，
    HybridPIBLS 的特征学习优势得以体现。
    """
    print("=" * 70)
    print(" 实验 1: 标准 PIBLS  vs  HybridPIBLS  (高频 Poisson 方程)")
    print("=" * 70)

    exact_fn = lambda x, y: (np.sin(3 * np.pi * x) * np.sin(3 * np.pi * y)
                              + 0.5 * np.sin(np.pi * x) * np.sin(np.pi * y))
    source_fn = lambda x, y: (-18 * np.pi ** 2 * np.sin(3 * np.pi * x) * np.sin(3 * np.pi * y)
                               - np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y))

    x_pde, y_pde = generate_interior_points(1200)
    x_bc, y_bc = generate_boundary_points(80)
    pde_data = (x_pde, y_pde)
    bc_data = (x_bc, y_bc)

    N1, N2 = 20, 20

    # ---- 标准 PIBLS (单激活) ----
    print("\n--- 标准 PIBLS (单次伪逆, tanh, N=20) ---")
    np.random.seed(42)
    model_std = PIBLS(N1, N2, map_func='tanh', enhance_func='sigmoid',
                      source_fn=source_fn, exact_solution_fn=exact_fn)
    model_std.fit(pde_data, bc_data)

    x_test, y_test, X, Y = generate_test_grid(50)
    u_std = model_std.predict(x_test, y_test)
    u_exact = exact_fn(x_test, y_test)
    rmse_std = np.sqrt(np.mean((u_std - u_exact) ** 2))
    print(f"  RMSE = {rmse_std:.6e}")

    # ---- HybridPIBLS (伪逆 + 梯度交替, 单激活tanh) ----
    # 注: 二阶PDE使用tanh (二阶导非零), 多激活(identity/ReLU)的二阶导为0不适合
    print("\n--- HybridPIBLS (伪逆 + SPSA梯度下降, N=20) ---")
    np.random.seed(42)
    model_hyb = HybridPIBLS(
        N1, N2, map_func='tanh', enhance_func='sigmoid',
        source_fn=source_fn, exact_solution_fn=exact_fn,
        lr=0.01, max_iter=50, lambda_bc=10.0, grad_method='spsa',
        n_grad_samples=15,
    )
    model_hyb.fit(pde_data, bc_data)

    u_hyb = model_hyb.predict(x_test, y_test)
    rmse_hyb = np.sqrt(np.mean((u_hyb - u_exact) ** 2))
    print(f"  RMSE = {rmse_hyb:.6e}")

    # ---- 结果对比 ----
    improve_hyb = (1 - rmse_hyb / rmse_std) * 100 if rmse_std > 0 else 0
    print(f"\n  HybridPIBLS vs 标准: {improve_hyb:+.1f}%  ({rmse_std:.4e} → {rmse_hyb:.4e})")

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    u_exact_grid = u_exact.reshape(50, 50)
    u_std_grid = u_std.reshape(50, 50)
    u_hyb_grid = u_hyb.reshape(50, 50)

    im0 = axes[0].contourf(X, Y, u_exact_grid, levels=30, cmap='viridis')
    axes[0].set_title('Exact Solution')
    plt.colorbar(im0, ax=axes[0])

    err_std_grid = np.abs(u_std_grid - u_exact_grid)
    im1 = axes[1].contourf(X, Y, err_std_grid, levels=30, cmap='hot')
    axes[1].set_title(f'PIBLS Error (RMSE={rmse_std:.2e})')
    plt.colorbar(im1, ax=axes[1])

    err_hyb_grid = np.abs(u_hyb_grid - u_exact_grid)
    im2 = axes[2].contourf(X, Y, err_hyb_grid, levels=30, cmap='hot')
    axes[2].set_title(f'HybridPIBLS Error (RMSE={rmse_hyb:.2e})')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig('test1_poisson_comparison.png', dpi=150)
    plt.close()
    print("  图表已保存: test1_poisson_comparison.png")

    # 收敛曲线
    if model_hyb.loss_history:
        plt.figure(figsize=(8, 4))
        plt.semilogy(model_hyb.loss_history, 'b-o', markersize=3)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('HybridPIBLS Convergence')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('test1_hybrid_convergence.png', dpi=150)
        plt.close()
        print("  图表已保存: test1_hybrid_convergence.png")

    return rmse_std, rmse_hyb


# =====================================================================
# 实验 2: NonlinearPIBLS 求解非线性方程
# =====================================================================

def test_nonlinear():
    """非线性方程: -u_xx - u_yy + u³ = f(x,y)  on [0,1]²
    精确解: u(x,y) = sin(πx)·sin(πy)
    源项:   f(x,y) = 2π²·sin(πx)·sin(πy) + sin³(πx)·sin³(πy)
    """
    print("\n" + "=" * 70)
    print(" 实验 2: NonlinearPIBLS (Newton-伪逆) 求解非线性方程")
    print("=" * 70)

    exact_fn = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    source_fn = lambda x, y: (
        2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        + np.sin(np.pi * x) ** 3 * np.sin(np.pi * y) ** 3
    )

    # PDE 残差: R = -u_xx - u_yy + u³ - f = 0
    def residual_fn(u, u_x, u_y, u_xx, u_yy, x, y):
        return -u_xx - u_yy + u ** 3 - source_fn(x, y)

    # 解析偏导数
    def dR_du(u, u_x, u_y, u_xx, u_yy, x, y):
        return 3.0 * u ** 2

    def dR_duxx(u, u_x, u_y, u_xx, u_yy, x, y):
        return -np.ones_like(u)

    def dR_duyy(u, u_x, u_y, u_xx, u_yy, x, y):
        return -np.ones_like(u)

    x_pde, y_pde = generate_interior_points(600)
    x_bc, y_bc = generate_boundary_points(50)
    pde_data = (x_pde, y_pde)
    bc_data = (x_bc, y_bc)

    N1, N2 = 40, 40

    # ---- NonlinearPIBLS (解析 Jacobian) ----
    print("\n--- NonlinearPIBLS (解析 Jacobian, N=40) ---")
    np.random.seed(42)
    model_nl = NonlinearPIBLS(
        N1, N2, map_func='tanh', enhance_func='sigmoid',
        residual_fn=residual_fn, bc_fn=exact_fn,
        dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
        lambda_bc=20.0,
    )
    model_nl.fit(pde_data, bc_data, max_iter=50, damping=0.8)

    x_test, y_test, X, Y = generate_test_grid(50)
    u_nl = model_nl.predict(x_test, y_test)
    u_exact = exact_fn(x_test, y_test)
    rmse_nl = np.sqrt(np.mean((u_nl - u_exact) ** 2))
    print(f"  RMSE = {rmse_nl:.6e}")

    # ---- NonlinearPIBLS (数值 Jacobian, 验证一致性) ----
    print("\n--- NonlinearPIBLS (数值 Jacobian, N=40) ---")
    np.random.seed(42)
    model_nl_num = NonlinearPIBLS(
        N1, N2, map_func='tanh', enhance_func='sigmoid',
        residual_fn=residual_fn, bc_fn=exact_fn,
        lambda_bc=20.0,
    )
    model_nl_num.fit(pde_data, bc_data, max_iter=50, damping=0.8)

    u_nl_num = model_nl_num.predict(x_test, y_test)
    rmse_nl_num = np.sqrt(np.mean((u_nl_num - u_exact) ** 2))
    print(f"  RMSE = {rmse_nl_num:.6e}")

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    u_exact_grid = u_exact.reshape(50, 50)
    u_nl_grid = u_nl.reshape(50, 50)

    im0 = axes[0].contourf(X, Y, u_exact_grid, levels=30, cmap='viridis')
    axes[0].set_title('Exact Solution')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, Y, u_nl_grid, levels=30, cmap='viridis')
    axes[1].set_title(f'NL-PIBLS Prediction')
    plt.colorbar(im1, ax=axes[1])

    err_grid = np.abs(u_nl_grid - u_exact_grid)
    im2 = axes[2].contourf(X, Y, err_grid, levels=30, cmap='hot')
    axes[2].set_title(f'Error (RMSE={rmse_nl:.2e})')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig('test2_nonlinear.png', dpi=150)
    plt.close()
    print("  图表已保存: test2_nonlinear.png")

    # Newton 收敛曲线
    if model_nl.loss_history:
        plt.figure(figsize=(8, 4))
        plt.semilogy(model_nl.loss_history, 'r-o', markersize=3, label='Analytical Jacobian')
        if model_nl_num.loss_history:
            plt.semilogy(model_nl_num.loss_history, 'b--s', markersize=3, label='Numerical Jacobian')
        plt.xlabel('Newton Iteration')
        plt.ylabel('Loss')
        plt.title('NonlinearPIBLS Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('test2_newton_convergence.png', dpi=150)
        plt.close()
        print("  图表已保存: test2_newton_convergence.png")

    return rmse_nl, rmse_nl_num


# =====================================================================
# 实验 3: NonlinearPIBLS 混合模式 (Newton + 特征学习)
# =====================================================================

def test_nonlinear_hybrid():
    """非线性方程 + 混合特征学习
    同实验 2 的方程, 但额外通过梯度下降优化内部权重
    """
    print("\n" + "=" * 70)
    print(" 实验 3: NonlinearPIBLS 混合模式 (Newton-伪逆 + 特征学习)")
    print("=" * 70)

    exact_fn = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    source_fn = lambda x, y: (
        2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        + np.sin(np.pi * x) ** 3 * np.sin(np.pi * y) ** 3
    )

    def residual_fn(u, u_x, u_y, u_xx, u_yy, x, y):
        return -u_xx - u_yy + u ** 3 - source_fn(x, y)

    def dR_du(u, u_x, u_y, u_xx, u_yy, x, y):
        return 3.0 * u ** 2

    def dR_duxx(u, u_x, u_y, u_xx, u_yy, x, y):
        return -np.ones_like(u)

    def dR_duyy(u, u_x, u_y, u_xx, u_yy, x, y):
        return -np.ones_like(u)

    x_pde, y_pde = generate_interior_points(600)
    x_bc, y_bc = generate_boundary_points(50)
    pde_data = (x_pde, y_pde)
    bc_data = (x_bc, y_bc)

    N1, N2 = 40, 40

    # ---- 纯 Newton (对照) ----
    print("\n--- 纯 Newton-伪逆 (对照, N=40) ---")
    np.random.seed(42)
    model_pure = NonlinearPIBLS(
        N1, N2, map_func='tanh', enhance_func='sigmoid',
        residual_fn=residual_fn, bc_fn=exact_fn,
        dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
        lambda_bc=20.0,
    )
    model_pure.fit(pde_data, bc_data, max_iter=50, damping=0.8)

    x_test, y_test, X, Y = generate_test_grid(50)
    u_pure = model_pure.predict(x_test, y_test)
    u_exact = exact_fn(x_test, y_test)
    rmse_pure = np.sqrt(np.mean((u_pure - u_exact) ** 2))
    print(f"  RMSE = {rmse_pure:.6e}")

    # ---- 混合模式 ----
    print("\n--- Newton-伪逆 + 特征学习 (混合, N=40) ---")
    np.random.seed(42)
    model_hyb = NonlinearPIBLS(
        N1, N2, map_func='tanh', enhance_func='sigmoid',
        residual_fn=residual_fn, bc_fn=exact_fn,
        dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
        lambda_bc=20.0,
    )
    model_hyb.fit_hybrid(
        pde_data, bc_data,
        outer_iters=5, inner_iters=30,
        lr=0.003, damping=0.8,
    )

    u_hyb = model_hyb.predict(x_test, y_test)
    rmse_hyb = np.sqrt(np.mean((u_hyb - u_exact) ** 2))
    print(f"  RMSE = {rmse_hyb:.6e}")

    improve = (1 - rmse_hyb / rmse_pure) * 100 if rmse_pure > 0 else 0
    print(f"\n  Improvement: {improve:+.1f}%  (纯 Newton {rmse_pure:.4e} → 混合 {rmse_hyb:.4e})")

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    u_exact_grid = u_exact.reshape(50, 50)
    err_pure_grid = np.abs(u_pure.reshape(50, 50) - u_exact_grid)
    err_hyb_grid = np.abs(u_hyb.reshape(50, 50) - u_exact_grid)

    im0 = axes[0].contourf(X, Y, u_exact_grid, levels=30, cmap='viridis')
    axes[0].set_title('Exact Solution')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, Y, err_pure_grid, levels=30, cmap='hot')
    axes[1].set_title(f'Pure Newton Error (RMSE={rmse_pure:.2e})')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(X, Y, err_hyb_grid, levels=30, cmap='hot')
    axes[2].set_title(f'Hybrid Error (RMSE={rmse_hyb:.2e})')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig('test3_hybrid_nonlinear.png', dpi=150)
    plt.close()
    print("  图表已保存: test3_hybrid_nonlinear.png")

    return rmse_pure, rmse_hyb


# =====================================================================
# 主程序
# =====================================================================

if __name__ == '__main__':
    print("HybridPIBLS & NonlinearPIBLS 验证测试")
    print("=" * 70)

    rmse_std, rmse_hyb = test_poisson()
    rmse_nl, rmse_nl_num = test_nonlinear()
    rmse_pure_nl, rmse_hyb_nl = test_nonlinear_hybrid()

    print("\n" + "=" * 70)
    print(" 总结")
    print("=" * 70)
    print(f"  实验1 (Poisson, 标准):    PIBLS RMSE = {rmse_std:.4e}")
    print(f"  实验1 (Poisson, Hybrid):  HybridPIBLS RMSE = {rmse_hyb:.4e}")
    print(f"  实验2 (非线性, 解析):     NL-PIBLS RMSE = {rmse_nl:.4e}")
    print(f"  实验2 (非线性, 数值):     NL-PIBLS RMSE = {rmse_nl_num:.4e}")
    print(f"  实验3 (非线性, 纯Newton): RMSE = {rmse_pure_nl:.4e}")
    print(f"  实验3 (非线性, 混合):     RMSE = {rmse_hyb_nl:.4e}")
