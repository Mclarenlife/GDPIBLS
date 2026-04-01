"""
公平性补充实验：回应三个关键质疑

Q1: HybridPIBLS 能否求解非线性 PDE？
    - 用 NonlinearPIBLS.fit_hybrid() 在困难非线性问题上测试

Q2: NL-PIBLS vs PINN 的对比是否公平？
    - 加入 L-BFGS 优化器（PINN 论文标配）
    - 加宽网络 (128 width)
    - 增加 epochs
    - 扫描 lambda_bc

Q3: HybridPIBLS 在非线性上能否超越 PINN？
    - 对比 NL-PIBLS (pure Newton) vs NL-PIBLS-Hybrid vs PINN-best
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')

np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device('cpu')

from advanced_pibls import NonlinearPIBLS


# =====================================================================
#  Deep PINN (支持 Adam + L-BFGS)
# =====================================================================

class DeepPINN(nn.Module):
    def __init__(self, layers, activation='tanh'):
        super().__init__()
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net.append(nn.Tanh() if activation == 'tanh' else nn.ReLU())
        self.net = nn.Sequential(*net)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        inp = torch.stack([x, y], dim=1)
        return self.net(inp).squeeze(-1)


class PINNSolverFair:
    """支持 Adam → L-BFGS 两阶段训练的 PINN 求解器"""

    def __init__(self, layers, pde_residual_fn, bc_fn,
                 lr=1e-3, epochs_adam=3000, epochs_lbfgs=2000,
                 lambda_bc=10.0, use_lbfgs=True):
        self.model = DeepPINN(layers).to(DEVICE)
        self.pde_residual_fn = pde_residual_fn
        self.bc_fn = bc_fn
        self.lr = lr
        self.epochs_adam = epochs_adam
        self.epochs_lbfgs = epochs_lbfgs
        self.lambda_bc = lambda_bc
        self.use_lbfgs = use_lbfgs
        self.loss_history = []
        self.train_time = 0.0

    def _compute_loss(self, x_pde, y_pde, x_bc, y_bc, u_bc_target):
        R = self.pde_residual_fn(self.model, x_pde, y_pde)
        loss_pde = torch.mean(R ** 2)
        u_bc_pred = self.model(x_bc, y_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc_target) ** 2)
        return loss_pde + self.lambda_bc * loss_bc, loss_pde, loss_bc

    def train(self, pde_data, bc_data):
        x_pde_np, y_pde_np = pde_data
        x_bc_np, y_bc_np = bc_data
        x_pde = torch.tensor(x_pde_np, dtype=torch.float64, requires_grad=True, device=DEVICE)
        y_pde = torch.tensor(y_pde_np, dtype=torch.float64, requires_grad=True, device=DEVICE)
        x_bc = torch.tensor(x_bc_np, dtype=torch.float64, device=DEVICE)
        y_bc = torch.tensor(y_bc_np, dtype=torch.float64, device=DEVICE)
        u_bc_target = torch.tensor(self.bc_fn(x_bc_np, y_bc_np), dtype=torch.float64, device=DEVICE)
        self.model.double()
        self.loss_history = []
        t0 = time.perf_counter()

        # Phase 1: Adam
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs_adam):
            optimizer.zero_grad()
            loss, loss_pde, loss_bc = self._compute_loss(x_pde, y_pde, x_bc, y_bc, u_bc_target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()
            self.loss_history.append(loss.item())
            if epoch % max(1, self.epochs_adam // 5) == 0:
                print(f"  [Adam] Epoch {epoch:>5d}: L_pde={loss_pde.item():.4e}  L_bc={loss_bc.item():.4e}")

        # Phase 2: L-BFGS (PINN 论文标配优化器)
        if self.use_lbfgs and self.epochs_lbfgs > 0:
            print(f"  [L-BFGS] Starting ({self.epochs_lbfgs} iters)...")
            optimizer_lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                lr=1.0,
                max_iter=20,
                history_size=50,
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
                line_search_fn='strong_wolfe',
            )
            for epoch in range(self.epochs_lbfgs):
                def closure():
                    optimizer_lbfgs.zero_grad()
                    loss, _, _ = self._compute_loss(x_pde, y_pde, x_bc, y_bc, u_bc_target)
                    loss.backward()
                    return loss
                loss_val = optimizer_lbfgs.step(closure)
                if loss_val is not None:
                    self.loss_history.append(loss_val.item())
                if epoch % max(1, self.epochs_lbfgs // 5) == 0:
                    cur = self.loss_history[-1] if self.loss_history else float('nan')
                    print(f"  [L-BFGS] Iter {epoch:>5d}: loss={cur:.4e}")

        self.train_time = time.perf_counter() - t0
        final_loss = self.loss_history[-1] if self.loss_history else float('nan')
        print(f"  Training done: {self.train_time:.2f}s, final loss={final_loss:.4e}")

    @torch.no_grad()
    def predict(self, x_np, y_np):
        self.model.eval()
        x = torch.tensor(x_np, dtype=torch.float64, device=DEVICE)
        y = torch.tensor(y_np, dtype=torch.float64, device=DEVICE)
        return self.model(x, y).cpu().numpy()

    def param_count(self):
        return sum(p.numel() for p in self.model.parameters())


# =====================================================================
#  PDE 残差 (torch)
# =====================================================================

def residual_cubic(model, x, y, source_fn):
    u = model(x, y)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return -u_xx - u_yy + u ** 3 - source_fn(x, y)

def residual_sin(model, x, y, source_fn):
    u = model(x, y)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return -u_xx - u_yy + torch.sin(u) - source_fn(x, y)


# =====================================================================
#  工具
# =====================================================================

def gen_interior(n):
    return np.random.uniform(0, 1, n), np.random.uniform(0, 1, n)

def gen_boundary(n_per_side):
    t = np.linspace(0, 1, n_per_side)
    x = np.concatenate([t, t, np.zeros(n_per_side), np.ones(n_per_side)])
    y = np.concatenate([np.zeros(n_per_side), np.ones(n_per_side), t, t])
    return x, y

def gen_test(n=50):
    xx = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xx, xx)
    return X.flatten(), Y.flatten()

def rmse(pred, exact):
    return np.sqrt(np.mean((pred - exact) ** 2))


# =====================================================================
#  实验 A: 公平性验证 — PINN 加强版 vs NL-PIBLS
#  在 Problem 3 (-Lap u + u^3 = f) 上测试
# =====================================================================

def experiment_A_fairness():
    """Q2: PINN 调优后是否仍不如 NL-PIBLS?"""
    print("\n" + "=" * 75)
    print("  实验 A: 公平性验证 — PINN 加强版 vs NL-PIBLS")
    print("  方程: -Lap(u) + u^3 = f,  精确解: sin(pi*x)*sin(pi*y)")
    print("  加强: L-BFGS 优化器, 更宽网络 (128), 更多配点 (2000)")
    print("=" * 75)

    exact_fn = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    source_np = lambda x, y: (2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
                               + np.sin(np.pi*x)**3 * np.sin(np.pi*y)**3)
    source_torch = lambda x, y: (2 * np.pi**2 * torch.sin(np.pi*x) * torch.sin(np.pi*y)
                                  + torch.sin(np.pi*x)**3 * torch.sin(np.pi*y)**3)

    def res_fn(u, u_x, u_y, u_xx, u_yy, x, y):
        return -u_xx - u_yy + u**3 - source_np(x, y)
    def dR_du(u, u_x, u_y, u_xx, u_yy, x, y):
        return 3.0 * u**2
    def dR_duxx(u, u_x, u_y, u_xx, u_yy, x, y):
        return -np.ones_like(u)
    def dR_duyy(u, u_x, u_y, u_xx, u_yy, x, y):
        return -np.ones_like(u)

    # 更多配点以给 PINN 更好的训练条件
    n_pde, n_bc = 2000, 100
    np.random.seed(42)
    x_pde, y_pde = gen_interior(n_pde)
    x_bc, y_bc = gen_boundary(n_bc)
    x_test, y_test = gen_test(50)
    u_exact = exact_fn(x_test, y_test)

    results = {}

    # ---- NL-PIBLS (基准) ----
    print("\n--- NL-PIBLS (N=40, Newton-pinv, damping=0.8) ---")
    np.random.seed(42)
    t0 = time.perf_counter()
    m = NonlinearPIBLS(40, 40, 'tanh', 'sigmoid',
                       residual_fn=res_fn, bc_fn=exact_fn,
                       dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
                       lambda_bc=20.0)
    m.fit((x_pde, y_pde), (x_bc, y_bc), max_iter=50, damping=0.8)
    t1 = time.perf_counter() - t0
    r = rmse(m.predict(x_test, y_test), u_exact)
    results['NL-PIBLS'] = {'rmse': r, 'time': t1}
    print(f"  RMSE = {r:.6e}, Time = {t1:.3f}s")

    # ---- NL-PIBLS (N=60, 更多节点) ----
    print("\n--- NL-PIBLS (N=60, Newton-pinv) ---")
    np.random.seed(42)
    t0 = time.perf_counter()
    m2 = NonlinearPIBLS(60, 60, 'tanh', 'sigmoid',
                        residual_fn=res_fn, bc_fn=exact_fn,
                        dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
                        lambda_bc=20.0)
    m2.fit((x_pde, y_pde), (x_bc, y_bc), max_iter=50, damping=0.8)
    t2 = time.perf_counter() - t0
    r2 = rmse(m2.predict(x_test, y_test), u_exact)
    results['NL-PIBLS-N60'] = {'rmse': r2, 'time': t2}
    print(f"  RMSE = {r2:.6e}, Time = {t2:.3f}s")

    # ---- PINN Adam-only (旧方案, 作为参照) ----
    print("\n--- PINN-4L-64W Adam-only 5000ep (原实验) ---")
    torch.manual_seed(42)
    sf = source_torch
    pinn_old = PINNSolverFair(
        [2, 64, 64, 64, 64, 1],
        pde_residual_fn=lambda m, x, y, _sf=sf: residual_cubic(m, x, y, _sf),
        bc_fn=exact_fn,
        lr=1e-3, epochs_adam=5000, epochs_lbfgs=0,
        lambda_bc=20.0, use_lbfgs=False,
    )
    pinn_old.train((x_pde, y_pde), (x_bc, y_bc))
    r_old = rmse(pinn_old.predict(x_test, y_test), u_exact)
    results['PINN-4L-64W-Adam'] = {'rmse': r_old, 'time': pinn_old.train_time,
                                    'params': pinn_old.param_count()}
    print(f"  RMSE = {r_old:.6e}, Params = {pinn_old.param_count()}")

    # ---- PINN Adam + L-BFGS (公平加强版) ----
    print("\n--- PINN-4L-64W Adam(3000) + L-BFGS(1000) ---")
    torch.manual_seed(42)
    pinn_lbfgs = PINNSolverFair(
        [2, 64, 64, 64, 64, 1],
        pde_residual_fn=lambda m, x, y, _sf=sf: residual_cubic(m, x, y, _sf),
        bc_fn=exact_fn,
        lr=1e-3, epochs_adam=3000, epochs_lbfgs=1000,
        lambda_bc=20.0, use_lbfgs=True,
    )
    pinn_lbfgs.train((x_pde, y_pde), (x_bc, y_bc))
    r_lbfgs = rmse(pinn_lbfgs.predict(x_test, y_test), u_exact)
    results['PINN-4L-64W-LBFGS'] = {'rmse': r_lbfgs, 'time': pinn_lbfgs.train_time,
                                     'params': pinn_lbfgs.param_count()}
    print(f"  RMSE = {r_lbfgs:.6e}")

    # ---- PINN 更宽 (128 width) + L-BFGS ----
    print("\n--- PINN-4L-128W Adam(3000) + L-BFGS(1000) ---")
    torch.manual_seed(42)
    pinn_wide = PINNSolverFair(
        [2, 128, 128, 128, 128, 1],
        pde_residual_fn=lambda m, x, y, _sf=sf: residual_cubic(m, x, y, _sf),
        bc_fn=exact_fn,
        lr=1e-3, epochs_adam=3000, epochs_lbfgs=1000,
        lambda_bc=20.0, use_lbfgs=True,
    )
    pinn_wide.train((x_pde, y_pde), (x_bc, y_bc))
    r_wide = rmse(pinn_wide.predict(x_test, y_test), u_exact)
    results['PINN-4L-128W-LBFGS'] = {'rmse': r_wide, 'time': pinn_wide.train_time,
                                      'params': pinn_wide.param_count()}
    print(f"  RMSE = {r_wide:.6e}, Params = {pinn_wide.param_count()}")

    # ---- PINN 浅层宽网络 (2L-128W) + L-BFGS ----
    print("\n--- PINN-2L-128W Adam(3000) + L-BFGS(1000) ---")
    torch.manual_seed(42)
    pinn_shallow = PINNSolverFair(
        [2, 128, 128, 1],
        pde_residual_fn=lambda m, x, y, _sf=sf: residual_cubic(m, x, y, _sf),
        bc_fn=exact_fn,
        lr=1e-3, epochs_adam=3000, epochs_lbfgs=1000,
        lambda_bc=20.0, use_lbfgs=True,
    )
    pinn_shallow.train((x_pde, y_pde), (x_bc, y_bc))
    r_sh = rmse(pinn_shallow.predict(x_test, y_test), u_exact)
    results['PINN-2L-128W-LBFGS'] = {'rmse': r_sh, 'time': pinn_shallow.train_time,
                                      'params': pinn_shallow.param_count()}
    print(f"  RMSE = {r_sh:.6e}, Params = {pinn_shallow.param_count()}")

    print_results("Experiment A: Fairness (P3 u^3)", results)
    return results


# =====================================================================
#  实验 B: NonlinearPIBLS 混合模式 (困难问题)
#  在 Problem 4 (-Lap u + sin(u) = f) 上测试
# =====================================================================

def experiment_B_hybrid_nonlinear():
    """Q1 & Q3: NL-PIBLS-Hybrid 在困难非线性问题上表现如何?"""
    print("\n" + "=" * 75)
    print("  实验 B: NL-PIBLS-Hybrid vs PINN-best")
    print("  方程: -Lap(u) + sin(u) = f,  精确解: sin(2pi*x)*sin(2pi*y)")
    print("  测试: Newton-伪逆 + 特征学习(SPSA) 是否进一步提升")
    print("=" * 75)

    exact_fn = lambda x, y: np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    source_np = lambda x, y: (8 * np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
                               + np.sin(np.sin(2*np.pi*x) * np.sin(2*np.pi*y)))
    source_torch = lambda x, y: (8 * np.pi**2 * torch.sin(2*np.pi*x) * torch.sin(2*np.pi*y)
                                  + torch.sin(torch.sin(2*np.pi*x) * torch.sin(2*np.pi*y)))

    def res_fn(u, u_x, u_y, u_xx, u_yy, x, y):
        return -u_xx - u_yy + np.sin(u) - source_np(x, y)
    def dR_du(u, u_x, u_y, u_xx, u_yy, x, y):
        return np.cos(u)
    def dR_duxx(u, u_x, u_y, u_xx, u_yy, x, y):
        return -np.ones_like(u)
    def dR_duyy(u, u_x, u_y, u_xx, u_yy, x, y):
        return -np.ones_like(u)

    n_pde, n_bc = 1000, 80
    np.random.seed(42)
    x_pde, y_pde = gen_interior(n_pde)
    x_bc, y_bc = gen_boundary(n_bc)
    x_test, y_test = gen_test(50)
    u_exact = exact_fn(x_test, y_test)

    results = {}

    # ---- NL-PIBLS (纯 Newton, N=50) ----
    print("\n--- NL-PIBLS (N=50, pure Newton) ---")
    np.random.seed(42)
    t0 = time.perf_counter()
    m1 = NonlinearPIBLS(50, 50, 'tanh', 'sigmoid',
                        residual_fn=res_fn, bc_fn=exact_fn,
                        dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
                        lambda_bc=20.0)
    m1.fit((x_pde, y_pde), (x_bc, y_bc), max_iter=60, damping=0.6)
    t1 = time.perf_counter() - t0
    r1 = rmse(m1.predict(x_test, y_test), u_exact)
    results['NL-PIBLS-N50'] = {'rmse': r1, 'time': t1}
    print(f"  RMSE = {r1:.6e}, Time = {t1:.3f}s")

    # ---- NL-PIBLS (纯 Newton, N=30, 节点偏少使特征学习有空间) ----
    print("\n--- NL-PIBLS (N=30, pure Newton, 节点偏少) ---")
    np.random.seed(42)
    t0 = time.perf_counter()
    m_small = NonlinearPIBLS(30, 30, 'tanh', 'sigmoid',
                             residual_fn=res_fn, bc_fn=exact_fn,
                             dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
                             lambda_bc=20.0)
    m_small.fit((x_pde, y_pde), (x_bc, y_bc), max_iter=60, damping=0.6)
    t_small = time.perf_counter() - t0
    r_small = rmse(m_small.predict(x_test, y_test), u_exact)
    results['NL-PIBLS-N30'] = {'rmse': r_small, 'time': t_small}
    print(f"  RMSE = {r_small:.6e}, Time = {t_small:.3f}s")

    # ---- NL-PIBLS-Hybrid (N=30, Newton + 特征学习) ----
    print("\n--- NL-PIBLS-Hybrid (N=30, Newton + SPSA feature learning) ---")
    np.random.seed(42)
    t0 = time.perf_counter()
    m_hyb = NonlinearPIBLS(30, 30, 'tanh', 'sigmoid',
                           residual_fn=res_fn, bc_fn=exact_fn,
                           dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
                           lambda_bc=20.0)
    m_hyb.fit_hybrid(
        (x_pde, y_pde), (x_bc, y_bc),
        outer_iters=8, inner_iters=30,
        lr=0.005, damping=0.6, mu=0.0,
        grad_method='spsa', verbose=True,
    )
    t_hyb = time.perf_counter() - t0
    r_hyb = rmse(m_hyb.predict(x_test, y_test), u_exact)
    results['NL-PIBLS-Hybrid-N30'] = {'rmse': r_hyb, 'time': t_hyb}
    print(f"  RMSE = {r_hyb:.6e}, Time = {t_hyb:.3f}s")

    # ---- PINN Adam+L-BFGS (最强配置) ----
    print("\n--- PINN-2L-128W Adam(3000) + L-BFGS(1000) ---")
    torch.manual_seed(42)
    sf = source_torch
    pinn = PINNSolverFair(
        [2, 128, 128, 1],
        pde_residual_fn=lambda m, x, y, _sf=sf: residual_sin(m, x, y, _sf),
        bc_fn=exact_fn,
        lr=1e-3, epochs_adam=3000, epochs_lbfgs=1000,
        lambda_bc=20.0, use_lbfgs=True,
    )
    pinn.train((x_pde, y_pde), (x_bc, y_bc))
    rp = rmse(pinn.predict(x_test, y_test), u_exact)
    results['PINN-2L-128W-LBFGS'] = {'rmse': rp, 'time': pinn.train_time,
                                      'params': pinn.param_count()}
    print(f"  RMSE = {rp:.6e}, Params = {pinn.param_count()}")

    # ---- PINN-4L-128W + L-BFGS ----
    print("\n--- PINN-4L-128W Adam(3000) + L-BFGS(1000) ---")
    torch.manual_seed(42)
    pinn4 = PINNSolverFair(
        [2, 128, 128, 128, 128, 1],
        pde_residual_fn=lambda m, x, y, _sf=sf: residual_sin(m, x, y, _sf),
        bc_fn=exact_fn,
        lr=1e-3, epochs_adam=3000, epochs_lbfgs=1000,
        lambda_bc=20.0, use_lbfgs=True,
    )
    pinn4.train((x_pde, y_pde), (x_bc, y_bc))
    rp4 = rmse(pinn4.predict(x_test, y_test), u_exact)
    results['PINN-4L-128W-LBFGS'] = {'rmse': rp4, 'time': pinn4.train_time,
                                      'params': pinn4.param_count()}
    print(f"  RMSE = {rp4:.6e}, Params = {pinn4.param_count()}")

    print_results("Experiment B: Hybrid nonlinear (P4 sin(u))", results)
    return results


# =====================================================================
#  打印
# =====================================================================

def print_results(title, results):
    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")
    print(f"  {'Method':<28s}  {'RMSE':>12s}  {'Time(s)':>10s}  {'Params':>8s}")
    print(f"  {'-'*28}  {'-'*12}  {'-'*10}  {'-'*8}")
    for name, r in results.items():
        rmse_s = f"{r['rmse']:.4e}"
        time_s = f"{r['time']:.2f}"
        params_s = str(r.get('params', '-'))
        print(f"  {name:<28s}  {rmse_s:>12s}  {time_s:>10s}  {params_s:>8s}")

    best = min(results, key=lambda k: results[k]['rmse'])
    print(f"\n  Best: {best} (RMSE = {results[best]['rmse']:.4e})")
    print(f"{'=' * 75}")


# =====================================================================
#  主函数
# =====================================================================

if __name__ == '__main__':
    print("=" * 75)
    print("  公平性补充实验: PIBLS 系列 vs Deep PINN (加强版)")
    print("=" * 75)

    rA = experiment_A_fairness()
    rB = experiment_B_hybrid_nonlinear()

    # 最终总结
    print("\n" + "=" * 75)
    print("  最终总结")
    print("=" * 75)

    print("\n  实验 A — Q2 回应: PINN 加强后是否仍不如 NL-PIBLS?")
    pibls_a = min([k for k in rA if 'NL-PIBLS' in k], key=lambda k: rA[k]['rmse'])
    pinn_a = min([k for k in rA if 'PINN' in k], key=lambda k: rA[k]['rmse'])
    ratio_a = rA[pinn_a]['rmse'] / rA[pibls_a]['rmse']
    speed_a = rA[pinn_a]['time'] / max(rA[pibls_a]['time'], 1e-6)
    print(f"    NL-PIBLS best: {pibls_a} = {rA[pibls_a]['rmse']:.4e}")
    print(f"    PINN best:     {pinn_a} = {rA[pinn_a]['rmse']:.4e}")
    if ratio_a > 1:
        print(f"    => NL-PIBLS 仍然领先 {ratio_a:.1f}x, 速度快 {speed_a:.0f}x")
    else:
        print(f"    => PINN 加强后反超 {1/ratio_a:.1f}x (但速度慢 {speed_a:.0f}x)")

    print("\n  实验 B — Q1/Q3 回应: Hybrid 在非线性上效果如何?")
    # 对比 N30-pure vs N30-hybrid
    if 'NL-PIBLS-N30' in rB and 'NL-PIBLS-Hybrid-N30' in rB:
        r_pure = rB['NL-PIBLS-N30']['rmse']
        r_hyb = rB['NL-PIBLS-Hybrid-N30']['rmse']
        if r_hyb < r_pure:
            print(f"    Hybrid vs Pure Newton (N=30): {r_pure:.4e} -> {r_hyb:.4e} (improve {(1-r_hyb/r_pure)*100:.1f}%)")
        else:
            print(f"    Hybrid vs Pure Newton (N=30): {r_pure:.4e} -> {r_hyb:.4e} (no improve)")

    pibls_b = min([k for k in rB if 'NL-PIBLS' in k], key=lambda k: rB[k]['rmse'])
    pinn_b = min([k for k in rB if 'PINN' in k], key=lambda k: rB[k]['rmse'])
    ratio_b = rB[pinn_b]['rmse'] / rB[pibls_b]['rmse']
    speed_b = rB[pinn_b]['time'] / max(rB[pibls_b]['time'], 1e-6)
    print(f"    NL-PIBLS best: {pibls_b} = {rB[pibls_b]['rmse']:.4e}")
    print(f"    PINN best:     {pinn_b} = {rB[pinn_b]['rmse']:.4e}")
    if ratio_b > 1:
        print(f"    => NL-PIBLS 仍然领先 {ratio_b:.1f}x, 速度快 {speed_b:.0f}x")
    else:
        print(f"    => PINN 加强后反超 {1/ratio_b:.1f}x")

    print("\n" + "=" * 75)
