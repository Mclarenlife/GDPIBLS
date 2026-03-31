"""只运行 Problem 3 和 Problem 4，补全 benchmark 结果"""
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
from advanced_pibls import NonlinearPIBLS

np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device('cpu')


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


class PINNSolver:
    def __init__(self, layers, pde_residual_fn, bc_fn,
                 lr=1e-3, epochs=5000, lambda_bc=10.0):
        self.model = DeepPINN(layers).to(DEVICE)
        self.pde_residual_fn = pde_residual_fn
        self.bc_fn = bc_fn
        self.lr = lr
        self.epochs = epochs
        self.lambda_bc = lambda_bc
        self.loss_history = []
        self.train_time = 0.0

    def train(self, pde_data, bc_data):
        x_pde_np, y_pde_np = pde_data
        x_bc_np, y_bc_np = bc_data
        x_pde = torch.tensor(x_pde_np, dtype=torch.float64, requires_grad=True, device=DEVICE)
        y_pde = torch.tensor(y_pde_np, dtype=torch.float64, requires_grad=True, device=DEVICE)
        x_bc = torch.tensor(x_bc_np, dtype=torch.float64, device=DEVICE)
        y_bc = torch.tensor(y_bc_np, dtype=torch.float64, device=DEVICE)
        u_bc_target = torch.tensor(self.bc_fn(x_bc_np, y_bc_np), dtype=torch.float64, device=DEVICE)
        self.model.double()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.lr * 0.01)
        self.loss_history = []
        t0 = time.perf_counter()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            R = self.pde_residual_fn(self.model, x_pde, y_pde)
            loss_pde = torch.mean(R ** 2)
            u_bc_pred = self.model(x_bc, y_bc)
            loss_bc = torch.mean((u_bc_pred - u_bc_target) ** 2)
            loss = loss_pde + self.lambda_bc * loss_bc
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            self.loss_history.append(loss.item())
            if epoch % max(1, self.epochs // 5) == 0:
                print(f"  [PINN] Epoch {epoch:>5d}: L_pde={loss_pde.item():.4e}  L_bc={loss_bc.item():.4e}  total={loss.item():.4e}")
        self.train_time = time.perf_counter() - t0
        print(f"  [PINN] Done: {self.train_time:.2f}s, final loss={self.loss_history[-1]:.4e}")

    @torch.no_grad()
    def predict(self, x_np, y_np):
        self.model.eval()
        x = torch.tensor(x_np, dtype=torch.float64, device=DEVICE)
        y = torch.tensor(y_np, dtype=torch.float64, device=DEVICE)
        return self.model(x, y).cpu().numpy()

    def param_count(self):
        return sum(p.numel() for p in self.model.parameters())


def generate_interior(n):
    return np.random.uniform(0, 1, n), np.random.uniform(0, 1, n)

def generate_boundary(n_per_side):
    t = np.linspace(0, 1, n_per_side)
    x_bc = np.concatenate([t, t, np.zeros(n_per_side), np.ones(n_per_side)])
    y_bc = np.concatenate([np.zeros(n_per_side), np.ones(n_per_side), t, t])
    return x_bc, y_bc

def generate_test_grid(n=50):
    xx = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xx, xx)
    return X.flatten(), Y.flatten()

def compute_rmse(u_pred, u_exact):
    return np.sqrt(np.mean((u_pred - u_exact) ** 2))

def nonlinear_residual_cubic(model, x, y, source_fn_torch):
    u = model(x, y)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return -u_xx - u_yy + u ** 3 - source_fn_torch(x, y)

def nonlinear_residual_sin(model, x, y, source_fn_torch):
    u = model(x, y)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return -u_xx - u_yy + torch.sin(u) - source_fn_torch(x, y)


# ===================== Problem 3: -Lap(u) + u^3 = f =====================

def problem3():
    print("\n" + "=" * 70)
    print("  Problem 3: -Lap(u) + u^3 = f")
    print("  Exact: u = sin(pi*x)*sin(pi*y)")
    print("=" * 70)

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

    n_pde, n_bc = 600, 50
    x_pde, y_pde = generate_interior(n_pde)
    x_bc, y_bc = generate_boundary(n_bc)
    x_test, y_test = generate_test_grid(50)
    u_exact = exact(x_test, y_test)

    results = {}

    # NL-PIBLS
    print("\n--- NonlinearPIBLS (N=40, Newton-pinv) ---")
    np.random.seed(42)
    t0 = time.perf_counter()
    m = NonlinearPIBLS(40, 40, 'tanh', 'sigmoid',
                       residual_fn=residual_fn, bc_fn=exact,
                       dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
                       lambda_bc=20.0)
    m.fit((x_pde, y_pde), (x_bc, y_bc), max_iter=50, damping=0.8)
    t1 = time.perf_counter() - t0
    rmse = compute_rmse(m.predict(x_test, y_test), u_exact)
    results['NL-PIBLS'] = {'rmse': rmse, 'time': t1, 'loss_history': m.loss_history}
    print(f"  RMSE = {rmse:.6e}, Time = {t1:.3f}s")

    # PINN baselines
    for nl, w, ep in [(2, 64, 5000), (4, 64, 5000), (6, 64, 5000)]:
        layers = [2] + [w]*nl + [1]
        label = f'PINN-{nl}L-{w}W'
        print(f"\n--- {label} (epochs={ep}) ---")
        torch.manual_seed(42)
        sf = source_torch
        pinn = PINNSolver(layers,
                          pde_residual_fn=lambda m, x, y, _sf=sf: nonlinear_residual_cubic(m, x, y, _sf),
                          bc_fn=exact, lr=1e-3, epochs=ep, lambda_bc=20.0)
        pinn.train((x_pde, y_pde), (x_bc, y_bc))
        rmse_p = compute_rmse(pinn.predict(x_test, y_test), u_exact)
        results[label] = {'rmse': rmse_p, 'time': pinn.train_time,
                          'params': pinn.param_count(), 'loss_history': pinn.loss_history}
        print(f"  RMSE = {rmse_p:.6e}, Params = {pinn.param_count()}")

    return results


# ===================== Problem 4: -Lap(u) + sin(u) = f =====================

def problem4():
    print("\n" + "=" * 70)
    print("  Problem 4: -Lap(u) + sin(u) = f  (strong nonlinear)")
    print("  Exact: u = sin(2*pi*x)*sin(2*pi*y)")
    print("=" * 70)

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

    n_pde, n_bc = 1000, 80
    x_pde, y_pde = generate_interior(n_pde)
    x_bc, y_bc = generate_boundary(n_bc)
    x_test, y_test = generate_test_grid(50)
    u_exact = exact(x_test, y_test)

    results = {}

    # NL-PIBLS
    print("\n--- NonlinearPIBLS (N=50, Newton-pinv) ---")
    np.random.seed(42)
    t0 = time.perf_counter()
    m = NonlinearPIBLS(50, 50, 'tanh', 'sigmoid',
                       residual_fn=residual_fn, bc_fn=exact,
                       dR_du=dR_du, dR_duxx=dR_duxx, dR_duyy=dR_duyy,
                       lambda_bc=20.0)
    m.fit((x_pde, y_pde), (x_bc, y_bc), max_iter=60, damping=0.6)
    t1 = time.perf_counter() - t0
    rmse = compute_rmse(m.predict(x_test, y_test), u_exact)
    results['NL-PIBLS'] = {'rmse': rmse, 'time': t1, 'loss_history': m.loss_history}
    print(f"  RMSE = {rmse:.6e}, Time = {t1:.3f}s")

    # PINN baselines
    for nl, w, ep in [(2, 64, 5000), (4, 64, 5000), (6, 64, 8000)]:
        layers = [2] + [w]*nl + [1]
        label = f'PINN-{nl}L-{w}W'
        print(f"\n--- {label} (epochs={ep}) ---")
        torch.manual_seed(42)
        sf = source_torch
        pinn = PINNSolver(layers,
                          pde_residual_fn=lambda m, x, y, _sf=sf: nonlinear_residual_sin(m, x, y, _sf),
                          bc_fn=exact, lr=1e-3, epochs=ep, lambda_bc=20.0)
        pinn.train((x_pde, y_pde), (x_bc, y_bc))
        rmse_p = compute_rmse(pinn.predict(x_test, y_test), u_exact)
        results[label] = {'rmse': rmse_p, 'time': pinn.train_time,
                          'params': pinn.param_count(), 'loss_history': pinn.loss_history}
        print(f"  RMSE = {rmse_p:.6e}, Params = {pinn.param_count()}")

    return results


def print_table(title, results):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Method':<22s}  {'RMSE':>12s}  {'Time(s)':>10s}  {'Params':>8s}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*10}  {'-'*8}")
    for name, r in results.items():
        rmse_s = f"{r['rmse']:.4e}"
        time_s = f"{r['time']:.2f}" if 'time' in r else '-'
        params_s = str(r.get('params', '-'))
        print(f"  {name:<22s}  {rmse_s:>12s}  {time_s:>10s}  {params_s:>8s}")
    best = min(results, key=lambda k: results[k]['rmse'])
    print(f"\n  Best: {best} (RMSE = {results[best]['rmse']:.4e})")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    print("=" * 70)
    print("  PIBLS vs Deep PINN: Problem 3 & 4 (Nonlinear PDEs)")
    print("=" * 70)

    r3 = problem3()
    print_table("Problem 3: -Lap(u) + u^3 = f", r3)

    r4 = problem4()
    print_table("Problem 4: -Lap(u) + sin(u) = f", r4)

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    for pname, res in [("P3", r3), ("P4", r4)]:
        pibls_best = min([k for k in res if 'PIBLS' in k], key=lambda k: res[k]['rmse'])
        pinn_best = min([k for k in res if 'PINN' in k], key=lambda k: res[k]['rmse'])
        ratio = res[pinn_best]['rmse'] / res[pibls_best]['rmse']
        speedup = res[pinn_best]['time'] / max(res[pibls_best]['time'], 1e-6)
        winner = "NL-PIBLS" if ratio > 1 else "PINN"
        factor = ratio if ratio > 1 else 1/ratio
        print(f"  {pname}: NL-PIBLS={res[pibls_best]['rmse']:.4e}  PINN={res[pinn_best]['rmse']:.4e}  -> {winner} wins by {factor:.1f}x, speed {speedup:.0f}x")
    print("=" * 70)
