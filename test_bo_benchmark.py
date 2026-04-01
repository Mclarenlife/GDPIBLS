"""
BO-PIBLS vs PINN Benchmark on Classical PDEs

B1: 2D Helmholtz        -Delta u - k^2 u = f
B2: 1D Burgers (x,t)    u_t + u*u_x - nu*u_xx = f
B3: 1D Allen-Cahn (x,t) u_t - eps^2*u_xx - u + u^3 = f
B4: 2D Navier-Stokes    Kovasznay flow (steady, Re=20)

All use manufactured solutions with known exact answers.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import time
import copy

from bo_pibls import BOPIBLS

torch.set_default_dtype(torch.float64)

# ================================================================
# Generic PINN
# ================================================================
class GenericPINN(nn.Module):
    def __init__(self, D_in, D_out, layers=[64, 64, 64, 64]):
        super().__init__()
        dims = [D_in] + layers + [D_out]
        self.linears = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])
        for lin in self.linears[:-1]:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.xavier_normal_(self.linears[-1].weight)
        nn.init.zeros_(self.linears[-1].bias)

    def forward(self, X):
        h = X
        for lin in self.linears[:-1]:
            h = torch.tanh(lin(h))
        return self.linears[-1](h)


def train_pinn(model, loss_fn, epochs_adam=3000, lr=1e-3,
               epochs_lbfgs=500, verbose=True):
    """Train PINN with Adam + L-BFGS."""
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    best_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs_adam):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (epoch % 500 == 0 or epoch == epochs_adam - 1):
            print(f"    Adam [{epoch:>4d}/{epochs_adam}] loss={loss.item():.4e}")

    model.load_state_dict(best_state)

    if epochs_lbfgs > 0:
        opt_lbfgs = torch.optim.LBFGS(
            params, lr=1.0, max_iter=20,
            line_search_fn='strong_wolfe', history_size=50
        )
        step = [0]
        def closure():
            opt_lbfgs.zero_grad()
            loss = loss_fn()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 10.0)
            if verbose and step[0] % 100 == 0:
                print(f"    L-BFGS [{step[0]}] loss={loss.item():.4e}")
            step[0] += 1
            return loss
        for _ in range(epochs_lbfgs):
            opt_lbfgs.step(closure)

        final_loss = loss_fn().item()
        if verbose:
            print(f"    L-BFGS final: loss={final_loss:.4e}")


# ================================================================
# Helpers
# ================================================================
def make_interior_2d(x_range, y_range, nx, ny):
    x = np.linspace(x_range[0], x_range[1], nx + 2)[1:-1]
    y = np.linspace(y_range[0], y_range[1], ny + 2)[1:-1]
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def make_boundary_2d(x_range, y_range, n_per_side):
    x0, x1 = x_range
    y0, y1 = y_range
    t = np.linspace(0, 1, n_per_side)
    sides = [
        np.column_stack([x0 + (x1 - x0) * t, np.full(n_per_side, y0)]),
        np.column_stack([x0 + (x1 - x0) * t, np.full(n_per_side, y1)]),
        np.column_stack([np.full(n_per_side, x0), y0 + (y1 - y0) * t]),
        np.column_stack([np.full(n_per_side, x1), y0 + (y1 - y0) * t]),
    ]
    return np.vstack(sides)


def make_boundary_xt(x_range, t_range, n_bc, n_ic):
    """For time-dependent 1D PDEs: spatial BC + initial condition."""
    x0, x1 = x_range
    t0, t1 = t_range
    # Spatial boundaries
    t_pts = np.linspace(t0, t1, n_bc)
    left = np.column_stack([np.full(n_bc, x0), t_pts])
    right = np.column_stack([np.full(n_bc, x1), t_pts])
    # Initial condition
    x_pts = np.linspace(x0, x1, n_ic)
    ic = np.column_stack([x_pts, np.full(n_ic, t0)])
    return np.vstack([left, right, ic])


def make_eval_2d(x_range, y_range, nx, ny):
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def rmse(pred, exact):
    return np.sqrt(np.mean((pred - exact) ** 2))


def rel_l2(pred, exact):
    return np.sqrt(np.sum((pred - exact)**2) / np.sum(exact**2 + 1e-30))


# ================================================================
# B1: Helmholtz  -Delta u - k^2 u = f  on [0,1]^2
# ================================================================
def run_helmholtz(k=3.0, verbose=True):
    print("\n" + "=" * 70)
    print("B1: 2D Helmholtz  -Delta u - k^2 u = f,  k =", k)
    print("=" * 70)

    exact_fn = lambda X: np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    source_fn = lambda X: (2 * np.pi**2 - k**2) * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    bc_fn = lambda X: np.zeros(len(X))  # sin(pi*0)=sin(pi*1)=0

    X_int = make_interior_2d([0, 1], [0, 1], 30, 30)
    X_bc = make_boundary_2d([0, 1], [0, 1], 50)
    X_eval = make_eval_2d([0, 1], [0, 1], 50, 50)
    exact_eval = exact_fn(X_eval)

    results = {'name': 'Helmholtz'}

    # --- BO-PIBLS ---
    print("\n>>> BO-PIBLS")
    bo = BOPIBLS(n_map=50, n_enh=50, ridge=1e-6, bc_weight=10.0,
                 lr=5e-3, epochs=300, lr_lbfgs=0.5, epochs_lbfgs=100,
                 seed=42, verbose=verbose)

    X_int_t = torch.tensor(X_int)
    X_bc_t = torch.tensor(X_bc)
    f_vals = torch.tensor(source_fn(X_int))
    g_vals = torch.tensor(bc_fn(X_bc))

    def helmholtz_loss():
        X_all = torch.cat([X_int_t, X_bc_t], dim=0)
        Ni = X_int_t.shape[0]
        H, H_xd, H_xdxd, H_lap = bo._build_features_full(X_all)
        Hi, Hli = H[:Ni], H_lap[:Ni]
        Hb = H[Ni:]
        # -Delta u - k^2 u = f  =>  (-H_lap - k^2 H) beta = f
        A_int = -Hli - k**2 * Hi
        A = torch.cat([A_int, bo.bc_weight * Hb], dim=0)
        b = torch.cat([f_vals, bo.bc_weight * g_vals], dim=0)
        beta = bo._solve_beta(A, b)
        # True residual
        u_int = Hi @ beta
        lap_u = Hli @ beta
        pde_res = -lap_u - k**2 * u_int - f_vals
        L_pde = torch.mean(pde_res ** 2)
        u_bc = Hb @ beta
        L_bc = torch.mean((u_bc - g_vals) ** 2)
        loss = L_pde + bo.bc_weight * L_bc
        return loss, beta, L_pde.item(), L_bc.item()

    t0 = time.time()
    bo.fit_custom(D=2, compute_loss_fn=helmholtz_loss)
    t_bo = time.time() - t0

    with torch.no_grad():
        Ht, _, _, _ = bo._build_features_full(torch.tensor(X_eval))
        pred_bo = (Ht @ bo._final_beta).numpy()
    r_bo = rmse(pred_bo, exact_eval)
    print(f"  BO-PIBLS  RMSE={r_bo:.4e}  time={t_bo:.1f}s")
    results['bo'] = {'rmse': r_bo, 'time': t_bo}

    # --- PINN ---
    print("\n>>> PINN (4L-64W)")
    torch.manual_seed(42)
    pinn = GenericPINN(2, 1, [64, 64, 64, 64])
    X_int_p = torch.tensor(X_int, requires_grad=True)
    X_bc_p = torch.tensor(X_bc)
    bc_vals_p = torch.tensor(bc_fn(X_bc)).unsqueeze(1)
    f_int_p = torch.tensor(source_fn(X_int)).unsqueeze(1)

    def pinn_loss_helm():
        u = pinn(X_int_p)
        grads = torch.autograd.grad(u.sum(), X_int_p, create_graph=True)[0]
        u_x, u_y = grads[:, 0:1], grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), X_int_p, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), X_int_p, create_graph=True)[0][:, 1:2]
        pde = -(u_xx + u_yy) - k**2 * u - f_int_p
        u_bc = pinn(X_bc_p)
        return torch.mean(pde**2) + 10.0 * torch.mean((u_bc - bc_vals_p)**2)

    t0 = time.time()
    train_pinn(pinn, pinn_loss_helm, epochs_adam=3000, lr=1e-3,
               epochs_lbfgs=500, verbose=verbose)
    t_pinn = time.time() - t0

    with torch.no_grad():
        pred_pinn = pinn(torch.tensor(X_eval)).numpy().ravel()
    r_pinn = rmse(pred_pinn, exact_eval)
    print(f"  PINN  RMSE={r_pinn:.4e}  time={t_pinn:.1f}s")
    results['pinn'] = {'rmse': r_pinn, 'time': t_pinn}

    return results


# ================================================================
# B2: Burgers  u_t + u*u_x - nu*u_xx = f  on [0,1]x[0,1]
# ================================================================
def run_burgers(nu=0.01 / np.pi, verbose=True):
    print("\n" + "=" * 70)
    print(f"B2: 1D Burgers  u_t + u*u_x - nu*u_xx = f,  nu={nu:.5f}")
    print("=" * 70)

    # Manufactured solution: u = exp(-t)*sin(pi*x)
    exact_fn = lambda X: np.exp(-X[:, 1]) * np.sin(np.pi * X[:, 0])
    source_fn = lambda X: (
        np.exp(-X[:, 1]) * np.sin(np.pi * X[:, 0]) * (nu * np.pi**2 - 1)
        + 0.5 * np.pi * np.exp(-2 * X[:, 1]) * np.sin(2 * np.pi * X[:, 0])
    )
    bc_fn = lambda X: exact_fn(X)  # exact on all boundaries/IC

    X_int = make_interior_2d([0, 1], [0, 1], 30, 30)
    X_bc = make_boundary_xt([0, 1], [0, 1], n_bc=40, n_ic=40)
    X_eval = make_eval_2d([0, 1], [0, 1], 50, 50)
    exact_eval = exact_fn(X_eval)

    results = {'name': 'Burgers'}

    # --- BO-PIBLS (Newton-in-the-loop for u*u_x nonlinearity) ---
    print("\n>>> BO-PIBLS (Newton-in-the-loop)")
    bo = BOPIBLS(n_map=50, n_enh=50, ridge=1e-6, bc_weight=10.0,
                 lr=5e-3, epochs=300, lr_lbfgs=0.5, epochs_lbfgs=100,
                 seed=42, verbose=verbose)

    X_int_t = torch.tensor(X_int)
    X_bc_t = torch.tensor(X_bc)
    f_vals = torch.tensor(source_fn(X_int))
    g_vals = torch.tensor(bc_fn(X_bc))
    beta_warm = [None]

    def burgers_loss():
        X_all = torch.cat([X_int_t, X_bc_t], dim=0)
        Ni = X_int_t.shape[0]
        H, H_xd, H_xdxd, H_lap = bo._build_features_full(X_all)
        Hi = H[:Ni]
        Hb = H[Ni:]
        H_x = H_xd[0][:Ni]    # du/dx (spatial)
        H_t = H_xd[1][:Ni]    # du/dt (time)
        H_xx = H_xdxd[0][:Ni]  # d2u/dx2

        n_feat = Hi.shape[1]

        # Initial beta (warmstart or linear solve)
        if beta_warm[0] is not None and beta_warm[0].shape[0] == n_feat:
            beta = beta_warm[0].detach().clone()
        else:
            A_lin = torch.cat([H_t - nu * H_xx, bo.bc_weight * Hb], dim=0)
            b_lin = torch.cat([f_vals, bo.bc_weight * g_vals], dim=0)
            with torch.no_grad():
                beta = bo._solve_beta(A_lin, b_lin).detach()

        # Newton iterations
        for _ in range(5):
            u_k = (Hi.detach() @ beta).detach()
            u_x_k = (H_x.detach() @ beta).detach()
            # Newton-linearized: H_t + diag(u_x_k)*H + diag(u_k)*H_x - nu*H_xx
            A_int = H_t + u_x_k.unsqueeze(1) * Hi + u_k.unsqueeze(1) * H_x - nu * H_xx
            b_int = f_vals + u_k * u_x_k  # RHS = f + u^k * u^k_x
            A = torch.cat([A_int, bo.bc_weight * Hb], dim=0)
            b = torch.cat([b_int, bo.bc_weight * g_vals], dim=0)
            beta = bo._solve_beta(A, b)

        beta_warm[0] = beta.detach().clone()

        # True residual
        u = Hi @ beta
        u_t = H_t @ beta
        u_x = H_x @ beta
        u_xx = H_xx @ beta
        pde_res = u_t + u * u_x - nu * u_xx - f_vals
        L_pde = torch.mean(pde_res ** 2)
        u_bc = Hb @ beta
        L_bc = torch.mean((u_bc - g_vals) ** 2)
        loss = L_pde + bo.bc_weight * L_bc
        return loss, beta, L_pde.item(), L_bc.item()

    t0 = time.time()
    bo.fit_custom(D=2, compute_loss_fn=burgers_loss)
    t_bo = time.time() - t0

    with torch.no_grad():
        Ht, _, _, _ = bo._build_features_full(torch.tensor(X_eval))
        pred_bo = (Ht @ bo._final_beta).numpy()
    r_bo = rmse(pred_bo, exact_eval)
    print(f"  BO-PIBLS  RMSE={r_bo:.4e}  time={t_bo:.1f}s")
    results['bo'] = {'rmse': r_bo, 'time': t_bo}

    # --- PINN ---
    print("\n>>> PINN (4L-64W)")
    torch.manual_seed(42)
    pinn = GenericPINN(2, 1, [64, 64, 64, 64])
    X_int_p = torch.tensor(X_int, requires_grad=True)
    X_bc_p = torch.tensor(X_bc)
    f_int_p = torch.tensor(source_fn(X_int)).unsqueeze(1)
    bc_vals_p = torch.tensor(bc_fn(X_bc)).unsqueeze(1)

    def pinn_loss_burg():
        u = pinn(X_int_p)
        grads = torch.autograd.grad(u.sum(), X_int_p, create_graph=True)[0]
        u_x, u_t = grads[:, 0:1], grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), X_int_p, create_graph=True)[0][:, 0:1]
        pde = u_t + u * u_x - nu * u_xx - f_int_p
        u_bc = pinn(X_bc_p)
        return torch.mean(pde**2) + 10.0 * torch.mean((u_bc - bc_vals_p)**2)

    t0 = time.time()
    train_pinn(pinn, pinn_loss_burg, epochs_adam=3000, lr=1e-3,
               epochs_lbfgs=500, verbose=verbose)
    t_pinn = time.time() - t0

    with torch.no_grad():
        pred_pinn = pinn(torch.tensor(X_eval)).numpy().ravel()
    r_pinn = rmse(pred_pinn, exact_eval)
    print(f"  PINN  RMSE={r_pinn:.4e}  time={t_pinn:.1f}s")
    results['pinn'] = {'rmse': r_pinn, 'time': t_pinn}

    return results


# ================================================================
# B3: Allen-Cahn  u_t - eps^2*u_xx - u + u^3 = f  on [0,1]x[0,1]
# ================================================================
def run_allen_cahn(eps=0.1, verbose=True):
    print("\n" + "=" * 70)
    print(f"B3: Allen-Cahn  u_t - eps^2*u_xx - u + u^3 = f,  eps={eps}")
    print("=" * 70)

    # Manufactured solution: u = exp(-t)*sin(pi*x)
    exact_fn = lambda X: np.exp(-X[:, 1]) * np.sin(np.pi * X[:, 0])
    source_fn = lambda X: (
        np.exp(-X[:, 1]) * np.sin(np.pi * X[:, 0]) * (eps**2 * np.pi**2 - 2)
        + np.exp(-3 * X[:, 1]) * np.sin(np.pi * X[:, 0])**3
    )
    bc_fn = lambda X: exact_fn(X)

    X_int = make_interior_2d([0, 1], [0, 1], 30, 30)
    X_bc = make_boundary_xt([0, 1], [0, 1], n_bc=40, n_ic=40)
    X_eval = make_eval_2d([0, 1], [0, 1], 50, 50)
    exact_eval = exact_fn(X_eval)

    results = {'name': 'Allen-Cahn'}

    # --- BO-PIBLS ---
    print("\n>>> BO-PIBLS (Newton-in-the-loop)")
    bo = BOPIBLS(n_map=50, n_enh=50, ridge=1e-6, bc_weight=10.0,
                 lr=5e-3, epochs=300, lr_lbfgs=0.5, epochs_lbfgs=100,
                 seed=42, verbose=verbose)

    X_int_t = torch.tensor(X_int)
    X_bc_t = torch.tensor(X_bc)
    f_vals = torch.tensor(source_fn(X_int))
    g_vals = torch.tensor(bc_fn(X_bc))
    beta_warm = [None]

    def ac_loss():
        X_all = torch.cat([X_int_t, X_bc_t], dim=0)
        Ni = X_int_t.shape[0]
        H, H_xd, H_xdxd, H_lap = bo._build_features_full(X_all)
        Hi = H[:Ni]
        Hb = H[Ni:]
        H_t = H_xd[1][:Ni]     # du/dt
        H_xx = H_xdxd[0][:Ni]  # d2u/dx2
        n_feat = Hi.shape[1]

        if beta_warm[0] is not None and beta_warm[0].shape[0] == n_feat:
            beta = beta_warm[0].detach().clone()
        else:
            A_lin = torch.cat([H_t - eps**2 * H_xx - Hi, bo.bc_weight * Hb], dim=0)
            b_lin = torch.cat([f_vals, bo.bc_weight * g_vals], dim=0)
            with torch.no_grad():
                beta = bo._solve_beta(A_lin, b_lin).detach()

        # Newton: u_t - eps^2*u_xx - u + u^3 = f
        # Linearized: H_t - eps^2*H_xx + (3u_k^2 - 1)*H
        for _ in range(5):
            u_k = (Hi.detach() @ beta).detach()
            coeff = 3.0 * u_k ** 2 - 1.0
            A_int = H_t - eps**2 * H_xx + coeff.unsqueeze(1) * Hi
            b_int = f_vals + 2.0 * u_k ** 3  # f + 2*u_k^3
            A = torch.cat([A_int, bo.bc_weight * Hb], dim=0)
            b = torch.cat([b_int, bo.bc_weight * g_vals], dim=0)
            beta = bo._solve_beta(A, b)

        beta_warm[0] = beta.detach().clone()

        # True residual
        u = Hi @ beta
        u_t_val = H_t @ beta
        u_xx_val = H_xx @ beta
        pde_res = u_t_val - eps**2 * u_xx_val - u + u**3 - f_vals
        L_pde = torch.mean(pde_res ** 2)
        u_bc = Hb @ beta
        L_bc = torch.mean((u_bc - g_vals) ** 2)
        loss = L_pde + bo.bc_weight * L_bc
        return loss, beta, L_pde.item(), L_bc.item()

    t0 = time.time()
    bo.fit_custom(D=2, compute_loss_fn=ac_loss)
    t_bo = time.time() - t0

    with torch.no_grad():
        Ht, _, _, _ = bo._build_features_full(torch.tensor(X_eval))
        pred_bo = (Ht @ bo._final_beta).numpy()
    r_bo = rmse(pred_bo, exact_eval)
    print(f"  BO-PIBLS  RMSE={r_bo:.4e}  time={t_bo:.1f}s")
    results['bo'] = {'rmse': r_bo, 'time': t_bo}

    # --- PINN ---
    print("\n>>> PINN (4L-64W)")
    torch.manual_seed(42)
    pinn = GenericPINN(2, 1, [64, 64, 64, 64])
    X_int_p = torch.tensor(X_int, requires_grad=True)
    X_bc_p = torch.tensor(X_bc)
    f_int_p = torch.tensor(source_fn(X_int)).unsqueeze(1)
    bc_vals_p = torch.tensor(bc_fn(X_bc)).unsqueeze(1)

    def pinn_loss_ac():
        u = pinn(X_int_p)
        grads = torch.autograd.grad(u.sum(), X_int_p, create_graph=True)[0]
        u_x, u_t = grads[:, 0:1], grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), X_int_p, create_graph=True)[0][:, 0:1]
        pde = u_t - eps**2 * u_xx - u + u**3 - f_int_p
        u_bc = pinn(X_bc_p)
        return torch.mean(pde**2) + 10.0 * torch.mean((u_bc - bc_vals_p)**2)

    t0 = time.time()
    train_pinn(pinn, pinn_loss_ac, epochs_adam=3000, lr=1e-3,
               epochs_lbfgs=500, verbose=verbose)
    t_pinn = time.time() - t0

    with torch.no_grad():
        pred_pinn = pinn(torch.tensor(X_eval)).numpy().ravel()
    r_pinn = rmse(pred_pinn, exact_eval)
    print(f"  PINN  RMSE={r_pinn:.4e}  time={t_pinn:.1f}s")
    results['pinn'] = {'rmse': r_pinn, 'time': t_pinn}

    return results


# ================================================================
# B4: Navier-Stokes (Kovasznay flow, 2D steady, Re=20)
# ================================================================
def run_navier_stokes(Re=20, verbose=True):
    print("\n" + "=" * 70)
    print(f"B4: Navier-Stokes (Kovasznay flow, Re={Re})")
    print("=" * 70)

    lam = Re / 2.0 - np.sqrt(Re**2 / 4.0 + 4 * np.pi**2)

    def exact_u(X):
        return 1.0 - np.exp(lam * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])

    def exact_v(X):
        return lam / (2 * np.pi) * np.exp(lam * X[:, 0]) * np.sin(2 * np.pi * X[:, 1])

    def exact_p(X):
        return 0.5 * (1.0 - np.exp(2 * lam * X[:, 0]))

    x_range = [-0.5, 1.0]
    y_range = [0.0, 1.0]

    X_int = make_interior_2d(x_range, y_range, 30, 30)
    X_bc = make_boundary_2d(x_range, y_range, 50)
    X_eval = make_eval_2d(x_range, y_range, 50, 50)

    u_exact_eval = exact_u(X_eval)
    v_exact_eval = exact_v(X_eval)
    p_exact_eval = exact_p(X_eval)

    results = {'name': 'Navier-Stokes'}

    # --- BO-PIBLS (multi-output Newton) ---
    print("\n>>> BO-PIBLS (system Newton, 3 outputs)")
    bo = BOPIBLS(n_map=50, n_enh=50, ridge=1e-6, bc_weight=10.0,
                 lr=5e-3, epochs=300, lr_lbfgs=0.5, epochs_lbfgs=100,
                 seed=42, verbose=verbose)

    X_int_t = torch.tensor(X_int)
    X_bc_t = torch.tensor(X_bc)
    u_bc_vals = torch.tensor(exact_u(X_bc))
    v_bc_vals = torch.tensor(exact_v(X_bc))
    # Pressure reference at first boundary point
    p_ref_val = torch.tensor([exact_p(X_bc[:1])[0]])

    beta_warm_ns = [None]

    def ns_loss():
        X_all = torch.cat([X_int_t, X_bc_t], dim=0)
        Ni = X_int_t.shape[0]
        Nb = X_bc_t.shape[0]

        H, H_xd, H_xdxd, H_lap = bo._build_features_full(X_all)
        Hi = H[:Ni]
        H_x = H_xd[0][:Ni]
        H_y = H_xd[1][:Ni]
        Hlap_i = H_lap[:Ni]
        Hb = H[Ni:]
        H_ref = H[Ni:Ni+1]  # First boundary point as pressure ref

        n_feat = Hi.shape[1]
        w = bo.bc_weight

        # Initialize stacked beta: [beta_u; beta_v; beta_p]
        if beta_warm_ns[0] is not None and beta_warm_ns[0].shape[0] == 3 * n_feat:
            beta_full = beta_warm_ns[0].detach().clone()
        else:
            beta_full = torch.zeros(3 * n_feat, dtype=torch.float64)

        # Newton iterations for coupled NS
        for _ in range(5):
            bu = beta_full[:n_feat].detach()
            bv = beta_full[n_feat:2*n_feat].detach()
            bp = beta_full[2*n_feat:].detach()

            u_k = (Hi.detach() @ bu)
            v_k = (Hi.detach() @ bv)
            u_x_k = (H_x.detach() @ bu)
            u_y_k = (H_y.detach() @ bu)
            v_x_k = (H_x.detach() @ bv)
            v_y_k = (H_y.detach() @ bv)

            zero_block = torch.zeros(Ni, n_feat, dtype=torch.float64)
            zero_bc = torch.zeros(Nb, n_feat, dtype=torch.float64)
            zero_ref = torch.zeros(1, n_feat, dtype=torch.float64)

            # Jacobian blocks (interior)
            J11 = (u_x_k.unsqueeze(1) * Hi + u_k.unsqueeze(1) * H_x
                    + v_k.unsqueeze(1) * H_y - (1.0/Re) * Hlap_i)
            J12 = u_y_k.unsqueeze(1) * Hi
            J13 = H_x

            J21 = v_x_k.unsqueeze(1) * Hi
            J22 = (u_k.unsqueeze(1) * H_x + v_y_k.unsqueeze(1) * Hi
                    + v_k.unsqueeze(1) * H_y - (1.0/Re) * Hlap_i)
            J23 = H_y

            J31 = H_x
            J32 = H_y
            J33 = zero_block

            # RHS = J*beta^k - R(beta^k) (detached)
            b1 = u_k * u_x_k + v_k * u_y_k
            b2 = u_k * v_x_k + v_k * v_y_k
            b3 = torch.zeros(Ni, dtype=torch.float64)

            # Assemble block system
            A_row1 = torch.cat([J11, J12, J13], dim=1)
            A_row2 = torch.cat([J21, J22, J23], dim=1)
            A_row3 = torch.cat([J31, J32, J33], dim=1)

            # BC rows
            A_bc_u = torch.cat([w * Hb, zero_bc, zero_bc], dim=1)
            A_bc_v = torch.cat([zero_bc, w * Hb, zero_bc], dim=1)

            # Pressure reference (first BC point)
            A_p_ref = torch.cat([zero_ref, zero_ref, H_ref], dim=1)

            A_sys = torch.cat([A_row1, A_row2, A_row3,
                              A_bc_u, A_bc_v, A_p_ref], dim=0)
            b_sys = torch.cat([b1, b2, b3,
                              w * u_bc_vals, w * v_bc_vals, p_ref_val], dim=0)

            beta_full = bo._solve_beta(A_sys, b_sys)

        beta_warm_ns[0] = beta_full.detach().clone()

        # True NS residuals
        bu_f = beta_full[:n_feat]
        bv_f = beta_full[n_feat:2*n_feat]
        bp_f = beta_full[2*n_feat:]

        u_ = Hi @ bu_f
        v_ = Hi @ bv_f
        u_x_ = H_x @ bu_f
        u_y_ = H_y @ bu_f
        v_x_ = H_x @ bv_f
        v_y_ = H_y @ bv_f
        p_x_ = H_x @ bp_f
        p_y_ = H_y @ bp_f
        lap_u = Hlap_i @ bu_f
        lap_v = Hlap_i @ bv_f

        R1 = u_ * u_x_ + v_ * u_y_ + p_x_ - (1.0/Re) * lap_u
        R2 = u_ * v_x_ + v_ * v_y_ + p_y_ - (1.0/Re) * lap_v
        R3 = u_x_ + v_y_

        L_pde = torch.mean(R1**2) + torch.mean(R2**2) + torch.mean(R3**2)

        u_bc_pred = Hb @ bu_f
        v_bc_pred = Hb @ bv_f
        L_bc = torch.mean((u_bc_pred - u_bc_vals)**2) + torch.mean((v_bc_pred - v_bc_vals)**2)

        loss = L_pde + w * L_bc
        return loss, beta_full, L_pde.item(), L_bc.item()

    t0 = time.time()
    bo.fit_custom(D=2, compute_loss_fn=ns_loss)
    t_bo = time.time() - t0

    # Predict
    with torch.no_grad():
        Ht, _, _, _ = bo._build_features_full(torch.tensor(X_eval))
        n_feat = bo.n_map + bo.n_enh
        beta_f = bo._final_beta
        pred_u = (Ht @ beta_f[:n_feat]).numpy()
        pred_v = (Ht @ beta_f[n_feat:2*n_feat]).numpy()
        pred_p = (Ht @ beta_f[2*n_feat:]).numpy()

    r_u = rmse(pred_u, u_exact_eval)
    r_v = rmse(pred_v, v_exact_eval)
    r_p = rmse(pred_p, p_exact_eval)
    r_total = np.sqrt((r_u**2 + r_v**2 + r_p**2) / 3)
    print(f"  BO-PIBLS  RMSE_u={r_u:.4e}  RMSE_v={r_v:.4e}  RMSE_p={r_p:.4e}  "
          f"avg={r_total:.4e}  time={t_bo:.1f}s")
    results['bo'] = {'rmse_u': r_u, 'rmse_v': r_v, 'rmse_p': r_p,
                     'rmse': r_total, 'time': t_bo}

    # --- PINN (3 outputs) ---
    print("\n>>> PINN (4L-64W, 3 outputs)")
    torch.manual_seed(42)
    pinn = GenericPINN(2, 3, [64, 64, 64, 64])
    X_int_p = torch.tensor(X_int, requires_grad=True)
    X_bc_p = torch.tensor(X_bc)
    u_bc_p = torch.tensor(exact_u(X_bc)).unsqueeze(1)
    v_bc_p = torch.tensor(exact_v(X_bc)).unsqueeze(1)

    def pinn_loss_ns():
        out = pinn(X_int_p)
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        grads_u = torch.autograd.grad(u.sum(), X_int_p, create_graph=True)[0]
        u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), X_int_p, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), X_int_p, create_graph=True)[0][:, 1:2]

        grads_v = torch.autograd.grad(v.sum(), X_int_p, create_graph=True)[0]
        v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2]
        v_xx = torch.autograd.grad(v_x.sum(), X_int_p, create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y.sum(), X_int_p, create_graph=True)[0][:, 1:2]

        grads_p = torch.autograd.grad(p.sum(), X_int_p, create_graph=True)[0]
        p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]

        R1 = u * u_x + v * u_y + p_x - (1.0/Re) * (u_xx + u_yy)
        R2 = u * v_x + v * v_y + p_y - (1.0/Re) * (v_xx + v_yy)
        R3 = u_x + v_y

        L_pde = torch.mean(R1**2) + torch.mean(R2**2) + torch.mean(R3**2)

        out_bc = pinn(X_bc_p)
        L_bc = (torch.mean((out_bc[:, 0:1] - u_bc_p)**2)
                + torch.mean((out_bc[:, 1:2] - v_bc_p)**2))

        return L_pde + 10.0 * L_bc

    t0 = time.time()
    train_pinn(pinn, pinn_loss_ns, epochs_adam=3000, lr=1e-3,
               epochs_lbfgs=500, verbose=verbose)
    t_pinn = time.time() - t0

    with torch.no_grad():
        out_eval = pinn(torch.tensor(X_eval))
        pred_u_p = out_eval[:, 0].numpy()
        pred_v_p = out_eval[:, 1].numpy()
        pred_p_p = out_eval[:, 2].numpy()

    r_u_p = rmse(pred_u_p, u_exact_eval)
    r_v_p = rmse(pred_v_p, v_exact_eval)
    r_p_p = rmse(pred_p_p, p_exact_eval)
    r_total_p = np.sqrt((r_u_p**2 + r_v_p**2 + r_p_p**2) / 3)
    print(f"  PINN  RMSE_u={r_u_p:.4e}  RMSE_v={r_v_p:.4e}  RMSE_p={r_p_p:.4e}  "
          f"avg={r_total_p:.4e}  time={t_pinn:.1f}s")
    results['pinn'] = {'rmse_u': r_u_p, 'rmse_v': r_v_p, 'rmse_p': r_p_p,
                       'rmse': r_total_p, 'time': t_pinn}

    return results


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    np.random.seed(42)

    all_results = []

    all_results.append(run_helmholtz(k=3.0, verbose=True))
    all_results.append(run_burgers(nu=0.01/np.pi, verbose=True))
    all_results.append(run_allen_cahn(eps=0.1, verbose=True))
    all_results.append(run_navier_stokes(Re=20, verbose=True))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Problem':<20s} {'BO-PIBLS RMSE':>14s} {'time':>7s} "
          f"{'PINN RMSE':>14s} {'time':>7s} {'BO vs PINN':>12s} {'Speedup':>8s}")
    print("-" * 90)

    for r in all_results:
        name = r['name']
        bo_r, bo_t = r['bo']['rmse'], r['bo']['time']
        pi_r, pi_t = r['pinn']['rmse'], r['pinn']['time']
        improve = (pi_r - bo_r) / pi_r * 100
        speedup = pi_t / bo_t if bo_t > 0 else float('inf')
        sign = '+' if improve > 0 else ''
        print(f"{name:<20s} {bo_r:>14.4e} {bo_t:>6.1f}s "
              f"{pi_r:>14.4e} {pi_t:>6.1f}s {sign}{improve:>10.1f}% {speedup:>7.1f}x")

    print("-" * 90)
    print("Done.")
