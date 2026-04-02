"""
综合对比实验：BO-PIBLS vs PI-ELM vs PINN vs RAR-PINN

实验内容：
1. 多种子统计（5 seeds: 42, 123, 456, 789, 1024）
2. PINN 加强版（更大网络、更多配点、更多 epochs）
3. PI-ELM 对比（= BO-PIBLS 去掉梯度步，θ 固定）
4. RAR-PINN（残差自适应配点加密）

测试问题：
P1: 低频 Poisson -Δu = f, u = sin(πx)sin(πy)
P2: 高频 Poisson -Δu = f, u = sin(3πx)sin(3πy) + 0.5sin(πx)sin(πy)
P3: 非线性 -Δu + u³ = f
P4: 强非线性 -Δu + sin(u) = f

作者: BO-PIBLS research
日期: 2026-04-01
"""

import numpy as np
import torch
import torch.nn as nn
import time
import os
import sys

# 防止 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from bo_pibls import BOPIBLS

# ============================================================
# 通用工具
# ============================================================

def make_grid_2d(n=30):
    """生成 [0,1]² 内部配点 + 边界配点"""
    x = np.linspace(0, 1, n + 2)[1:-1]
    xx, yy = np.meshgrid(x, x)
    X_int = np.column_stack([xx.ravel(), yy.ravel()])

    nb = 4 * n
    b1 = np.column_stack([np.linspace(0, 1, nb), np.zeros(nb)])
    b2 = np.column_stack([np.linspace(0, 1, nb), np.ones(nb)])
    b3 = np.column_stack([np.zeros(nb), np.linspace(0, 1, nb)])
    b4 = np.column_stack([np.ones(nb), np.linspace(0, 1, nb)])
    X_bc = np.vstack([b1, b2, b3, b4])
    return X_int, X_bc


# ============================================================
# 测试问题定义
# ============================================================

class Problem:
    """PDE 测试问题基类"""
    def __init__(self, name, source_fn, bc_fn, exact_fn,
                 g_fn_np=None, g_fn_torch=None, is_nonlinear=False):
        self.name = name
        self.source_fn = source_fn
        self.bc_fn = bc_fn
        self.exact_fn = exact_fn
        self.g_fn_np = g_fn_np
        self.g_fn_torch = g_fn_torch
        self.is_nonlinear = is_nonlinear


def get_problems():
    """返回 4 个标准测试问题"""
    problems = []

    # P1: 低频 Poisson
    def exact_p1(X):
        return np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    def source_p1(X):
        return 2 * np.pi**2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    def bc_p1(X):
        return exact_p1(X)
    problems.append(Problem("P1_LowFreq_Poisson", source_p1, bc_p1, exact_p1))

    # P2: 高频 Poisson
    def exact_p2(X):
        return (np.sin(3*np.pi*X[:, 0]) * np.sin(3*np.pi*X[:, 1])
                + 0.5 * np.sin(np.pi*X[:, 0]) * np.sin(np.pi*X[:, 1]))
    def source_p2(X):
        return (18*np.pi**2 * np.sin(3*np.pi*X[:, 0]) * np.sin(3*np.pi*X[:, 1])
                + np.pi**2 * np.sin(np.pi*X[:, 0]) * np.sin(np.pi*X[:, 1]))
    def bc_p2(X):
        return exact_p2(X)
    problems.append(Problem("P2_HighFreq_Poisson", source_p2, bc_p2, exact_p2))

    # P3: 非线性 u^3
    def exact_p3(X):
        return np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    def source_p3(X):
        u = exact_p3(X)
        return 2 * np.pi**2 * u + u**3
    def bc_p3(X):
        return exact_p3(X)
    def g_fn_np_p3(u):
        return u**3
    def g_fn_torch_p3(u):
        return u**3
    problems.append(Problem("P3_Nonlinear_u3", source_p3, bc_p3, exact_p3,
                            g_fn_np_p3, g_fn_torch_p3, is_nonlinear=True))

    # P4: 强非线性 sin(u)
    def exact_p4(X):
        return np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    def source_p4(X):
        u = exact_p4(X)
        return 2 * np.pi**2 * u + np.sin(u)
    def bc_p4(X):
        return exact_p4(X)
    def g_fn_np_p4(u):
        return np.sin(u)
    def g_fn_torch_p4(u):
        return torch.sin(u)
    problems.append(Problem("P4_Nonlinear_sinu", source_p4, bc_p4, exact_p4,
                            g_fn_np_p4, g_fn_torch_p4, is_nonlinear=True))

    return problems


# ============================================================
# PI-ELM 实现（= BO-PIBLS 去掉梯度步，θ 固定不学习）
# ============================================================

class PIELM:
    """
    Physics-Informed Extreme Learning Machine (Dwivedi 2020, JCP)

    与 BO-PIBLS 唯一区别：θ=(ω,b) 随机初始化后固定不变，
    只用伪逆 一次性求解 β。无梯度下降、无训练循环。
    """

    def __init__(self, n_hidden=100, ridge=1e-6, bc_weight=10.0,
                 seed=42, activation='fourier'):
        self.n_hidden = n_hidden
        self.ridge = ridge
        self.bc_weight = bc_weight
        self.seed = seed
        self.activation = activation
        self.W = None
        self.b = None
        self.beta = None

    def _init_weights(self, D):
        rng = np.random.RandomState(self.seed)
        # 多尺度随机频率（与 BO-PIBLS 初始化方式一致）
        n_low = self.n_hidden // 3
        n_mid = self.n_hidden // 3
        n_high = self.n_hidden - n_low - n_mid
        W_parts = []
        if n_low > 0:
            W_parts.append(rng.randn(D, n_low) * 1.0)
        if n_mid > 0:
            W_parts.append(rng.randn(D, n_mid) * 3.0)
        if n_high > 0:
            W_parts.append(rng.randn(D, n_high) * 6.0)
        self.W = np.hstack(W_parts)
        self.b = rng.rand(self.n_hidden) * 2 * np.pi

    def _features_and_laplacian(self, X):
        """sin features + 解析拉普拉斯"""
        Z = X @ self.W + self.b
        H = np.sin(Z)
        # Laplacian: ΔH_j = -||W_j||² · H_j
        W_sq_sum = np.sum(self.W**2, axis=0, keepdims=True)
        H_lap = -W_sq_sum * H
        return H, H_lap

    def fit_linear(self, X_int, X_bc, source_fn, bc_fn):
        D = X_int.shape[1]
        self._init_weights(D)

        Hi, Hli = self._features_and_laplacian(X_int)
        Hb, _ = self._features_and_laplacian(X_bc)

        f_vals = source_fn(X_int)
        g_vals = bc_fn(X_bc)

        A = np.vstack([-Hli, self.bc_weight * Hb])
        b = np.concatenate([f_vals, self.bc_weight * g_vals])

        ATA = A.T @ A + self.ridge * np.eye(A.shape[1])
        ATb = A.T @ b
        self.beta = np.linalg.solve(ATA, ATb)
        return self

    def fit_nonlinear(self, X_int, X_bc, g_fn, source_fn, bc_fn,
                      max_iter=20, damping=0.8):
        """Newton-伪逆迭代求解非线性 PDE（θ固定）"""
        D = X_int.shape[1]
        self._init_weights(D)

        Hi, Hli = self._features_and_laplacian(X_int)
        Hb, _ = self._features_and_laplacian(X_bc)

        f_vals = source_fn(X_int)
        g_vals = bc_fn(X_bc)

        # 线性初始猜测
        A0 = np.vstack([-Hli, self.bc_weight * Hb])
        b0 = np.concatenate([f_vals, self.bc_weight * g_vals])
        ATA = A0.T @ A0 + self.ridge * np.eye(A0.shape[1])
        self.beta = np.linalg.solve(ATA, A0.T @ b0)

        for k in range(max_iter):
            u_cur = Hi @ self.beta
            g_u = g_fn(u_cur)

            # 数值 g'(u)
            eps = 1e-7
            gp_u = (g_fn(u_cur + eps) - g_fn(u_cur - eps)) / (2 * eps)

            A_int = -Hli + gp_u[:, None] * Hi
            b_int = f_vals - g_u + gp_u * u_cur

            A_k = np.vstack([A_int, self.bc_weight * Hb])
            b_k = np.concatenate([b_int, self.bc_weight * g_vals])

            ATA = A_k.T @ A_k + self.ridge * np.eye(A_k.shape[1])
            beta_new = np.linalg.solve(ATA, A_k.T @ b_k)
            self.beta = (1 - damping) * self.beta + damping * beta_new

            # 收敛检查
            res = -Hli @ self.beta + g_fn(Hi @ self.beta) - f_vals
            if np.mean(res**2) < 1e-14:
                break
        return self

    def predict(self, X):
        Z = X @ self.W + self.b
        H = np.sin(Z)
        return H @ self.beta


# ============================================================
# PINN 实现（含 RAR 自适应配点加密变体）
# ============================================================

class PINN(nn.Module):
    """标准 Physics-Informed Neural Network"""

    def __init__(self, layers, activation='tanh'):
        super().__init__()
        self.nets = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.nets.append(nn.Linear(layers[i], layers[i+1]))
        self._act = torch.tanh if activation == 'tanh' else torch.relu

        # Xavier 初始化
        for m in self.nets:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        for i, layer in enumerate(self.nets[:-1]):
            x = self._act(layer(x))
        return self.nets[-1](x).squeeze(-1)


def train_pinn(problem, X_int, X_bc, layers, seed=42,
               epochs_adam=5000, epochs_lbfgs=500, lr=1e-3,
               rar_enabled=False, rar_interval=1000, rar_n_add=100,
               rar_n_cand=10000, verbose=True):
    """
    训练 PINN（支持 RAR 自适应配点加密）

    Parameters
    ----------
    rar_enabled : bool
        是否启用 RAR（Residual-based Adaptive Refinement）
    rar_interval : int
        每隔多少 epochs 执行一次 RAR
    rar_n_add : int
        每次 RAR 添加的配点数
    rar_n_cand : int
        RAR 候选配点池大小
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PINN(layers).double()
    X_int_t = torch.tensor(X_int, dtype=torch.float64, requires_grad=True)
    X_bc_t = torch.tensor(X_bc, dtype=torch.float64)

    f_vals = torch.tensor(problem.source_fn(X_int), dtype=torch.float64)
    g_vals = torch.tensor(problem.bc_fn(X_bc), dtype=torch.float64)

    bc_weight = 10.0

    def compute_loss(X_pde, f_pde):
        u = model(X_pde)
        u_sum = u.sum()
        grad_u = torch.autograd.grad(u_sum, X_pde, create_graph=True)[0]
        u_xx = torch.autograd.grad(grad_u[:, 0].sum(), X_pde, create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(grad_u[:, 1].sum(), X_pde, create_graph=True)[0][:, 1]
        lap_u = u_xx + u_yy

        if problem.is_nonlinear:
            pde_res = -lap_u + problem.g_fn_torch(u) - f_pde
        else:
            pde_res = -lap_u - f_pde
        L_pde = torch.mean(pde_res**2)

        u_bc = model(X_bc_t)
        L_bc = torch.mean((u_bc - g_vals)**2)
        return L_pde + bc_weight * L_bc, L_pde.item(), L_bc.item()

    # ------ Phase 1: Adam ------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_adam, eta_min=lr * 0.01
    )

    current_X_int = X_int_t
    current_f = f_vals

    for epoch in range(epochs_adam):
        optimizer.zero_grad()
        loss, lpde, lbc = compute_loss(current_X_int, current_f)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        if verbose and (epoch % 500 == 0 or epoch == epochs_adam - 1):
            print(f"    Adam [{epoch:>5d}/{epochs_adam}] loss={loss.item():.4e} "
                  f"pde={lpde:.4e} bc={lbc:.4e}")

        # RAR: 每隔 rar_interval 步添加残差最大处的配点
        if rar_enabled and epoch > 0 and epoch % rar_interval == 0:
            cand = np.random.rand(rar_n_cand, 2)
            cand_t = torch.tensor(cand, dtype=torch.float64, requires_grad=True)
            with torch.no_grad():
                u_c = model(cand_t)
            # 需要重新计算梯度
            cand_t2 = torch.tensor(cand, dtype=torch.float64, requires_grad=True)
            u_c2 = model(cand_t2)
            g_c = torch.autograd.grad(u_c2.sum(), cand_t2, create_graph=True)[0]
            u_xx_c = torch.autograd.grad(g_c[:, 0].sum(), cand_t2, create_graph=True)[0][:, 0]
            u_yy_c = torch.autograd.grad(g_c[:, 1].sum(), cand_t2, create_graph=True)[0][:, 1]
            lap_c = u_xx_c + u_yy_c
            f_c = torch.tensor(problem.source_fn(cand), dtype=torch.float64)
            if problem.is_nonlinear:
                res_c = (-lap_c + problem.g_fn_torch(u_c2) - f_c).detach().abs()
            else:
                res_c = (-lap_c - f_c).detach().abs()

            # 取残差最大的 rar_n_add 个点
            _, topk_idx = torch.topk(res_c, min(rar_n_add, len(res_c)))
            new_pts = cand[topk_idx.numpy()]
            new_pts_t = torch.tensor(new_pts, dtype=torch.float64, requires_grad=True)
            new_f = torch.tensor(problem.source_fn(new_pts), dtype=torch.float64)

            current_X_int = torch.tensor(
                np.vstack([current_X_int.detach().numpy(), new_pts]),
                dtype=torch.float64, requires_grad=True
            )
            current_f = torch.cat([current_f, new_f])

            if verbose:
                print(f"    RAR: added {len(new_pts)} points, "
                      f"total={current_X_int.shape[0]}")

    # ------ Phase 2: L-BFGS ------
    if epochs_lbfgs > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0,
            max_iter=20, line_search_fn='strong_wolfe',
            history_size=50
        )
        lbfgs_step = [0]

        def closure():
            optimizer_lbfgs.zero_grad()
            loss, lpde, lbc = compute_loss(current_X_int, current_f)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            lbfgs_step[0] += 1
            return loss

        for _ in range(epochs_lbfgs):
            optimizer_lbfgs.step(closure)

    return model


# ============================================================
# 评估函数
# ============================================================

def evaluate_rmse(pred, exact):
    return np.sqrt(np.mean((pred - exact)**2))


def run_bo_pibls(problem, X_int, X_bc, X_eval, seed):
    """运行 BO-PIBLS"""
    solver = BOPIBLS(
        n_map=50, n_enh=50, ridge=1e-6, bc_weight=10.0,
        lr=1e-2, epochs=400, lr_lbfgs=1.0, epochs_lbfgs=200,
        seed=seed, verbose=False, freq_init_scale=1.0
    )
    t0 = time.time()
    if problem.is_nonlinear:
        solver.fit_nonlinear(X_int, X_bc, problem.g_fn_torch,
                             problem.source_fn, problem.bc_fn,
                             n_newton=5, damping=0.8)
    else:
        solver.fit_linear(X_int, X_bc, problem.source_fn, problem.bc_fn)
    dt = time.time() - t0
    pred = solver.predict(X_eval)
    rmse = evaluate_rmse(pred, problem.exact_fn(X_eval))
    return rmse, dt


def run_pi_elm(problem, X_int, X_bc, X_eval, seed):
    """运行 PI-ELM（= BO-PIBLS 去掉梯度步）"""
    solver = PIELM(n_hidden=100, ridge=1e-6, bc_weight=10.0,
                   seed=seed, activation='fourier')
    t0 = time.time()
    if problem.is_nonlinear:
        solver.fit_nonlinear(X_int, X_bc, problem.g_fn_np,
                             problem.source_fn, problem.bc_fn)
    else:
        solver.fit_linear(X_int, X_bc, problem.source_fn, problem.bc_fn)
    dt = time.time() - t0
    pred = solver.predict(X_eval)
    rmse = evaluate_rmse(pred, problem.exact_fn(X_eval))
    return rmse, dt


def run_pinn(problem, X_int, X_bc, X_eval, seed, label="PINN",
             layers=None, epochs_adam=5000, epochs_lbfgs=500, lr=1e-3,
             rar_enabled=False):
    """运行 PINN 或 RAR-PINN"""
    if layers is None:
        layers = [2, 128, 128, 128, 128, 1]

    t0 = time.time()
    model = train_pinn(
        problem, X_int, X_bc, layers=layers, seed=seed,
        epochs_adam=epochs_adam, epochs_lbfgs=epochs_lbfgs, lr=lr,
        rar_enabled=rar_enabled, verbose=False
    )
    dt = time.time() - t0

    X_eval_t = torch.tensor(X_eval, dtype=torch.float64)
    with torch.no_grad():
        pred = model(X_eval_t).numpy()
    rmse = evaluate_rmse(pred, problem.exact_fn(X_eval))
    return rmse, dt


# ============================================================
# 主实验
# ============================================================

def main():
    print("=" * 80)
    print("综合对比实验: BO-PIBLS vs PI-ELM vs PINN vs RAR-PINN")
    print("=" * 80)

    seeds = [42, 123, 456, 789, 1024]
    problems = get_problems()

    X_int, X_bc = make_grid_2d(n=30)  # 784 内部 + 120 边界
    X_eval, _ = make_grid_2d(n=50)    # 2401 评估点

    # 方法配置
    methods = {
        'BO-PIBLS': lambda prob, s: run_bo_pibls(
            prob, X_int, X_bc, X_eval, s),
        'PI-ELM': lambda prob, s: run_pi_elm(
            prob, X_int, X_bc, X_eval, s),
        'PINN-4L-128W': lambda prob, s: run_pinn(
            prob, X_int, X_bc, X_eval, s,
            layers=[2, 128, 128, 128, 128, 1],
            epochs_adam=5000, epochs_lbfgs=500, lr=1e-3),
        'PINN-6L-128W': lambda prob, s: run_pinn(
            prob, X_int, X_bc, X_eval, s,
            layers=[2, 128, 128, 128, 128, 128, 128, 1],
            epochs_adam=5000, epochs_lbfgs=500, lr=1e-3),
        'RAR-PINN-4L': lambda prob, s: run_pinn(
            prob, X_int, X_bc, X_eval, s,
            layers=[2, 128, 128, 128, 128, 1],
            epochs_adam=8000, epochs_lbfgs=500, lr=1e-3,
            rar_enabled=True),
    }

    # 存储所有结果
    all_results = {}  # {problem_name: {method: {'rmses': [], 'times': []}}}

    for prob in problems:
        print(f"\n{'='*70}")
        print(f"Problem: {prob.name}")
        print(f"{'='*70}")

        all_results[prob.name] = {}

        for method_name, method_fn in methods.items():
            rmses = []
            times = []
            print(f"\n  Method: {method_name}")

            for i, seed in enumerate(seeds):
                print(f"    Seed {seed} ({i+1}/{len(seeds)})...", end=" ", flush=True)
                try:
                    rmse, dt = method_fn(prob, seed)
                    rmses.append(rmse)
                    times.append(dt)
                    print(f"RMSE={rmse:.4e}, time={dt:.1f}s")
                except Exception as e:
                    print(f"FAILED: {e}")
                    rmses.append(float('nan'))
                    times.append(float('nan'))

            all_results[prob.name][method_name] = {
                'rmses': rmses, 'times': times
            }

            # 单方法统计
            valid = [r for r in rmses if not np.isnan(r)]
            if valid:
                mean_r = np.mean(valid)
                std_r = np.std(valid)
                mean_t = np.nanmean(times)
                print(f"    => RMSE: {mean_r:.4e} +/- {std_r:.4e}  "
                      f"Time: {mean_t:.1f}s  (n={len(valid)}/{len(seeds)})")

    # ============================================================
    # 汇总表格
    # ============================================================
    print("\n\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 100)

    print(f"\n{'Problem':<25} {'Method':<20} {'RMSE (mean)':<14} "
          f"{'RMSE (std)':<14} {'Time (mean)':<12} {'Seeds OK':<10}")
    print("-" * 100)

    for prob_name in all_results:
        for method_name in all_results[prob_name]:
            data = all_results[prob_name][method_name]
            valid_r = [r for r in data['rmses'] if not np.isnan(r)]
            valid_t = [t for t in data['times'] if not np.isnan(t)]

            if valid_r:
                mean_r = np.mean(valid_r)
                std_r = np.std(valid_r)
                mean_t = np.mean(valid_t)
                n_ok = len(valid_r)
            else:
                mean_r, std_r, mean_t, n_ok = float('nan'), float('nan'), float('nan'), 0

            print(f"  {prob_name:<23} {method_name:<20} {mean_r:<14.4e} "
                  f"{std_r:<14.4e} {mean_t:<12.1f} {n_ok}/{len(seeds)}")
        print()

    # ============================================================
    # 消融分析：BO-PIBLS vs PI-ELM（证明梯度学习的价值）
    # ============================================================
    print("\n" + "=" * 80)
    print("ABLATION: BO-PIBLS vs PI-ELM (gradient learning value)")
    print("=" * 80)
    for prob_name in all_results:
        bo = all_results[prob_name].get('BO-PIBLS', {})
        elm = all_results[prob_name].get('PI-ELM', {})

        bo_valid = [r for r in bo.get('rmses', []) if not np.isnan(r)]
        elm_valid = [r for r in elm.get('rmses', []) if not np.isnan(r)]

        if bo_valid and elm_valid:
            bo_mean = np.mean(bo_valid)
            elm_mean = np.mean(elm_valid)
            improvement = (elm_mean - bo_mean) / elm_mean * 100
            speedup = np.mean(elm.get('times', [1])) / max(np.mean(bo.get('times', [1])), 0.01)
            print(f"  {prob_name:<25} PI-ELM={elm_mean:.4e} -> BO-PIBLS={bo_mean:.4e}  "
                  f"Improvement: {improvement:+.1f}%  SpeedRatio: {speedup:.1f}x")

    print("\n\nExperiment completed!")


if __name__ == '__main__':
    main()
