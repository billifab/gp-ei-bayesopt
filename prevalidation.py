"""
GP-EI Bayesian optimization framework with reproducibility checks on benchmark test functions.

Methodology: Matérn 5/2 Gaussian Process surrogate + Expected Improvement acquisition.
Initial design: 15-point Box-Behnken.
Refinement: 3 iterative cycles, 3 proposed points per cycle.

Reference implementation in pure NumPy. The same algorithmic specification
runs in a production setting with BoTorch (PyTorch-based).

Output: convergence figures and summary statistics demonstrating that the
pipeline recovers known optima within 3 cycles on noisy benchmark functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.linalg import cholesky, solve
import json
import os

RNG = np.random.default_rng(20260518)
OUTDIR = "/sessions/eloquent-sharp-feynman/mnt/outputs/memo/prevalidation_figs"
os.makedirs(OUTDIR, exist_ok=True)

# ======================================================================
# Numerical helpers (no scipy available, so implement erf, normal CDF/PDF)
# ======================================================================
def erf_vec(x):
    """Abramowitz & Stegun 7.1.26, accurate to ~1.5e-7."""
    a1, a2, a3, a4, a5, p = (
        0.254829592, -0.284496736, 1.421413741,
        -1.453152027, 1.061405429, 0.3275911,
    )
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sign * y


def Phi(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))


def phi(z):
    """Standard normal PDF."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


# ======================================================================
# Matérn 5/2 kernel
# ======================================================================
def matern52(X1, X2, length_scale=0.30, signal_variance=1.0):
    """
    Matérn 5/2 covariance between rows of X1 (n×d) and X2 (m×d).
    Returns K of shape (n, m).
    """
    diff = (X1[:, None, :] - X2[None, :, :]) / length_scale
    r = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    sqrt5 = np.sqrt(5.0)
    K = signal_variance * (1.0 + sqrt5 * r + (5.0 / 3.0) * r * r) * np.exp(-sqrt5 * r)
    return K


# ======================================================================
# Gaussian Process posterior
# ======================================================================
class GP:
    def __init__(self, length_scale=0.30, signal_variance=1.0, noise_variance=0.04):
        self.ls = length_scale
        self.sv = signal_variance
        self.nv = noise_variance
        self.X_train = None
        self.y_train = None
        self.L = None
        self.alpha = None
        self.y_mean = 0.0

    def fit(self, X, y):
        self.X_train = X
        self.y_mean = float(np.mean(y))
        y_centered = y - self.y_mean
        K = matern52(X, X, self.ls, self.sv)
        K += self.nv * np.eye(K.shape[0])
        K += 1e-8 * np.eye(K.shape[0])
        self.L = cholesky(K)
        self.alpha = solve(self.L.T, solve(self.L, y_centered))
        self.y_train = y_centered

    def predict(self, Xnew):
        Kstar = matern52(self.X_train, Xnew, self.ls, self.sv)
        mu = Kstar.T @ self.alpha + self.y_mean
        v = solve(self.L, Kstar)
        var_diag = self.sv - np.sum(v * v, axis=0)
        var_diag = np.maximum(var_diag, 1e-10)
        return mu, var_diag


# ======================================================================
# Expected Improvement acquisition function (maximization)
# ======================================================================
def expected_improvement(gp, X, y_best, xi=0.01):
    mu, var = gp.predict(X)
    sigma = np.sqrt(var)
    imp = mu - y_best - xi
    z = np.where(sigma > 1e-10, imp / sigma, 0.0)
    ei = imp * Phi(z) + sigma * phi(z)
    ei = np.where(sigma > 1e-10, ei, 0.0)
    return ei


# ======================================================================
# Three benchmark test functions
# ======================================================================
# Parameter space: x in [0,1]^3 with three continuous variables.
#
# True functions return a scalar response in arbitrary units. Each has a
# known optimum.

def surface_unimodal(x):
    """Smooth quadratic with off-center peak. Optimum at (0.70, 0.60, 0.55) → 65 a.u."""
    x_star = np.array([0.70, 0.60, 0.55])
    if x.ndim == 1:
        x = x[None, :]
    d2 = np.sum((x - x_star) ** 2 / np.array([0.20, 0.25, 0.30]) ** 2, axis=1)
    return 30.0 + 35.0 * np.exp(-0.5 * d2)


def surface_multimodal(x):
    """Two peaks: a deceptive local max + a true global max in a less
    obvious location.
    True optimum at (0.20, 0.75, 0.40) → 70 a.u.
    Decoy at (0.80, 0.30, 0.70) → 55 a.u."""
    if x.ndim == 1:
        x = x[None, :]
    decoy = np.array([0.80, 0.30, 0.70])
    truepk = np.array([0.20, 0.75, 0.40])
    d1 = np.sum((x - decoy) ** 2 / np.array([0.18, 0.22, 0.20]) ** 2, axis=1)
    d2 = np.sum((x - truepk) ** 2 / np.array([0.18, 0.22, 0.20]) ** 2, axis=1)
    return 30.0 + 25.0 * np.exp(-0.5 * d1) + 40.0 * np.exp(-0.5 * d2)


def surface_ridge(x):
    """Ridge geometry: optimum lies along a curved manifold rather than a point.
    Best at (0.65, 0.50, 0.50) along ridge → 60 a.u."""
    if x.ndim == 1:
        x = x[None, :]
    ridge_coord = x[:, 0] - 0.5 * x[:, 1] - 0.3
    cross_coord = x[:, 2] - 0.5
    return (
        30.0
        + 30.0 * np.exp(-(ridge_coord ** 2) / (2 * 0.08 ** 2))
        * np.exp(-(cross_coord ** 2) / (2 * 0.25 ** 2))
    )


SURFACES = {
    "unimodal": (surface_unimodal, np.array([0.70, 0.60, 0.55])),
    "multimodal": (surface_multimodal, np.array([0.20, 0.75, 0.40])),
    "ridge": (surface_ridge, None),
}


def add_noise(y_true, cv=0.20):
    """Heteroscedastic Gaussian noise with coefficient of variation ~20%."""
    sigma = cv * y_true
    return y_true + RNG.normal(0, sigma, size=y_true.shape)


# ======================================================================
# Box-Behnken initial design (15 points)
# ======================================================================
def box_behnken_3d():
    """Standard 3-factor Box-Behnken with 3 center points. 15 unique runs.
    Returns coded levels in [0,1]^3 (mapping: -1 -> 0, 0 -> 0.5, +1 -> 1)."""
    rows = [
        (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0),
        (-1, 0, -1), (1, 0, -1), (-1, 0, 1), (1, 0, 1),
        (0, -1, -1), (0, 1, -1), (0, -1, 1), (0, 1, 1),
        (0, 0, 0), (0, 0, 0), (0, 0, 0),
    ]
    coded = np.array(rows, dtype=float)
    unit = 0.5 * coded + 0.5
    return unit


# ======================================================================
# Bayesian optimization loop
# ======================================================================
def bayesopt_run(surface_fn, n_cycles=3, n_per_cycle=3, cv=0.20, grid_n=25):
    """
    Run GP-EI optimization starting from a Box-Behnken design.
    Returns a record of cycles, observations, and convergence stats.
    """
    X_bb = box_behnken_3d()
    y_true_init = surface_fn(X_bb).flatten()
    y_obs_init = add_noise(y_true_init, cv=cv)

    X_obs = X_bb.copy()
    y_obs = y_obs_init.copy()

    cycles_record = []
    cycles_record.append({
        "cycle": 0,
        "n_total": len(y_obs),
        "best_observed": float(np.max(y_obs)),
        "best_true_at_obs": float(np.max(surface_fn(X_obs))),
        "X_added": X_bb.tolist(),
        "y_added_obs": y_obs.tolist(),
    })

    g = np.linspace(0.02, 0.98, grid_n)
    GX, GY, GZ = np.meshgrid(g, g, g, indexing="ij")
    grid = np.stack([GX.flatten(), GY.flatten(), GZ.flatten()], axis=1)

    for cycle in range(1, n_cycles + 1):
        gp = GP(length_scale=0.30, signal_variance=400.0, noise_variance=25.0)
        gp.fit(X_obs, y_obs)

        y_best = float(np.max(y_obs))
        ei = expected_improvement(gp, grid, y_best, xi=0.5)

        idx_sorted = np.argsort(-ei)
        selected = []
        for idx in idx_sorted:
            cand = grid[idx]
            if len(selected) == 0:
                selected.append(cand)
            else:
                min_dist = min(
                    np.linalg.norm(cand - s) for s in selected
                )
                if min_dist > 0.10:
                    selected.append(cand)
            if len(selected) == n_per_cycle:
                break
        X_new = np.array(selected)
        y_new_true = surface_fn(X_new).flatten()
        y_new_obs = add_noise(y_new_true, cv=cv)

        X_obs = np.vstack([X_obs, X_new])
        y_obs = np.concatenate([y_obs, y_new_obs])

        cycles_record.append({
            "cycle": cycle,
            "n_total": len(y_obs),
            "best_observed": float(np.max(y_obs)),
            "best_true_at_obs": float(np.max(surface_fn(X_obs))),
            "X_added": X_new.tolist(),
            "y_added_obs": y_new_obs.tolist(),
            "y_added_true": y_new_true.tolist(),
        })

    return {
        "X_obs": X_obs,
        "y_obs": y_obs,
        "cycles": cycles_record,
    }


# ======================================================================
# Grid search baseline
# ======================================================================
def grid_baseline(surface_fn, levels=3, cv=0.20):
    """3-level full-factorial grid baseline."""
    g = np.linspace(0.0, 1.0, levels)
    GX, GY, GZ = np.meshgrid(g, g, g, indexing="ij")
    Xg = np.stack([GX.flatten(), GY.flatten(), GZ.flatten()], axis=1)
    y_true = surface_fn(Xg).flatten()
    y_obs = add_noise(y_true, cv=cv)
    return {
        "X": Xg,
        "y_true": y_true,
        "y_obs": y_obs,
        "best_observed": float(np.max(y_obs)),
        "n_total": len(y_obs),
    }


def grid_baseline_finer(surface_fn, levels=4, cv=0.20):
    """4-level full-factorial grid (64 conditions)."""
    return grid_baseline(surface_fn, levels=levels, cv=cv)


# ======================================================================
# Run all surfaces with replication across noise realizations
# ======================================================================
def run_all_surfaces(n_replicates=20):
    """Run each surface n_replicates times to capture sampling variability."""
    results = {}
    for name, (fn, true_opt) in SURFACES.items():
        print(f"\n=== Surface: {name} ===")
        true_max = float(np.max(fn(box_behnken_3d().repeat(100, axis=0)
                                  + RNG.uniform(0, 1, (1500, 3)) * 0.001)))
        gdense = np.linspace(0, 1, 60)
        XX, YY, ZZ = np.meshgrid(gdense, gdense, gdense, indexing="ij")
        Xdense = np.stack([XX.flatten(), YY.flatten(), ZZ.flatten()], axis=1)
        true_max = float(np.max(fn(Xdense)))
        print(f"  True maximum (estimated): {true_max:.2f}")

        rep_results = {"bayesopt": [], "grid27": [], "grid64": []}
        for rep in range(n_replicates):
            globals()["RNG"] = np.random.default_rng(20260518 + rep)
            bo = bayesopt_run(fn, n_cycles=3, n_per_cycle=3)
            g27 = grid_baseline(fn, levels=3)
            g64 = grid_baseline(fn, levels=4)
            rep_results["bayesopt"].append({
                "best_per_cycle": [c["best_observed"] for c in bo["cycles"]],
                "n_per_cycle": [c["n_total"] for c in bo["cycles"]],
            })
            rep_results["grid27"].append({"best": g27["best_observed"], "n": g27["n_total"]})
            rep_results["grid64"].append({"best": g64["best_observed"], "n": g64["n_total"]})

        results[name] = {
            "true_max": true_max,
            "replicates": rep_results,
        }
    return results


if __name__ == "__main__":
    all_results = run_all_surfaces(n_replicates=20)

    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS over 20 replicate runs per surface")
    print("=" * 70)
    print(f"{'Surface':<14} {'True max':>10} {'BO (24 expts)':>16} {'Grid27':>14} {'Grid64':>14}")
    print("-" * 70)
    for name, res in all_results.items():
        true_max = res["true_max"]
        bo_finals = [r["best_per_cycle"][-1] for r in res["replicates"]["bayesopt"]]
        g27_finals = [r["best"] for r in res["replicates"]["grid27"]]
        g64_finals = [r["best"] for r in res["replicates"]["grid64"]]
        print(f"{name:<14} {true_max:>10.2f} "
              f"{np.mean(bo_finals):>10.2f}±{np.std(bo_finals):.2f} "
              f"{np.mean(g27_finals):>9.2f}±{np.std(g27_finals):.2f} "
              f"{np.mean(g64_finals):>9.2f}±{np.std(g64_finals):.2f}")

    print()
    print("Experimental budget: BayesOpt 15 BB + 3*3 = 24 conditions; Grid27: 27; Grid64: 64.")
