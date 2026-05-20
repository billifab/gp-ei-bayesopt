"""
Pre-validation of GP-EI Bayesian optimization on synthetic 3D response surfaces.

Methodology:
- Matérn 5/2 Gaussian Process surrogate
- Expected Improvement acquisition
- Initial design: 15-point Box-Behnken
- Refinement: 3 iterative cycles, 3 proposed points per cycle

This reference implementation uses only NumPy so the workflow remains transparent,
auditable, and easy to reproduce in minimal Python environments.

Output:
- results.json
- printed summary statistics comparing GP-EI BayesOpt against grid-search baselines
"""

from pathlib import Path
import json

import numpy as np
from numpy.linalg import cholesky, solve


RNG = np.random.default_rng(20260518)

HERE = Path(__file__).resolve().parent
OUTDIR = HERE
OUTDIR.mkdir(exist_ok=True)


# ======================================================================
# Numerical helpers
# ======================================================================

def erf_vec(x):
    """Vectorized approximation to the error function.

    Abramowitz & Stegun 7.1.26, accurate to approximately 1.5e-7.
    """
    a1, a2, a3, a4, a5, p = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
        0.3275911,
    )
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(
        -ax * ax
    )
    return sign * y


def Phi(z):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))


def phi(z):
    """Standard normal probability density function."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


# ======================================================================
# Matérn 5/2 kernel
# ======================================================================

def matern52(X1, X2, length_scale=0.30, signal_variance=1.0):
    """Matérn 5/2 covariance between rows of X1 and X2.

    Parameters
    ----------
    X1 : ndarray, shape (n, d)
        First set of input points.
    X2 : ndarray, shape (m, d)
        Second set of input points.
    length_scale : float
        Isotropic length scale in normalized input space.
    signal_variance : float
        Kernel signal variance.

    Returns
    -------
    ndarray, shape (n, m)
        Covariance matrix.
    """
    diff = (X1[:, None, :] - X2[None, :, :]) / length_scale
    r = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)

    sqrt5 = np.sqrt(5.0)
    K = signal_variance * (1.0 + sqrt5 * r + (5.0 / 3.0) * r * r) * np.exp(
        -sqrt5 * r
    )
    return K


# ======================================================================
# Gaussian Process posterior
# ======================================================================

class GP:
    """Simple Gaussian Process regression model with a Matérn 5/2 kernel."""

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
        """Fit the GP to observed input-output pairs."""
        self.X_train = X
        self.y_mean = float(np.mean(y))
        y_centered = y - self.y_mean

        K = matern52(X, X, self.ls, self.sv)
        K += self.nv * np.eye(K.shape[0])
        K += 1e-8 * np.eye(K.shape[0])  # numerical jitter

        self.L = cholesky(K)
        self.alpha = solve(self.L.T, solve(self.L, y_centered))
        self.y_train = y_centered

    def predict(self, Xnew):
        """Return posterior mean and marginal variance at new points."""
        Kstar = matern52(self.X_train, Xnew, self.ls, self.sv)
        mu = Kstar.T @ self.alpha + self.y_mean

        v = solve(self.L, Kstar)
        var_diag = self.sv - np.sum(v * v, axis=0)
        var_diag = np.maximum(var_diag, 1e-10)

        return mu, var_diag


# ======================================================================
# Expected Improvement acquisition function
# ======================================================================

def expected_improvement(gp, X, y_best, xi=0.01):
    """Expected Improvement acquisition function for maximization."""
    mu, var = gp.predict(X)
    sigma = np.sqrt(var)

    imp = mu - y_best - xi
    z = np.where(sigma > 1e-10, imp / sigma, 0.0)

    ei = imp * Phi(z) + sigma * phi(z)
    ei = np.where(sigma > 1e-10, ei, 0.0)

    return ei


# ======================================================================
# Synthetic response surfaces
# ======================================================================

# Parameter space:
# x in [0, 1]^3 represents three normalized experimental input variables.
#
# Response:
# Each function returns a simulated response in arbitrary units.
# The surfaces are designed to exercise distinct optimization geometries:
# smooth unimodal, deceptive multimodal, and ridge-like.


def surface_unimodal(x):
    """Smooth quadratic-like response with a single off-center peak."""
    x_star = np.array([0.70, 0.60, 0.55])

    if x.ndim == 1:
        x = x[None, :]

    d2 = np.sum(
        (x - x_star) ** 2 / np.array([0.20, 0.25, 0.30]) ** 2,
        axis=1,
    )
    return 30.0 + 35.0 * np.exp(-0.5 * d2)


def surface_multimodal(x):
    """Two-peak response with a deceptive local maximum and a true global maximum."""
    if x.ndim == 1:
        x = x[None, :]

    decoy = np.array([0.80, 0.30, 0.70])
    true_peak = np.array([0.20, 0.75, 0.40])

    d1 = np.sum(
        (x - decoy) ** 2 / np.array([0.18, 0.22, 0.20]) ** 2,
        axis=1,
    )
    d2 = np.sum(
        (x - true_peak) ** 2 / np.array([0.18, 0.22, 0.20]) ** 2,
        axis=1,
    )

    return 30.0 + 25.0 * np.exp(-0.5 * d1) + 40.0 * np.exp(-0.5 * d2)


def surface_ridge(x):
    """Ridge response where high values lie along a curved manifold."""
    if x.ndim == 1:
        x = x[None, :]

    ridge_coord = x[:, 0] - 0.5 * x[:, 1] - 0.3
    cross_coord = x[:, 2] - 0.5

    return (
        30.0
        + 30.0
        * np.exp(-(ridge_coord ** 2) / (2 * 0.08 ** 2))
        * np.exp(-(cross_coord ** 2) / (2 * 0.25 ** 2))
    )


SURFACES = {
    "unimodal": (surface_unimodal, np.array([0.70, 0.60, 0.55])),
    "multimodal": (surface_multimodal, np.array([0.20, 0.75, 0.40])),
    "ridge": (surface_ridge, None),
}


def add_noise(y_true, cv=0.20):
    """Add heteroscedastic Gaussian noise with coefficient of variation cv."""
    sigma = cv * y_true
    return y_true + RNG.normal(0, sigma, size=y_true.shape)


# ======================================================================
# Box-Behnken initial design
# ======================================================================

def box_behnken_3d():
    """Standard 3-factor Box-Behnken design with 3 center points.

    Returns
    -------
    ndarray, shape (15, 3)
        Coded levels mapped from (-1, 0, +1) to (0, 0.5, 1).
    """
    rows = [
        (-1, -1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (1, 1, 0),
        (-1, 0, -1),
        (1, 0, -1),
        (-1, 0, 1),
        (1, 0, 1),
        (0, -1, -1),
        (0, 1, -1),
        (0, -1, 1),
        (0, 1, 1),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    ]

    coded = np.array(rows, dtype=float)
    unit = 0.5 * coded + 0.5

    return unit


# ======================================================================
# Bayesian optimization loop
# ======================================================================

def bayesopt_run(surface_fn, n_cycles=3, n_per_cycle=3, cv=0.20, grid_n=25):
    """Run GP-EI optimization starting from a Box-Behnken design.

    Parameters
    ----------
    surface_fn : callable
        Synthetic response surface.
    n_cycles : int
        Number of sequential optimization cycles.
    n_per_cycle : int
        Number of points proposed per cycle.
    cv : float
        Coefficient of variation for heteroscedastic noise.
    grid_n : int
        Number of grid points per dimension for acquisition maximization.

    Returns
    -------
    dict
        Observations and per-cycle convergence records.
    """
    X_bb = box_behnken_3d()
    y_true_init = surface_fn(X_bb).flatten()
    y_obs_init = add_noise(y_true_init, cv=cv)

    X_obs = X_bb.copy()
    y_obs = y_obs_init.copy()

    cycles_record = [
        {
            "cycle": 0,
            "n_total": len(y_obs),
            "best_observed": float(np.max(y_obs)),
            "best_true_at_obs": float(np.max(surface_fn(X_obs))),
            "X_added": X_bb.tolist(),
            "y_added_obs": y_obs.tolist(),
        }
    ]

    # Candidate grid for Expected Improvement maximization.
    # Avoid exact boundaries during acquisition search.
    g = np.linspace(0.02, 0.98, grid_n)
    GX, GY, GZ = np.meshgrid(g, g, g, indexing="ij")
    grid = np.stack([GX.flatten(), GY.flatten(), GZ.flatten()], axis=1)

    for cycle in range(1, n_cycles + 1):
        gp = GP(length_scale=0.30, signal_variance=400.0, noise_variance=25.0)
        gp.fit(X_obs, y_obs)

        y_best = float(np.max(y_obs))
        ei = expected_improvement(gp, grid, y_best, xi=0.5)

        # Greedy batch selection by EI with a minimum spacing constraint.
        idx_sorted = np.argsort(-ei)
        selected = []

        for idx in idx_sorted:
            cand = grid[idx]

            if not selected:
                selected.append(cand)
            else:
                min_dist = min(np.linalg.norm(cand - s) for s in selected)
                if min_dist > 0.10:
                    selected.append(cand)

            if len(selected) == n_per_cycle:
                break

        X_new = np.array(selected)
        y_new_true = surface_fn(X_new).flatten()
        y_new_obs = add_noise(y_new_true, cv=cv)

        X_obs = np.vstack([X_obs, X_new])
        y_obs = np.concatenate([y_obs, y_new_obs])

        cycles_record.append(
            {
                "cycle": cycle,
                "n_total": len(y_obs),
                "best_observed": float(np.max(y_obs)),
                "best_true_at_obs": float(np.max(surface_fn(X_obs))),
                "X_added": X_new.tolist(),
                "y_added_obs": y_new_obs.tolist(),
                "y_added_true": y_new_true.tolist(),
            }
        )

    return {
        "X_obs": X_obs,
        "y_obs": y_obs,
        "cycles": cycles_record,
    }


# ======================================================================
# Grid-search baselines
# ======================================================================

def grid_baseline(surface_fn, levels=3, cv=0.20):
    """Full-factorial grid-search baseline."""
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
    """Convenience wrapper for a finer full-factorial grid-search baseline."""
    return grid_baseline(surface_fn, levels=levels, cv=cv)


# ======================================================================
# Run all surfaces with replicated noise realizations
# ======================================================================

def estimate_true_max(surface_fn, grid_n=60):
    """Estimate the noiseless true maximum by dense grid evaluation."""
    gdense = np.linspace(0, 1, grid_n)
    XX, YY, ZZ = np.meshgrid(gdense, gdense, gdense, indexing="ij")
    Xdense = np.stack([XX.flatten(), YY.flatten(), ZZ.flatten()], axis=1)
    return float(np.max(surface_fn(Xdense)))


def run_all_surfaces(n_replicates=20):
    """Run all synthetic surfaces across replicated noise realizations."""
    global RNG

    results = {}

    for name, (fn, _) in SURFACES.items():
        print(f"\n=== Surface: {name} ===")
        true_max = estimate_true_max(fn, grid_n=60)
        print(f"  True maximum (estimated): {true_max:.2f}")

        rep_results = {
            "bayesopt": [],
            "grid27": [],
            "grid64": [],
        }

        for rep in range(n_replicates):
            RNG = np.random.default_rng(20260518 + rep)

            bo = bayesopt_run(fn, n_cycles=3, n_per_cycle=3)
            g27 = grid_baseline(fn, levels=3)
            g64 = grid_baseline(fn, levels=4)

            rep_results["bayesopt"].append(
                {
                    "best_per_cycle": [
                        c["best_observed"] for c in bo["cycles"]
                    ],
                    "n_per_cycle": [
                        c["n_total"] for c in bo["cycles"]
                    ],
                }
            )
            rep_results["grid27"].append(
                {
                    "best": g27["best_observed"],
                    "n": g27["n_total"],
                }
            )
            rep_results["grid64"].append(
                {
                    "best": g64["best_observed"],
                    "n": g64["n_total"],
                }
            )

        results[name] = {
            "true_max": true_max,
            "replicates": rep_results,
        }

    return results


def summarize_results(results):
    """Print summary statistics for the full pre-validation run."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS — over 20 replicate runs per surface")
    print("=" * 70)
    print(
        f"{'Surface':<14} {'True max':>10} "
        f"{'BO (24 evals)':>16} {'Grid27':>14} {'Grid64':>14}"
    )
    print("-" * 70)

    for name, res in results.items():
        true_max = res["true_max"]
        bo_finals = [
            r["best_per_cycle"][-1]
            for r in res["replicates"]["bayesopt"]
        ]
        g27_finals = [
            r["best"]
            for r in res["replicates"]["grid27"]
        ]
        g64_finals = [
            r["best"]
            for r in res["replicates"]["grid64"]
        ]

        print(
            f"{name:<14} {true_max:>10.2f} "
            f"{np.mean(bo_finals):>10.2f}±{np.std(bo_finals):.2f} "
            f"{np.mean(g27_finals):>9.2f}±{np.std(g27_finals):.2f} "
            f"{np.mean(g64_finals):>9.2f}±{np.std(g64_finals):.2f}"
        )

    print()
    print(
        "Evaluation budget — BayesOpt: 15 Box-Behnken + 3*3 = 24 evaluations; "
        "Grid27: 27 evaluations; Grid64: 64 evaluations."
    )
    print(
        "Performance is reported as best noisy observed response. "
        "Best-observed values may exceed the noiseless true maximum because of simulated measurement noise."
    )


if __name__ == "__main__":
    all_results = run_all_surfaces(n_replicates=20)

    with open(OUTDIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    summarize_results(all_results)
