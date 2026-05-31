from pathlib import Path
import csv
import json
import numpy as np

# -----------------------------
# Global configuration
# -----------------------------
BASE_SEED = 20260525
OUT_JSON = "/Users/yiqun/Downloads/aim3_power_results.json"
OUT_CSV = "/Users/yiqun/Downloads/aim3_power_summary.csv"

# Simulation defaults. Increase N_REPLICATES to 500-1000 for final reporting.
N_REPLICATES = 10
N_INITIAL_DONORS = 3       # Aim 2 donor count
N_VALIDATION_DONORS = 5    # Aim 3 validation donor count
N_BO_CYCLES = 3
N_PER_CYCLE = 3
GRID_N = 25                # candidate grid for BO acquisition optimization
DENSE_GRID_N = 15          # grid used to estimate true maxima and optima

# Biological success thresholds
MEANINGFUL_IMPROVEMENT = 0.10  # BayesOpt must be >=10% better than RSM
NEAR_OPTIMUM_MARGIN = 0.15     # selected true response within 15% of true maximum
T_CRIT_ONE_SIDED_DF4 = 2.132   # one-sided alpha=0.05, df=4; for n=5 paired validation

# -----------------------------
# Math utilities
# -----------------------------
def erf_vec(x):
    """Abramowitz & Stegun 7.1.26 approximation, accurate to ~1.5e-7."""
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
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sign * y


def Phi(z):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))


def phi(z):
    """Standard normal probability density function."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


# -----------------------------
# Synthetic response surfaces
# -----------------------------
def surface_unimodal(x):
    """Smooth surface with a single off-center peak."""
    x = np.atleast_2d(x)
    x_star = np.array([0.70, 0.60, 0.55])
    d2 = np.sum((x - x_star) ** 2 / np.array([0.20, 0.25, 0.30]) ** 2, axis=1)
    return 30.0 + 35.0 * np.exp(-0.5 * d2)


def surface_multimodal(x):
    """Two-peak surface with a deceptive local maximum and a true global maximum."""
    x = np.atleast_2d(x)
    decoy = np.array([0.80, 0.30, 0.70])
    truepk = np.array([0.20, 0.75, 0.40])
    d1 = np.sum((x - decoy) ** 2 / np.array([0.18, 0.22, 0.20]) ** 2, axis=1)
    d2 = np.sum((x - truepk) ** 2 / np.array([0.18, 0.22, 0.20]) ** 2, axis=1)
    return 30.0 + 25.0 * np.exp(-0.5 * d1) + 40.0 * np.exp(-0.5 * d2)


def surface_ridge(x):
    """Ridge geometry with high response along a curved manifold."""
    x = np.atleast_2d(x)
    ridge_coord = x[:, 0] - 0.5 * x[:, 1] - 0.3
    cross_coord = x[:, 2] - 0.5
    return 30.0 + 30.0 * np.exp(-(ridge_coord**2) / (2 * 0.08**2)) * np.exp(
        -(cross_coord**2) / (2 * 0.25**2)
    )


def surface_null_quadratic(x):
    """
    Smooth quadratic-like surface where RSM should be close to optimal.
    This is useful for estimating false-positive or overclaiming behavior.
    """
    x = np.atleast_2d(x)
    center = np.array([0.55, 0.55, 0.55])
    z = x - center
    return 55.0 - 12.0 * z[:, 0] ** 2 - 10.0 * z[:, 1] ** 2 - 8.0 * z[:, 2] ** 2


SURFACES = {
    "unimodal": surface_unimodal,
    "multimodal": surface_multimodal,
    "ridge": surface_ridge,
    "null_quadratic": surface_null_quadratic,
}


# -----------------------------
# Designs and model matrices
# -----------------------------
def box_behnken_3d():
    """Standard 3-factor Box-Behnken design with three center points, returned in [0,1]^3."""
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
    return 0.5 * coded + 0.5


def candidate_grid(n=25, low=0.02, high=0.98):
    g = np.linspace(low, high, n)
    GX, GY, GZ = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)


def rsm_design_matrix(X):
    """Quadratic RSM matrix in coded coordinates z=-1..1."""
    z = 2.0 * X - 1.0
    f, inten, dur = z[:, 0], z[:, 1], z[:, 2]
    return np.column_stack([
        np.ones(len(X)),
        f, inten, dur,
        f * f, inten * inten, dur * dur,
        f * inten, f * dur, inten * dur,
    ])


def fit_rsm_surface(X_condition, y_donor, donor_ids, ridge=1e-8):
    """
    Fit quadratic RSM with donor fixed intercepts for numerical simplicity.
    Donor dummies remove donor-specific baseline shifts; the surface coefficients
    are then used for prediction at the average/reference donor.
    """
    Xr = rsm_design_matrix(X_condition)
    donor_ids = np.asarray(donor_ids)
    unique_donors = np.unique(donor_ids)
    # Use donor dummies excluding the first donor.
    dummies = []
    for d in unique_donors[1:]:
        dummies.append((donor_ids == d).astype(float))
    if dummies:
        Z = np.column_stack([Xr] + dummies)
    else:
        Z = Xr

    penalty = ridge * np.eye(Z.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(Z.T @ Z + penalty, Z.T @ y_donor)
    surface_beta = beta[:10]
    return surface_beta


def predict_rsm(beta, Xnew):
    return rsm_design_matrix(Xnew) @ beta


# -----------------------------
# GP implementation
# -----------------------------
def matern52(X1, X2, length_scale=0.30, signal_variance=400.0):
    diff = (X1[:, None, :] - X2[None, :, :]) / length_scale
    r = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    sqrt5 = np.sqrt(5.0)
    return signal_variance * (1.0 + sqrt5 * r + (5.0 / 3.0) * r * r) * np.exp(-sqrt5 * r)


class GP:
    """Gaussian-process regressor with optional per-observation noise variances."""

    def __init__(self, length_scale=0.30, signal_variance=400.0, noise_variance=25.0):
        self.ls = length_scale
        self.sv = signal_variance
        self.default_nv = noise_variance
        self.X_train = None
        self.y_mean = 0.0
        self.L = None
        self.alpha = None

    def fit(self, X, y, noise_var=None):
        self.X_train = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.y_mean = float(np.mean(y))
        yc = y - self.y_mean

        if noise_var is None:
            noise_var = np.full(len(y), self.default_nv, dtype=float)
        else:
            noise_var = np.asarray(noise_var, dtype=float)
            noise_var = np.maximum(noise_var, 1e-6)

        K = matern52(self.X_train, self.X_train, self.ls, self.sv)
        K += np.diag(noise_var)
        K += 1e-8 * np.eye(K.shape[0])
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yc))

    def predict(self, Xnew):
        Xnew = np.asarray(Xnew, dtype=float)
        Kstar = matern52(self.X_train, Xnew, self.ls, self.sv)
        mu = Kstar.T @ self.alpha + self.y_mean
        v = np.linalg.solve(self.L, Kstar)
        var = self.sv - np.sum(v * v, axis=0)
        return mu, np.maximum(var, 1e-10)


def expected_improvement(gp, X, y_best, xi=0.5):
    """Standard EI. """
    mu, var = gp.predict(X)
    sigma = np.sqrt(var)
    imp = mu - y_best - xi
    z = np.where(sigma > 1e-10, imp / sigma, 0.0)
    ei = imp * Phi(z) + sigma * phi(z)
    return np.where(sigma > 1e-10, ei, 0.0)


# -----------------------------
# Biological data simulation
# -----------------------------
def simulate_donor_data(surface_fn, X_conditions, n_donors, residual_cv, donor_cv, rng):
    """
    Simulate donor-level observations at each condition.
    Y_ij = f(x_j) + donor_shift_i + residual_error_ij

    donor_cv and residual_cv are expressed relative to the mean true response.
    """
    true_y = surface_fn(X_conditions)
    grand_mean = float(np.mean(true_y))
    donor_sd = donor_cv * grand_mean

    rows_X, rows_y, donor_ids = [], [], []
    donor_shifts = rng.normal(0.0, donor_sd, size=n_donors)
    for i in range(n_donors):
        for j, x in enumerate(X_conditions):
            resid_sd = residual_cv * max(true_y[j], 1e-6)
            y = true_y[j] + donor_shifts[i] + rng.normal(0.0, resid_sd)
            rows_X.append(x)
            rows_y.append(y)
            donor_ids.append(i)
    return np.asarray(rows_X), np.asarray(rows_y), np.asarray(donor_ids)


def condition_means_and_vars(X_rows, y_rows):
    """Collapse donor rows to condition-level means and variances for GP training."""
    unique_X, inverse = np.unique(np.round(X_rows, 10), axis=0, return_inverse=True)
    means, vars_ = [], []
    for k in range(len(unique_X)):
        vals = y_rows[inverse == k]
        means.append(float(np.mean(vals)))
        if len(vals) > 1:
            # variance of the condition mean; lower bound avoids zero at repeated center artifacts
            vars_.append(float(np.var(vals, ddof=1) / len(vals) + 1e-6))
        else:
            vars_.append(float(np.var(y_rows) * 0.10 + 1e-6))
    return unique_X, np.asarray(means), np.asarray(vars_)


# -----------------------------
# One full Aim 3 simulation replicate
# -----------------------------
def one_replicate(surface_fn, residual_cv, donor_cv, rng, grid):
    X_bbd = box_behnken_3d()

    # 1) Simulate Aim 2 donor-level data.
    X_rows, y_rows, donor_ids = simulate_donor_data(
        surface_fn, X_bbd, N_INITIAL_DONORS, residual_cv, donor_cv, rng
    )

    # 2) Fit RSM comparator and identify RSM optimum.
    beta_rsm = fit_rsm_surface(X_rows, y_rows, donor_ids)
    rsm_pred_grid = predict_rsm(beta_rsm, grid)
    x_rsm = grid[int(np.argmax(rsm_pred_grid))]

    # 3) Initialize GP using condition-level means and heteroscedastic mean variances.
    X_gp, y_gp, noise_gp = condition_means_and_vars(X_rows, y_rows)

    # 4) Run fixed-budget BO.
    for _cycle in range(N_BO_CYCLES):
        gp = GP(length_scale=0.30, signal_variance=400.0, noise_variance=25.0)
        gp.fit(X_gp, y_gp, noise_var=noise_gp)
        y_best = float(np.max(y_gp))
        ei = expected_improvement(gp, grid, y_best, xi=0.5)

        # Avoid re-selecting already observed points and avoid near duplicates within batch.
        selected = []
        for idx in np.argsort(-ei):
            cand = grid[idx]
            if np.min(np.linalg.norm(X_gp - cand, axis=1)) < 0.07:
                continue
            if selected and min(np.linalg.norm(cand - s) for s in selected) < 0.10:
                continue
            selected.append(cand)
            if len(selected) == N_PER_CYCLE:
                break
        X_new = np.asarray(selected)

        # Simulate 3 donor-level responses for the newly proposed conditions.
        X_new_rows, y_new_rows, _ = simulate_donor_data(
            surface_fn, X_new, N_INITIAL_DONORS, residual_cv, donor_cv, rng
        )
        X_new_gp, y_new_gp, noise_new_gp = condition_means_and_vars(X_new_rows, y_new_rows)
        X_gp = np.vstack([X_gp, X_new_gp])
        y_gp = np.concatenate([y_gp, y_new_gp])
        noise_gp = np.concatenate([noise_gp, noise_new_gp])

    # 5) Final GP-selected BayesOpt optimum.
    gp_final = GP(length_scale=0.30, signal_variance=400.0, noise_variance=25.0)
    gp_final.fit(X_gp, y_gp, noise_var=noise_gp)
    gp_mu, _ = gp_final.predict(grid)
    x_bo = grid[int(np.argmax(gp_mu))]

    # 6) True performance of selected optima.
    true_max = float(np.max(surface_fn(grid)))
    true_rsm = float(surface_fn(x_rsm)[0])
    true_bo = float(surface_fn(x_bo)[0])
    true_rel_improvement = (true_bo - true_rsm) / max(abs(true_rsm), 1e-6)

    # 7) Simulate independent validation donors at RSM and BO optima.
    # Paired design: same donors measured under both conditions.
    val_conditions = np.vstack([x_rsm, x_bo])
    true_vals = surface_fn(val_conditions)
    grand_mean = float(np.mean(true_vals))
    donor_sd = donor_cv * grand_mean
    diffs = []
    for _i in range(N_VALIDATION_DONORS):
        donor_shift = rng.normal(0.0, donor_sd)
        y_rsm = true_vals[0] + donor_shift + rng.normal(0.0, residual_cv * true_vals[0])
        y_bo = true_vals[1] + donor_shift + rng.normal(0.0, residual_cv * true_vals[1])
        diffs.append(y_bo - y_rsm)
    diffs = np.asarray(diffs)
    mean_diff = float(np.mean(diffs))
    mean_rsm_est = float(true_vals[0])  # denominator for relative effect; use true to define OC cleanly
    obs_rel_diff = mean_diff / max(abs(mean_rsm_est), 1e-6)
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else np.nan
    tstat = mean_diff / (sd_diff / np.sqrt(N_VALIDATION_DONORS) + 1e-12)

    return {
        "true_max": true_max,
        "true_rsm": true_rsm,
        "true_bo": true_bo,
        "true_rel_improvement": true_rel_improvement,
        "identified_meaningful_improvement": bool(true_rel_improvement >= MEANINGFUL_IMPROVEMENT),
        "bo_near_true_optimum": bool(true_bo >= (1.0 - NEAR_OPTIMUM_MARGIN) * true_max),
        "rsm_near_true_optimum": bool(true_rsm >= (1.0 - NEAR_OPTIMUM_MARGIN) * true_max),
        "validation_obs_rel_diff": obs_rel_diff,
        "validation_tstat": float(tstat),
        "validation_supports_improvement": bool(
            (obs_rel_diff >= MEANINGFUL_IMPROVEMENT) and (tstat > T_CRIT_ONE_SIDED_DF4)
        ),
        "x_rsm": x_rsm.tolist(),
        "x_bo": x_bo.tolist(),
    }


# -----------------------------
# Run simulation scenarios
# -----------------------------
def summarize_records(records):
    arr = records
    return {
        "n_replicates": len(arr),
        "mean_true_rel_improvement": float(np.mean([r["true_rel_improvement"] for r in arr])),
        "sd_true_rel_improvement": float(np.std([r["true_rel_improvement"] for r in arr], ddof=1)),
        "prob_identify_ge_10pct_true_improvement": float(np.mean([r["identified_meaningful_improvement"] for r in arr])),
        "prob_validation_supports_ge_10pct_improvement": float(np.mean([r["validation_supports_improvement"] for r in arr])),
        "prob_bo_within_15pct_true_optimum": float(np.mean([r["bo_near_true_optimum"] for r in arr])),
        "prob_rsm_within_15pct_true_optimum": float(np.mean([r["rsm_near_true_optimum"] for r in arr])),
        "mean_validation_obs_rel_diff": float(np.mean([r["validation_obs_rel_diff"] for r in arr])),
    }


def main():
    grid = candidate_grid(DENSE_GRID_N, low=0.0, high=1.0)
    residual_cvs = [0.20]  # edit to [0.15, 0.20, 0.25, 0.30] for final analysis
    donor_cvs = [0.10]  # edit to [0.05, 0.10, 0.15] for final analysis

    all_results = {}
    summary_rows = []

    for surface_name, surface_fn in SURFACES.items():
        for residual_cv in residual_cvs:
            for donor_cv in donor_cvs:
                key = f"{surface_name}|residCV={residual_cv:.2f}|donorCV={donor_cv:.2f}"
                records = []
                for rep in range(N_REPLICATES):
                    seed = BASE_SEED + rep + int(residual_cv * 1000) + int(donor_cv * 10000)
                    seed += 100000 * list(SURFACES.keys()).index(surface_name)
                    rng = np.random.default_rng(seed)
                    records.append(one_replicate(surface_fn, residual_cv, donor_cv, rng, grid))
                summ = summarize_records(records)
                all_results[key] = {"summary": summ, "records": records}
                row = {
                    "surface": surface_name,
                    "residual_cv": residual_cv,
                    "donor_cv": donor_cv,
                    **summ,
                }
                summary_rows.append(row)
                print(
                    f"{surface_name:<15} residCV={residual_cv:.2f} donorCV={donor_cv:.2f} "
                    f"P(true BO>=RSM+10%)={summ['prob_identify_ge_10pct_true_improvement']:.2f} "
                    f"P(validation supports)={summ['prob_validation_supports_ge_10pct_improvement']:.2f} "
                    f"P(BO near true max)={summ['prob_bo_within_15pct_true_optimum']:.2f}"
                )

    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved detailed results to: {OUT_JSON}")
    print(f"Saved summary table to:   {OUT_CSV}")


if __name__ == "__main__":
    main()
