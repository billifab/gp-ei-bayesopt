"""Generate figures for GP-EI Bayesian optimization pre-validation."""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from prevalidation import SURFACES, GP

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "prevalidation_figs"
OUTDIR.mkdir(exist_ok=True)

RESULTS_PATH = HERE / "results.json"
if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        "results.json was not found. Run `python3 prevalidation.py` first, "
        "or place results.json in the repository folder."
    )

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

with open(RESULTS_PATH, "r") as f:
    results = json.load(f)


# ======================================================================
# Figure 1: Convergence trajectory
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=False)

colors = {"bayesopt": "#1f77b4", "grid27": "#ff7f0e", "grid64": "#2ca02c"}

for ax_i, name in enumerate(["unimodal", "multimodal", "ridge"]):
    ax = axes[ax_i]
    res = results[name]
    true_max = res["true_max"]

    bo_trajs = np.array([r["best_per_cycle"] for r in res["replicates"]["bayesopt"]])
    bo_n = res["replicates"]["bayesopt"][0]["n_per_cycle"]
    bo_mean = bo_trajs.mean(axis=0)
    bo_std = bo_trajs.std(axis=0)

    ax.plot(
        bo_n,
        bo_mean,
        marker="o",
        color=colors["bayesopt"],
        linewidth=2,
        markersize=7,
        label="GP-EI BayesOpt",
    )
    ax.fill_between(bo_n, bo_mean - bo_std, bo_mean + bo_std, color=colors["bayesopt"], alpha=0.2)

    g27 = [r["best"] for r in res["replicates"]["grid27"]]
    ax.errorbar(
        27,
        np.mean(g27),
        yerr=np.std(g27),
        marker="s",
        color=colors["grid27"],
        markersize=9,
        linewidth=2,
        capsize=4,
        label="3×3×3 grid",
    )

    g64 = [r["best"] for r in res["replicates"]["grid64"]]
    ax.errorbar(
        64,
        np.mean(g64),
        yerr=np.std(g64),
        marker="^",
        color=colors["grid64"],
        markersize=9,
        linewidth=2,
        capsize=4,
        label="4×4×4 grid",
    )

    ax.axhline(true_max, color="gray", linestyle="--", linewidth=1, label=f"Noiseless true maximum ({true_max:.1f})")

    ax.set_xlabel("Cumulative number of experiments")
    if ax_i == 0:
        ax.set_ylabel("Best observed response\n(arbitrary units)")

    title_map = {
        "unimodal": "(A) Unimodal surface",
        "multimodal": "(B) Multimodal surface",
        "ridge": "(C) Ridge surface",
    }
    ax.set_title(title_map[name])
    ax.set_xlim(10, 72)
    ax.grid(alpha=0.3)

    if ax_i == 0:
        ax.legend(loc="lower right", framealpha=0.9, fontsize=8)

plt.suptitle("GP-EI Bayesian optimization on noisy synthetic 3D response surfaces", y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig(OUTDIR / "fig1_convergence.png")
plt.savefig(OUTDIR / "fig1_convergence.pdf")
plt.close()
print("Saved fig1_convergence.png/pdf")


# ======================================================================
# Figure 2: 2D slice of GP posterior
# ======================================================================
RNG_FIG = np.random.default_rng(42)


def representative_run(surface_fn):
    """Generate a representative trajectory for posterior visualization."""
    import prevalidation as pv

    pv.RNG = RNG_FIG
    return pv.bayesopt_run(surface_fn, n_cycles=3, n_per_cycle=3)


fig, axes = plt.subplots(2, 3, figsize=(11.5, 7), constrained_layout=True)

for col, (name, (fn, _true_opt)) in enumerate(SURFACES.items()):
    bo = representative_run(fn)
    X_obs = bo["X_obs"]
    y_obs = bo["y_obs"]

    idx_best = int(np.argmax(y_obs))
    z_slice = X_obs[idx_best, 2]

    gp = GP(length_scale=0.30, signal_variance=400.0, noise_variance=25.0)
    gp.fit(X_obs, y_obs)

    g = np.linspace(0, 1, 60)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    Xgrid = np.stack([GX.flatten(), GY.flatten(), np.full(GX.size, z_slice)], axis=1)

    y_true = fn(Xgrid).reshape(GX.shape)
    mu, _var = gp.predict(Xgrid)
    mu = mu.reshape(GX.shape)

    ax = axes[0, col]
    im = ax.contourf(g, g, y_true.T, levels=15, cmap="viridis", vmin=30, vmax=75)
    ax.set_title(f"True response — {name}\n(Input 3 slice ≈ {z_slice:.2f})")
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    plt.colorbar(im, ax=ax, label="Response (a.u.)", shrink=0.85)

    ax = axes[1, col]
    im = ax.contourf(g, g, mu.T, levels=15, cmap="viridis", vmin=30, vmax=75)

    near_slice = np.abs(X_obs[:, 2] - z_slice) < 0.15
    X_near = X_obs[near_slice]
    if len(X_near) > 0:
        ax.scatter(
            X_near[:, 0],
            X_near[:, 1],
            c="white",
            edgecolors="black",
            s=40,
            linewidth=1.2,
            zorder=10,
            label="Observations",
        )

    ax.set_title(f"GP posterior mean — {name}")
    ax.set_xlabel("Input 1")
    if col == 0:
        ax.set_ylabel("Input 2")
    plt.colorbar(im, ax=ax, label="Predicted response (a.u.)", shrink=0.85)
    ax.legend(loc="lower left", framealpha=0.85, fontsize=8)

plt.suptitle("GP posterior recovery from 24 noisy observations", fontsize=11)
plt.savefig(OUTDIR / "fig2_posterior.png")
plt.savefig(OUTDIR / "fig2_posterior.pdf")
plt.close()
print("Saved fig2_posterior.png/pdf")


# ======================================================================
# Figure 3: Summary efficiency bar chart
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 4.2))

surface_names = ["Unimodal", "Multimodal", "Ridge"]
bo_means, bo_stds = [], []
g27_means, g27_stds = [], []
g64_means, g64_stds = [], []

for name in ["unimodal", "multimodal", "ridge"]:
    res = results[name]
    bo_finals = [r["best_per_cycle"][-1] for r in res["replicates"]["bayesopt"]]
    g27_finals = [r["best"] for r in res["replicates"]["grid27"]]
    g64_finals = [r["best"] for r in res["replicates"]["grid64"]]

    bo_means.append(np.mean(bo_finals))
    bo_stds.append(np.std(bo_finals))
    g27_means.append(np.mean(g27_finals))
    g27_stds.append(np.std(g27_finals))
    g64_means.append(np.mean(g64_finals))
    g64_stds.append(np.std(g64_finals))

x_pos = np.arange(len(surface_names))
width = 0.27

ax.bar(x_pos - width, g27_means, width, yerr=g27_stds, capsize=4, color="#ff7f0e", alpha=0.85, label="3×3×3 grid (27 expts)")
ax.bar(x_pos, bo_means, width, yerr=bo_stds, capsize=4, color="#1f77b4", alpha=0.85, label="GP-EI BayesOpt (24 expts)")
ax.bar(x_pos + width, g64_means, width, yerr=g64_stds, capsize=4, color="#2ca02c", alpha=0.85, label="4×4×4 grid (64 expts)")

true_maxes = [results[n]["true_max"] for n in ["unimodal", "multimodal", "ridge"]]
for i, tm in enumerate(true_maxes):
    ax.hlines(tm, i - 1.5 * width, i + 1.5 * width, colors="black", linestyles="--", linewidth=1.2)
    if i == 0:
        ax.text(i - 1.4 * width, tm + 0.8, "Noiseless true max", fontsize=8, color="black")

ax.set_xticks(x_pos)
ax.set_xticklabels(surface_names)
ax.set_ylabel("Mean best observed response\n(arbitrary units)")
ax.set_title("BayesOpt achieves grid-64 performance with fewer experiments")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(30, 80)

plt.tight_layout()
plt.savefig(OUTDIR / "fig3_summary.png")
plt.savefig(OUTDIR / "fig3_summary.pdf")
plt.close()
print("Saved fig3_summary.png/pdf")

print(f"\nAll figures generated in: {OUTDIR}")
