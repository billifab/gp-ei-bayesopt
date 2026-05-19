"""Generate publication-quality figures for the GP-EI pre-validation."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.insert(0, "/sessions/eloquent-sharp-feynman/mnt/outputs/memo")
from prevalidation import (
    SURFACES, GP, expected_improvement, matern52,
    box_behnken_3d, bayesopt_run, grid_baseline, add_noise
)

OUTDIR = "/sessions/eloquent-sharp-feynman/mnt/outputs/memo/prevalidation_figs"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
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
})

with open(os.path.join(OUTDIR, "results.json"), "r") as f:
    results = json.load(f)


# ================== FIGURE 1: Convergence trajectory ==================
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=False)

colors = {"bayesopt": "#1f77b4", "grid27": "#ff7f0e", "grid64": "#2ca02c"}

for ax_i, name in enumerate(["unimodal", "multimodal", "ridge"]):
    ax = axes[ax_i]
    res = results[name]
    true_max = res["true_max"]

    # BO trajectories: best_per_cycle has length 4 (BB baseline + 3 cycles)
    # n_per_cycle has same length: [15, 18, 21, 24]
    bo_trajs = np.array([r["best_per_cycle"] for r in res["replicates"]["bayesopt"]])
    bo_n = res["replicates"]["bayesopt"][0]["n_per_cycle"]  # [15, 18, 21, 24]
    bo_mean = bo_trajs.mean(axis=0)
    bo_std = bo_trajs.std(axis=0)

    ax.plot(bo_n, bo_mean, marker="o", color=colors["bayesopt"],
            linewidth=2, markersize=7, label="GP-EI BayesOpt (BB + 3 cycles)")
    ax.fill_between(bo_n, bo_mean - bo_std, bo_mean + bo_std,
                    color=colors["bayesopt"], alpha=0.2)

    # Grid27 — point at n=27
    g27 = [r["best"] for r in res["replicates"]["grid27"]]
    ax.errorbar(27, np.mean(g27), yerr=np.std(g27), marker="s",
                color=colors["grid27"], markersize=9, linewidth=2,
                capsize=4, label="3×3×3 grid (27 expts)")

    # Grid64 — point at n=64
    g64 = [r["best"] for r in res["replicates"]["grid64"]]
    ax.errorbar(64, np.mean(g64), yerr=np.std(g64), marker="^",
                color=colors["grid64"], markersize=9, linewidth=2,
                capsize=4, label="4×4×4 grid (64 expts)")

    # True maximum reference line
    ax.axhline(true_max, color="gray", linestyle="--", linewidth=1,
               label=f"True maximum ({true_max:.1f})")

    ax.set_xlabel("Cumulative number of experiments")
    if ax_i == 0:
        ax.set_ylabel("Best observed chondrogenic response\n(simulated sGAG, a.u.)")
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

plt.suptitle("Pre-validation: GP-EI BayesOpt converges within 3 cycles on noisy synthetic "
             "3D response surfaces", y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig1_convergence.png"))
plt.savefig(os.path.join(OUTDIR, "fig1_convergence.pdf"))
plt.close()
print("Saved fig1_convergence.png")


# ================== FIGURE 2: 2D slice of GP posterior ==================
# Run one representative BayesOpt trajectory for each surface and visualize
# the GP posterior mean over the 2D (frequency × intensity) slice at the
# optimal duration.

RNG_FIG = np.random.default_rng(42)

def representative_run(surface_fn, target_dim=2):
    globals()["RNG"] = RNG_FIG
    # Re-import to refresh RNG inside prevalidation module
    import prevalidation as pv
    pv.RNG = RNG_FIG
    bo = pv.bayesopt_run(surface_fn, n_cycles=3, n_per_cycle=3)
    return bo


fig, axes = plt.subplots(2, 3, figsize=(11.5, 7), constrained_layout=True)

for col, (name, (fn, true_opt)) in enumerate(SURFACES.items()):
    bo = representative_run(fn)
    X_obs = bo["X_obs"]
    y_obs = bo["y_obs"]

    # Find best observation duration (slice the 3D space at that x[2])
    idx_best = int(np.argmax(y_obs))
    z_slice = X_obs[idx_best, 2]

    # Fit GP to all observations
    gp = GP(length_scale=0.30, signal_variance=400.0, noise_variance=25.0)
    gp.fit(X_obs, y_obs)

    # 2D grid at z = z_slice
    g = np.linspace(0, 1, 60)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    Xgrid = np.stack([GX.flatten(), GY.flatten(),
                      np.full(GX.size, z_slice)], axis=1)

    # True surface
    y_true = fn(Xgrid).reshape(GX.shape)
    # GP posterior mean
    mu, var = gp.predict(Xgrid)
    mu = mu.reshape(GX.shape)

    # Map normalized [0,1] back to natural units
    freq_natural = 15 + 60 * g       # 15–75 Hz
    int_natural = 0.5 + 2.5 * g      # 0.5–3 mT

    # Top row: true surface
    ax = axes[0, col]
    im = ax.contourf(freq_natural, int_natural, y_true.T, levels=15,
                     cmap="viridis", vmin=30, vmax=75)
    ax.set_title(f"True response — {name}\n(duration slice ≈ {30 + 150*z_slice:.0f} min/day)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Intensity (mT)")
    plt.colorbar(im, ax=ax, label="sGAG (a.u.)", shrink=0.85)

    # Bottom row: GP posterior + observations
    ax = axes[1, col]
    im = ax.contourf(freq_natural, int_natural, mu.T, levels=15,
                     cmap="viridis", vmin=30, vmax=75)
    # Overlay observations within ±0.10 of the slice
    near_slice = np.abs(X_obs[:, 2] - z_slice) < 0.15
    X_near = X_obs[near_slice]
    if len(X_near) > 0:
        ax.scatter(15 + 60 * X_near[:, 0], 0.5 + 2.5 * X_near[:, 1],
                   c="white", edgecolors="black", s=40,
                   linewidth=1.2, zorder=10, label="Observations")
    ax.set_title(f"GP posterior mean — {name}")
    ax.set_xlabel("Frequency (Hz)")
    if col == 0:
        ax.set_ylabel("Intensity (mT)")
    plt.colorbar(im, ax=ax, label="Predicted sGAG (a.u.)", shrink=0.85)
    ax.legend(loc="lower left", framealpha=0.85, fontsize=8)

plt.suptitle("Pre-validation: GP posterior recovers the response surface from 24 observations",
             fontsize=11)
plt.savefig(os.path.join(OUTDIR, "fig2_posterior.png"))
plt.savefig(os.path.join(OUTDIR, "fig2_posterior.pdf"))
plt.close()
print("Saved fig2_posterior.png")


# ================== FIGURE 3: Summary efficiency bar chart ==================
fig, ax = plt.subplots(figsize=(7, 4.2))

surface_names = ["Unimodal", "Multimodal", "Ridge"]
bo_means = []
bo_stds = []
g27_means = []
g27_stds = []
g64_means = []
g64_stds = []

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

ax.bar(x_pos - width, g27_means, width, yerr=g27_stds, capsize=4,
       color="#ff7f0e", alpha=0.85, label="3×3×3 grid (27 expts)")
ax.bar(x_pos, bo_means, width, yerr=bo_stds, capsize=4,
       color="#1f77b4", alpha=0.85, label="GP-EI BayesOpt (24 expts)")
ax.bar(x_pos + width, g64_means, width, yerr=g64_stds, capsize=4,
       color="#2ca02c", alpha=0.85, label="4×4×4 grid (64 expts)")

# Add true max line
true_maxes = [results[n]["true_max"] for n in ["unimodal", "multimodal", "ridge"]]
for i, tm in enumerate(true_maxes):
    ax.hlines(tm, i - 1.5 * width, i + 1.5 * width,
              colors="black", linestyles="--", linewidth=1.2)
    if i == 0:
        ax.text(i - 1.4 * width, tm + 0.8, "True max",
                fontsize=8, color="black")

ax.set_xticks(x_pos)
ax.set_xticklabels(surface_names)
ax.set_ylabel("Mean best observed response\n(simulated sGAG, a.u.)")
ax.set_title("BayesOpt achieves grid-64 performance with 60% fewer experiments")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(30, 80)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig3_summary.png"))
plt.savefig(os.path.join(OUTDIR, "fig3_summary.pdf"))
plt.close()
print("Saved fig3_summary.png")

print("\nAll figures generated in:", OUTDIR)
