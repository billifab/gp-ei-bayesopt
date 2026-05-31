# GP-EI Bayesian Optimization Framework with Benchmark Reproducibility

**Version 2.0** (2026-06)

This deposit accompanies the NIH R21 application *A Platform for Data-Efficient Optimization of Biophysical Stimulation Parameters in Regenerative Cell Engineering* (Billi, contact PI; Péault, co-PI; Jiang, co-I).

## What is new in version 2.0

Version 2.0 adds the Aim 3 power-analysis simulation framework and its production operating-characteristic outputs. The original Version 1 contents (GP-EI pre-validation on three synthetic 3D response surfaces, used as the methodological proof of concept cited in §3.1 of the Research Strategy) are retained unchanged.

New files in this version:

- `aim3_simulation_power.py` — self-contained Python implementation of the Aim 3 power simulation. Authored by Yiqun Jiang; integrated into the application as the operating-characteristic backbone for the >10% superiority margin in the Bayesian optimization validation phase.
- `aim3_power_results.json` — full per-replicate records across the 48 simulated scenarios (four response-surface geometries × four within-donor residual coefficients of variation × three between-donor coefficients of variation, 500 Monte Carlo replicates each).
- `aim3_power_summary.csv` — operating-characteristic summary table corresponding to Supplementary Table S1 of the Research Strategy.

## Scope of the Aim 3 power simulation

The simulation evaluates the operating characteristics of the proposed two-stage RSM-plus-BayesOpt workflow when applied to a three-factor PEMF parameter space under realistic donor-level and residual biological variability. Four canonical response-surface geometries are tested:

- **Unimodal**: smooth surface with a single off-center peak.
- **Multimodal**: two-peak surface with a deceptive local maximum and a true global maximum.
- **Ridge**: high response along a curved manifold, where neither classical RSM nor BayesOpt is expected to identify a sharp optimum.
- **Null-quadratic**: smooth quadratic surface where classical RSM is expected to perform near-optimally; included to estimate the false-positive rate for BayesOpt superiority claims under conditions where no genuine improvement is possible.

For each scenario, the simulation executes the full proposed workflow: a 15-run Box-Behnken design across three initial donors, three sequential Bayesian optimization cycles proposing three new conditions per cycle, and a paired-design validation in five independent donors with one-sided t-test of the BayesOpt-versus-RSM contrast (df = 4, α = 0.05). Decision thresholds match the proposed validation framework: a 10% relative improvement is the minimum biologically meaningful effect; a candidate optimum is considered "near the true maximum" if it lies within 15% of the true global maximum on the simulated surface.

## Reported operating characteristics

The Research Strategy §4.3 Power and design justification subsection reports a range summary of the probability of identifying a true ≥10% improvement over the RSM-predicted optimum when distinct optima are present (approximately 68–92%), low false-positive rates under null-quadratic surfaces, and robustness across the residual-CV range tested. The complete table of probabilities across all 48 scenarios is provided in `aim3_power_summary.csv` and referenced in the Research Strategy as Supplementary Table S1.

## Reproducibility

The simulation uses a fixed master seed (`BASE_SEED = 20260525`) with deterministic per-scenario seed derivation. Re-running `aim3_simulation_power.py` with the same configuration reproduces `aim3_power_results.json` and `aim3_power_summary.csv` exactly. Default settings in the file are `N_REPLICATES = 10` for fast iteration during development; reproducing the production outputs requires setting `N_REPLICATES = 500` and the full coefficient-of-variation sweep `residual_cvs = [0.15, 0.20, 0.25, 0.30]` and `donor_cvs = [0.05, 0.10, 0.15]` in the `main()` function.

The simulation depends only on `numpy`. No external Bayesian-optimization or statistical libraries are required, ensuring portability across computing environments.

## Methodological notes

The Gaussian process surrogate uses a Matérn 5/2 covariance kernel with fixed hyperparameters (length scale 0.30 on the [0, 1] unit cube; signal variance 400; default observation noise variance 25), with condition-specific noise variances estimated from donor-level replicate responses where available. Expected Improvement with exploration constant ξ = 0.5 is used as the acquisition function and evaluated over a dense 15×15×15 candidate grid; near-duplicate candidates (within Euclidean distance 0.07 of observed points or 0.10 of within-batch selections) are suppressed.

Fixed hyperparameters are appropriate for operating-characteristic analysis, where the goal is to characterize workflow performance under realistic scenarios rather than to optimize the surrogate model itself. The full production implementation in Aim 3 will use marginal-likelihood optimization of the GP hyperparameters via the BoTorch framework, as described in the Research Strategy §4.3 Computational methodology.

## Authorship and contributions

- **Fabrizio Billi, PhD** (UCLA Orthopaedic Surgery, Billi Laboratory) — original GP-EI pre-validation framework (Version 1.0), project conception, repository maintenance.
- **Yiqun Jiang, PhD** (UCLA Department of Medicine) — Aim 3 power simulation framework, operating-characteristic analysis (Version 2.0).

## License and citation

The deposit is released under the same license as Version 1.0. When citing this work in publications or grant applications, use the concept DOI (10.5281/zenodo.20296093), which always resolves to the latest version. The version-specific DOI for Version 2.0 is minted at publication and printed in the Zenodo record header.
