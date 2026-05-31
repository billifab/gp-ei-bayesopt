# Changelog

## v2.0.0 — 2026-06-XX

Added: Aim 3 Bayesian-optimization power simulation framework (`aim3_power/aim3_simulation_power.py`) by Yiqun Jiang, with full
operating-characteristic outputs across 48 scenarios (4 response-surface geometries × 4 residual coefficients of variation × 3 donor coefficients of variation; 500 Monte Carlo replicates each). Production outputs included as `aim3_power/aim3_power_results.json` and `aim3_power/aim3_power_summary.csv`. The original GP-EI pre-validation framework (v1.0) is retained unchanged.

## Public reproducibility update

- Replaced session-specific absolute paths with repository-relative paths.
- Generalized application-specific terminology to keep the repository application-neutral.
- Clarified that best observed noisy responses can exceed the noiseless true maximum because the reported performance metric is based on noisy observations.
- Numerical results, algorithmic specification, and simulation workflow are unchanged.
