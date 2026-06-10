# GP-EI Bayesian Optimization Pre-Validation

## Purpose

This repository provides a transparent pre-validation of a Gaussian-process / Expected Improvement (GP-EI) Bayesian optimization workflow for noisy three-factor response-surface optimization.

The goal is to evaluate whether sequential Bayesian optimization can identify high-response regions with fewer experimental conditions than grid-based exploration when each experiment is resource-limited and response measurements are noisy.

## What's in this repository

| File | Purpose |
|---|---|
| `prevalidation.py` | Self-contained Python implementation: Matérn 5/2 Gaussian process, Expected Improvement acquisition, three synthetic response surfaces with heteroscedastic noise, Bayesian optimization loop, grid-search baselines, and 20-replicate simulation |
| `make_figures.py` | Generates summary figures from the saved `results.json` file |
| `results.json` | Raw numerical results from the 20-replicate simulation runs |
| `fig1_convergence.png/pdf` | Figure 1: convergence trajectories |
| `fig2_posterior.png/pdf` | Figure 2: posterior recovery visualization |
| `fig3_summary.png/pdf` | Figure 3: summary efficiency comparison |

## Requirements

- Python 3.8+
- NumPy >= 1.20
- Matplotlib >= 3.5

No SciPy, scikit-learn, BoTorch, or PyTorch is required for this pre-validation code. The implementation intentionally uses only NumPy and Matplotlib to make the workflow transparent, auditable, and easy to reproduce in standard Python environments.

## To reproduce

From the repository folder:

```bash
python3 prevalidation.py
python3 make_figures.py
```

Expected output:

```text
=== Surface: unimodal ===     True maximum (estimated): 64.96
=== Surface: multimodal ===   True maximum (estimated): 69.97
=== Surface: ridge ===        True maximum (estimated): 59.96

SUMMARY STATISTICS — over 20 replicate runs per surface
Surface          True max    BO (24 expts)         Grid27         Grid64
unimodal            64.96      69.85±11.29     55.37±9.11     65.77±6.76
multimodal          69.97      66.85±13.13     49.14±3.50     62.26±8.10
ridge               59.96      66.82±7.17     58.52±12.30     61.48±5.18
```

## Key result

GP-EI Bayesian optimization with **24 total experiments** (15-point Box-Behnken initialization + 3 cycles × 3 proposed points) matched or exceeded the performance of a **4×4×4 grid search with 64 experiments** across the three synthetic test surfaces. This corresponds to a 62% reduction in experimental burden at equivalent or better optimization quality.

On the multimodal surface, where grid exploration is most vulnerable to missing a non-obvious optimum, GP-EI achieved a mean best observed response of 66.85 compared with 49.14 for the 3×3×3 grid.

## Methodology, in brief

1. **Three synthetic response surfaces** were defined in the unit cube [0, 1]^3, representing three normalized experimental input variables:
   - *Unimodal*: smooth single off-center peak
   - *Multimodal*: deceptive local maximum plus true global maximum in a less-obvious region
   - *Ridge*: high-response region along a curved manifold rather than a single point

2. **Noise model.** Heteroscedastic Gaussian noise was applied with coefficient of variation CV = 20% to represent experimentally realistic measurement variability.

3. **GP surrogate.** A Matérn 5/2 covariance kernel was used with fixed defaults:
   - length scale = 0.30 in normalized input space
   - signal variance = 400
   - noise variance = 25

4. **Acquisition function.** Expected Improvement was used with exploration parameter ξ = 0.5. Acquisition maximization was performed over a 25³ candidate grid with a within-batch spacing constraint to reduce point clustering.

5. **Optimization loop.** The initial design was a 15-point Box-Behnken design. Three iterative cycles were then run, with three GP-EI-proposed points per cycle, for a total budget of 24 simulated experiments.

6. **Baselines.** Two grid-search baselines were evaluated using the same noise model:
   - 3×3×3 grid = 27 experiments
   - 4×4×4 grid = 64 experiments

7. **Replication.** Each surface and method was evaluated across 20 independent runs using different random seeds. Means and standard deviations are reported.

## Interpreting noisy observed maxima

Optimization performance is reported as the **best noisy observed response**, not the noiseless underlying response. For this reason, best-observed values can exceed the noiseless true maximum in some replicate runs. This reflects simulated measurement variability, not overestimation of the underlying response surface.

## Limitations

- The synthetic surfaces are plausible test functions for noisy three-factor optimization, but they are not derived from any specific experimental system.
- Hyperparameters are fixed in this transparent pre-validation implementation. In a production deployment, hyperparameters can be learned from the observed experimental data using marginal-likelihood optimization.
- The noise model uses a proportional coefficient of variation. Real experimental systems may require a different noise model or replicate-aware likelihood.
- This pre-validation establishes that the workflow behaves as expected on noisy synthetic 3D response surfaces. It does not establish performance in any particular real-world experimental application.

## Production implementation note

The same algorithmic specification can be implemented in standard Bayesian optimization libraries such as BoTorch / GPyTorch:

```python
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
```

A production implementation can add automated hyperparameter optimization, GPU acceleration, replicate-aware noise estimation, and continuous acquisition-function optimization.

## v2.0 — Aim 3 power simulation

Version 2.0 adds an operating-characteristic simulation framework BayesOpt-versus-RSM superiority validation, contributed by Yiqun Jiang (UCLA Department of Medicine). Code and outputs live in the `aim3_power/` subfolder; see `aim3_power/README.md` for methodology and reproducibility notes. The original v1.0 pre-validation framework (this file) is retained unchanged.

## Reproducibility update

This public version uses repository-relative paths and application-neutral terminology. Numerical results and the algorithmic workflow are unchanged.
