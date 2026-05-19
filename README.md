# GP-EI Bayesian Optimization Framework

This repository contains a Gaussian-Process / Expected-Improvement (GP-EI) Bayesian-optimization implementation for noisy, expensive black-box functions in moderate-dimensional continuous parameter spaces.

The implementation includes a reproducibility framework that tests algorithm behavior on standard benchmark test functions with realistic noise characteristics, and compares performance against full-factorial grid baselines at matched and larger experimental budgets.

A manuscript describing an application of this framework is in preparation.

## Contents

| File | Purpose |
|---|---|
| `prevalidation.py` | Self-contained Python implementation: Matérn 5/2 GP, Expected Improvement acquisition, three benchmark test functions with heteroscedastic noise, BayesOpt loop, grid-search baselines, 20-replicate run harness |
| `make_figures.py` | Generates the convergence-trajectory, GP-posterior, and summary-comparison figures from the saved results |
| `results.json` | Raw numerical results from the 20-replicate runs |
| `fig1_convergence.png/pdf` | Convergence trajectories across the three benchmark functions |
| `fig2_posterior.png/pdf` | GP posterior visualization on representative runs |
| `fig3_summary.png/pdf` | Summary efficiency comparison versus grid-search baselines |

## Requirements

Python 3.8+, NumPy ≥ 1.20, Matplotlib ≥ 3.5. No other dependencies required. The code intentionally avoids SciPy, scikit-learn, BoTorch, and PyTorch so it runs in any minimal Python environment.

## Quick reproduction (~5 minutes)

```bash
python3 prevalidation.py     # runs the optimization, prints summary, saves results.json
python3 make_figures.py      # generates the figures from results.json
```

Expected console output:

```
Surface          True max    BO (24 expts)         Grid27         Grid64
unimodal            64.96      69.85±11.29     55.37±9.11     65.77±6.76
multimodal          69.97      66.85±13.13     49.14±3.50     62.26±8.10
ridge               59.96      66.82±7.17     58.52±12.30     61.48±5.18
```

## Headline result

GP-EI BayesOpt with **24 evaluations** (15-point Box-Behnken initial design + 3 cycles × 3 GP-EI-proposed points) matches or exceeds the performance of **4×4×4 grid search with 64 evaluations** across all three benchmark functions — a 62% reduction in evaluation budget at equivalent or better optimization quality. On the multimodal benchmark, where grid search is most vulnerable to a deceptive local maximum, GP-EI outperforms the 3×3×3 grid (27 evaluations) by 36%.

## Methodology

1. **Three benchmark test functions** on the unit cube [0, 1]³:
   - *Unimodal*: smooth single off-center peak (true max ≈ 65)
   - *Multimodal*: deceptive local maximum with the true global maximum in a less-obvious region (true max ≈ 70)
   - *Ridge*: optimum along a curved manifold rather than a point (true max ≈ 60)
2. **Noise model.** Heteroscedastic Gaussian, coefficient of variation 20%, intended to mimic the order of magnitude of replicate noise typical of expensive experimental evaluations.
3. **GP surrogate.** Matérn 5/2 covariance kernel with fixed hyperparameters (length scale ℓ = 0.30, signal variance σ_f² = 400, noise variance σ_n² = 25). In a production deployment using BoTorch, these would become priors for marginal-likelihood optimization.
4. **Acquisition.** Expected Improvement with exploration parameter ξ = 0.5. Maximization over a 25³ = 15,625-point evaluation grid with within-batch spacing constraint (≥ 0.10 in normalized coordinates) to prevent point clustering.
5. **Optimization loop.** Initial design = 15-point Box-Behnken. Three iterative cycles, three GP-EI-proposed points per cycle. Total: 24 simulated evaluations.
6. **Baselines.** 3×3×3 = 27-point grid and 4×4×4 = 64-point grid with identical noise model and replication.
7. **Replication.** 20 independent runs per benchmark with different RNG seeds; means ± SD reported.

## Production implementation (BoTorch)

The algorithmic specification implemented here is identical to the production version, which uses BoTorch (PyTorch-based) for marginal-likelihood hyperparameter learning and automatic differentiation. The custom NumPy implementation in this repository is provided for transparency, audit, and minimal-dependency reproducibility. A typical BoTorch deployment uses:

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

## Limitations

- Fixed hyperparameters are used here. A production deployment will optimize hyperparameters via marginal log-likelihood; the fixed values used here represent reasonable defaults that demonstrate correct algorithmic behavior.
- Heteroscedastic noise is modeled as proportional to the response. A production likelihood will be re-estimated from observed replicate variance in the deployment domain.
- The benchmark functions are designed to exercise distinct geometric features (single peak, deceptive local maximum, ridge manifold) but are not derived from any specific empirical system.

## License

MIT License.
