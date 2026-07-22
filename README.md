# CRIM-S: Spectral Compensation on the Probability Simplex

Reproducible code for all numerical figures in:

> **Exact stability threshold and steady-state diversity for stochastic consensus dynamics on the probability simplex**
>

## Quickest way to reproduce (Google Colab)

All scripts run in [Google Colab](https://colab.research.google.com/) without any setup.
Colab provides `numpy`, `scipy`, and `matplotlib` by default.

1. Open a new Colab notebook.
2. Paste any script below into a cell and run.
3. Output figures are saved in the Colab working directory and also displayed inline.

No package installation is needed. Total runtime for all 6 scripts is approximately 45–70 minutes, depending on the Colab instance assigned.

## Local installation (alternative)

If you prefer to run locally:

```bash
pip install numpy scipy matplotlib
```

Requires Python ≥ 3.9.

## Scripts

| Script | Output | Runtime* | Description |
|---|---|---|---|
| `colab_figure1.py` | `Figure1.png` | ~2 min | Diversity Φ(t) with and without regularization |
| `colab_figure2.py` | `Figure2.png` | ~16 min | Spectral-sum prediction, collapse plot, single-mode comparison |
| `colab_figure3_v2.py` | `Figure3.png` | ~1 min | Time traces (stable / critical / unstable) and (α, λ₂) phase diagram |
| `colab_exp1.py` | `exp1.png` | ~5 min | Simplex-constrained vs. unconstrained dynamics |
| `colab_exp2.py` | `exp2.png` | ~25 min | Boundary contact fraction and state-component distributions |
| `colab_exp3.py` | `exp3.png` | ~20 min | Additive vs. multiplicative noise |

*Runtimes on a standard Colab CPU instance. Local machines may differ.


## Reproducibility

All simulations use fixed random seeds — either a global `np.random.seed(42)` or per-run `np.random.RandomState(seed)` with deterministic seed values. Running any script multiple times, on any machine, produces identical numerical output and identical figures.

## Expected output for the core result

`colab_figure2.py` prints verification numbers to the console:

```
  spectral sum : c = 0.693   R^2 = 0.9967
  single mode  : c = 11.62    R^2 = -0.3492   (wrong shape)
  unconstrained: c = 1.148   (linear theory predicts c = 1)
  projection suppresses Phi* by 40% relative to the unconstrained system
```

`colab_exp3.py` prints:

```
    Additive noise         c = 0.720   R^2 = 0.9869
    Multiplicative noise   c = 0.240   R^2 = 0.9864
```

## Notes on simulation parameters

Baseline parameters are N = 200, S = 3, Δt = 0.01, η = 1, γσ = 0.1, k_R = 1.

**Steady-state experiments (Figs. 2, 4–6)** use T = 500 (300 burn-in + 200 averaging), with all quantities averaged over 20 independent realizations per parameter point.

**Transient trajectories and phase diagram (Figs. 1, 3)** use T = 100–200. The trajectories in Fig. 3(a) are individual sample paths; each grid point of the phase diagram in Fig. 3(b) is a single run classified by its steady-state diversity.

## License

MIT
