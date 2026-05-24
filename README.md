# QWO: Quantum Wavefunction Optimizer for Continuous Machine Learning Landscapes

> A simulator-based research prototype of a new optimization method. The
> optimizer represents continuous parameters on a discretised quantum
> register, propagates a complex wavefunction over parameter space using a
> loss-dependent phase oracle and a kinetic mixer, optionally couples
> registers with an entangling phase, samples candidate solutions from
> `|psi(theta)|^2`, and refines the top candidates with a classical local
> optimiser.
>
> As an **exploratory comparison**, we measure where QWO reaches
> lower loss, higher success rate, or fewer objective evaluations than
> matched classical baselines under the same evaluation budget. We make
> **no claim of quantum advantage.**

## Pipeline

For an objective `L(theta)` with `theta in R^d`:

1. Discretise `theta` into a grid `Theta_h` of `N = prod_k n_k` points.
2. Prepare a uniform complex wavefunction
   `|Psi_0> = (1/sqrt(N)) sum_j |theta_j>`.
3. Apply a phase oracle `U_L(eta) |theta_j> = exp(-i eta L(theta_j)) |theta_j>`.
4. Apply a kinetic mixer - either FFT/QFT-style free-particle evolution
   `exp(-i tau K^2 / 2)` in momentum space, or a finite-difference Laplacian
   Crank-Nicolson step. A no-mixer ablation is available.
5. Optionally apply an entangling ZZ-style phase
   `exp(-i chi theta_a theta_b)` between parameter axes.
6. Renormalise. Sample candidate `theta` values from `|Psi(theta)|^2`.
7. Locally refine each candidate with `scipy.optimize.minimize`
   (Nelder-Mead). Keep the running best.
8. Repeat for `K` iterations.

## Quantum ingredients

- Discrete complex wavefunctions over parameter space.
- Phase encoding of the objective via `U_L(eta)`.
- FFT-based momentum-space kinetic mixing and finite-difference
  Crank-Nicolson mixing.
- Optional two-register entangling phase `exp(-i chi theta_a theta_b)`.
- Sampling from `|Psi|^2`, treated as a discrete probability over the grid.
- Entropy and concentration diagnostics on `|Psi|^2` as a function of
  iteration.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_all_experiments.py --quick   # 2 seeds, ~5 seconds
python scripts/build_release_media.py           # figures + GIF + MP4
pytest -q
```

## The 20 experiments

| #  | Name                          | Objective              | Tests                                       |
|---:|-------------------------------|------------------------|---------------------------------------------|
| 1  | `01_double_well`              | `double_well`          | 1-D bimodal landscape                       |
| 2  | `02_rugged_sinusoid`          | `rugged_sinusoid`      | many shallow local minima                   |
| 3  | `03_rastrigin_2d`             | `rastrigin_2d`         | classical multimodal benchmark              |
| 4  | `04_ackley_2d`                | `ackley_2d`            | flat outer, sharp basin                     |
| 5  | `05_rosenbrock`               | `rosenbrock`           | curved narrow valley                        |
| 6  | `06_himmelblau`               | `himmelblau`           | four equal global minima                    |
| 7  | `07_rastrigin_4d`             | `rastrigin_4d`         | 4-D multimodal                              |
| 8  | `08_ackley_4d`                | `ackley_4d`            | 4-D mostly flat                             |
| 9  | `09_two_moons_logreg`         | `two_moons_logreg`     | ML loss, non-convex in (w1, w2)             |
| 10 | `10_circles_logreg`           | `circles_logreg`       | ML loss, flat near origin                   |
| 11 | `11_fourier_regression`       | `fourier_regression`   | MSE of 2 Fourier coefficients               |
| 12 | `12_noisy_quadratic`          | `noisy_quadratic`      | additive Gaussian noise                     |
| 13 | `13_sparse_reward`            | `sparse_reward`        | zero gradient outside a tiny ball           |
| 14 | `14_wide_vs_narrow`           | `wide_vs_narrow`       | wide attractor vs narrow global min         |
| 15 | `15_qwo_no_mixer`             | `rastrigin_2d`         | ablation: mixer = none                      |
| 16 | `16_qwo_no_phase`             | `rastrigin_2d`         | ablation: eta = 0                           |
| 17 | `17_qwo_with_zz`              | `rosenbrock`           | optional ZZ entangling phase                |
| 18 | `18_qwo_separable`            | `rosenbrock`           | per-axis Crank-Nicolson only                |
| 19 | `19_qwo_vs_spsa`              | `noisy_quadratic`      | QWO vs SPSA, head-to-head                   |
| 20 | `20_qwo_vs_sa_rs`             | `rastrigin_2d`         | QWO vs SA and random search                 |

Numerical results land in `outputs/experiments/`:
- `all_metrics.csv` - per-seed long table
- `all_metrics.json` - nested raw payload
- `bootstrap_summary.csv` - mean / signal rows
- `per_experiment_plots/*.png` - convergence curves per experiment

## Optimization signal

For each experiment, define
```
optimization_signal = mean(best_classical_loss) - mean(qwo_loss)
```
across seeds, picking the *best* classical method per experiment by its
own mean best-loss. Positive signal means QWO is ahead.

There is also an *evaluations-saved* metric:
```
evaluations_saved = evals_classical - evals_qwo
```
on the runs where both methods reached the user-specified loss threshold.

## Honest read of the results

QWO is competitive on a handful of synthetic 1-D / 2-D landscapes and is
noticeably **outperformed by classical baselines** on the harder benchmarks
- particularly Rastrigin and Ackley in 4-D, where the discretised grid is
too coarse to localise the global minimum and well-tuned simulated
annealing / SPSA pull ahead. The ablations also do not show a clean win
for entanglement or for the kinetic mixer at the modest grid sizes used in
`--quick` mode. We report this honestly: a quantum-inspired structure is
not a free win, and a careful matched-budget benchmark is the right tool
for telling the story.

## Limitations

- **Simulator only.** The wavefunction is a complex numpy array of size
  `N = prod_k n_k`. We exploit FFTs and Crank-Nicolson for the mixers.
  At `n_q <= log2(N) ~ 13` qubits everything is classically tractable.
- **Local refinement matters.** Most QWO "wins" come from the local
  Nelder-Mead step, not from the wavefunction. A future version should
  isolate the contribution of phase + mixing.
- **No noise model.** Real-hardware estimation of `|Psi|^2` would
  introduce shot noise that this prototype ignores.
- **Heuristic schedules.** `eta`, `tau`, `chi`, and the number of samples
  per step are fixed defaults. A proper hyperparameter sweep would tighten
  the comparison.

## Honesty note

We do **not** claim any quantum advantage. The signal we report is a
matched-budget performance difference on a controlled simulator-based
benchmark. A real-world claim requires a problem with no efficient
classical algorithm, accounting for shot noise, and an experimental
protocol the classical baseline cannot match. Our prototype does none of
those.

## License

Research prototype. No license file is shipped.
