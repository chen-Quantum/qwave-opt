"""Matched-budget classical baselines for the QWO comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np
from scipy import optimize

from . import SEED
from .grids import Grid
from .objectives import Objective


@dataclass
class BaselineRun:
    name: str
    best_theta: np.ndarray
    best_loss: float
    history: List[float]
    evals_per_step: List[int]
    total_evals: int


def _wrap(history: List[float], evals: List[int]) -> Tuple[List[float], List[int]]:
    """Convert per-step lists into a cumulative-best trace and matching eval counts."""
    best = float("inf")
    out_hist: List[float] = []
    for v in history:
        best = min(best, v)
        out_hist.append(best)
    return out_hist, evals


# -----------------------------------------------------------------------
# Random search
# -----------------------------------------------------------------------

def random_search(grid: Grid, objective: Objective, budget: int,
                  seed: int = SEED) -> BaselineRun:
    rng = np.random.default_rng(seed)
    history: List[float] = []
    evals: List[int] = []
    best_theta = None
    best = float("inf")
    for _ in range(int(budget)):
        theta = np.asarray([rng.uniform(lo, hi) for (lo, hi) in grid.bounds],
                            dtype=np.float64)
        val = float(objective(theta))
        history.append(val)
        evals.append(len(history))
        if val < best:
            best = val; best_theta = theta
    out_hist, out_evals = _wrap(history, evals)
    return BaselineRun("random_search", best_theta if best_theta is not None else np.zeros(grid.d),
                       best, out_hist, out_evals, len(history))


# -----------------------------------------------------------------------
# Grid search (equal eval budget by sub-sampling the grid)
# -----------------------------------------------------------------------

def grid_search(grid: Grid, objective: Objective, budget: int,
                seed: int = SEED) -> BaselineRun:
    rng = np.random.default_rng(seed)
    all_pts = grid.all_points()
    if budget < all_pts.shape[0]:
        idx = rng.choice(all_pts.shape[0], size=int(budget), replace=False)
        pts = all_pts[idx]
    else:
        # repeat the full grid if budget exceeds N.
        reps = int(np.ceil(budget / all_pts.shape[0]))
        pts = np.concatenate([all_pts] * reps, axis=0)[:int(budget)]
    history: List[float] = []
    evals: List[int] = []
    best = float("inf"); best_theta = None
    for t in pts:
        v = float(objective(t))
        history.append(v); evals.append(len(history))
        if v < best:
            best = v; best_theta = t
    out_hist, out_evals = _wrap(history, evals)
    return BaselineRun("grid_search", best_theta if best_theta is not None else np.zeros(grid.d),
                       best, out_hist, out_evals, len(history))


# -----------------------------------------------------------------------
# Simulated annealing
# -----------------------------------------------------------------------

def simulated_annealing(grid: Grid, objective: Objective, budget: int,
                         step: float = 0.3, T0: float = 1.0, alpha: float = 0.995,
                         seed: int = SEED) -> BaselineRun:
    rng = np.random.default_rng(seed)
    theta = np.asarray([rng.uniform(lo, hi) for (lo, hi) in grid.bounds],
                        dtype=np.float64)
    cur = float(objective(theta))
    best = cur; best_theta = theta.copy()
    T = float(T0)
    history: List[float] = [cur]; evals: List[int] = [1]
    for _ in range(int(budget) - 1):
        cand = theta + step * rng.standard_normal(grid.d)
        # Reflect off the box bounds.
        for k in range(grid.d):
            lo, hi = grid.bounds[k]
            if cand[k] < lo: cand[k] = 2 * lo - cand[k]
            if cand[k] > hi: cand[k] = 2 * hi - cand[k]
            cand[k] = max(min(cand[k], hi), lo)
        v = float(objective(cand))
        if v < cur or rng.random() < np.exp(-(v - cur) / max(T, 1e-9)):
            theta = cand; cur = v
            if v < best:
                best = v; best_theta = cand.copy()
        history.append(cur); evals.append(len(history))
        T = T * alpha
    out_hist, out_evals = _wrap(history, evals)
    return BaselineRun("simulated_annealing", best_theta, best, out_hist, out_evals,
                       len(history))


# -----------------------------------------------------------------------
# SPSA
# -----------------------------------------------------------------------

def spsa(grid: Grid, objective: Objective, budget: int, a: float = 0.10,
          c: float = 0.10, A: float = 5.0, gamma: float = 0.101,
          alpha: float = 0.602, seed: int = SEED) -> BaselineRun:
    rng = np.random.default_rng(seed)
    theta = np.asarray([rng.uniform(lo, hi) for (lo, hi) in grid.bounds],
                        dtype=np.float64)
    history: List[float] = []; evals: List[int] = []
    best = float("inf"); best_theta = theta.copy()
    n_iter = int(budget) // 2  # each iter uses 2 evals
    for k in range(1, n_iter + 1):
        ak = a / (k + A) ** alpha
        ck = c / k ** gamma
        delta = (2 * rng.integers(0, 2, size=grid.d) - 1).astype(np.float64)
        f_plus = float(objective(theta + ck * delta))
        f_minus = float(objective(theta - ck * delta))
        ghat = (f_plus - f_minus) / (2 * ck) * delta
        theta = theta - ak * ghat
        for j in range(grid.d):
            lo, hi = grid.bounds[j]
            theta[j] = max(min(theta[j], hi), lo)
        v = float(objective(theta))
        history.extend([f_plus, f_minus])
        evals.extend([len(history) - 1, len(history)])
        if v < best:
            best = v; best_theta = theta.copy()
    out_hist, out_evals = _wrap(history, evals)
    return BaselineRun("spsa", best_theta, best, out_hist, out_evals, len(history))


# -----------------------------------------------------------------------
# scipy minimize (Nelder-Mead) with random starts
# -----------------------------------------------------------------------

def scipy_minimize(grid: Grid, objective: Objective, budget: int,
                    n_starts: int = 5, seed: int = SEED) -> BaselineRun:
    rng = np.random.default_rng(seed)
    history: List[float] = []
    evals: List[int] = []
    best = float("inf"); best_theta = None
    per_start_budget = max(8, int(budget) // max(int(n_starts), 1))
    for _ in range(int(n_starts)):
        x0 = np.asarray([rng.uniform(lo, hi) for (lo, hi) in grid.bounds],
                         dtype=np.float64)
        res = optimize.minimize(
            lambda t: objective(t), x0=x0, method="Nelder-Mead",
            options=dict(maxiter=per_start_budget, xatol=1e-5),
        )
        history.append(float(res.fun))
        evals.append(len(history))
        if float(res.fun) < best:
            best = float(res.fun); best_theta = np.asarray(res.x, dtype=np.float64)
        if len(history) >= budget:
            break
    out_hist, out_evals = _wrap(history, evals)
    return BaselineRun("scipy_minimize", best_theta if best_theta is not None else np.zeros(grid.d),
                       best, out_hist, out_evals, len(history))


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

REGISTRY: dict[str, Callable[..., BaselineRun]] = {
    "random_search": random_search,
    "grid_search": grid_search,
    "simulated_annealing": simulated_annealing,
    "spsa": spsa,
    "scipy_minimize": scipy_minimize,
}
