"""Metrics: best loss, evaluations-to-threshold, AUC, optimization signal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def best_loss(history: Sequence[float]) -> float:
    if not history:
        return float("inf")
    return float(min(history))


def evaluations_to_threshold(history: Sequence[float], evals: Sequence[int],
                             threshold: float) -> int | None:
    """First number of evaluations at which the cumulative-best loss is <= threshold.

    Returns None if the threshold is never reached.
    """
    for v, e in zip(history, evals):
        if v <= threshold:
            return int(e)
    return None


def area_under_best_curve(history: Sequence[float], evals: Sequence[int]) -> float:
    """Crude trapezoidal area of the (eval-count, best-loss-so-far) curve."""
    if len(history) < 2:
        return float("inf")
    e = np.asarray(evals, dtype=np.float64)
    h = np.asarray(history, dtype=np.float64)
    order = np.argsort(e)
    e = e[order]; h = h[order]
    return float(np.trapezoid(h, e))


def success_rate(per_seed_best_losses: Sequence[float], threshold: float) -> float:
    if not per_seed_best_losses:
        return 0.0
    return float(np.mean(np.asarray(per_seed_best_losses) <= threshold))


# -----------------------------------------------------------------------
# Optimization signal
# -----------------------------------------------------------------------

def optimization_signal(qwo_best: Sequence[float],
                         classical_best_by_method: Dict[str, Sequence[float]]
                         ) -> Dict[str, Any]:
    """Mean-loss difference between QWO and the *best* classical method (lowest
    mean best-loss). Negative numbers mean QWO is better.
    """
    qwo = list(qwo_best)
    if not classical_best_by_method:
        return {"signal_mean": float("nan"), "best_classical": None,
                 "qwo_mean": float(np.mean(qwo)) if qwo else float("nan"),
                 "best_classical_mean": float("nan"), "n": len(qwo)}
    method_means = {m: float(np.mean(v)) for m, v in classical_best_by_method.items()}
    # The "best" classical method is the one with the LOWEST mean loss.
    best_method = min(method_means, key=method_means.get)
    best = list(classical_best_by_method[best_method])
    if len(best) == len(qwo) and len(best) > 0:
        diffs = np.asarray(best, dtype=np.float64) - np.asarray(qwo, dtype=np.float64)
        return {
            "signal_mean": float(diffs.mean()),
            "signal_std": float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0,
            "signal_lo": float(np.quantile(diffs, 0.025)) if diffs.size > 3 else float("nan"),
            "signal_hi": float(np.quantile(diffs, 0.975)) if diffs.size > 3 else float("nan"),
            "best_classical": best_method,
            "qwo_mean": float(np.mean(qwo)),
            "best_classical_mean": method_means[best_method],
            "n": len(qwo),
        }
    return {"signal_mean": float("nan"), "best_classical": best_method, "n": len(qwo)}


def evaluations_saved(qwo_evals_to_threshold: Sequence[int | None],
                       classical_evals_to_threshold_by_method:
                           Dict[str, Sequence[int | None]]) -> Dict[str, Any]:
    """How many evaluations the best classical method needed beyond QWO, on
    runs where both reached the threshold."""
    qwo = [int(x) for x in qwo_evals_to_threshold if x is not None]
    if not classical_evals_to_threshold_by_method:
        return {"saved_mean": float("nan"), "best_classical": None, "n": 0}
    method_med: Dict[str, float] = {}
    for m, vals in classical_evals_to_threshold_by_method.items():
        finite = [int(x) for x in vals if x is not None]
        if finite:
            method_med[m] = float(np.mean(finite))
    if not method_med:
        return {"saved_mean": float("nan"), "best_classical": None, "n": 0}
    best_method = min(method_med, key=method_med.get)
    best = [int(x) for x in classical_evals_to_threshold_by_method[best_method] if x is not None]
    if not best or not qwo:
        return {"saved_mean": float("nan"), "best_classical": best_method, "n": 0}
    diffs = np.asarray(best, dtype=np.float64) - np.asarray(qwo[:len(best)], dtype=np.float64)
    return {
        "saved_mean": float(diffs.mean()) if diffs.size else float("nan"),
        "saved_std": float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0,
        "best_classical": best_method,
        "qwo_evals_mean": float(np.mean(qwo)),
        "best_classical_evals_mean": float(np.mean(best)),
        "n": int(min(len(best), len(qwo))),
    }
