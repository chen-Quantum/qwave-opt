"""Metrics: best_loss, evals_to_threshold, signal."""

from __future__ import annotations

import numpy as np

from src.metrics import (
    area_under_best_curve, best_loss, evaluations_saved,
    evaluations_to_threshold, optimization_signal,
)


def test_best_loss() -> None:
    assert best_loss([5.0, 2.0, 3.0]) == 2.0


def test_evaluations_to_threshold() -> None:
    hist = [10.0, 4.0, 1.0]
    evs = [100, 200, 300]
    assert evaluations_to_threshold(hist, evs, 5.0) == 200
    assert evaluations_to_threshold(hist, evs, 0.5) is None


def test_optimization_signal_paired() -> None:
    sig = optimization_signal(
        qwo_best=[0.10, 0.12, 0.09],
        classical_best_by_method={
            "random_search": [0.20, 0.22, 0.21],
            "simulated_annealing": [0.15, 0.16, 0.14],
        },
    )
    # Best classical = simulated_annealing (lowest mean). Signal = mean(SA - QWO).
    assert sig["best_classical"] == "simulated_annealing"
    expected = float(np.mean([0.15 - 0.10, 0.16 - 0.12, 0.14 - 0.09]))
    assert abs(sig["signal_mean"] - expected) < 1e-9


def test_evaluations_saved() -> None:
    out = evaluations_saved(
        qwo_evals_to_threshold=[80, 90, None],
        classical_evals_to_threshold_by_method={
            "random_search": [150, 200, 180],
            "simulated_annealing": [120, None, 130],
        },
    )
    # Best classical = simulated_annealing (lowest mean of finite values).
    assert out["best_classical"] == "simulated_annealing"
    # Paired: QWO (80, 90) vs SA (120, 130). Saved mean = ((120-80)+(130-90))/2 = 40.
    assert abs(out["saved_mean"] - 40.0) < 1e-9
