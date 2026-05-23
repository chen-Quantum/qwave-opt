"""QWO smoke tests: tiny optimisation runs end-to-end and beats the uniform
sample on at least one simple objective.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments import Experiment, run_experiment, _grid_for
from src.objectives import get_objective
from src.qwo import QWOConfig, run_qwo


def test_qwo_runs_on_double_well() -> None:
    obj = get_objective("double_well")
    grid = _grid_for(obj, 32)
    cfg = QWOConfig(iterations=4, samples_per_step=4, local_refine_steps=10,
                     seed=0)
    run = run_qwo(grid, obj, cfg)
    # Best loss should be <= the grid's minimum value at worst, and very likely
    # strictly better after a few local refines.
    grid_min = float(obj.vectorised(grid.all_points()).min())
    assert run.best_loss <= grid_min + 1e-6
    assert run.total_evals > grid.size  # spent some refinement evals


def test_qwo_improves_on_uniform_baseline() -> None:
    obj = get_objective("rosenbrock")
    grid = _grid_for(obj, 24)
    cfg = QWOConfig(iterations=6, samples_per_step=6, local_refine_steps=20,
                     seed=0)
    run = run_qwo(grid, obj, cfg)
    grid_min = float(obj.vectorised(grid.all_points()).min())
    assert run.best_loss < grid_min  # Nelder-Mead climbed off the grid.


def test_experiment_end_to_end() -> None:
    exp = Experiment(
        name="t_smoke", objective="double_well", grid_points=16,
        qwo=QWOConfig(iterations=3, samples_per_step=3, local_refine_steps=8,
                      seed=0),
        baselines=("random_search",), seeds=(0,),
        threshold=-0.4, notes="smoke",
    )
    out = run_experiment(exp, seeds=(0,))
    assert "per_seed" in out and len(out["per_seed"]) == 1
    s = out["per_seed"][0]
    assert "qwo_best_loss" in s
    assert "random_search" in s["baselines"]
    assert 0.0 <= s["qwo_evals"] < 10_000
