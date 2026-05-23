"""Phase oracle is unitary."""

from __future__ import annotations

import numpy as np
import pytest

from src.grids import linear_grid
from src.objectives import get_objective
from src.phase_oracle import PhaseOracle
from src.wavefunction import uniform_wavefunction


def test_phase_preserves_norm() -> None:
    g = linear_grid(-2.0, 2.0, 16)
    obj = get_objective("double_well")
    wf = uniform_wavefunction(g)
    oracle = PhaseOracle(grid=g, objective=obj)
    n_before = wf.norm()
    oracle.apply(wf, eta=0.7)
    n_after = wf.norm()
    assert pytest.approx(n_before, abs=1e-12) == n_after


def test_phase_only_modulates_phase_not_amplitude() -> None:
    g = linear_grid(-1.0, 1.0, 8)
    obj = get_objective("double_well")
    wf = uniform_wavefunction(g)
    oracle = PhaseOracle(grid=g, objective=obj)
    abs_before = np.abs(wf.psi).copy()
    oracle.apply(wf, eta=1.3)
    abs_after = np.abs(wf.psi)
    assert np.allclose(abs_before, abs_after, atol=1e-12)


def test_phase_evaluation_count() -> None:
    g = linear_grid(-1.0, 1.0, 32)
    obj = get_objective("double_well")
    oracle = PhaseOracle(grid=g, objective=obj)
    assert oracle.evaluations == 0
    oracle.evaluate_grid()
    assert oracle.evaluations == 32
    # Subsequent calls reuse the cache.
    oracle.evaluate_grid()
    assert oracle.evaluations == 32
