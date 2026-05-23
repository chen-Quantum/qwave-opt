"""Wavefunction normalisation, probabilities, sampling."""

from __future__ import annotations

import numpy as np
import pytest

from src.grids import linear_grid, square_grid
from src.wavefunction import Wavefunction, uniform_wavefunction


def test_uniform_norm_is_one() -> None:
    wf = uniform_wavefunction(linear_grid(-1.0, 1.0, 8))
    assert pytest.approx(1.0, abs=1e-12) == wf.norm()


def test_probabilities_sum_to_one() -> None:
    wf = uniform_wavefunction(square_grid(-1.0, 1.0, 6, d=2))
    p = wf.probabilities()
    assert p.shape == (6, 6)
    assert pytest.approx(1.0, abs=1e-12) == float(p.sum())


def test_sample_size_and_range() -> None:
    rng = np.random.default_rng(0)
    wf = uniform_wavefunction(linear_grid(-1.0, 1.0, 10))
    idx = wf.sample(50, rng)
    assert idx.shape == (50,)
    assert idx.min() >= 0 and idx.max() < 10


def test_entropy_of_uniform_is_log_N() -> None:
    wf = uniform_wavefunction(linear_grid(-1.0, 1.0, 16))
    assert pytest.approx(np.log(16), abs=1e-9) == wf.shannon_entropy()


def test_normalize_after_perturbation() -> None:
    wf = uniform_wavefunction(linear_grid(-1.0, 1.0, 8))
    wf.psi = wf.psi * 3.5  # scale up
    wf.normalize()
    assert pytest.approx(1.0, abs=1e-12) == wf.norm()
