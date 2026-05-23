"""Mixers preserve norm."""

from __future__ import annotations

import numpy as np
import pytest

from src.grids import linear_grid, square_grid
from src.mixers import fft_mixer, finite_difference_mixer, no_mixer, zz_coupling
from src.wavefunction import uniform_wavefunction


def _random_wf(grid, seed: int):
    wf = uniform_wavefunction(grid)
    rng = np.random.default_rng(seed)
    wf.psi = rng.standard_normal(grid.shape) + 1j * rng.standard_normal(grid.shape)
    wf.normalize()
    return wf


def test_fft_mixer_preserves_norm_1d() -> None:
    wf = _random_wf(linear_grid(-1.0, 1.0, 16), seed=0)
    fft_mixer(wf, tau=0.5)
    assert pytest.approx(1.0, abs=1e-10) == wf.norm()


def test_fft_mixer_preserves_norm_2d() -> None:
    wf = _random_wf(square_grid(-1.0, 1.0, 8, d=2), seed=1)
    fft_mixer(wf, tau=0.3)
    assert pytest.approx(1.0, abs=1e-10) == wf.norm()


def test_finite_difference_mixer_preserves_norm_1d() -> None:
    wf = _random_wf(linear_grid(-1.0, 1.0, 16), seed=2)
    finite_difference_mixer(wf, tau=0.3)
    assert pytest.approx(1.0, abs=1e-9) == wf.norm()


def test_no_mixer_is_identity() -> None:
    wf = _random_wf(linear_grid(-1.0, 1.0, 8), seed=3)
    psi_before = wf.psi.copy()
    no_mixer(wf, tau=0.5)
    assert np.allclose(wf.psi, psi_before)


def test_zz_coupling_preserves_amplitudes() -> None:
    wf = _random_wf(square_grid(-1.0, 1.0, 8, d=2), seed=4)
    amps_before = np.abs(wf.psi).copy()
    zz_coupling(wf, chi=0.5, axis_a=0, axis_b=1)
    assert np.allclose(np.abs(wf.psi), amps_before, atol=1e-12)
