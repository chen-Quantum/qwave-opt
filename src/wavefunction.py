"""Complex wavefunction over a parameter grid.

A wavefunction psi has the same shape as the grid's point grid:
psi.shape == grid.shape (e.g. (n_1, n_2) for a 2-D grid). It stores complex
amplitudes and provides utilities for normalisation, probability extraction,
sampling, and basic information-theoretic diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import SEED
from .grids import Grid


@dataclass
class Wavefunction:
    grid: Grid
    psi: np.ndarray  # complex array, shape = grid.shape

    @classmethod
    def uniform(cls, grid: Grid) -> "Wavefunction":
        amp = 1.0 / np.sqrt(grid.size)
        psi = np.full(grid.shape, amp + 0j, dtype=np.complex128)
        return cls(grid=grid, psi=psi)

    def normalize(self) -> None:
        nrm = float(np.linalg.norm(self.psi.flatten()))
        if nrm > 1e-12:
            self.psi /= nrm

    def norm(self) -> float:
        return float(np.linalg.norm(self.psi.flatten()))

    def probabilities(self) -> np.ndarray:
        p = np.abs(self.psi) ** 2
        s = float(p.sum())
        if s < 1e-12:
            return np.full_like(p, 1.0 / self.grid.size)
        return p / s

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n integer flat indices according to |psi|^2."""
        p = self.probabilities().flatten()
        # safety: renormalise after rounding errors.
        p = np.clip(p, 0.0, None)
        s = float(p.sum())
        if s < 1e-12:
            return rng.integers(0, self.grid.size, size=n)
        p = p / s
        return rng.choice(self.grid.size, size=n, p=p)

    def best_index(self) -> int:
        return int(np.argmax(self.probabilities()))

    def shannon_entropy(self) -> float:
        p = self.probabilities().flatten()
        p = np.clip(p, 1e-30, 1.0)
        return float(-np.sum(p * np.log(p)))

    def concentration(self) -> float:
        """Max(|psi|^2) - a quick scalar diagnostic of how peaked the wavefunction is."""
        return float(np.max(self.probabilities()))

    def copy(self) -> "Wavefunction":
        return Wavefunction(grid=self.grid, psi=self.psi.copy())


def hamming_distance_array(grid: Grid) -> np.ndarray:
    """Return a (N, d) array of grid coordinates. Cached helper for visualisations."""
    return grid.all_points()


def uniform_wavefunction(grid: Grid) -> Wavefunction:
    return Wavefunction.uniform(grid)


def sample_decoded(wf: Wavefunction, n: int, seed: int = SEED) -> np.ndarray:
    """Sample n grid points and decode each into continuous coordinates."""
    rng = np.random.default_rng(seed)
    idx = wf.sample(n, rng)
    return np.stack([wf.grid.decode(int(k)) for k in idx], axis=0)
