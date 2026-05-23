"""Phase oracle: U_L(eta) |theta> = exp(-i eta L(theta)) |theta>.

We cache the per-grid-point loss array so that subsequent phase steps are
free in objective evaluations after the first. Each unique call to
``evaluate_grid`` charges ``grid.size`` evaluations to the optimiser's
budget; reusing the cached array adds zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .grids import Grid
from .objectives import Objective
from .wavefunction import Wavefunction


@dataclass
class PhaseOracle:
    grid: Grid
    objective: Objective
    loss_grid: Optional[np.ndarray] = None  # shape = grid.shape, dtype float64
    evaluations: int = 0

    def evaluate_grid(self) -> np.ndarray:
        """Compute L(theta_j) at every grid point. Counts grid.size evals once."""
        if self.loss_grid is None:
            pts = self.grid.all_points()
            losses = np.asarray([self.objective(p) for p in pts], dtype=np.float64)
            self.loss_grid = losses.reshape(self.grid.shape)
            self.evaluations += self.grid.size
        return self.loss_grid

    def apply(self, wf: Wavefunction, eta: float) -> None:
        """Multiply psi by exp(-i eta L) elementwise."""
        L = self.evaluate_grid()
        wf.psi = wf.psi * np.exp(-1j * float(eta) * L)

    def best_grid_value(self) -> float:
        """Smallest loss seen at any grid point. Useful as a sanity reference."""
        return float(self.evaluate_grid().min())

    def reset_cache(self) -> None:
        self.loss_grid = None
