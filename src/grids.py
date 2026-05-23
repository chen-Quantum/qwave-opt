"""Grid encoding for continuous parameter spaces.

A Grid is a tensor product of regularly-spaced 1-D axes:

    theta in [low_k, high_k]   discretised into n_k points,  k = 1, ..., d.

The total number of grid points is N = prod_k n_k. We index them as a flat
array of length N and as a multi-index of shape (n_1, ..., n_d) when we need
per-axis operations such as the FFT mixer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Grid:
    bounds: Tuple[Tuple[float, float], ...]   # ((low_1, high_1), ..., (low_d, high_d))
    points_per_axis: Tuple[int, ...]           # (n_1, ..., n_d)

    @property
    def d(self) -> int:
        return len(self.bounds)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.points_per_axis)

    @property
    def size(self) -> int:
        out = 1
        for n in self.points_per_axis:
            out *= int(n)
        return out

    @property
    def spacings(self) -> Tuple[float, ...]:
        out: List[float] = []
        for (lo, hi), n in zip(self.bounds, self.points_per_axis):
            out.append((hi - lo) / max(n - 1, 1))
        return tuple(out)

    def axis_values(self, axis: int) -> np.ndarray:
        lo, hi = self.bounds[axis]
        return np.linspace(lo, hi, self.points_per_axis[axis], dtype=np.float64)

    def all_points(self) -> np.ndarray:
        """Return an (N, d) array of all grid points in C-order (last axis fastest)."""
        axes = [self.axis_values(k) for k in range(self.d)]
        mesh = np.meshgrid(*axes, indexing="ij")
        flat = [m.flatten() for m in mesh]
        return np.stack(flat, axis=-1)

    def encode(self, theta: np.ndarray) -> int:
        """Nearest grid index for a single continuous point theta of shape (d,)."""
        theta = np.asarray(theta, dtype=np.float64).flatten()
        if theta.size != self.d:
            raise ValueError(f"theta must have {self.d} components")
        idx_per_axis: List[int] = []
        for k in range(self.d):
            lo, hi = self.bounds[k]
            n = self.points_per_axis[k]
            t = (theta[k] - lo) / max(hi - lo, 1e-12) * (n - 1)
            idx_per_axis.append(int(np.clip(round(t), 0, n - 1)))
        return self.ravel(tuple(idx_per_axis))

    def decode(self, flat_index: int) -> np.ndarray:
        """Continuous coordinates of grid point at the given flat index."""
        multi = self.unravel(flat_index)
        return np.asarray(
            [self.axis_values(k)[multi[k]] for k in range(self.d)],
            dtype=np.float64,
        )

    def ravel(self, multi_index: Sequence[int]) -> int:
        return int(np.ravel_multi_index(tuple(multi_index), self.points_per_axis))

    def unravel(self, flat_index: int) -> Tuple[int, ...]:
        return tuple(int(x) for x in np.unravel_index(int(flat_index), self.points_per_axis))


def linear_grid(low: float, high: float, n: int) -> Grid:
    """Convenience constructor for a 1-D grid."""
    return Grid(bounds=((float(low), float(high)),), points_per_axis=(int(n),))


def square_grid(low: float, high: float, n: int, d: int = 2) -> Grid:
    """Convenience constructor for an isotropic d-D grid."""
    bounds = tuple([(float(low), float(high))] * d)
    pts = tuple([int(n)] * d)
    return Grid(bounds=bounds, points_per_axis=pts)
