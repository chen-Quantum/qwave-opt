"""Grid encode/decode correctness."""

from __future__ import annotations

import numpy as np
import pytest

from src.grids import Grid, linear_grid, square_grid


def test_linear_grid_size_and_decode_roundtrip() -> None:
    g = linear_grid(-1.0, 1.0, 5)
    assert g.size == 5
    assert g.shape == (5,)
    # End-points map to extreme indices.
    assert g.encode(np.array([-1.0])) == 0
    assert g.encode(np.array([1.0])) == 4
    # Decode every index gives back the axis values.
    axis = g.axis_values(0)
    for i, v in enumerate(axis):
        assert pytest.approx(v, abs=1e-12) == g.decode(i)[0]


def test_2d_grid_size_and_indexing() -> None:
    g = square_grid(-2.0, 2.0, 4, d=2)
    assert g.size == 16
    assert g.shape == (4, 4)
    pts = g.all_points()
    assert pts.shape == (16, 2)
    # Re-encode the decoded point gives the same flat index.
    for i in range(g.size):
        theta = g.decode(i)
        assert g.encode(theta) == i


def test_grid_clipping() -> None:
    g = linear_grid(0.0, 1.0, 11)
    # Out-of-bounds inputs clip to valid indices.
    assert g.encode(np.array([-5.0])) == 0
    assert g.encode(np.array([5.0])) == 10
