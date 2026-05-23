"""Kinetic and entangling mixers for the QWO wavefunction.

Three options + an entangling phase:

1. ``fft_mixer``                 - FFT/QFT-style free-particle evolution. For
                                   each axis, transform to momentum space,
                                   multiply by exp(-i tau k^2 / 2), inverse
                                   transform. Norm-preserving by construction.
2. ``finite_difference_mixer``   - second-order finite-difference Laplacian
                                   stencil per axis. Approximates the same
                                   evolution to first order in tau via a
                                   single-step Crank-Nicolson update.
3. ``no_mixer``                  - identity. Used for ablations.
4. ``zz_coupling``               - phase only: exp(-i chi theta_i theta_j)
                                   between any two parameter axes. Not a
                                   kinetic mixer, but adds a non-trivial
                                   entangling-style phase between registers.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift

from .grids import Grid
from .wavefunction import Wavefunction


# -----------------------------------------------------------------------
# Momentum grid helpers
# -----------------------------------------------------------------------

def _momentum_axis(n: int, dx: float) -> np.ndarray:
    """Standard FFT momentum frequencies in cycles-per-unit, shifted so that
    zero frequency is at index 0 (matches numpy's fftn convention).
    """
    return 2 * np.pi * np.fft.fftfreq(n, d=dx)


def _momentum_squared(grid: Grid) -> np.ndarray:
    """K^2 array with the same shape as the grid, broadcasting per-axis k_i^2 sums."""
    ks = [_momentum_axis(n, dx) for n, dx in zip(grid.points_per_axis, grid.spacings)]
    K2 = np.zeros(grid.shape, dtype=np.float64)
    for axis, k in enumerate(ks):
        shape = [1] * grid.d
        shape[axis] = -1
        K2 = K2 + (k.reshape(shape)) ** 2
    return K2


# -----------------------------------------------------------------------
# Mixers
# -----------------------------------------------------------------------

def fft_mixer(wf: Wavefunction, tau: float) -> None:
    """In-place evolution by exp(-i tau (-1/2) Laplacian) via FFT."""
    K2 = _momentum_squared(wf.grid)
    psi_k = fftn(wf.psi)
    psi_k = psi_k * np.exp(-1j * float(tau) * 0.5 * K2)
    wf.psi = ifftn(psi_k)


def finite_difference_mixer(wf: Wavefunction, tau: float) -> None:
    """One Crank-Nicolson step of the discrete-Laplacian Schroedinger equation.

    Implements (I + i tau/4 D) psi_{n+1} = (I - i tau/4 D) psi_n, where D is
    the negative discrete Laplacian. Norm-preserving up to numerical precision.
    For small grids we factor explicitly along each axis (operator splitting).
    """
    psi = wf.psi
    for axis in range(wf.grid.d):
        n = wf.grid.points_per_axis[axis]
        dx = wf.grid.spacings[axis]
        # Build the 1-D second-difference matrix on this axis (periodic boundary).
        diag = -2.0 * np.ones(n)
        off = np.ones(n)
        D = (1.0 / (dx * dx)) * (
            np.diag(diag) + np.diag(off[:-1], 1) + np.diag(off[:-1], -1)
        )
        # Periodic wrap (corners).
        D[0, -1] = D[-1, 0] = 1.0 / (dx * dx)
        T = -0.5 * D
        # Build Crank-Nicolson matrices.
        I = np.eye(n)
        A = I + 0.25j * tau * T
        B = I - 0.25j * tau * T
        # Move the active axis to the front for easy matmul.
        psi_swap = np.moveaxis(psi, axis, 0)
        flat = psi_swap.reshape(n, -1)
        rhs = B @ flat
        sol = np.linalg.solve(A, rhs)
        psi_swap = sol.reshape(psi_swap.shape)
        psi = np.moveaxis(psi_swap, 0, axis)
    wf.psi = psi


def no_mixer(wf: Wavefunction, tau: float) -> None:
    """Ablation: do nothing."""
    return None


def zz_coupling(wf: Wavefunction, chi: float, axis_a: int = 0, axis_b: int = 1) -> None:
    """Phase factor exp(-i chi theta_a theta_b) - couples two registers."""
    if wf.grid.d < 2:
        return None
    a = wf.grid.axis_values(axis_a)
    b = wf.grid.axis_values(axis_b)
    shape_a = [1] * wf.grid.d
    shape_a[axis_a] = -1
    shape_b = [1] * wf.grid.d
    shape_b[axis_b] = -1
    phase = np.exp(-1j * float(chi) * a.reshape(shape_a) * b.reshape(shape_b))
    wf.psi = wf.psi * phase


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

MIXER_REGISTRY = {
    "fft": fft_mixer,
    "finite_difference": finite_difference_mixer,
    "none": no_mixer,
}


def get_mixer(name: str):
    if name not in MIXER_REGISTRY:
        raise KeyError(f"Unknown mixer {name!r}. Choices: {sorted(MIXER_REGISTRY)}")
    return MIXER_REGISTRY[name]
