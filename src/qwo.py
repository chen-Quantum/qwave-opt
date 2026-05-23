"""Quantum Wavefunction Optimizer - main loop.

Pseudocode:

    psi <- uniform wavefunction over grid
    for iter = 1..K:
        apply phase oracle U_L(eta_t)
        apply mixer (FFT / finite-difference / none)
        optionally apply ZZ coupling
        sample top-K candidates from |psi|^2
        locally refine each candidate with scipy.optimize.minimize
        keep the running best
    return best candidate

We track total objective evaluations (grid evaluations + each refinement
step) so the optimiser can be compared head-to-head with classical baselines
at a fixed budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

from . import SEED
from .grids import Grid
from .mixers import fft_mixer, finite_difference_mixer, no_mixer, zz_coupling
from .objectives import Objective
from .phase_oracle import PhaseOracle
from .wavefunction import Wavefunction, uniform_wavefunction


MIXER_BY_NAME = {
    "fft": fft_mixer,
    "finite_difference": finite_difference_mixer,
    "none": no_mixer,
}


@dataclass
class QWOConfig:
    iterations: int = 12
    eta: float = 0.8           # phase strength per step
    tau: float = 0.4           # kinetic mixer strength per step
    mixer: str = "fft"
    use_zz: bool = False
    chi: float = 0.3
    samples_per_step: int = 8
    local_refine_steps: int = 25
    seed: int = SEED


@dataclass
class QWORun:
    best_theta: np.ndarray
    best_loss: float
    history: List[float]                  # cumulative best loss after each iteration
    evals_per_step: List[int]             # total objective evals after each iteration
    iter_concentration: List[float]       # max |psi|^2 per iteration
    iter_entropy: List[float]
    total_evals: int
    psi_frames: List[np.ndarray] = field(default_factory=list)


def _grid_objective_eval_count(grid: Grid) -> int:
    return grid.size


def run_qwo(grid: Grid, objective: Objective, config: QWOConfig,
            record_frames: bool = False) -> QWORun:
    """Run the QWO loop and return a record of progress."""
    rng = np.random.default_rng(int(config.seed))
    wf = uniform_wavefunction(grid)
    oracle = PhaseOracle(grid=grid, objective=objective)

    # Pre-charge the cache.
    oracle.evaluate_grid()
    total_evals = oracle.evaluations

    mixer = MIXER_BY_NAME[config.mixer]

    best_theta = grid.decode(int(np.argmin(oracle.evaluate_grid())))
    best_loss = float(objective(best_theta))
    history: List[float] = [best_loss]
    evals_log: List[int] = [total_evals]
    conc_log: List[float] = [wf.concentration()]
    ent_log: List[float] = [wf.shannon_entropy()]
    frames: List[np.ndarray] = []
    if record_frames:
        frames.append(np.abs(wf.psi) ** 2)

    for it in range(config.iterations):
        oracle.apply(wf, eta=config.eta)
        mixer(wf, tau=config.tau)
        if config.use_zz and grid.d >= 2:
            zz_coupling(wf, chi=config.chi, axis_a=0, axis_b=1)
        wf.normalize()

        # Sample candidates from |psi|^2 and locally refine each.
        idx = wf.sample(config.samples_per_step, rng)
        for k in idx:
            theta0 = grid.decode(int(k))
            res = optimize.minimize(
                lambda t: objective(t),
                x0=theta0,
                method="Nelder-Mead",
                options=dict(maxiter=int(config.local_refine_steps), xatol=1e-4),
            )
            total_evals += int(res.nfev)
            if float(res.fun) < best_loss:
                best_loss = float(res.fun)
                best_theta = np.asarray(res.x, dtype=np.float64)

        history.append(best_loss)
        evals_log.append(total_evals)
        conc_log.append(wf.concentration())
        ent_log.append(wf.shannon_entropy())
        if record_frames:
            frames.append(np.abs(wf.psi) ** 2)

    return QWORun(
        best_theta=best_theta,
        best_loss=float(best_loss),
        history=history,
        evals_per_step=evals_log,
        iter_concentration=conc_log,
        iter_entropy=ent_log,
        total_evals=int(total_evals),
        psi_frames=frames,
    )
