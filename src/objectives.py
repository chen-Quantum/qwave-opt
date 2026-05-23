"""Objective functions for the QWO benchmark suite.

Every objective is a callable returning a scalar loss for a 1-D numpy array
of parameter values, plus a recommended ``Grid`` bounds tuple. Objectives
expose their globally optimal value (when known) so experiments can report
optimality gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from sklearn.datasets import make_circles, make_moons

from . import SEED


# -----------------------------------------------------------------------
# Shared container
# -----------------------------------------------------------------------

@dataclass
class Objective:
    name: str
    fn: Callable[[np.ndarray], float]
    d: int
    bounds: Tuple[Tuple[float, float], ...]
    minimum_loss: float
    has_minimum_at: Optional[np.ndarray] = None
    notes: str = ""

    def __call__(self, theta: np.ndarray) -> float:
        return float(self.fn(np.asarray(theta, dtype=np.float64).flatten()))

    def vectorised(self, thetas: np.ndarray) -> np.ndarray:
        """Vectorised over the leading axis. ``thetas`` has shape (N, d)."""
        thetas = np.asarray(thetas, dtype=np.float64)
        if thetas.ndim == 1:
            thetas = thetas.reshape(-1, self.d)
        return np.fromiter((self.fn(t) for t in thetas), dtype=np.float64, count=thetas.shape[0])


# -----------------------------------------------------------------------
# 1-D and 2-D textbook landscapes
# -----------------------------------------------------------------------

def double_well() -> Objective:
    def fn(theta: np.ndarray) -> float:
        x = theta[0]
        return (x * x - 4.0) ** 2 + 0.3 * x
    return Objective(
        "double_well", fn, d=1, bounds=((-3.0, 3.0),),
        minimum_loss=-0.5921, has_minimum_at=np.array([-1.985]),
        notes="(x^2 - 4)^2 + 0.3 x. Two near-degenerate basins around +-2.",
    )


def rugged_sinusoid() -> Objective:
    def fn(theta: np.ndarray) -> float:
        x = theta[0]
        return 0.10 * x * x + np.sin(4.0 * x) + 0.3 * np.sin(11.0 * x)
    return Objective(
        "rugged_sinusoid", fn, d=1, bounds=((-4.0, 4.0),),
        minimum_loss=-1.27, notes="Many shallow local minima inside a weak quadratic envelope.",
    )


def rastrigin(d: int = 2, A: float = 10.0) -> Objective:
    def fn(theta: np.ndarray) -> float:
        return float(A * d + np.sum(theta * theta - A * np.cos(2 * np.pi * theta)))
    bounds = tuple([(-5.12, 5.12)] * d)
    return Objective(
        f"rastrigin_{d}d", fn, d=d, bounds=bounds, minimum_loss=0.0,
        has_minimum_at=np.zeros(d), notes="Highly multimodal radial bumps; global min at origin.",
    )


def ackley(d: int = 2) -> Objective:
    def fn(theta: np.ndarray) -> float:
        n = theta.size
        s1 = float(np.sum(theta * theta))
        s2 = float(np.sum(np.cos(2 * np.pi * theta)))
        return -20.0 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20.0 + np.e
    bounds = tuple([(-5.0, 5.0)] * d)
    return Objective(
        f"ackley_{d}d", fn, d=d, bounds=bounds, minimum_loss=0.0,
        has_minimum_at=np.zeros(d), notes="Mostly flat outer plateau, sharp basin at origin.",
    )


def rosenbrock() -> Objective:
    def fn(theta: np.ndarray) -> float:
        x, y = theta[0], theta[1]
        return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2
    return Objective(
        "rosenbrock", fn, d=2, bounds=((-2.0, 2.0), (-1.0, 3.0)),
        minimum_loss=0.0, has_minimum_at=np.array([1.0, 1.0]),
        notes="Curved narrow valley; descent followed by hard-to-reach minimum.",
    )


def himmelblau() -> Objective:
    def fn(theta: np.ndarray) -> float:
        x, y = theta[0], theta[1]
        return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2
    return Objective(
        "himmelblau", fn, d=2, bounds=((-5.0, 5.0), (-5.0, 5.0)),
        minimum_loss=0.0, notes="Four equal global minima; classic multimodal test.",
    )


# -----------------------------------------------------------------------
# Stochastic / pathological objectives
# -----------------------------------------------------------------------

def noisy_quadratic(noise: float = 0.5, seed: int = SEED) -> Objective:
    rng = np.random.default_rng(seed)

    def fn(theta: np.ndarray) -> float:
        return float(0.5 * np.sum(theta * theta) + noise * rng.standard_normal())
    return Objective(
        "noisy_quadratic", fn, d=2, bounds=((-3.0, 3.0), (-3.0, 3.0)),
        minimum_loss=0.0, has_minimum_at=np.zeros(2),
        notes=f"Quadratic with Gaussian observation noise sigma = {noise}.",
    )


def sparse_reward() -> Objective:
    def fn(theta: np.ndarray) -> float:
        # Loss is 1 - reward; reward is 1 inside a tiny ball, 0 elsewhere.
        r2 = float(np.sum((theta - 1.0) ** 2))
        return 1.0 - float(r2 < 0.04)
    return Objective(
        "sparse_reward", fn, d=2, bounds=((-3.0, 3.0), (-3.0, 3.0)),
        minimum_loss=0.0, has_minimum_at=np.array([1.0, 1.0]),
        notes="Reward = 1 inside |theta - 1|^2 < 0.04; zero gradient elsewhere.",
    )


def wide_vs_narrow() -> Objective:
    def fn(theta: np.ndarray) -> float:
        # Wide shallow basin at (-2, -2), narrow deep basin at (2, 2).
        x, y = theta[0], theta[1]
        wide = -0.5 * np.exp(-((x + 2.0) ** 2 + (y + 2.0) ** 2) / 4.0)
        narrow = -1.0 * np.exp(-((x - 2.0) ** 2 + (y - 2.0) ** 2) / 0.20)
        return float(wide + narrow + 0.5)
    return Objective(
        "wide_vs_narrow", fn, d=2, bounds=((-4.0, 4.0), (-4.0, 4.0)),
        minimum_loss=-0.5, has_minimum_at=np.array([2.0, 2.0]),
        notes="Wide attractor vs narrow global minimum.",
    )


# -----------------------------------------------------------------------
# Machine-learning objectives
# -----------------------------------------------------------------------

def _logreg_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Binary logistic regression NLL with no bias term (regularised slightly)."""
    z = X @ theta
    # numerically stable log(1 + exp(-y * z))
    yz = y * z
    return float(np.mean(np.logaddexp(0.0, -yz)) + 1e-3 * float(theta @ theta))


def two_moons_logreg(n: int = 80, noise: float = 0.15, seed: int = SEED) -> Objective:
    X, y_raw = make_moons(n_samples=n, noise=noise, random_state=seed)
    y = 2 * y_raw.astype(np.float64) - 1.0  # {-1, +1}

    def fn(theta: np.ndarray) -> float:
        return _logreg_loss(X.astype(np.float64), y, theta)
    return Objective(
        "two_moons_logreg", fn, d=2, bounds=((-6.0, 6.0), (-6.0, 6.0)),
        minimum_loss=0.10, notes="Logistic regression on two-moons; non-convex in (w1, w2).",
    )


def circles_logreg(n: int = 80, noise: float = 0.10, seed: int = SEED) -> Objective:
    X, y_raw = make_circles(n_samples=n, noise=noise, factor=0.4, random_state=seed)
    y = 2 * y_raw.astype(np.float64) - 1.0

    def fn(theta: np.ndarray) -> float:
        return _logreg_loss(X.astype(np.float64), y, theta)
    return Objective(
        "circles_logreg", fn, d=2, bounds=((-6.0, 6.0), (-6.0, 6.0)),
        minimum_loss=0.55, notes="Logistic regression on circles; flat near origin.",
    )


def fourier_regression(seed: int = SEED) -> Objective:
    rng = np.random.default_rng(seed)
    x = np.linspace(-np.pi, np.pi, 64)
    a_true, b_true = 0.8, -0.6
    y = a_true * np.sin(x) + b_true * np.cos(2 * x) + 0.05 * rng.standard_normal(x.size)

    def fn(theta: np.ndarray) -> float:
        a, b = theta[0], theta[1]
        return float(np.mean((a * np.sin(x) + b * np.cos(2 * x) - y) ** 2))
    return Objective(
        "fourier_regression", fn, d=2, bounds=((-2.0, 2.0), (-2.0, 2.0)),
        minimum_loss=0.0025, has_minimum_at=np.array([a_true, b_true]),
        notes="MSE for fitting two Fourier coefficients to a noisy target.",
    )


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

REGISTRY = {
    "double_well": double_well,
    "rugged_sinusoid": rugged_sinusoid,
    "rastrigin_2d": lambda: rastrigin(2),
    "ackley_2d": lambda: ackley(2),
    "rosenbrock": rosenbrock,
    "himmelblau": himmelblau,
    "rastrigin_4d": lambda: rastrigin(4),
    "ackley_4d": lambda: ackley(4),
    "two_moons_logreg": two_moons_logreg,
    "circles_logreg": circles_logreg,
    "fourier_regression": fourier_regression,
    "noisy_quadratic": noisy_quadratic,
    "sparse_reward": sparse_reward,
    "wide_vs_narrow": wide_vs_narrow,
}


def get_objective(name: str) -> Objective:
    if name not in REGISTRY:
        raise KeyError(f"Unknown objective {name!r}. Choices: {sorted(REGISTRY)}")
    return REGISTRY[name]()
