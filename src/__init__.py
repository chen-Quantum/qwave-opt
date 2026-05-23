"""QWO: Quantum Wavefunction Optimizer for Continuous Machine Learning Landscapes.

A simulator-based research prototype. The optimizer represents continuous
parameters on a discretised grid, propagates a complex wavefunction over the
grid using a loss-dependent phase oracle and a kinetic mixer, and samples
candidate solutions from the resulting probability distribution. We measure a
"quantum optimization signal" - matched-budget performance against classical
baselines. We make NO claim of true quantum advantage.
"""

SEED = 0xABBA1357  # 2880541015


__all__ = [
    "grids",
    "objectives",
    "wavefunction",
    "phase_oracle",
    "mixers",
    "qwo",
    "baselines",
    "metrics",
    "visualize",
    "experiments",
]
