"""Quick interactive demo: run QWO on a 2-D landscape and dump artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "experiments" / "toy_demo"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from src.experiments import _grid_for  # type: ignore
    from src.objectives import get_objective
    from src.qwo import QWOConfig, run_qwo
    from src.visualize import (
        loss_landscape_2d, wavefunction_evolution_gif,
    )
    obj = get_objective("rastrigin_2d")
    grid = _grid_for(obj, 32)
    cfg = QWOConfig(iterations=8, samples_per_step=8, local_refine_steps=20)
    run = run_qwo(grid, obj, cfg, record_frames=True)
    print(f"best_loss = {run.best_loss:.4f}  theta = {run.best_theta}")
    landscape_path = OUT / "landscape.png"
    loss_landscape_2d(obj, grid, landscape_path,
                       samples=None,
                       best=run.best_theta,
                       title="Rastrigin 2D")
    wavefunction_evolution_gif(run.psi_frames, OUT / "wavefunction.gif", fps=4)
    print(f"Outputs in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
