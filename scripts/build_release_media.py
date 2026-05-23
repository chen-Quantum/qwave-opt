"""Render all release media into outputs/release_media/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_EXP = ROOT / "outputs" / "experiments"
OUT_REL = ROOT / "outputs" / "release_media"


def _ensure_metrics() -> dict:
    p = OUT_EXP / "all_metrics.json"
    if not p.exists():
        from scripts.run_all_experiments import main as run_main
        run_main(["--quick"])
    with p.open() as fh:
        return json.load(fh)


def main() -> int:
    OUT_REL.mkdir(parents=True, exist_ok=True)
    summary = _ensure_metrics()
    from src.experiments import _grid_for, long_table  # type: ignore
    from src.objectives import get_objective
    from src.qwo import QWOConfig, run_qwo
    from src.visualize import (
        ablation_summary, cinematic_summary, convergence_curves,
        evaluations_to_threshold_plot, loss_landscape_2d, pipeline_overview,
        readme_hero, wavefunction_evolution_gif,
    )
    rows = long_table(summary)

    print("[QWO media] 1/7  pipeline overview ...")
    pipeline_overview(OUT_REL / "pipeline_overview.png")

    print("[QWO media] 2/7  landscape + QWO samples (Rastrigin 2D) ...")
    obj = get_objective("rastrigin_2d")
    grid = _grid_for(obj, 32)
    cfg = QWOConfig(iterations=8, samples_per_step=8, local_refine_steps=20)
    run = run_qwo(grid, obj, cfg, record_frames=True)
    # Sample 60 candidates from the final |psi|^2 to overlay on the landscape.
    rng = np.random.default_rng(0)
    from src.wavefunction import Wavefunction
    psi = run.psi_frames[-1] if run.psi_frames else None
    samples_xy = None
    if psi is not None:
        p = psi.flatten()
        p = np.clip(p, 0.0, None); s = p.sum()
        if s > 1e-12:
            p = p / s
            idx = rng.choice(p.size, size=60, p=p)
            samples_xy = np.stack([grid.decode(int(k)) for k in idx], axis=0)
    loss_landscape_2d(obj, grid, OUT_REL / "loss_landscape_with_samples.png",
                      samples=samples_xy, best=run.best_theta,
                      title="Rastrigin 2D landscape with QWO samples")

    print("[QWO media] 3/7  wavefunction evolution GIF ...")
    wavefunction_evolution_gif(run.psi_frames,
                                OUT_REL / "wavefunction_evolution.gif", fps=4)

    print("[QWO media] 4/7  convergence curves vs baselines (Rastrigin 2D) ...")
    target = next((r for r in summary["results"] if r["name"] == "03_rastrigin_2d"), None)
    if target is None and summary["results"]:
        target = summary["results"][0]
    if target is not None:
        convergence_curves(target["per_seed"], OUT_REL / "qwo_vs_baselines.png",
                           title=f"QWO vs baselines: {target['name']}",
                           threshold=target.get("threshold"))

    print("[QWO media] 5/7  evaluations-to-threshold summary ...")
    evaluations_to_threshold_plot(rows, OUT_REL / "evaluations_to_threshold.png")

    print("[QWO media] 6/7  ablation summary ...")
    ablation_summary(rows, OUT_REL / "ablation_summary.png")

    print("[QWO media] 7/7  README hero + cinematic summary ...")
    readme_hero(rows, OUT_REL / "loss_landscape_with_samples.png",
                OUT_REL / "qwo_vs_baselines.png", OUT_REL / "readme_hero.png")
    cinematic_summary(rows, OUT_REL / "loss_landscape_with_samples.png",
                      run.psi_frames, OUT_REL)

    print("\n[QWO media] Done.  Outputs:")
    for p in sorted(OUT_REL.iterdir()):
        if p.is_file():
            print(f"  {p.name:42s}  {p.stat().st_size:>10d} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
