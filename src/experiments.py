"""Experiment registry and runner."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from . import SEED
from .baselines import REGISTRY as BASELINES, BaselineRun
from .grids import Grid, square_grid, linear_grid
from .metrics import (
    best_loss, evaluations_to_threshold, optimization_signal, evaluations_saved,
)
from .objectives import REGISTRY as OBJECTIVES, Objective, get_objective
from .qwo import QWOConfig, run_qwo


# -----------------------------------------------------------------------
# Experiment spec
# -----------------------------------------------------------------------

@dataclass
class Experiment:
    name: str
    objective: str                      # key into objectives.REGISTRY
    grid_points: int = 32               # per axis (capped per total size below)
    qwo: QWOConfig = field(default_factory=QWOConfig)
    baselines: Sequence[str] = field(default_factory=lambda: (
        "random_search", "grid_search", "simulated_annealing", "spsa", "scipy_minimize"))
    seeds: Sequence[int] = (0, 1, 2)
    threshold: float | None = None
    notes: str = ""


# Default seeds used by all experiments.
QUICK_SEEDS = (0, 1)
FULL_SEEDS = (0, 1, 2, 3, 4)


def _grid_for(obj: Objective, n_per_axis: int) -> Grid:
    bounds = obj.bounds
    n = int(min(n_per_axis, max(4, int(round(2 ** (16 / max(obj.d, 1)))))))
    # Hard cap on total grid size to keep memory + FFT cheap.
    while np.prod([n] * obj.d) > 4096:
        n = max(4, n - 1)
    return Grid(bounds=bounds, points_per_axis=tuple([int(n)] * obj.d))


def _budget(grid: Grid, qwo_cfg: QWOConfig) -> int:
    # Approximate the QWO eval budget so baselines run with a matched count.
    refine = int(qwo_cfg.local_refine_steps) * int(qwo_cfg.samples_per_step)
    return int(grid.size + qwo_cfg.iterations * refine)


def run_experiment(exp: Experiment, seeds: Sequence[int],
                   record_frames: bool = False) -> Dict[str, Any]:
    obj = get_objective(exp.objective)
    grid = _grid_for(obj, exp.grid_points)
    out: Dict[str, Any] = {
        "name": exp.name, "objective": exp.objective, "d": obj.d,
        "grid_size": grid.size, "grid_shape": list(grid.shape),
        "seeds": list(seeds), "qwo_config": asdict(exp.qwo),
        "per_seed": [], "notes": exp.notes,
        "threshold": float(exp.threshold) if exp.threshold is not None else None,
    }
    for s in seeds:
        cfg = QWOConfig(**{**asdict(exp.qwo), "seed": int(s)})
        q = run_qwo(grid, obj, cfg, record_frames=record_frames)
        budget = _budget(grid, cfg)
        per: Dict[str, Any] = {
            "seed": int(s),
            "qwo_best_loss": q.best_loss,
            "qwo_evals": q.total_evals,
            "qwo_history": q.history,
            "qwo_evals_per_step": q.evals_per_step,
            "qwo_entropy": q.iter_entropy,
            "qwo_concentration": q.iter_concentration,
            "baselines": {},
        }
        if exp.threshold is not None:
            per["qwo_evals_to_threshold"] = evaluations_to_threshold(
                q.history, q.evals_per_step, exp.threshold
            )
        for b_name in exp.baselines:
            if b_name not in BASELINES:
                continue
            b: BaselineRun = BASELINES[b_name](grid, obj, budget=budget, seed=int(s))
            entry = {
                "best_loss": b.best_loss, "evals": b.total_evals,
                "history": b.history, "evals_per_step": b.evals_per_step,
            }
            if exp.threshold is not None:
                entry["evals_to_threshold"] = evaluations_to_threshold(
                    b.history, b.evals_per_step, exp.threshold
                )
            per["baselines"][b_name] = entry
        out["per_seed"].append(per)
    return out


# -----------------------------------------------------------------------
# 20 experiments
# -----------------------------------------------------------------------

def all_experiments() -> List[Experiment]:
    base_qwo = QWOConfig()
    exps: List[Experiment] = []

    # 1) 1D double well
    exps.append(Experiment("01_double_well", "double_well", grid_points=64,
                            qwo=QWOConfig(iterations=10), threshold=-0.4,
                            notes="1-D double well; threshold near second-best minimum."))
    # 2) 1D rugged sinusoid
    exps.append(Experiment("02_rugged_sinusoid", "rugged_sinusoid", grid_points=64,
                            qwo=QWOConfig(iterations=12), threshold=-1.0,
                            notes="1-D rugged sinusoid."))
    # 3-4) 2D Rastrigin / Ackley
    exps.append(Experiment("03_rastrigin_2d", "rastrigin_2d", grid_points=32,
                            qwo=QWOConfig(iterations=12), threshold=2.0))
    exps.append(Experiment("04_ackley_2d", "ackley_2d", grid_points=32,
                            qwo=QWOConfig(iterations=12), threshold=1.5))
    # 5-6) 2D Rosenbrock / Himmelblau
    exps.append(Experiment("05_rosenbrock", "rosenbrock", grid_points=32,
                            qwo=QWOConfig(iterations=12), threshold=1.0))
    exps.append(Experiment("06_himmelblau", "himmelblau", grid_points=32,
                            qwo=QWOConfig(iterations=10), threshold=2.0))
    # 7-8) 4D Rastrigin / Ackley
    exps.append(Experiment("07_rastrigin_4d", "rastrigin_4d", grid_points=8,
                            qwo=QWOConfig(iterations=10), threshold=8.0))
    exps.append(Experiment("08_ackley_4d", "ackley_4d", grid_points=8,
                            qwo=QWOConfig(iterations=10), threshold=3.0))
    # 9-11) ML objectives
    exps.append(Experiment("09_two_moons_logreg", "two_moons_logreg", grid_points=32,
                            qwo=QWOConfig(iterations=10), threshold=0.30))
    exps.append(Experiment("10_circles_logreg", "circles_logreg", grid_points=32,
                            qwo=QWOConfig(iterations=10), threshold=0.66))
    exps.append(Experiment("11_fourier_regression", "fourier_regression", grid_points=32,
                            qwo=QWOConfig(iterations=10), threshold=0.05))
    # 12-14) pathological cases
    exps.append(Experiment("12_noisy_quadratic", "noisy_quadratic", grid_points=32,
                            qwo=QWOConfig(iterations=10), threshold=0.30))
    exps.append(Experiment("13_sparse_reward", "sparse_reward", grid_points=32,
                            qwo=QWOConfig(iterations=12), threshold=0.5))
    exps.append(Experiment("14_wide_vs_narrow", "wide_vs_narrow", grid_points=32,
                            qwo=QWOConfig(iterations=12), threshold=-0.30))
    # 15) QWO without mixer
    exps.append(Experiment("15_qwo_no_mixer", "rastrigin_2d", grid_points=32,
                            qwo=QWOConfig(iterations=12, mixer="none"),
                            threshold=2.0, notes="Mixer ablation: no kinetic mixing."))
    # 16) QWO without phase oracle (eta = 0)
    exps.append(Experiment("16_qwo_no_phase", "rastrigin_2d", grid_points=32,
                            qwo=QWOConfig(iterations=12, eta=0.0),
                            threshold=2.0, notes="Phase ablation: eta = 0."))
    # 17) QWO with entangling coupling
    exps.append(Experiment("17_qwo_with_zz", "rosenbrock", grid_points=32,
                            qwo=QWOConfig(iterations=12, use_zz=True, chi=0.4),
                            threshold=1.0, notes="ZZ entangling phase across the two axes."))
    # 18) QWO separable registers (no ZZ, no kinetic across axes - use finite difference)
    exps.append(Experiment("18_qwo_separable", "rosenbrock", grid_points=32,
                            qwo=QWOConfig(iterations=12, mixer="finite_difference"),
                            threshold=1.0, notes="Per-axis Crank-Nicolson mixing only."))
    # 19) QWO vs SPSA on the noisy objective
    exps.append(Experiment("19_qwo_vs_spsa", "noisy_quadratic", grid_points=32,
                            qwo=QWOConfig(iterations=10),
                            baselines=("spsa", "random_search"), threshold=0.30))
    # 20) QWO vs SA + RS on Rastrigin 2D
    exps.append(Experiment("20_qwo_vs_sa_rs", "rastrigin_2d", grid_points=32,
                            qwo=QWOConfig(iterations=12),
                            baselines=("simulated_annealing", "random_search"),
                            threshold=2.0))
    return exps


# -----------------------------------------------------------------------
# Long table / signal aggregation
# -----------------------------------------------------------------------

def long_table(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the nested summary into per-(experiment, method) rows."""
    rows: List[Dict[str, Any]] = []
    for exp in summary["results"]:
        per_seed: List[Dict[str, Any]] = exp["per_seed"]
        if not per_seed:
            continue
        # QWO
        qwo_best = [s["qwo_best_loss"] for s in per_seed]
        rows.append({
            "experiment": exp["name"], "method": "qwo",
            "mean": float(np.mean(qwo_best)),
            "std": float(np.std(qwo_best, ddof=1)) if len(qwo_best) > 1 else 0.0,
            "n": len(qwo_best),
        })
        # Baselines
        baselines_collected: Dict[str, List[float]] = {}
        for s in per_seed:
            for b_name, entry in s["baselines"].items():
                baselines_collected.setdefault(b_name, []).append(entry["best_loss"])
        for b_name, vals in baselines_collected.items():
            rows.append({
                "experiment": exp["name"], "method": f"classical_{b_name}",
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            })
        # Signal
        sig = optimization_signal(qwo_best, baselines_collected)
        rows.append({
            "experiment": exp["name"], "method": "OPTIMIZATION_SIGNAL",
            "best_classical": sig.get("best_classical"),
            "qwo_mean": sig.get("qwo_mean"),
            "best_classical_mean": sig.get("best_classical_mean"),
            "signal_mean": sig.get("signal_mean"),
            "signal_lo": sig.get("signal_lo", float("nan")),
            "signal_hi": sig.get("signal_hi", float("nan")),
            "mean": sig.get("signal_mean"),
            "n": sig.get("n"),
        })
        # Evals saved (if threshold defined)
        if exp.get("threshold") is not None:
            qwo_evals = [s.get("qwo_evals_to_threshold") for s in per_seed]
            cls_evals: Dict[str, List[int | None]] = {}
            for s in per_seed:
                for b_name, entry in s["baselines"].items():
                    cls_evals.setdefault(b_name, []).append(entry.get("evals_to_threshold"))
            saved = evaluations_saved(qwo_evals, cls_evals)
            rows.append({
                "experiment": exp["name"], "method": "EVALS_SAVED",
                "best_classical": saved.get("best_classical"),
                "qwo_evals_mean": saved.get("qwo_evals_mean"),
                "best_classical_evals_mean": saved.get("best_classical_evals_mean"),
                "mean": saved.get("saved_mean"),
                "n": saved.get("n"),
            })
    return rows


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

def run_suite(experiments: Sequence[Experiment], seeds: Sequence[int],
               out_dir: Path,
               on_progress: Callable[[int, int, str], None] | None = None
               ) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for k, exp in enumerate(experiments):
        if on_progress:
            on_progress(k, len(experiments), exp.name)
        try:
            r = run_experiment(exp, seeds)
        except Exception as e:  # pragma: no cover
            r = {"name": exp.name, "error": str(e), "per_seed": []}
        results.append(r)
    elapsed = time.time() - t0
    return {"elapsed_seconds": elapsed, "n_experiments": len(experiments),
             "seeds": list(seeds), "results": results}
