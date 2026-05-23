"""Run the full QWO experiment suite and write metrics + plots.

    python scripts/run_all_experiments.py            # 3 seeds, full suite
    python scripts/run_all_experiments.py --quick    # 2 seeds, faster
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs" / "experiments"


def _write_metrics(summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "all_metrics.json").open("w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o:
                  float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    # Per-seed long table.
    flat: List[Dict[str, Any]] = []
    for r in summary["results"]:
        if r.get("error"):
            flat.append({"experiment": r["name"], "method": "ERROR", "value": r["error"]})
            continue
        for s in r["per_seed"]:
            flat.append({
                "experiment": r["name"], "objective": r["objective"],
                "d": r["d"], "grid_size": r["grid_size"],
                "seed": s["seed"], "method": "qwo",
                "best_loss": s["qwo_best_loss"], "evals": s["qwo_evals"],
                "evals_to_threshold": s.get("qwo_evals_to_threshold"),
            })
            for b_name, entry in s["baselines"].items():
                flat.append({
                    "experiment": r["name"], "objective": r["objective"],
                    "d": r["d"], "grid_size": r["grid_size"],
                    "seed": s["seed"], "method": b_name,
                    "best_loss": entry["best_loss"], "evals": entry["evals"],
                    "evals_to_threshold": entry.get("evals_to_threshold"),
                })
    if flat:
        keys = list({k for row in flat for k in row.keys()})
        with (OUT_DIR / "all_metrics.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in flat:
                w.writerow(row)
    keys = ["experiment", "method", "best_classical", "qwo_mean",
            "best_classical_mean", "qwo_evals_mean", "best_classical_evals_mean",
            "mean", "std", "n"]
    with (OUT_DIR / "bootstrap_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _per_experiment_plots(summary: Dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    out = OUT_DIR / "per_experiment_plots"
    out.mkdir(parents=True, exist_ok=True)
    from src.visualize import convergence_curves
    for exp in summary["results"]:
        if exp.get("error"):
            continue
        per_seed = exp["per_seed"]
        if not per_seed:
            continue
        convergence_curves(
            per_seed, out / f"{exp['name']}.png",
            title=f"{exp['name']}  ({exp['objective']})",
            threshold=exp.get("threshold"),
        )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                    help="2 seeds + fewer QWO iterations / refines.")
    args = p.parse_args(argv)

    from src.experiments import QUICK_SEEDS, FULL_SEEDS, all_experiments, long_table, run_suite

    seeds = QUICK_SEEDS if args.quick else FULL_SEEDS
    exps = all_experiments()
    if args.quick:
        for e in exps:
            e.qwo.iterations = max(4, e.qwo.iterations // 2)
            e.qwo.samples_per_step = max(4, e.qwo.samples_per_step // 2)
            e.qwo.local_refine_steps = max(8, e.qwo.local_refine_steps // 2)

    print(f"[QWO] Running {len(exps)} experiments x {len(seeds)} seeds "
          f"({'quick' if args.quick else 'full'} mode)")
    t0 = time.time()

    def progress(k: int, total: int, name: str) -> None:
        dt = time.time() - t0
        print(f"  [{k + 1:>2}/{total}]  {name:34s}  elapsed {dt:5.1f}s")

    summary = run_suite(exps, seeds, OUT_DIR, on_progress=progress)
    rows = long_table(summary)
    _write_metrics(summary, rows)
    _per_experiment_plots(summary)
    print(f"\n[QWO] suite finished in {summary['elapsed_seconds']:.1f}s")
    print(f"[QWO] outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
