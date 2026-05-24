# QWO: Quantum Wavefunction Optimizer for Continuous Machine Learning Landscapes

A simulator-based research prototype exploring a sampling-based optimizer over a
discretised parameter grid, compared against classical optimizers on synthetic
loss landscapes.

<p align="center">
  <img src="outputs/public_readme/hero.png" width="700">
</p>

> Exploratory simulator-based research prototype. No quantum-advantage claim.

## What this explores

Whether a global, distribution-based search over a discretised parameter grid
behaves usefully on rugged synthetic landscapes, compared with standard
classical optimizers (e.g. SPSA, simulated annealing, random search) under a
matched evaluation budget. Everything runs on a classical simulator at small
scale.

## Selected visuals

<p align="center">
  <img src="outputs/public_readme/selected_result.png" width="560">
</p>

*Exploratory outcome tally across synthetic landscapes (simulator; lower loss is
better). Classical baselines win or tie on most tasks — an honest, mixed result.*

<p align="center">
  <img src="outputs/public_readme/gallery.png" width="760">
</p>

*Selected synthetic optimization landscapes.*

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_all_experiments.py --quick
pytest -q
```

## Honest status

- This is an exploratory research prototype.
- Results are simulator-based.
- No quantum advantage is claimed.
- No state-of-the-art claim is made.
- Some classical baselines match or outperform the prototype (often, here).
