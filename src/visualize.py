"""Figures + GIFs + cinematic media for QWO."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from .grids import Grid
from .objectives import Objective

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR = {
    "deep_blue": "#1F4E79",
    "blue": "#2E75B6",
    "orange": "#C65911",
    "green": "#548235",
    "purple": "#8064A2",
    "grey": "#595959",
    "panel": "#0E1B2C",
    "panel_light": "#1F2D44",
    "ink": "#F5F7FA",
    "amber": "#F2BD46",
    "red": "#C00000",
}


# -----------------------------------------------------------------------
# Loss landscape with samples
# -----------------------------------------------------------------------

def loss_landscape_2d(obj: Objective, grid: Grid, out_path: Path | str,
                      samples: np.ndarray | None = None,
                      best: np.ndarray | None = None,
                      title: str | None = None) -> None:
    if obj.d != 2:
        return
    pts = grid.all_points()
    L = obj.vectorised(pts).reshape(grid.shape)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    extent = (grid.bounds[1][0], grid.bounds[1][1],
              grid.bounds[0][0], grid.bounds[0][1])
    im = ax.imshow(L, extent=extent, origin="lower", cmap="viridis",
                    aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="L(theta)")
    if samples is not None and samples.size:
        ax.scatter(samples[:, 1], samples[:, 0], s=12, c=COLOR["amber"],
                    edgecolor="black", linewidth=0.4, label="QWO samples")
    if best is not None:
        ax.scatter([best[1]], [best[0]], s=80, c=COLOR["red"],
                    marker="*", edgecolor="black", label="best")
    ax.set_xlabel("theta_2"); ax.set_ylabel("theta_1")
    ax.set_title(title or f"Loss landscape: {obj.name}", color=COLOR["deep_blue"])
    if samples is not None or best is not None:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# QWO vs baselines convergence curves
# -----------------------------------------------------------------------

def convergence_curves(per_seed: List[Dict[str, Any]],
                       out_path: Path | str, title: str = "QWO vs baselines",
                       threshold: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    colors = {
        "qwo": COLOR["deep_blue"],
        "random_search": COLOR["orange"],
        "grid_search": COLOR["green"],
        "simulated_annealing": COLOR["purple"],
        "spsa": COLOR["red"],
        "scipy_minimize": COLOR["amber"],
    }
    # Plot one trace per method, averaged across seeds by interpolation onto
    # a common eval grid.
    all_methods: Dict[str, List[Tuple[List[int], List[float]]]] = {"qwo": []}
    for s in per_seed:
        all_methods["qwo"].append((s["qwo_evals_per_step"], s["qwo_history"]))
        for b_name, entry in s["baselines"].items():
            all_methods.setdefault(b_name, []).append((entry["evals_per_step"], entry["history"]))
    if not any(all_methods.values()):
        return
    eval_min = max(min(min(e) for e, _ in traces) for traces in all_methods.values() if traces)
    eval_max = min(max(max(e) for e, _ in traces) for traces in all_methods.values() if traces)
    grid_evals = np.linspace(eval_min, eval_max, 80)
    for m, traces in all_methods.items():
        if not traces:
            continue
        ys: List[np.ndarray] = []
        for evs, hist in traces:
            ys.append(np.interp(grid_evals, np.asarray(evs, dtype=float),
                                 np.asarray(hist, dtype=float)))
        ys_arr = np.stack(ys, axis=0)
        mean = ys_arr.mean(axis=0)
        std = ys_arr.std(axis=0)
        c = colors.get(m, COLOR["grey"])
        ax.plot(grid_evals, mean, color=c, lw=2.0, label=m)
        ax.fill_between(grid_evals, mean - std, mean + std, color=c, alpha=0.15)
    if threshold is not None:
        ax.axhline(threshold, color="black", lw=0.8, ls="--", label="threshold")
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("cumulative best loss")
    ax.set_title(title, color=COLOR["deep_blue"])
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# Evaluations-to-threshold summary
# -----------------------------------------------------------------------

def evaluations_to_threshold_plot(rows: List[Dict[str, Any]],
                                    out_path: Path | str) -> None:
    saved = [r for r in rows if r.get("method") == "EVALS_SAVED" and r.get("mean") is not None]
    if not saved:
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No EVALS_SAVED rows (no threshold-positive runs).",
                  ha="center", va="center"); plt.axis("off")
        plt.savefig(out_path); plt.close(fig); return
    saved = sorted(saved, key=lambda r: r["mean"])
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.32 * len(saved))))
    names = [r["experiment"] for r in saved]
    means = [r["mean"] for r in saved]
    colors = [COLOR["green"] if m > 0 else COLOR["red"] for m in means]
    y = np.arange(len(saved))
    ax.barh(y, means, color=colors, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("evals(best_classical) - evals(qwo)  (positive = QWO faster)")
    ax.set_title("Evaluations-to-threshold saved by QWO", color=COLOR["deep_blue"])
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# Ablation summary
# -----------------------------------------------------------------------

def ablation_summary(rows: List[Dict[str, Any]], out_path: Path | str) -> None:
    keep_exps = ("15_qwo_no_mixer", "16_qwo_no_phase", "17_qwo_with_zz",
                 "18_qwo_separable", "03_rastrigin_2d", "05_rosenbrock")
    rs = [r for r in rows
          if r["experiment"] in keep_exps and r["method"] == "qwo"]
    if not rs:
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No ablation rows.", ha="center", va="center")
        plt.axis("off"); plt.savefig(out_path); plt.close(fig); return
    rs = sorted(rs, key=lambda r: r["experiment"])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    names = [r["experiment"] for r in rs]
    means = [r["mean"] for r in rs]
    ax.bar(np.arange(len(rs)), means, color=COLOR["deep_blue"], alpha=0.85)
    ax.set_xticks(np.arange(len(rs)))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("QWO best loss (mean over seeds)")
    ax.set_title("Ablation: QWO best loss across configurations",
                  color=COLOR["deep_blue"])
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# Pipeline overview diagram
# -----------------------------------------------------------------------

def pipeline_overview(out_path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(16, 4.6), dpi=170)
    ax.set_xlim(0, 16); ax.set_ylim(0, 4.6); ax.set_axis_off()
    stages = [
        ("Continuous loss", "$L(\\theta),\\ \\theta\\in\\mathbb{R}^d$", COLOR["deep_blue"]),
        ("Grid", "$\\Theta_h$ discretised", COLOR["blue"]),
        ("Wavefunction", "$|\\Psi_0\\rangle=\\frac{1}{\\sqrt{N}}\\sum_j|\\theta_j\\rangle$", COLOR["purple"]),
        ("Phase oracle", "$e^{-i\\eta L(\\theta_j)}$", COLOR["orange"]),
        ("Mixer", "FFT / FD Laplacian / no-op", COLOR["orange"]),
        ("Measurement", "sample from $|\\Psi|^2$", COLOR["green"]),
        ("Local refine", "Nelder-Mead", COLOR["deep_blue"]),
    ]
    n = len(stages)
    x0 = 0.35; gap = 0.30
    box_w = (16 - 2 * x0 - gap * (n - 1)) / n
    y_c = 2.4; box_h = 2.05
    centers = []
    for i, (title, body, c) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        ax.add_patch(FancyBboxPatch((x, y_c - box_h / 2), box_w, box_h,
                                     boxstyle="round,pad=0.04,rounding_size=0.10",
                                     facecolor=c, edgecolor="none", alpha=0.94))
        ax.text(x + box_w / 2, y_c + 0.55, title, ha="center", va="center",
                 fontsize=11.5, weight="bold", color="white", linespacing=1.0)
        ax.text(x + box_w / 2, y_c - 0.40, body, ha="center", va="center",
                 fontsize=8.5, color="white", linespacing=1.25)
        centers.append(x + box_w / 2)
    for cx0, cx1 in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=(cx1 - box_w / 2 - 0.04, y_c),
                     xytext=(cx0 + box_w / 2 + 0.04, y_c),
                     arrowprops=dict(arrowstyle="-|>,head_length=0.28,head_width=0.16",
                                      color=COLOR["grey"], lw=1.8))
    ax.text(8.0, 4.2, "QWO pipeline overview", ha="center",
            fontsize=17, weight="bold", color=COLOR["deep_blue"])
    ax.text(8.0, 0.30,
            "Phase oracle $\\cdot$ kinetic mixer $\\cdot$ measurement $\\cdot$ local refine.  Simulator-based, no true quantum-advantage claim.",
            ha="center", fontsize=10, color=COLOR["grey"], style="italic")
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)


# -----------------------------------------------------------------------
# README hero
# -----------------------------------------------------------------------

def readme_hero(rows: List[Dict[str, Any]], landscape_path: Path | str,
                 curve_path: Path | str, out_path: Path | str) -> None:
    fig = plt.figure(figsize=(13.5, 6.0), dpi=180, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.1],
                           hspace=0.4, wspace=0.30,
                           left=0.05, right=0.98, top=0.80, bottom=0.10)
    fig.text(0.05, 0.93,
              "QWO: Quantum Wavefunction Optimizer",
              fontsize=20, weight="bold", color=COLOR["deep_blue"])
    fig.text(0.05, 0.89,
              "Continuous machine-learning loss landscapes via phase oracles + kinetic mixers",
              fontsize=11.5, color=COLOR["grey"])
    fig.text(0.05, 0.86,
              "simulator-based prototype  *  optimization-signal metric, not true quantum advantage",
              fontsize=9.5, color=COLOR["grey"], style="italic")
    # Panel 1: loss landscape with samples
    ax1 = fig.add_subplot(gs[0, 0])
    if Path(landscape_path).exists():
        img = np.asarray(Image.open(landscape_path).convert("RGB"))
        ax1.imshow(img)
    ax1.set_axis_off()
    ax1.set_title("Loss landscape + QWO samples", color=COLOR["deep_blue"], pad=8)
    # Panel 2: convergence curve preview
    ax2 = fig.add_subplot(gs[0, 1])
    if Path(curve_path).exists():
        img = np.asarray(Image.open(curve_path).convert("RGB"))
        ax2.imshow(img)
    ax2.set_axis_off()
    ax2.set_title("Convergence vs baselines", color=COLOR["deep_blue"], pad=8)
    # Panel 3: signal preview
    ax3 = fig.add_subplot(gs[0, 2])
    sig = [r for r in rows if r.get("method") == "OPTIMIZATION_SIGNAL"]
    sig = [r for r in sig if r.get("mean") is not None and r["mean"] is not None]
    sig = sorted(sig, key=lambda r: -abs(r["mean"]))[:8]
    if sig:
        names = [r["experiment"].split("_", 1)[0] + "..." for r in sig]
        means = [r["mean"] for r in sig]
        colors = [COLOR["green"] if m > 0 else COLOR["red"] for m in means]
        y = np.arange(len(sig))
        ax3.barh(y, means, color=colors, alpha=0.85)
        ax3.set_yticks(y); ax3.set_yticklabels(names, fontsize=8.5)
        ax3.axvline(0, color="black", lw=0.7)
        ax3.set_xlabel("best_classical - qwo", fontsize=9)
        ax3.set_title("Optimization signal (subset)", color=COLOR["deep_blue"], pad=8)
    else:
        ax3.set_axis_off()
        ax3.text(0.5, 0.5, "(signal pending)", ha="center", va="center")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------------
# Wavefunction evolution GIF
# -----------------------------------------------------------------------

def wavefunction_evolution_gif(frames: List[np.ndarray], out_path: Path | str,
                                 fps: int = 4) -> None:
    """Save a GIF of |psi|^2 frames over QWO iterations. Works for 1-D or 2-D grids."""
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"imageio not available: {exc}"); return
    if not frames:
        return
    rendered: List[np.ndarray] = []
    if frames[0].ndim == 1:
        x = np.arange(frames[0].size)
        for k, f in enumerate(frames):
            fig, ax = plt.subplots(figsize=(6.0, 3.2), facecolor="white")
            ax.bar(x, f / max(f.max(), 1e-12),
                    color=COLOR["deep_blue"], alpha=0.85)
            ax.set_ylim(0, 1.05); ax.set_xlim(-1, x[-1] + 1)
            ax.set_xlabel("grid index"); ax.set_ylabel("normalised probability")
            ax.set_title(f"|psi|^2 at iteration {k}",
                          color=COLOR["deep_blue"])
            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            rendered.append(rgba[..., :3].copy())
            plt.close(fig)
    elif frames[0].ndim == 2:
        for k, f in enumerate(frames):
            fig, ax = plt.subplots(figsize=(4.6, 4.0), facecolor="white")
            ax.imshow(f / max(f.max(), 1e-12), cmap="viridis", origin="lower")
            ax.set_axis_off()
            ax.set_title(f"|psi|^2 at iteration {k}",
                          color=COLOR["deep_blue"])
            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            rendered.append(rgba[..., :3].copy())
            plt.close(fig)
    else:
        return
    imageio.mimsave(out_path, rendered, fps=fps, loop=0)


# -----------------------------------------------------------------------
# Cinematic summary
# -----------------------------------------------------------------------

def cinematic_summary(rows: List[Dict[str, Any]],
                        landscape_path: Path | str,
                        evolution_frames: List[np.ndarray],
                        out_dir: Path,
                        fps_mp4: int = 24, fps_gif: int = 6) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"imageio missing: {exc}"); return
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "cinematic_summary.mp4"
    gif = out_dir / "cinematic_summary.gif"
    writer = imageio.get_writer(mp4, fps=fps_mp4, codec="libx264", quality=9,
                                  macro_block_size=1)
    captured: List[np.ndarray] = []
    counter = 0
    every = 4

    def slide():
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=COLOR["panel"])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(COLOR["panel"])
        ax.set_xlim(0, 1280); ax.set_ylim(0, 720)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return fig, ax

    def to_arr(fig):
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return rgba[..., :3].copy()

    def emit(frame):
        nonlocal counter
        writer.append_data(frame)
        if counter % every == 0:
            captured.append(frame)
        counter += 1

    try:
        # Title (3s)
        for _ in range(fps_mp4 * 3):
            fig, ax = slide()
            ax.add_patch(FancyBboxPatch((40, 260), 18, 360, boxstyle="round,pad=0",
                                          facecolor=COLOR["amber"], edgecolor="none"))
            ax.text(110, 600, "QWO", color=COLOR["ink"], fontsize=72, weight="bold")
            ax.text(110, 540, "Quantum Wavefunction Optimizer",
                     color=COLOR["ink"], fontsize=28)
            ax.text(110, 500, "for continuous ML loss landscapes",
                     color=COLOR["ink"], fontsize=24)
            ax.text(110, 430, "Phase oracle, kinetic mixer, measurement, local refine.",
                     color=COLOR["amber"], fontsize=20)
            ax.text(110, 340, "Simulator prototype  *  no true quantum advantage",
                     color=COLOR["grey"], fontsize=16)
            emit(to_arr(fig)); plt.close(fig)

        # Loss landscape display (4s)
        if Path(landscape_path).exists():
            img = np.asarray(Image.open(landscape_path).convert("RGB"))
            for _ in range(fps_mp4 * 4):
                fig, ax = slide()
                ax.add_patch(FancyBboxPatch((40, 620), 18, 80, boxstyle="round,pad=0",
                                              facecolor=COLOR["amber"], edgecolor="none"))
                ax.text(80, 660, "Step 1 - the loss landscape",
                         color=COLOR["ink"], fontsize=26, weight="bold")
                h_im, w_im = img.shape[:2]
                target_h = 480
                ratio = target_h / h_im
                target_w = int(w_im * ratio)
                ax.imshow(img, extent=(160, 160 + target_w, 80, 80 + target_h))
                ax.text(160 + target_w + 50, 480,
                         "Continuous loss\nL(theta) on a grid",
                         color=COLOR["ink"], fontsize=18)
                emit(to_arr(fig)); plt.close(fig)

        # Wavefunction evolution (6s)
        for k in range(fps_mp4 * 6):
            t = k / max(fps_mp4 * 6 - 1, 1)
            idx = int(t * (len(evolution_frames) - 1)) if evolution_frames else 0
            fig, ax = slide()
            ax.add_patch(FancyBboxPatch((40, 620), 18, 80, boxstyle="round,pad=0",
                                          facecolor=COLOR["amber"], edgecolor="none"))
            ax.text(80, 660, "Step 2 - wavefunction concentrates with each step",
                     color=COLOR["ink"], fontsize=24, weight="bold")
            if evolution_frames:
                arr = evolution_frames[idx]
                if arr.ndim == 1:
                    bar_h = 350
                    x = np.linspace(180, 1140, arr.size)
                    bar_w = (1140 - 180) / arr.size * 0.7
                    h = arr / max(arr.max(), 1e-12) * bar_h
                    for j, hh in enumerate(h):
                        ax.add_patch(FancyBboxPatch((x[j] - bar_w / 2, 120), bar_w, hh,
                                                      boxstyle="round,pad=0",
                                                      facecolor=COLOR["amber"], edgecolor="none"))
                else:
                    p = arr / max(arr.max(), 1e-12)
                    ax.imshow(p, cmap="viridis", origin="lower",
                                extent=(280, 1080, 80, 580))
            ax.text(80, 100, f"iteration {idx} / {max(len(evolution_frames) - 1, 0)}",
                     color=COLOR["grey"], fontsize=14, family="monospace")
            emit(to_arr(fig)); plt.close(fig)

        # Optimization signal chart (5s)
        sig = [r for r in rows if r.get("method") == "OPTIMIZATION_SIGNAL" and r.get("mean") is not None]
        sig = sorted(sig, key=lambda r: r["mean"])
        for _ in range(fps_mp4 * 5):
            fig, ax = slide()
            ax.add_patch(FancyBboxPatch((40, 620), 18, 80, boxstyle="round,pad=0",
                                          facecolor=COLOR["amber"], edgecolor="none"))
            ax.text(80, 660,
                     "Step 3 - optimization signal across experiments",
                     color=COLOR["ink"], fontsize=24, weight="bold")
            n = min(12, len(sig))
            x0 = 280; y0 = 100; bar_w = 600; row_h = 36
            ax.plot([x0 + bar_w / 2, x0 + bar_w / 2], [y0 - row_h * 0.5, y0 + n * row_h],
                     color=COLOR["grey"], lw=0.7)
            for i in range(n):
                r = sig[i]
                m = r["mean"]
                color = COLOR["green"] if m > 0 else COLOR["red"]
                bx = x0 + bar_w / 2
                scale = max(0.05, max(abs(rr["mean"]) for rr in sig[:n]))
                length = (m / scale) * (bar_w / 2)
                length = max(min(length, bar_w / 2 - 6), -(bar_w / 2 - 6))
                yy = y0 + (n - 1 - i) * row_h
                ax.add_patch(FancyBboxPatch((min(bx, bx + length), yy),
                                              abs(length), row_h * 0.7,
                                              boxstyle="round,pad=0",
                                              facecolor=color, edgecolor="none", alpha=0.9))
                ax.text(x0 - 10, yy + row_h * 0.4, r["experiment"], ha="right",
                          color=COLOR["ink"], fontsize=10)
                ax.text(x0 + bar_w + 30, yy + row_h * 0.4, f"{m:+.3f}",
                          ha="left", color=color, fontsize=11, weight="bold")
            emit(to_arr(fig)); plt.close(fig)

        # Closing (3s)
        for _ in range(fps_mp4 * 3):
            fig, ax = slide()
            ax.add_patch(FancyBboxPatch((40, 260), 18, 360, boxstyle="round,pad=0",
                                          facecolor=COLOR["amber"], edgecolor="none"))
            ax.text(110, 560, "Simulator prototype.",
                     color=COLOR["ink"], fontsize=34, weight="bold")
            ax.text(110, 510, "No true quantum-advantage claim.",
                     color=COLOR["amber"], fontsize=22)
            ax.text(110, 420, "Reproduce:", color=COLOR["grey"], fontsize=16)
            ax.text(110, 385, "    python scripts/run_all_experiments.py --quick",
                     color=COLOR["ink"], fontsize=16, family="monospace")
            ax.text(110, 355, "    python scripts/build_release_media.py",
                     color=COLOR["ink"], fontsize=16, family="monospace")
            ax.text(110, 325, "    pytest -q",
                     color=COLOR["ink"], fontsize=16, family="monospace")
            emit(to_arr(fig)); plt.close(fig)
    finally:
        writer.close()
    # GIF (downsampled)
    try:
        target_w = 640
        gif_frames: List[np.ndarray] = []
        for f in captured:
            img = Image.fromarray(f)
            ratio = target_w / img.width
            img = img.resize((target_w, int(img.height * ratio)), Image.BILINEAR)
            gif_frames.append(np.array(img))
        imageio.mimsave(gif, gif_frames, fps=fps_gif, loop=0)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"GIF write failed: {exc}")
