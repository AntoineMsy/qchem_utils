"""
compare_runs.py
===============
Loads the serialised NPZ data produced by the plot callbacks from one or more
run directories and generates comparison figures.

Usage
-----
    python compare_runs.py --config compare_config.yaml

Or pass run_paths directly on the command line (they override the yaml list):

    python compare_runs.py path/to/run1 path/to/run2 [--config compare_config.yaml]

The script is intentionally plain Python (no Hydra) to keep logging clean.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_hydra_config(run_path: str) -> dict:
    """Read .hydra/config.yaml inside *run_path*. Returns {} on failure."""
    cfg_path = os.path.join(run_path, ".hydra", "config.yaml")
    if not os.path.isfile(cfg_path):
        warnings.warn(f"No .hydra/config.yaml found in {run_path}")
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def make_label(run_path: str, cfg: dict, label_fields: list[str]) -> str:
    """Build a compact legend label from selected config fields."""
    parts = []
    for field in label_fields:
        # Support dotted keys like "system.U"
        keys = field.split(".")
        val = cfg
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                val = None
                break
        if val is not None:
            parts.append(f"{keys[-1]}={val}")
    if not parts:
        # Fall back to last two path components
        parts = [os.path.join(*run_path.rstrip("/").split("/")[-2:])]
    return ", ".join(parts)


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    padded = np.concatenate([np.full(w - 1, x[0]), x])
    return np.convolve(padded, kernel, mode="valid")


# ─────────────────────────────────────────────────────────────────────────────
# Per-run data loader
# ─────────────────────────────────────────────────────────────────────────────

_NPZ_FILES = [
    "training_energy.npz",
    "energy_per_site.npz",
    "energy_per_site_error.npz",
    "alpha_snr.npz",
]


def load_run(run_path: str, plots_subdir: str = "plots") -> dict:
    """
    Returns a dict with keys:
        cfg          – hydra config dict
        label_raw    – run_path (for fallback)
        data         – { npz_stem: np.NpzFile } for each file found
    """
    plots_dir = os.path.join(run_path, plots_subdir)
    data = {}
    for fname in _NPZ_FILES:
        fpath = os.path.join(plots_dir, fname)
        if os.path.isfile(fpath):
            data[fname.replace(".npz", "")] = np.load(fpath)
        else:
            warnings.warn(f"  [missing] {fpath}")
    cfg = load_hydra_config(run_path)
    return {"cfg": cfg, "label_raw": run_path, "data": data}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting routines
# ─────────────────────────────────────────────────────────────────────────────

def _color_cycle(n: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def plot_training_energy(runs: list[dict], cfg_plot: dict, out_dir: str) -> None:
    """Energy (total or per-site, vs iteration and vs time) comparison."""
    window = cfg_plot.get("moving_average_window", 20)
    use_per_site = cfg_plot.get("energy_per_site", True)
    x_axis = cfg_plot.get("x_axis", "steps")          # "steps" | "time"
    show_raw = cfg_plot.get("show_raw_scatter", False)
    exact_energy = cfg_plot.get("exact_energy", None)  # per-site; optional override

    colors = _color_cycle(len(runs))

    fig_e, ax_e = plt.subplots(figsize=(9, 5))
    fig_r, ax_r = plt.subplots(figsize=(9, 3))
    fig_ess, ax_ess = plt.subplots(figsize=(9, 3))
    has_rhat = False
    has_ess = False

    for run, color in zip(runs, colors):
        d = run["data"].get("training_energy")
        if d is None:
            warnings.warn(f"training_energy.npz not found for {run['label_raw']}")
            continue
        label = run["label"]

        steps = d["steps"]
        times = d["times"] if "times" in d else None
        means = d["means"]
        rhats = d["rhats"] if "rhats" in d else None
        ess   = d["ess"]   if "ess"   in d else None

        # Select y values
        if use_per_site and "energy_per_site" in d:
            y = d["energy_per_site"]
        else:
            y = means

        # Select x values
        if x_axis == "time" and times is not None:
            x = times
            xlabel = "Optimisation time (s)"
        else:
            x = steps
            xlabel = "Iteration"

        ma = _moving_average(y, window)
        if show_raw:
            ax_e.scatter(x, y, s=3, color=color, alpha=0.2)
        ax_e.plot(x, ma, color=color, lw=1.8, label=label)

        if rhats is not None:
            ax_r.plot(x, rhats, color=color, lw=1.2, label=label)
            has_rhat = True

        if ess is not None and not np.all(np.isnan(ess)):
            ess_ma = _moving_average(ess, window)
            ax_ess.plot(x, ess_ma, color=color, lw=1.5, label=label)
            has_ess = True

    # Exact energy reference
    if exact_energy is not None:
        ax_e.axhline(exact_energy, color="black", ls="-.", lw=1.2,
                     label=rf"exact = {exact_energy:.4f}")

    ylabel = r"$E / N_{\rm sites}$" if use_per_site else r"$E$ (training)"
    ax_e.set_xlabel(xlabel)
    ax_e.set_ylabel(ylabel)
    ax_e.set_title("Training energy comparison")
    ax_e.legend(fontsize=8)
    ax_e.grid(True, alpha=0.3)
    if x_axis == "time":
        ax_e.set_xscale("log")
    fig_e.tight_layout()
    fig_e.savefig(os.path.join(out_dir, "compare_training_energy.png"),
                  dpi=150, bbox_inches="tight")
    plt.close(fig_e)

    if has_rhat:
        ax_r.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax_r.axhline(1.05, color="red", ls=":", lw=0.8, alpha=0.7)
        ax_r.set_xlabel(xlabel)
        ax_r.set_ylabel(r"$\hat{R}$")
        ax_r.set_title(r"$\hat{R}$ comparison")
        ax_r.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax_r.legend(fontsize=8)
        ax_r.grid(True, alpha=0.3)
        fig_r.tight_layout()
        fig_r.savefig(os.path.join(out_dir, "compare_rhat.png"),
                      dpi=150, bbox_inches="tight")
    plt.close(fig_r)

    if has_ess:
        ax_ess.axhline(0.1, color="gray", ls=":", lw=0.8, label="ESS = 0.1")
        ax_ess.set_xlabel(xlabel)
        ax_ess.set_ylabel("ESS")
        ax_ess.set_ylim(0, 1)
        ax_ess.set_title("ESS comparison")
        ax_ess.legend(fontsize=8)
        ax_ess.grid(True, alpha=0.3)
        fig_ess.tight_layout()
        fig_ess.savefig(os.path.join(out_dir, "compare_ess.png"),
                        dpi=150, bbox_inches="tight")
    plt.close(fig_ess)


def plot_energy_per_site(runs: list[dict], cfg_plot: dict, out_dir: str) -> None:
    """P²-based energy per site comparison (with uncertainty bands)."""
    exact_energy = cfg_plot.get("exact_energy", None)
    colors = _color_cycle(len(runs))

    fig, ax = plt.subplots(figsize=(9, 5))
    for run, color in zip(runs, colors):
        d = run["data"].get("energy_per_site")
        if d is None:
            continue
        label = run["label"]
        steps = d["steps"]
        means = d["means"]
        errs  = d["errs"] if "errs" in d else None
        ax.plot(steps, means, color=color, lw=1.8, label=label)
        if errs is not None:
            ax.fill_between(steps, means - errs, means + errs,
                            alpha=0.15, color=color)

    if exact_energy is not None:
        ax.axhline(exact_energy, color="black", ls="-.", lw=1.2,
                   label=rf"exact = {exact_energy:.4f}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$E / N_{\rm sites}$")
    ax.set_title(r"$P^2$ energy per site comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_energy_per_site.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_energy_error(runs: list[dict], cfg_plot: dict, out_dir: str) -> None:
    """Relative energy error comparison (log scale)."""
    colors = _color_cycle(len(runs))
    fig, ax = plt.subplots(figsize=(9, 4))
    plotted = False
    for run, color in zip(runs, colors):
        d = run["data"].get("energy_per_site_error")
        if d is None:
            continue
        label = run["label"]
        steps   = d["steps"]
        rel_err = d["rel_err"]
        errs    = d["errs"] if "errs" in d else None
        exact   = float(d["exact_energy"]) if "exact_energy" in d else None
        ax.plot(steps, rel_err, color=color, lw=1.5, label=label)
        if errs is not None and exact is not None and exact != 0:
            ax.fill_between(
                steps,
                np.maximum(rel_err - np.abs(errs / exact), 1e-12),
                rel_err + np.abs(errs / exact),
                alpha=0.15, color=color,
            )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative error")
    ax.set_title("Relative energy error comparison (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_energy_error.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_snr(runs: list[dict], cfg_plot: dict, out_dir: str) -> None:
    """Alpha and SNR comparison for overdispersed runs."""
    colors = _color_cycle(len(runs))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    plotted = False
    for run, color in zip(runs, colors):
        d = run["data"].get("alpha_snr")
        if d is None:
            continue
        label = run["label"]
        steps  = d["steps"]
        alphas = d["alphas"]
        snrs   = d["snrs"] if "snrs" in d else None
        axes[0].plot(steps, alphas, color=color, lw=1.5, label=label)
        if snrs is not None:
            valid = ~np.isnan(snrs)
            if valid.any():
                axes[1].plot(steps[valid], snrs[valid], color=color, lw=1.5, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axes[0].set_ylabel(r"$\alpha$ (overdispersion)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel(r"SNR = $|E| / \sigma_E$")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("Iteration")
    fig.suptitle("Alpha & SNR comparison", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_alpha_snr.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare optimisation runs from their saved NPZ data."
    )
    parser.add_argument(
        "run_paths",
        nargs="*",
        help="One or more run directories (override the yaml list when supplied).",
    )
    parser.add_argument(
        "--config", "-c",
        default=os.path.join(os.path.dirname(__file__), "compare_config.yaml"),
        help="Path to compare_config.yaml (default: same directory as this script).",
    )
    args = parser.parse_args()

    # ── Load plot config ──────────────────────────────────────────────────
    cfg_plot: dict = {}
    if os.path.isfile(args.config):
        with open(args.config) as f:
            cfg_plot = yaml.safe_load(f) or {}
        print(f"Loaded config: {args.config}")
    else:
        print(f"Config file not found at {args.config}, using defaults.")

    # ── Determine run paths ───────────────────────────────────────────────
    run_paths: list[str] = args.run_paths or cfg_plot.get("run_paths", [])
    if not run_paths:
        print("No run paths provided. Supply them as positional args or in the yaml.")
        sys.exit(1)

    plots_subdir = cfg_plot.get("plots_subdir", "plots")
    label_fields = cfg_plot.get("label_fields", ["sampler_net.name", "training.n_s", "system.U"])
    out_dir = cfg_plot.get("out_dir", "compare_output")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load all runs ─────────────────────────────────────────────────────
    runs: list[dict] = []
    for rp in run_paths:
        print(f"Loading {rp} …")
        run = load_run(rp, plots_subdir=plots_subdir)
        run["label"] = make_label(rp, run["cfg"], label_fields)
        runs.append(run)
        # Print a brief summary of what was found
        print(f"  label    : {run['label']}")
        print(f"  data keys: {list(run['data'].keys())}")
        # Print relevant config fields
        for field in label_fields:
            keys = field.split(".")
            val = run["cfg"]
            for k in keys:
                val = val.get(k) if isinstance(val, dict) else None
            print(f"  {field}: {val}")

    if not runs:
        print("No runs loaded. Exiting.")
        sys.exit(1)

    # ── Generate comparison figures ────────────────────────────────────────
    print(f"\nSaving figures to: {out_dir}")
    plot_training_energy(runs, cfg_plot, out_dir)
    plot_energy_per_site(runs, cfg_plot, out_dir)
    plot_energy_error(runs, cfg_plot, out_dir)
    plot_alpha_snr(runs, cfg_plot, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
