"""
Two plotting callbacks:

  PlotEnergyFromPsiCallback
      Plots `energy_per_site` logged by EnergyPerSiteCallback.
      Shows: energy ± error_of_mean (shaded band) and R̂ in a bottom panel.
      Written only at sparse intervals to avoid I/O overhead.

  PlotTrainingEnergyCallback
      Plots the driver's own training energy (`log_data['Energy']`), which is
      the IS-reweighted estimate coming from Q-samples.  Uses a rolling
      average for smoothing and shows R̂ in a bottom panel.  No fill-between
      (the raw values are shown as a faint scatter).

Both save a single fixed-name PNG file to `out_dir` (overwritten each time).
"""

import os
import time
import copy

import numpy as np
import jax
import jax.numpy as jnp
import netket.jax as nkjax
import netket as nk
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for cluster use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from advanced_drivers._src.callbacks.base import AbstractCallback
from netket.utils import struct


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Causal moving average with window `w`."""
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    # pad left with first value so output length matches input
    padded = np.concatenate([np.full(w - 1, x[0]), x])
    return np.convolve(padded, kernel, mode="valid")


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Energy-per-site from P² with uncertainty band
# ──────────────────────────────────────────────────────────────────────────────

class PlotEnergyFromPsiCallback(AbstractCallback, mutable=True):
    r"""
    Computes the energy per site directly from `driver.state_p` by calling
    `state_p.expect(driver._ham)` every `compute_every` steps, then plots:

    energy_per_site.png
      top    – energy per site with ±1 σ_mean shaded band; variance per site
               on a secondary y-axis; tail-mean and (if provided) exact energy
               annotated as dashed lines
      bottom – R̂ diagnostic (dashed at 1.0; warn region > 1.05 shaded red)

    energy_per_site_error.png  (only when exact_energy is provided)
      single panel – |(E - E_exact) / E_exact| vs iteration, log y-scale

    Both figures are overwritten each time.

    Parameters
    ----------
    out_dir           : directory for saved figures
    plot_every        : how often (in steps) to refresh the figures
    compute_every     : how often (in steps) to call expect() on state_p
    n_s               : number of samples used for the expect() call
                        (restores original n_samples afterwards)
    exact_energy      : exact ground-state energy *per site* for reference
                        (pass total_energy / n_sites from e.g. Lanczos ED)
    """

    _out_dir: str = struct.field(pytree_node=False)
    _plot_every: int = struct.field(pytree_node=False, default=50)
    _compute_every: int = struct.field(pytree_node=False, default=50)
    _n_s: int = struct.field(pytree_node=False, default=2**14)
    _exact_energy: float = struct.field(pytree_node=False, default=None)
    _steps: list = struct.field(pytree_node=False, default_factory=list)
    _means: list = struct.field(pytree_node=False, default_factory=list)
    _errs: list = struct.field(pytree_node=False, default_factory=list)
    _vars: list = struct.field(pytree_node=False, default_factory=list)
    _rhats: list = struct.field(pytree_node=False, default_factory=list)

    def __init__(
        self,
        out_dir: str,
        plot_every: int = 50,
        compute_every: int = 50,
        n_s: int = 2**14,
        exact_energy: float = None,
    ):
        super().__init__()
        os.makedirs(out_dir, exist_ok=True)
        self._out_dir = out_dir
        self._plot_every = plot_every
        self._compute_every = compute_every
        self._n_s = n_s
        self._exact_energy = exact_energy
        self._steps: list[int] = []
        self._means: list[float] = []
        self._errs: list[float] = []
        self._vars: list[float] = []
        self._rhats: list[float] = []

    def on_step_end(self, step, log_data, driver):
        if step % self._compute_every != 0:
            return

        # Support both NIS driver (state_p / _ham) and standard VMC (state / hamiltonian)
        state = getattr(driver, "state_p", None) or driver.state
        ham   = getattr(driver, "_ham",    None) or driver._ham_jax
        n_sites = state.hilbert.n_orbitals

        # Temporarily increase sample count for a more accurate estimate
        n_s_orig = state.n_samples
        state.n_samples = self._n_s
        stats = state.expect(ham)
        state.n_samples = n_s_orig

        self._steps.append(step)
        self._means.append(stats.mean.real / n_sites)
        self._errs.append(stats.error_of_mean / n_sites)
        self._vars.append(stats.variance / n_sites**2)
        self._rhats.append(stats.R_hat)

        if step % self._plot_every != 0 or len(self._steps) < 2:
            return

        steps  = np.array(self._steps)
        means  = np.array(self._means)
        errs   = np.array(self._errs)
        vars_  = np.array(self._vars)
        rhats  = np.array(self._rhats)

        fig, (ax_e, ax_r) = plt.subplots(
            2, 1, figsize=(8, 5), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # ── top panel: relative error (log) if exact known, else energy/site
        if self._exact_energy is not None:
            _floor = 1e-12
            rel_err = np.maximum(np.abs((means - self._exact_energy) / self._exact_energy), _floor)
            ax_e.plot(steps, rel_err, color="steelblue",
                      label=r"$|E/N - E^{\rm exact}/N|\,/\,|E^{\rm exact}/N|$")
            ax_e.fill_between(
                steps,
                np.maximum(rel_err - np.abs(errs / self._exact_energy), _floor),
                np.maximum(rel_err + np.abs(errs / self._exact_energy), _floor),
                alpha=0.2, color="steelblue",
            )
            ax_e.set_yscale("log")
            ax_e.set_ylabel("Relative error")
            fig.suptitle(
                rf"P² energy error  ($E^{{\rm exact}}/N = {self._exact_energy:.4f}$)",
                fontsize=11,
            )
        else:
            ax_e.plot(steps, means, color="steelblue", label=r"$\langle E \rangle / N$")
            ax_e.fill_between(
                steps, means - errs, means + errs,
                alpha=0.25, color="steelblue", label=r"$\pm\sigma_{\rm mean}$",
            )
            # Variance on secondary axis
            ax_v = ax_e.twinx()
            ax_v.plot(steps, vars_, color="darkorange", lw=1, ls=":", alpha=0.7)
            ax_v.set_ylabel(r"Var$(E)/N^2$", color="darkorange", fontsize=8)
            ax_v.tick_params(axis="y", labelcolor="darkorange", labelsize=7)
            ax_v.set_yscale("log")
            ax_e.set_ylabel(r"$E / N$")
            fig.suptitle("Energy per site – P² estimate", fontsize=11)

        ax_e.legend(fontsize=8, loc="upper right")
        ax_e.grid(True, alpha=0.3)

        # ── bottom panel: R̂ ───────────────────────────────────────────────
        ax_r.plot(steps, rhats, color="purple", lw=1.2)
        ax_r.axhline(1.0, color="gray", ls="--", lw=0.8)
        rhat_max = float(np.nanmax(rhats)) if not np.all(np.isnan(rhats)) else 1.1
        rhat_top = max(rhat_max * 1.05, 1.1)
        ax_r.axhspan(1.05, rhat_top, color="red", alpha=0.07, label=r"$\hat{R} > 1.05$")
        ax_r.set_ylabel(r"$\hat{R}$")
        ax_r.set_xlabel("Iteration")
        ax_r.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax_r.set_ylim(bottom=0.99, top=rhat_top)
        ax_r.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(
            os.path.join(self._out_dir, "energy_per_site.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

        # ── serialise data ─────────────────────────────────────────────────
        save_kw = dict(steps=steps, means=means, errs=errs, vars=vars_, rhats=rhats)
        if self._exact_energy is not None:
            save_kw["exact_energy"] = np.array(self._exact_energy)
            save_kw["rel_err"] = rel_err
        np.savez(os.path.join(self._out_dir, "energy_per_site.npz"), **save_kw)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Training energy (IS-reweighted, from the driver's own 'Energy' key)
# ──────────────────────────────────────────────────────────────────────────────

class PlotTrainingEnergyCallback(AbstractCallback, mutable=True):
    r"""
    Reads the 'Energy' stats logged by the IS-VMC driver on every step and
    produces a two-panel (or three-panel) figure:
      top    – raw energy values (faint scatter) + moving average (solid line)
      middle – R̂ diagnostic
      bottom – ESS moving average (only when an ESS key is present in log_data)

    The figure is saved to a single fixed file (overwritten each time).

    Parameters
    ----------
    out_dir        : directory for saved figures
    plot_every     : how often to flush the figure to disk
    window         : moving-average window width (in steps)
    energy_log_key : key in log_data for the driver energy (default: 'Energy')
    ess_log_key    : key for ESS; set to None to skip
    """

    _out_dir: str = struct.field(pytree_node=False)
    _plot_every: int = struct.field(pytree_node=False, default=50)
    _window: int = struct.field(pytree_node=False, default=20)
    _energy_key: str = struct.field(pytree_node=False, default="Energy")
    _ess_key: str = struct.field(pytree_node=False, default="ESS")
    _n_sites: int = struct.field(pytree_node=False, default=None)
    _exact_energy: float = struct.field(pytree_node=False, default=None)
    _steps: list = struct.field(pytree_node=False, default_factory=list)
    _means: list = struct.field(pytree_node=False, default_factory=list)
    _rhats: list = struct.field(pytree_node=False, default_factory=list)
    _ess: list = struct.field(pytree_node=False, default_factory=list)
    _times: list = struct.field(pytree_node=False, default_factory=list)
    _start_time: float = struct.field(pytree_node=False, default=None)

    def __init__(
        self,
        out_dir: str,
        plot_every: int = 50,
        window: int = 20,
        energy_log_key: str = "Energy",
        ess_log_key: str = "ESS",
        n_sites: int = None,
        exact_energy: float = None,
    ):
        super().__init__()
        os.makedirs(out_dir, exist_ok=True)
        self._out_dir = out_dir
        self._plot_every = plot_every
        self._window = window
        self._energy_key = energy_log_key
        self._ess_key = ess_log_key
        self._n_sites = n_sites
        self._exact_energy = exact_energy
        self._steps: list[int] = []
        self._means: list[float] = []
        self._rhats: list[float] = []
        self._ess: list[float] = []
        self._times: list[float] = []
        self._start_time: float = None

    def on_step_end(self, step, log_data, driver):
        if self._energy_key not in log_data:
            return
        # print(log_data)
        if self._start_time is None:
            self._start_time = time.time()

        stats = log_data[self._energy_key]
        self._steps.append(step)
        self._means.append(stats.mean)
        self._rhats.append(stats.R_hat)
        self._times.append(time.time() - self._start_time)

        if self._ess_key and self._ess_key in log_data:
            self._ess.append(_safe_float(log_data[self._ess_key]))
        else:
            self._ess.append(float("nan"))

        if step % self._plot_every != 0 or len(self._steps) < 2:
            return

        steps  = np.array(self._steps)
        means  = np.array(self._means)
        rhats  = np.array(self._rhats)
        ess    = np.array(self._ess)
        times  = np.array(self._times)
        ma     = _moving_average(means, self._window)

        has_ess = not np.all(np.isnan(ess))
        n_panels = 3 if has_ess else 2
        height_ratios = [3, 1, 1] if has_ess else [3, 1]
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(8, 4 + n_panels),
            sharex=True, gridspec_kw={"height_ratios": height_ratios},
        )
        ax_e = axes[0]
        ax_r = axes[1]
        ax_ess = axes[2] if has_ess else None

        # ── top panel: relative error (log) if exact known, else raw energy ─
        if self._exact_energy is not None:
            # resolve per-site energy for the relative error
            _n_sites_top = self._n_sites
            if _n_sites_top is None:
                _n_sites_top = driver.state.hilbert.size // 2
            eps_top = means / _n_sites_top if (_n_sites_top is not None and _n_sites_top > 0) else means
            rel_err_top = np.abs((eps_top - self._exact_energy) / self._exact_energy)
            rel_err_top_ma = _moving_average(rel_err_top, self._window)
            ax_e.scatter(steps, rel_err_top, s=4, color="steelblue", alpha=0.3, label="raw")
            ax_e.plot(steps, rel_err_top_ma, color="steelblue", lw=1.8,
                      label=f"MA (w={self._window})")
            ax_e.set_yscale("log")
            # ax_e.set_ylim(bottom=_floor)
            ax_e.set_ylabel("Relative error")
            fig.suptitle(
                rf"Training energy error  ($E^{{\rm exact}}/N = {self._exact_energy:.4f}$)",
                fontsize=11,
            )
        else:
            ax_e.scatter(steps, means, s=4, color="steelblue", alpha=0.3, label="raw")
            ax_e.plot(steps, ma, color="steelblue", lw=1.8, label=f"MA (w={self._window})")
            tail = max(1, len(means) // 4)
            e_avg = float(np.nanmean(means[-tail:]))
            ax_e.axhline(e_avg, color="tomato", ls="--", lw=1,
                         label=rf"tail mean = {e_avg:.4f}")
            ax_e.set_ylabel(r"$E$ (training)")
            fig.suptitle("Training energy (IS-reweighted)", fontsize=11)

        ax_e.legend(fontsize=8, loc="upper right")
        ax_e.grid(True, which="both" if self._exact_energy is not None else "major", alpha=0.3)

        # ── middle panel: R̂ ───────────────────────────────────────────────
        ax_r.plot(steps, rhats, color="purple", lw=1.2)
        ax_r.axhline(1.0, color="gray", ls="--", lw=0.8)
        rhat_max = float(np.nanmax(rhats)) if not np.all(np.isnan(rhats)) else 1.1
        rhat_top = max(rhat_max * 1.05, 1.1)
        if rhat_max > 1.05:
            ax_r.axhspan(1.05, rhat_top, color="red", alpha=0.07)
        ax_r.set_ylabel(r"$\hat{R}$")
        ax_r.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax_r.set_ylim(bottom=0.99, top=rhat_top)
        ax_r.grid(True, alpha=0.3)

        # ── bottom panel: ESS ──────────────────────────────────────────────
        if has_ess and ax_ess is not None:
            ess_ma = _moving_average(ess, self._window)
            ax_ess.plot(steps, ess_ma, color="seagreen", lw=1.5, label=f"ESS MA (w={self._window})")
            ax_ess.axhline(0.1, color="gray", ls=":", lw=0.8, label="ESS = 0.1")
            ax_ess.set_ylabel("ESS")
            ax_ess.set_ylim(0, 1)
            ax_ess.legend(fontsize=8)
            ax_ess.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Iteration")
        plt.tight_layout()
        fig.savefig(
            os.path.join(self._out_dir, "training_energy.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

        # ── second figure: energy per site vs optimisation time ────────────
        # Resolve n_sites: use stored value, or try to infer from driver
        n_sites = self._n_sites
        if n_sites is None:
            n_sites = driver.state.hilbert.size // 2

        if n_sites is not None and n_sites > 0:
            energy_per_site = means / n_sites
        else:
            energy_per_site = means

        fig2, ax2 = plt.subplots(figsize=(8, 4))

        if self._exact_energy is not None:
            # ── relative error vs time (log scale) ────────────────────────
            _floor = 1e-12
            rel_err = np.maximum(np.abs((energy_per_site - self._exact_energy) / self._exact_energy), _floor)
            rel_err_ma = np.maximum(_moving_average(rel_err, self._window), _floor)
            ax2.scatter(times, rel_err, s=4, color="steelblue", alpha=0.3, label="raw")
            ax2.plot(times, rel_err_ma, color="steelblue", lw=1.8,
                     label=f"MA (w={self._window})")
            ax2.set_yscale("log")
            
            ax2.set_ylabel("Relative error")
            fig2.suptitle(
                rf"Energy error vs time  ($E^{{\rm exact}}/N = {self._exact_energy:.4f}$)",
                fontsize=11,
            )
        else:
            # ── energy per site vs time ────────────────────────────────────
            ma_per_site = _moving_average(energy_per_site, self._window)
            ax2.scatter(times, energy_per_site, s=4, color="steelblue", alpha=0.3, label="raw")
            ax2.plot(times, ma_per_site, color="steelblue", lw=1.8,
                     label=f"MA (w={self._window})")
            tail = max(1, len(energy_per_site) // 4)
            e_avg_ps = float(np.nanmean(energy_per_site[-tail:]))
            ax2.axhline(e_avg_ps, color="tomato", ls="--", lw=1,
                        label=rf"tail mean = {e_avg_ps:.4f}")
            ylabel = (
                rf"$E / N_{{\rm sites}}$  ($N={n_sites}$)"
                if n_sites is not None else r"$E$ (training)"
            )
            ax2.set_ylabel(ylabel)
            fig2.suptitle("Variational energy per site vs optimisation time", fontsize=11)

        ax2.set_xlabel("Optimisation time (s)")
        # ax2.set_xscale("log")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, which="both" if self._exact_energy is not None else "major", alpha=0.3)
        plt.tight_layout()
        fig2.savefig(
            os.path.join(self._out_dir, "training_energy_per_site_vs_time.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig2)

        # ── serialise data ─────────────────────────────────────────────────
        save_kw = dict(
            steps=steps, means=means, rhats=rhats, ess=ess, times=times,
        )
        if n_sites is not None:
            save_kw["n_sites"] = np.array(n_sites)
            save_kw["energy_per_site"] = energy_per_site
        if self._exact_energy is not None:
            save_kw["exact_energy"] = np.array(self._exact_energy)
            save_kw["rel_err"] = rel_err
        np.savez(os.path.join(self._out_dir, "training_energy.npz"), **save_kw)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Alpha exponent + SNR for overdispersed sampling distributions
# ──────────────────────────────────────────────────────────────────────────────

class PlotAlphaCallback(AbstractCallback, mutable=True):
    r"""
    Tracks the overdispersion exponent `alpha` from `driver.state_q` (assumes
    an `OverdispersedWrapper` whose Flax parameter is named ``alpha``) and the
    signal-to-noise ratio SNR = |E| / σ_mean from the training energy.

    Produces a single fixed figure `alpha_snr.png` with two panels:
      top    – alpha exponent vs iteration
      bottom – SNR vs iteration (log scale)

    Parameters
    ----------
    out_dir        : directory for saved figures
    plot_every     : how often (in steps) to refresh the figure
    energy_log_key : key in log_data for the energy stats (default: 'Energy')
    """

    _out_dir: str = struct.field(pytree_node=False)
    _plot_every: int = struct.field(pytree_node=False, default=50)
    _energy_key: str = struct.field(pytree_node=False, default="Energy")
    _steps: list = struct.field(pytree_node=False, default_factory=list)
    _alphas: list = struct.field(pytree_node=False, default_factory=list)
    _snrs: list = struct.field(pytree_node=False, default_factory=list)

    def __init__(
        self,
        out_dir: str,
        plot_every: int = 50,
        energy_log_key: str = "Energy",
    ):
        super().__init__()
        os.makedirs(out_dir, exist_ok=True)
        self._out_dir = out_dir
        self._plot_every = plot_every
        self._energy_key = energy_log_key
        self._steps: list[int] = []
        self._alphas: list[float] = []
        self._snrs: list[float] = []

    def on_step_end(self, step, log_data, driver):
        # ── extract alpha from state_q parameters ──────────────────────────
        try:
            alpha = float(
                driver.state_q.variables["params"]["alpha"]
            )
        except (KeyError, AttributeError):
            return  # not an overdispersed state, skip silently

        # ── SNR from energy stats ──────────────────────────────────────────
        snr = log_data['snr_info']['SNR']
        if self._energy_key in log_data:
            stats = log_data[self._energy_key]
            mean = stats.mean.real
            err = stats.error_of_mean
            if err > 0:
                snr = abs(mean) / err

        self._steps.append(step)
        self._alphas.append(alpha)
        self._snrs.append(snr)

        if step % self._plot_every != 0 or len(self._steps) < 2:
            return

        steps  = np.array(self._steps)
        alphas = np.array(self._alphas)
        snrs   = np.array(self._snrs)

        has_snr = not np.all(np.isnan(snrs))
        n_panels = 2 if has_snr else 1
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(8, 3 * n_panels),
            sharex=True,
        )
        if n_panels == 1:
            axes = [axes]

        fig.suptitle("Overdispersed sampler diagnostics", fontsize=11)

        # ── top panel: alpha ───────────────────────────────────────────────
        axes[0].plot(steps, alphas, color="darkorange", lw=1.5)
        axes[0].set_ylabel(r"$\alpha$ (overdispersion)")
        axes[0].grid(True, alpha=0.3)

        # ── bottom panel: SNR ──────────────────────────────────────────────
        if has_snr:
            axes[1].plot(steps, snrs, color="steelblue", lw=1.5)
            axes[1].set_ylabel(r"SNR = $|E| / \sigma_E$")
            axes[1].set_yscale("log")
            axes[1].grid(True, which="both", alpha=0.3)

        axes[-1].set_xlabel("Iteration")
        plt.tight_layout()
        fig.savefig(
            os.path.join(self._out_dir, "alpha_snr.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

        # ── serialise data ─────────────────────────────────────────────────
        np.savez(
            os.path.join(self._out_dir, "alpha_snr.npz"),
            steps=steps, alphas=alphas, snrs=snrs,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4.  SNR vs alpha sweep – overdispersion landscape
# ──────────────────────────────────────────────────────────────────────────────

class SNRAlphaCallback(AbstractCallback, mutable=True):
    r"""
    At every `compute_every` steps, builds a FullSumState from the current
    driver state and sweeps the overdispersion exponent alpha over
    ``[alpha_min, alpha_max]`` (``n_alpha`` points).  For each alpha it
    computes the SNR of the variational gradient estimator when sampling
    from q(x) \propto |psi(x)|^alpha.

    Tracks and plots:
      snr_alpha.png    – two panels
        top    – SNR(alpha) curve at the most recent step (current landscape)
        bottom – time series of max_SNR, SNR(alpha=2), SNR(grad-optimal q)
      snr_alpha.npz    – full serialised data

    Parameters
    ----------
    out_dir        : directory for saved figures / NPZ
    H_sp           : sparse Hamiltonian matrix (e.g. from
                     ``hamiltonian.to_sparse()``)
    compute_every  : how often (in steps) to run the SNR sweep
    plot_every     : how often (in steps) to flush the figure
    alpha_min      : lower bound of the alpha sweep (default 0.5)
    alpha_max      : upper bound of the alpha sweep (default 2.0)
    n_alpha        : number of alpha grid points   (default 200)
    chunk_size_jac : chunk size passed to nkjax.jacobian
    """

    _out_dir: str       = struct.field(pytree_node=False)
    _H_sp: object       = struct.field(pytree_node=False)  # sparse matrix
    _compute_every: int = struct.field(pytree_node=False, default=10)
    _plot_every: int    = struct.field(pytree_node=False, default=50)
    _alpha_min: float   = struct.field(pytree_node=False, default=0.5)
    _alpha_max: float   = struct.field(pytree_node=False, default=2.0)
    _n_alpha: int       = struct.field(pytree_node=False, default=200)
    _chunk_size_jac: int = struct.field(pytree_node=False, default=200)
    # lazy-initialised on first call
    _fs_state: object   = struct.field(pytree_node=False, default=None)
    # history
    _steps: list         = struct.field(pytree_node=False, default_factory=list)
    _snr_psi_sq: list    = struct.field(pytree_node=False, default_factory=list)
    _max_snr_a: list     = struct.field(pytree_node=False, default_factory=list)
    _argmax_snr_a: list  = struct.field(pytree_node=False, default_factory=list)
    _snr_grad: list      = struct.field(pytree_node=False, default_factory=list)
    # stores the *subsampled* SNR-vs-alpha curve at each recorded step
    _snr_curves: list    = struct.field(pytree_node=False, default_factory=list)
    _a_vals_sub: object  = struct.field(pytree_node=False, default=None)

    def __init__(
        self,
        out_dir: str,
        H_sp,
        compute_every: int = 10,
        plot_every: int = 50,
        alpha_min: float = 0.0,
        alpha_max: float = 2.0,
        n_alpha: int = 200,
        chunk_size_jac: int = 200,
    ):
        super().__init__()
        os.makedirs(out_dir, exist_ok=True)
        self._out_dir = out_dir
        self._H_sp = H_sp
        self._compute_every = compute_every
        self._plot_every = plot_every
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._n_alpha = n_alpha
        self._chunk_size_jac = chunk_size_jac
        self._fs_state = None
        self._steps: list[int] = []
        self._snr_psi_sq: list[float] = []
        self._max_snr_a: list[float] = []
        self._argmax_snr_a: list[float] = []
        self._snr_grad: list[float] = []
        self._snr_curves: list = []
        self._a_vals_sub = None

    # ───────────────────────────────────────────────────────────────────
    def _get_fs_state(self, driver):
        """Lazy-init a FullSumState that mirrors the driver's variational state."""
        # Resolve the live MC state (NIS drivers expose state_p, standard VMC uses state)
        mc_state = getattr(driver, "state_p", None) or driver.state
        if self._fs_state is None:
            self._fs_state = nk.vqs.FullSumState(
                hilbert=mc_state.hilbert,
                model=mc_state.model,
                chunk_size=None,
                seed=0,
            )
        self._fs_state.variables = copy.deepcopy(mc_state.variables)
        return self._fs_state

    # ───────────────────────────────────────────────────────────────────
    def _compute_snr_sweep(self, fs_state):
        """
        Returns a dict with keys:
            snr_a        – full SNR curve (n_alpha,)
            snr_psi_sq   – SNR under Born sampling (alpha=2)
            max_snr_a    – max SNR over alpha grid
            argmax_snr_a – alpha that achieves max SNR
            snr_grad     – SNR under gradient-magnitude-weighted q
        """
        pdf       = fs_state.probability_distribution()          # (ℕ_states,)
        psi_arr   = fs_state.to_array()                          # (ℕ_states,) complex

        # Centred local energy
        Hloc      = self._H_sp @ psi_arr / psi_arr               # (ℕ_states,) complex
        Hloc_c    = Hloc - jnp.sum(Hloc * pdf)

        # Jacobian O_{x,k} = d log psi / d theta_k  (un-centred)
        jacobian  = nkjax.jacobian(
            fs_state._apply_fun,
            fs_state.parameters,
            fs_state.hilbert.all_states(),
            fs_state.model_state,
            pdf=pdf,
            mode="complex",
            dense=True,
            center=False,
            chunk_size=self._chunk_size_jac,
            _sqrt_rescale=False,
        )                                                         # (ℕ_states, n_p) complex

        # Reshape (x, 2) → (2x,) for both Hloc and Jacobian so we handle real/imag
        Hloc_ri   = jax.lax.collapse(
            jnp.stack([jnp.real(Hloc_c), jnp.imag(Hloc_c)], axis=-1), 0, 2
        )                                                         # (2*ℕ_states,)
        jac_c     = jacobian - jnp.sum(
            jacobian * jnp.expand_dims(pdf, list(range(1, jacobian.ndim))),
            axis=0,
        )
        jac_ri    = jax.lax.collapse(jac_c, 0, 2)               # (2*ℕ_states, n_p)

        # Local gradient per sample:  O_x^{\dag} * E_{loc,x}
        # shape: (n_p, 2*ℕ_states) then sum paired real/imag columns
        loc_grad  = jac_ri.T * Hloc_ri                           # (n_p, 2*ℕ_states)
        loc_grad  = loc_grad[:, ::2] + loc_grad[:, 1::2]         # (n_p, ℕ_states)

        # Mean gradient under P^2 (the target estimand)
        mean_grad = jnp.sum(pdf * loc_grad, axis=1)              # (n_p,)

        def unnorm_pdf(alpha):
            return jnp.abs(psi_arr) ** alpha                     # (ℕ_states,)

        def compute_snr(q):
            """SNR of gradient estimator under sampling dist proportional to q."""
            q_pdf  = q / jnp.sum(q)                              # normalise
            p2_q   = unnorm_pdf(2.0) / q                         # importance weights w(x) = P^2 / q
            w_mean = jnp.sum(q_pdf * p2_q) ** 2
            # per-parameter variance
            var    = jnp.sum(
                q_pdf
                * p2_q ** 2
                * jnp.abs(loc_grad - mean_grad[:, None]) ** 2,
                axis=1,
            ) / w_mean                                           # (n_p,)
            # SNR averaged across parameters
            return float(jnp.mean(jnp.abs(mean_grad) / jnp.sqrt(var + 1e-30)))

        a_vals  = jnp.linspace(self._alpha_min, self._alpha_max, self._n_alpha)
        snr_a   = jnp.array([compute_snr(unnorm_pdf(a)) for a in a_vals])

        argmax_idx  = int(jnp.argmax(snr_a))
        argmax_a    = float(a_vals[argmax_idx])
        # gradient-magnitude-weighted optimal q
        q_grad      = jnp.mean(unnorm_pdf(2.0) * jnp.abs(loc_grad), axis=0)
        snr_grad    = compute_snr(q_grad)

        return dict(
            snr_a        = np.array(snr_a),
            snr_psi_sq   = compute_snr(unnorm_pdf(2.0)),
            max_snr_a    = float(jnp.max(snr_a)),
            argmax_snr_a = argmax_a,
            snr_grad     = snr_grad,
            a_vals       = np.array(a_vals),
        )

    # ───────────────────────────────────────────────────────────────────
    def on_step_end(self, step, log_data, driver):
        if step % self._compute_every != 0:
            return

        try:
            fs_state = self._get_fs_state(driver)
            result   = self._compute_snr_sweep(fs_state)
        except Exception as exc:
            import warnings
            warnings.warn(f"SNRAlphaCallback: SNR sweep failed at step {step}: {exc}")
            return

        # Store scalar summaries in log_data for other callbacks / logger
        log_data["snr_psi_sq"]      = result["snr_psi_sq"]
        log_data["max_snr_alpha"]   = result["max_snr_a"]
        log_data["argmax_snr_alpha"]= result["argmax_snr_a"]
        log_data["snr_grad"]        = result["snr_grad"]

        # Subsample the curve (every 10 points → 20 values for n_alpha=200)
        stride = max(1, self._n_alpha // 20)
        self._steps.append(step)
        self._snr_psi_sq.append(result["snr_psi_sq"])
        self._max_snr_a.append(result["max_snr_a"])
        self._argmax_snr_a.append(result["argmax_snr_a"])
        self._snr_grad.append(result["snr_grad"])
        self._snr_curves.append(result["snr_a"][::stride])
        if self._a_vals_sub is None:
            self._a_vals_sub = result["a_vals"][::stride]

        if step % self._plot_every != 0 or len(self._steps) < 2:
            return

        steps         = np.array(self._steps)
        snr_psi_sq    = np.array(self._snr_psi_sq)
        max_snr_a     = np.array(self._max_snr_a)
        argmax_snr_a  = np.array(self._argmax_snr_a)
        snr_grad_arr  = np.array(self._snr_grad)
        snr_curves    = np.array(self._snr_curves)   # (n_steps, n_alpha_sub)
        a_vals_sub    = self._a_vals_sub             # (n_alpha_sub,)

        # ── figure 1: current SNR(alpha) landscape + time-series ───────────
        fig, (ax_cur, ax_ts) = plt.subplots(
            2, 1, figsize=(8, 6),
            gridspec_kw={"height_ratios": [2, 3]},
        )
        fig.suptitle("SNR vs overdispersion exponent α", fontsize=11)

        # Top: latest SNR(alpha) curve
        ax_cur.plot(a_vals_sub, snr_curves[-1], color="steelblue", lw=1.8,
                    label=f"step {steps[-1]}")
        ax_cur.axvline(argmax_snr_a[-1], color="tomato", ls="--", lw=1.2,
                       label=rf"$\alpha^* = {argmax_snr_a[-1]:.3f}$")
        ax_cur.axvline(2.0, color="seagreen", ls=":", lw=1.2, label=r"$\alpha=2$ (Born)")
        ax_cur.set_xlabel(r"$\alpha$")
        ax_cur.set_ylabel("SNR")
        ax_cur.legend(fontsize=8)
        ax_cur.grid(True, alpha=0.3)

        # Bottom: scalar time series
        ax_ts.plot(steps, max_snr_a,    color="tomato",    lw=1.5, label=r"max SNR($\alpha$)")
        ax_ts.plot(steps, snr_psi_sq,   color="seagreen",  lw=1.5, label=r"SNR($\alpha=2$, Born)")
        ax_ts.plot(steps, snr_grad_arr, color="darkorange", lw=1.5, ls="--", label="SNR(grad-opt q)")
        ax_ts.set_xlabel("Iteration")
        ax_ts.set_ylabel("SNR")
        ax_ts.legend(fontsize=8)
        ax_ts.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(
            os.path.join(self._out_dir, "snr_alpha.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

        # ── figure 2: heatmap of SNR(alpha) over optimisation time ──────
        if len(steps) >= 4:
            fig2, ax2 = plt.subplots(figsize=(9, 4))
            im = ax2.imshow(
                snr_curves.T,
                aspect="auto",
                origin="lower",
                extent=[steps[0], steps[-1], float(a_vals_sub[0]), float(a_vals_sub[-1])],
                cmap="viridis",
            )
            ax2.plot(steps, argmax_snr_a, color="tomato", lw=1.5, label=r"$\alpha^*$")
            ax2.axhline(2.0, color="white", ls=":", lw=1.2, label=r"$\alpha=2$")
            plt.colorbar(im, ax=ax2, label="SNR")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel(r"$\alpha$")
            ax2.set_title("SNR landscape over training")
            ax2.legend(fontsize=8)
            plt.tight_layout()
            fig2.savefig(
                os.path.join(self._out_dir, "snr_alpha_heatmap.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig2)

        # ── serialise ──────────────────────────────────────────────────
        np.savez(
            os.path.join(self._out_dir, "snr_alpha.npz"),
            steps=steps,
            a_vals_sub=a_vals_sub,
            snr_curves=snr_curves,
            snr_psi_sq=snr_psi_sq,
            max_snr_a=max_snr_a,
            argmax_snr_a=argmax_snr_a,
            snr_grad=snr_grad_arr,
        )