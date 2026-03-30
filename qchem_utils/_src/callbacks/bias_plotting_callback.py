from __future__ import annotations

import os
from typing import Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt

from advanced_drivers._src.callbacks.base import AbstractCallback
from netket.utils import struct


class ESSSNRBiasPlotCallback(AbstractCallback, mutable=True):
    r"""
    Callback that accumulates and plots ESS / SNR / Bias during training.

    Data sources:
      - ESS:  log_data["ESS"] if present, otherwise tries log_data["info"]["ESS"]
      - Bias: log_data["Bias"] if present, otherwise tries log_data["info"]["Bias"]
      - SNR:  tries (in this order):
          * log_data["SNR"]
          * log_data["snr"]
          * log_data["info"]["snr"]
          * computed from info as |grad| / sqrt(Var_g) if both exist:
                mean(|grad|) / sqrt(mean(Var_g))

    Saves a PDF at `output_dir/filename.pdf` (updated every `plot_every` steps).
    """

    output_dir: str = struct.field(pytree_node=False, default="outputs/plots")
    filename: str = struct.field(pytree_node=False, default="ess_snr_bias.pdf")
    plot_every: int = struct.field(pytree_node=False, default=10)

    _steps : list[int] = struct.field(pytree_node=False, default_factory=list)
    _ess : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _bias : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _snr : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)

    _pdf_path: str = struct.field(pytree_node=False, default="")

    def __init__(
        self,
        *,
        output_dir: str = "outputs/plots",
        filename: str = "ess_snr_bias.pdf",
        plot_every: int = 10,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.filename = filename
        self.plot_every = plot_every

        self._steps = []
        self._ess = []
        self._bias = []
        self._snr = []

        os.makedirs(self.output_dir, exist_ok=True)
        self._pdf_path = os.path.join(self.output_dir, self.filename)

    @staticmethod
    def _get_nested(d: dict, *keys):
        cur = d
        for k in keys:
            if cur is None or k not in cur:
                return None
            cur = cur[k]
        return cur

    @staticmethod
    def _to_float_or_none(x) -> Optional[float]:
        if x is None:
            return None
        try:
            return float(jnp.asarray(x))
        except Exception:
            return None

    def on_compute_update_end(self, step, log_data, driver):
        info = log_data.get("info", {}) if isinstance(log_data, dict) else {}

        ess = (
            log_data.get("ESS", None)
            if isinstance(log_data, dict)
            else None
        )
        if ess is None:
            ess = self._get_nested(log_data, "info", "ESS")
        ess_f = self._to_float_or_none(ess)

        bias = (
            log_data.get("Bias", None)
            if isinstance(log_data, dict)
            else None
        )
        if bias is None:
            bias = self._get_nested(log_data, "info", "Bias")
        bias_f = self._to_float_or_none(bias)

        snr_f = self._get_nested(log_data, 'snr_info', 'SNR', 'Mean')
        snr_f = self._get_nested(log_data, 'Energy', 'SNR', 'Mean')

        # Only append if at least one metric is present this step
        if ess_f is None and bias_f is None and snr_f is None:
            return

        self._steps.append(int(step))
        self._ess.append(ess_f if ess_f is not None else float("nan"))
        self._bias.append(bias_f if bias_f is not None else float("nan"))
        self._snr.append(snr_f if snr_f is not None else float("nan"))

        if step % self.plot_every != 0:
            return

        # --- plot ---
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)

        axes[0].plot(self._steps, self._ess, color="tab:blue", lw=2)
        axes[0].set_ylabel("ESS (fraction)")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0.0, 1.05)

        axes[1].plot(self._steps, self._snr, color="tab:green", lw=2)
        axes[1].set_ylabel("SNR")
        axes[1].grid(True, alpha=0.3)
        # SNR can span orders of magnitude; keep linear by default
        # axes[1].set_yscale("log")

        axes[2].plot(self._steps, self._bias, color="tab:red", lw=2)
        axes[2].set_ylabel("Bias")
        axes[2].set_xlabel("Step")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale("log")

        fig.suptitle("ESS / SNR / Bias", y=0.995)
        fig.tight_layout()

        fig.savefig(self._pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Updated plot saved to {self._pdf_path}")