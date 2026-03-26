from __future__ import annotations

import os
from typing import Optional

from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import matplotlib.pyplot as plt

from advanced_drivers._src.callbacks.base import AbstractCallback
from netket.utils import struct

from netket.vqs import FullSumState
import netket.jax as nkjax

class IterativeNormalizationCallback(AbstractCallback, mutable=True):
    r"""
    Callback that checks if the iterative normalization algorithm is actually correct 
    by comparing the actual ratio in full sum with the exact iterative approximation.
    """

    output_dir: str = struct.field(pytree_node=False, default="outputs/plots")
    filename: str = struct.field(pytree_node=False, default="iterative_normalization.pdf")
    plot_every: int = struct.field(pytree_node=False, default=10)

    _steps : list[int] = struct.field(pytree_node=False, default_factory=list)
    _exact_ratio_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _iterative_ratio_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _relative_update_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)

    _pdf_path: str = struct.field(pytree_node=False, default="")
    _iterative_ratio_next: Optional[float] = struct.field(pytree_node=False, default=None)

    fullsum_state_p: Optional[FullSumState] = struct.field(pytree_node=False, default=None)
    fullsum_state_q: Optional[FullSumState] = struct.field(pytree_node=False, default=None)

    def __init__(
        self,
        *,
        output_dir: str = "outputs/plots",
        filename: str = "iterative_normalization.pdf",
        plot_every: int = 10,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.filename = filename
        self.plot_every = plot_every
        self.fullsum_state_p = None
        self.fullsum_state_q = None

        self._steps = []
        self._exact_ratio_t = []
        self._iterative_ratio_t = []
        self._relative_update_t = []
        self._iterative_ratio_next = None

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
    def _to_float(x) -> float:
        return float(jnp.asarray(x))

    @staticmethod
    def _get_learning_rate(driver, step: int) -> float:
        lr = None

        optimizer_state = getattr(driver, "_optimizer_state", None)
        if optimizer_state is not None and hasattr(optimizer_state, "hyperparams"):
            hyperparams = optimizer_state.hyperparams
            if isinstance(hyperparams, dict) and "learning_rate" in hyperparams:
                lr = hyperparams["learning_rate"]

        if lr is None:
            optimizer = getattr(driver, "optimizer", None)
            lr = getattr(optimizer, "learning_rate", None)

        if lr is None:
            raise AttributeError("Could not find optimizer learning rate.")

        if callable(lr):
            lr = lr(step)

        return float(jnp.asarray(lr))

    @staticmethod
    def _real_jacobian(jac):
        jac = jnp.asarray(jac)
        if jac.ndim == 3 and jac.shape[1] == 2:
            # In NetKet complex mode, axis=1 stores [real, imag] components.
            return jac[:, 0, :]
        if jac.ndim == 2:
            return jnp.real(jac)
        return jnp.real(jac).reshape(jac.shape[0], -1)

    def on_compute_update_end(self, step, log_data, driver):
        info = log_data.get("info", {}) if isinstance(log_data, dict) else {}
        # cache for later use with MC estimator and not full summation quantities
        # cache = driver._nis_driver.cached_dict
        # # weights = cache["weights"]
        # jacobian_psi = cache["jacobian_psi"] #\partial_\theta \log \psi(x) shape N_s, 2, N_p stacked real and imag derivatives
        # jacobian_q = driver.info['is_jac'] #\partial_\theta \log q(x) shape N_s, N_\phi

        log_psi = driver.state_p._apply_fun
        log_q = driver.state_q._apply_fun

        if self.fullsum_state_p is None:
            self.fullsum_state_p = FullSumState(hilbert=driver.state_p.hilbert, model=driver.state_p.model, chunk_size=None, seed=0)

        self.fullsum_state_p.variables = driver.state_p.variables
        fs_state_p = self.fullsum_state_p
        psi_theta = fs_state_p.to_array()
        Z_p = jnp.sum(jnp.abs(psi_theta)**2)
        
        if self.fullsum_state_q is None:
            self.fullsum_state_q = FullSumState(hilbert=driver.state_q.hilbert, model=driver.state_q.model, chunk_size=None, seed=0) 
        self.fullsum_state_q.variables = driver.state_q.variables
        fs_state_q = self.fullsum_state_q 
        q_phi = fs_state_q.to_array() #q_phi always positive
        Z_q = jnp.sum(q_phi)

        exact_ratio = float(Z_p / Z_q)

        current_iterative_ratio = (
            exact_ratio if self._iterative_ratio_next is None else self._iterative_ratio_next
        )

        all_states = driver.state_q.hilbert.all_states()
        fs_jac_logq_phi =  nkjax.jacobian(
        driver.state_q._apply_fun,
        driver.state_q.parameters,
        all_states,
        driver.state_q.model_state,
        mode="real",
        dense=True,
        center=False,
    )  # (#ns, np') with np' = number of parameters of the sampler q
        fs_jac_logpsi_theta =  nkjax.jacobian(
        driver.state_p._apply_fun,
        driver.state_p.parameters,
        all_states,
        driver.state_p.model_state,
        mode="complex",
        dense=True,
        center=False,
    )  # (#ns, np') with np' = number of parameters of the sampler p

        jac_logq = self._real_jacobian(fs_jac_logq_phi)
        jac_logpsi_re = self._real_jacobian(fs_jac_logpsi_theta)

        alpha_p = self._get_learning_rate(driver, int(step))
        alpha_q = self._get_learning_rate(driver.driver_nis, int(step))

        dtheta = -alpha_p * ravel_pytree(driver._dp)[0]
        dphi = -alpha_q * ravel_pytree(driver.driver_nis._dp)[0]

        if jac_logpsi_re.shape[1] != dtheta.shape[0] or jac_logq.shape[1] != dphi.shape[0]:
            print(
                "IterativeNormalizationCallback: skipping step due to shape mismatch "
                f"(jac_logpsi={jac_logpsi_re.shape}, dtheta={dtheta.shape}, "
                f"jac_logq={jac_logq.shape}, dphi={dphi.shape})."
            )
            return

        q_phi = jnp.real(q_phi)
        q_norm = q_phi / jnp.sum(q_phi)

        weights = jnp.abs(psi_theta)**2 / q_phi

        # Implements Eq. R' = R * (1 + E_q[2 δθ^T W Re(∇_θ log ψ) - δφ^T ∇_φ log q]).
        theta_inner = 2.0 * weights * (jac_logpsi_re @ dtheta)
        phi_inner = jac_logq @ dphi
        relative_update = jnp.sum(q_norm * (theta_inner - phi_inner))

        next_iterative_ratio = current_iterative_ratio * (1.0 + float(relative_update))

        self._steps.append(int(step))
        self._exact_ratio_t.append(exact_ratio)
        self._iterative_ratio_t.append(float(current_iterative_ratio))
        self._relative_update_t.append(float(relative_update))

        self._iterative_ratio_next = float(next_iterative_ratio)

        if step % self.plot_every != 0:
            return

        # --- plot ---
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)

        axes[0].plot(self._steps, self._exact_ratio_t, color="tab:blue", lw=2)
        axes[0].plot(self._steps, self._iterative_ratio_t, color="tab:green", lw=2, ls="--")
        axes[0].set_ylabel("Z_p / Z_q")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(["Exact (full sum)", "Iterative estimate"], loc="best")

        exact_arr = jnp.asarray(self._exact_ratio_t)
        iterative_arr = jnp.asarray(self._iterative_ratio_t)
        ratio_err = jnp.abs(iterative_arr / exact_arr - 1.0)
        axes[1].plot(self._steps, ratio_err, color="tab:orange", lw=2)
        axes[1].set_ylabel("|R_iter / R_exact - 1|")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale("log")

        axes[2].plot(self._steps, self._relative_update_t, color="tab:red", lw=2)
        axes[2].set_ylabel("E_q[first-order term]")
        axes[2].set_xlabel("Step")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("Iterative normalization ratio tracker (full summation diagnostics)", y=0.995)
        fig.tight_layout()

        fig.savefig(self._pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Updated plot saved to {self._pdf_path}")