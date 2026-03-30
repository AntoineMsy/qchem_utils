from __future__ import annotations

import os
from typing import Optional
from typing import Any
import numpy as np

from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.tree_util import tree_map
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
    _exact_ratio_next_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _iterative_ratio_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _iterative_ratio_next_recursive_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _iterative_ratio_stochastic_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _iterative_ratio_next_stochastic_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _raw_ratio_stochastic_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _relative_update_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _relative_update_stochastic_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _ess_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _q_lin_rel_disc_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _p_lin_rel_disc_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _fullsum_energy_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)
    _fullsum_energy_relerr_t : list[Optional[float]] = struct.field(pytree_node=False, default_factory=list)

    _pdf_path: str = struct.field(pytree_node=False, default="")
    _iterative_ratio_next: Optional[float] = struct.field(pytree_node=False, default=None)
    _iterative_ratio_next_stochastic: Optional[float] = struct.field(pytree_node=False, default=None)
    _ham_sparse: Any = struct.field(pytree_node=False, default=None)
    _iterative_enabled: bool = struct.field(pytree_node=False, default=False)
    _last_fullsum_energy: Optional[float] = struct.field(pytree_node=False, default=None)
    _n_samples_from_driver: Optional[int] = struct.field(pytree_node=False, default=None)

    nis_lr: Any = struct.field(pytree_node=False, default=None)
    vmc_lr: Any = struct.field(pytree_node=False, default=None)
    E_exact: Optional[float] = struct.field(pytree_node=False, default=None)
    iterative_start_step: int = struct.field(pytree_node=False, default=50)
    auto_enable_iterative: bool = struct.field(pytree_node=False, default=False)
    energy_rel_delta_tol: float = struct.field(pytree_node=False, default=1e-4)
    n_sites: Optional[int] = struct.field(pytree_node=False, default=None)

    def __init__(
        self,
        *,
        output_dir: str = "outputs/plots",
        filename: str = "iterative_normalization.pdf",
        plot_every: int = 10,
        nis_lr=None,
        vmc_lr=None,
        E_exact: Optional[float] = None,
        iterative_start_step: int = 30,
        auto_enable_iterative: bool = False,
        energy_rel_delta_tol: float = 1e-4,
        n_sites: Optional[int] = None,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.filename = filename
        self.plot_every = plot_every
        self._steps = []
        self._exact_ratio_t = []
        self._exact_ratio_next_t = []
        self._iterative_ratio_t = []
        self._iterative_ratio_next_recursive_t = []
        self._iterative_ratio_stochastic_t = []
        self._iterative_ratio_next_stochastic_t = []
        self._raw_ratio_stochastic_t = []
        self._relative_update_t = []
        self._relative_update_stochastic_t = []
        self._ess_t = []
        self._q_lin_rel_disc_t = []
        self._p_lin_rel_disc_t = []
        self._fullsum_energy_t = []
        self._fullsum_energy_relerr_t = []
        self._iterative_ratio_next = None
        self._iterative_ratio_next_stochastic = None
        self._ham_sparse = None
        self._iterative_enabled = False
        self._last_fullsum_energy = None
        self._n_samples_from_driver = None
        self.nis_lr = nis_lr
        self.vmc_lr = vmc_lr
        self.E_exact = None if E_exact is None else float(E_exact)
        self.iterative_start_step = int(iterative_start_step)
        self.auto_enable_iterative = bool(auto_enable_iterative)
        self.energy_rel_delta_tol = float(energy_rel_delta_tol)
        self.n_sites = None if n_sites is None else int(n_sites)
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
    def _resolve_schedule(value, step: int) -> Optional[float]:
        """Resolve a fixed scalar or Optax/JAX schedule exactly like diag_shift handling."""
        if value is None:
            return None
        if callable(value):
            value = value(step)
        return float(jnp.asarray(value))

    @staticmethod
    def _real_jacobian(jac):
        jac = jnp.asarray(jac)
        if jac.ndim == 3 and jac.shape[1] == 2:
            # In NetKet complex mode, axis=1 stores [real, imag] components.
            return jac[:, 0, :]
        if jac.ndim == 2:
            return jnp.real(jac)
        return jnp.real(jac).reshape(jac.shape[0], -1)

    @staticmethod
    def _with_updated_params(variables, params):
        model_state = {k: v for k, v in variables.items() if k != "params"}
        return {"params": params, **model_state}

    @staticmethod
    def _mean_relative_discrepancy(approx, exact):
        den = jnp.clip(jnp.abs(exact), a_min=jnp.finfo(jnp.asarray(exact).dtype).tiny)
        return jnp.sum((jnp.exp(approx) - jnp.exp(exact))**2) / (jnp.sum(jnp.exp(exact))**2)

    def _update_n_samples_from_driver(self, driver):
        if self._n_samples_from_driver is not None:
            return

        n_samples = getattr(driver.state_q, "n_samples", None)
        if n_samples is not None:
            self._n_samples_from_driver = int(n_samples)
            return

        samples_q = getattr(driver.state_q, "samples", None)
        if samples_q is None:
            return

        samples_q = jnp.asarray(samples_q)
        if samples_q.ndim >= 2:
            self._n_samples_from_driver = int(samples_q.shape[0] * samples_q.shape[1])
        elif samples_q.ndim == 1:
            self._n_samples_from_driver = int(samples_q.shape[0])

    def _plot_diagnostics(self):
        fig, axes = plt.subplots(4, 1, figsize=(8.5, 11.5), sharex=True)

        axes[0].plot(self._steps, self._exact_ratio_t, color="tab:blue", lw=2)
        axes[0].plot(self._steps, self._iterative_ratio_t, color="tab:green", lw=2, ls="--")
        axes[0].plot(self._steps, self._iterative_ratio_stochastic_t, color="tab:purple", lw=2, ls=":")
        axes[0].plot(self._steps, self._raw_ratio_stochastic_t, color="tab:orange", lw=1.8, ls="-.")
        axes[0].set_ylabel("Z_p / Z_q")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend([
            "Exact current",
            "Iterative recursive (current)",
            "Iterative stochastic (current)",
            "Raw stochastic ratio",
        ], loc="best")

        exact_next = jnp.asarray(self._exact_ratio_next_t)
        exact_current = jnp.asarray(self._exact_ratio_t)
        iter_next_recursive = jnp.asarray(self._iterative_ratio_next_recursive_t)
        iter_next_stochastic = jnp.asarray(self._iterative_ratio_next_stochastic_t)
        raw_stochastic = jnp.asarray(self._raw_ratio_stochastic_t)
        eps = jnp.finfo(exact_next.dtype).tiny

        rel_err_recursive = jnp.abs(iter_next_recursive / jnp.clip(jnp.abs(exact_next), a_min=eps) - 1.0)
        rel_err_stochastic = jnp.abs(iter_next_stochastic / jnp.clip(jnp.abs(exact_next), a_min=eps) - 1.0)
        rel_err_raw_stochastic = jnp.abs(raw_stochastic / jnp.clip(jnp.abs(exact_current), a_min=eps) - 1.0)

        axes[1].plot(self._steps, rel_err_recursive, color="tab:orange", lw=2)
        axes[1].plot(self._steps, rel_err_stochastic, color="tab:purple", lw=2, ls=":")
        axes[1].plot(self._steps, rel_err_raw_stochastic, color="tab:brown", lw=2, ls="-.")
        axes[1].set_ylabel("Abs. rel. error")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale("log")
        axes[1].legend(["recursive (next)", "stochastic (next)", "raw stochastic (current)"], loc="best")

        ess_line = axes[2].plot(self._steps, self._ess_t, color="tab:blue", lw=2, label="ESS (raw w)")[0]
        axes[2].set_ylabel("ESS (raw w)", color="tab:blue")
        axes[2].tick_params(axis="y", labelcolor="tab:blue")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale("log")

        ax2_right = axes[2].twinx()
        q_disc_line = ax2_right.plot(
            self._steps,
            jnp.clip(jnp.array(self._q_lin_rel_disc_t), max=1e3),
            color="tab:green",
            lw=2,
            ls="--",
            label=r"$\epsilon_q^{\mathrm{lin}}$",
        )[0]
        p_disc_line = ax2_right.plot(
            self._steps,
            jnp.clip(jnp.array(self._p_lin_rel_disc_t), max=1e3),
            color="tab:purple",
            lw=2,
            ls=":",
            label=r"$\epsilon_{\psi}^{\mathrm{lin}}$",
        )[0]
        ax2_right.set_ylabel("Linearization discrepancy", color="tab:green")
        ax2_right.tick_params(axis="y", labelcolor="tab:green")
        ax2_right.set_yscale("log")

        axes[2].legend([ess_line, q_disc_line, p_disc_line], [
            "ESS (raw w)",
            r"$\epsilon_q^{\mathrm{lin}}$",
            r"$\epsilon_{\psi}^{\mathrm{lin}}$",
        ], loc="best")

        if self.E_exact is not None:
            axes[3].plot(self._steps, self._fullsum_energy_relerr_t, color="tab:brown", lw=2)
            axes[3].set_ylabel("|E_fullsum - E_exact| / |E_exact|")
            axes[3].set_yscale("log")
            axes[3].legend(["Full-sum energy rel. error"], loc="best")
        else:
            axes[3].plot(self._steps, self._fullsum_energy_t, color="tab:brown", lw=2)
            axes[3].set_ylabel("E_fullsum")
            axes[3].legend(["Full-sum energy"], loc="best")
        axes[3].set_xlabel("Step")
        axes[3].grid(True, alpha=0.3)

        title = "Iterative normalization ratio tracker (full summation diagnostics)"
        meta = []
        if self._n_samples_from_driver is not None:
            meta.append(f"n_samples={self._n_samples_from_driver}")
        if self.n_sites is not None:
            meta.append(f"L={self.n_sites}")
        if meta:
            title = f"{title} | " + ", ".join(meta)

        fig.suptitle(title, y=0.995)
        fig.tight_layout()

        fig.savefig(self._pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Updated plot saved to {self._pdf_path}")

    def on_parameter_update(self, step, log_data, driver):
        self._update_n_samples_from_driver(driver)

        all_states = driver.state_q.hilbert.all_states()
        logq_phi = driver.state_q._apply_fun(driver.state_q.variables, all_states)
        lopsi_theta = driver.state_p._apply_fun(driver.state_p.variables, all_states)

        q_phi = jnp.exp(logq_phi)
        p_theta = jnp.exp(2*jnp.real(lopsi_theta))
        Z_p = jnp.sum(p_theta)
        Z_q = jnp.sum(q_phi)

        exact_ratio = float(Z_p / Z_q)

        # Full-sum energy from current state and exact sparse Hamiltonian.
        fullsum_energy = float("nan")
        fullsum_relerr = float("nan")
        ham = getattr(driver, "_ham", None)
        if ham is not None and hasattr(ham, "to_sparse"):
            if self._ham_sparse is None:
                self._ham_sparse = ham.to_sparse()
            psi_theta = np.asarray(jnp.exp(lopsi_theta))
            den = np.vdot(psi_theta, psi_theta)
            if np.isfinite(np.abs(den)) and np.abs(den) > 0.0:
                e_num = np.vdot(psi_theta, self._ham_sparse @ psi_theta)
                fullsum_energy = float(np.real(e_num / den))
                if self.E_exact is not None:
                    denom = max(abs(self.E_exact), np.finfo(float).tiny)
                    fullsum_relerr = abs(fullsum_energy - self.E_exact) / denom

        energy_rel_delta = float("nan")
        if np.isfinite(fullsum_energy):
            if self._last_fullsum_energy is not None and np.isfinite(self._last_fullsum_energy):
                denom_prev = max(abs(self._last_fullsum_energy), np.finfo(float).tiny)
                energy_rel_delta = abs(fullsum_energy - self._last_fullsum_energy) / denom_prev
            self._last_fullsum_energy = fullsum_energy

        # Diagnostics that are meaningful even before enabling recursive iterative updates.
        q_safe = jnp.clip(q_phi, a_min=jnp.finfo(q_phi.dtype).tiny)
        w = p_theta / q_safe
        w_sq_sum = jnp.sum(w**2)
        ess_fraction = float((jnp.sum(w) ** 2) / (w.size * jnp.clip(w_sq_sum, a_min=jnp.finfo(w_sq_sum.dtype).tiny)))

        q_lin_rel_disc = float("nan")
        p_lin_rel_disc = float("nan")
        nis_driver_for_disc = getattr(driver, "driver_nis", None)
        if nis_driver_for_disc is None:
            nis_driver_for_disc = getattr(driver, "_nis_driver", None)

        if nis_driver_for_disc is not None:
            try:
                alpha_p_disc = self._resolve_schedule(self.vmc_lr, int(step))
                alpha_q_disc = self._resolve_schedule(self.nis_lr, int(step))
                if alpha_p_disc is None:
                    alpha_p_disc = self._get_learning_rate(driver, int(step))
                if alpha_q_disc is None:
                    alpha_q_disc = self._get_learning_rate(nis_driver_for_disc, int(step))

                delta_theta_tree_disc = tree_map(lambda x: -alpha_p_disc * x, driver._dp)
                delta_phi_tree_disc = tree_map(lambda x: -alpha_q_disc * x, nis_driver_for_disc._dp)
                dtheta_disc = ravel_pytree(delta_theta_tree_disc)[0]
                dphi_disc = ravel_pytree(delta_phi_tree_disc)[0]
                dtheta_stacked_disc = (
                    jnp.concatenate([dtheta_disc.real, dtheta_disc.imag])
                    if jnp.iscomplexobj(dtheta_disc)
                    else dtheta_disc
                )

                fs_jac_logq_phi_disc = nkjax.jacobian(
                    driver.state_q._apply_fun,
                    driver.state_q.parameters,
                    all_states,
                    driver.state_q.model_state,
                    mode="real",
                    dense=True,
                    center=False,
                )
                fs_jac_logpsi_theta_disc = nkjax.jacobian(
                    driver.state_p._apply_fun,
                    driver.state_p.parameters,
                    all_states,
                    driver.state_p.model_state,
                    mode="complex",
                    dense=True,
                    center=False,
                )
                jac_logpsi_complex_disc = (
                    fs_jac_logpsi_theta_disc[:, 0, :] + 1j * fs_jac_logpsi_theta_disc[:, 1, :]
                )

                delta_corr_phi_disc = fs_jac_logq_phi_disc @ dphi_disc
                delta_corr_psi_disc = jac_logpsi_complex_disc @ dtheta_stacked_disc

                updated_theta_params_disc = tree_map(jnp.add, driver.state_p.parameters, delta_theta_tree_disc)
                updated_phi_params_disc = tree_map(jnp.add, driver.state_q.parameters, delta_phi_tree_disc)
                logq_phi_next_disc = driver.state_q._apply_fun(
                    self._with_updated_params(driver.state_q.variables, updated_phi_params_disc),
                    all_states,
                )
                lopsi_theta_next_disc = driver.state_p._apply_fun(
                    self._with_updated_params(driver.state_p.variables, updated_theta_params_disc),
                    all_states,
                )

                q_next_lin_disc = logq_phi + delta_corr_phi_disc
                q_next_exact_disc = logq_phi_next_disc
                q_lin_rel_disc = self._mean_relative_discrepancy(q_next_lin_disc, q_next_exact_disc)

                p_next_lin_disc = jnp.real(lopsi_theta + delta_corr_psi_disc)
                p_next_exact_disc =  jnp.real(lopsi_theta_next_disc)
                p_lin_rel_disc = self._mean_relative_discrepancy(p_next_lin_disc, p_next_exact_disc)
            except Exception:
                # Keep NaN discrepancies if update/Jacobian objects are unavailable at this step.
                pass

        if not self._iterative_enabled:
            enable_iterative = int(step) >= self.iterative_start_step
            if self.auto_enable_iterative:
                enable_iterative = (
                    enable_iterative
                    and np.isfinite(energy_rel_delta)
                    and (energy_rel_delta <= self.energy_rel_delta_tol)
                )
            if enable_iterative:
                self._iterative_enabled = True
                print(
                    "IterativeNormalizationCallback: iterative updates enabled "
                    f"at step {int(step)} (delta_E_rel={energy_rel_delta:.3e})."
                )

        if not self._iterative_enabled:
            nan = float("nan")
            self._steps.append(int(step))
            self._exact_ratio_t.append(nan)
            self._exact_ratio_next_t.append(nan)
            self._iterative_ratio_t.append(nan)
            self._iterative_ratio_next_recursive_t.append(nan)
            self._iterative_ratio_stochastic_t.append(nan)
            self._iterative_ratio_next_stochastic_t.append(nan)
            self._raw_ratio_stochastic_t.append(nan)
            self._relative_update_t.append(nan)
            self._relative_update_stochastic_t.append(nan)
            self._ess_t.append(ess_fraction)
            self._q_lin_rel_disc_t.append(q_lin_rel_disc)
            self._p_lin_rel_disc_t.append(p_lin_rel_disc)
            self._fullsum_energy_t.append(fullsum_energy)
            self._fullsum_energy_relerr_t.append(fullsum_relerr)

            if step % self.plot_every == 0:
                self._plot_diagnostics()
            return

        current_iterative_ratio = (
            exact_ratio if self._iterative_ratio_next is None else self._iterative_ratio_next
        )

        fs_jac_logq_phi =  nkjax.jacobian(
        driver.state_q._apply_fun,
        driver.state_q.parameters,
        all_states,
        driver.state_q.model_state,
        mode="real",
        dense=True,
        center=False,
    ) 
        # (#ns, np') with np' = number of parameters of the sampler q
        fs_jac_logpsi_theta =  nkjax.jacobian(
        driver.state_p._apply_fun,
        driver.state_p.parameters,
        all_states,
        driver.state_p.model_state,
        mode="complex",
        dense=True,
        center=False,
    )  # (#ns, np') with np' = number of parameters of the sampler p
        jac_logpsi_complex = fs_jac_logpsi_theta[:, 0, :] + 1j * fs_jac_logpsi_theta[:, 1, :]

        alpha_p = self._resolve_schedule(self.vmc_lr, int(step))
        alpha_q = self._resolve_schedule(self.nis_lr, int(step))

        nis_driver = getattr(driver, "driver_nis", None)
        if nis_driver is None:
            nis_driver = getattr(driver, "_nis_driver", None)
        if nis_driver is None:
            print("IterativeNormalizationCallback: skipping step (NIS driver not found).")
            return

        if alpha_p is None:
            alpha_p = self._get_learning_rate(driver, int(step))
        if alpha_q is None:
            alpha_q = self._get_learning_rate(nis_driver, int(step))

        delta_theta_tree = tree_map(lambda x: -alpha_p * x, driver._dp)
        delta_phi_tree = tree_map(lambda x: -alpha_q * x, nis_driver._dp)
        dtheta = ravel_pytree(delta_theta_tree)[0]
        dphi = ravel_pytree(delta_phi_tree)[0]

        dtheta_stacked = jnp.concatenate([dtheta.real, dtheta.imag]) if jnp.iscomplexobj(dtheta) else dtheta


        delta_corr_phi = fs_jac_logq_phi @ dphi
        delta_corr_psi = jac_logpsi_complex @ dtheta_stacked

        # Correct reweighting under normalized q(x) = q_phi(x) / Z_q.
        q_safe = jnp.clip(q_phi, a_min=jnp.finfo(q_phi.dtype).tiny)
        q_pdf = q_safe / jnp.sum(q_safe)
        w = p_theta / q_safe

        exp_psi = jnp.exp(2.0 * jnp.real(delta_corr_psi))
        exp_q = jnp.exp(delta_corr_phi)

        e_w = jnp.sum(q_pdf * w)
        e_w_exp = jnp.sum(q_pdf * w * exp_psi)
        e_q_exp = jnp.sum(q_pdf * exp_q)

        # Multiplicative correction so that we can propagate the recursive estimate.
        correction_factor = (e_w_exp / jnp.clip(e_w, a_min=jnp.finfo(e_w.dtype).tiny)) / jnp.clip(
            e_q_exp, a_min=jnp.finfo(e_q_exp.dtype).tiny
        )

        R_old = current_iterative_ratio
        R_est = R_old * correction_factor

        updated_theta_params = tree_map(jnp.add, driver.state_p.parameters, delta_theta_tree)
        updated_phi_params = tree_map(jnp.add, driver.state_q.parameters, delta_phi_tree)

        logq_phi_next = driver.state_q._apply_fun(self._with_updated_params(driver.state_q.variables, updated_phi_params), all_states)
        lopsi_theta_next = driver.state_p._apply_fun(self._with_updated_params(driver.state_p.variables, updated_theta_params), all_states)

        Z_p_next = jnp.sum(jnp.exp(2*jnp.real(lopsi_theta_next)))
        Z_q_next = jnp.sum(jnp.exp(logq_phi_next))

        exact_ratio_next = float(Z_p_next / Z_q_next)

        # ESS of the raw estimator weights w = p/q (normalized by number of states).
        w_sq_sum = jnp.sum(w**2)
        ess_fraction = float((jnp.sum(w) ** 2) / (w.size * jnp.clip(w_sq_sum, a_min=jnp.finfo(w_sq_sum.dtype).tiny)))

        # Mean relative discrepancy between linearized model and exact updated model.
        q_next_lin = logq_phi + delta_corr_phi
        q_next_exact = logq_phi_next
        q_lin_rel_disc = self._mean_relative_discrepancy(q_next_lin, q_next_exact)

        p_next_lin = jnp.real(lopsi_theta + delta_corr_psi)
        p_next_exact =  jnp.real(lopsi_theta_next)
        p_lin_rel_disc = self._mean_relative_discrepancy(p_next_lin, p_next_exact)

        # Keep a scalar summary of the multiplicative update for diagnostics.
        self._relative_update_t.append(float(correction_factor - 1.0))

        next_iterative_ratio_recursive = R_est

        # --- independent stochastic iterative tracker from q samples only ---
        current_iterative_ratio_stochastic = (
            exact_ratio if self._iterative_ratio_next_stochastic is None else self._iterative_ratio_next_stochastic
        )
        next_iterative_ratio_stochastic = current_iterative_ratio_stochastic
        stoch_rel_update = float("nan")
        raw_ratio_stochastic = float("nan")

        samples_q = getattr(driver.state_q, "samples", None)
        if samples_q is not None:
            samples_q = jnp.asarray(samples_q)
            if samples_q.ndim == 1:
                samples_q = samples_q[None, :]
            else:
                samples_q = samples_q.reshape(-1, samples_q.shape[-1])

            logq_samples = driver.state_q._apply_fun(driver.state_q.variables, samples_q)
            logpsi_samples = driver.state_p._apply_fun(driver.state_p.variables, samples_q)

            q_samples = jnp.exp(logq_samples)
            p_samples = jnp.exp(2.0 * jnp.real(logpsi_samples))

            jac_logq_samples = nkjax.jacobian(
                driver.state_q._apply_fun,
                driver.state_q.parameters,
                samples_q,
                driver.state_q.model_state,
                mode="real",
                dense=True,
                center=False,
            )
            jac_logpsi_samples = nkjax.jacobian(
                driver.state_p._apply_fun,
                driver.state_p.parameters,
                samples_q,
                driver.state_p.model_state,
                mode="complex",
                dense=True,
                center=False,
            )

            jac_logpsi_samples_complex = (
                jac_logpsi_samples[:, 0, :] + 1j * jac_logpsi_samples[:, 1, :]
            )

            delta_corr_q_samples = jac_logq_samples @ dphi
            delta_corr_psi_samples = jac_logpsi_samples_complex @ dtheta_stacked

            q_samples_safe = jnp.clip(q_samples, a_min=jnp.finfo(q_samples.dtype).tiny)
            w_samples = p_samples / q_samples_safe

            exp_psi_samples = jnp.exp(2.0 * jnp.real(delta_corr_psi_samples))
            exp_q_samples = jnp.exp(delta_corr_q_samples)

            mean_w = jnp.mean(w_samples)
            mean_w_exp = jnp.mean(w_samples * exp_psi_samples)
            mean_q_exp = jnp.mean(exp_q_samples)

            if bool(jnp.isfinite(mean_w)):
                raw_ratio_stochastic = float(mean_w)

            stoch_correction_factor = (
                mean_w_exp / jnp.clip(mean_w, a_min=jnp.finfo(mean_w.dtype).tiny)
            ) / jnp.clip(mean_q_exp, a_min=jnp.finfo(mean_q_exp.dtype).tiny)

            if bool(jnp.isfinite(stoch_correction_factor)):
                next_iterative_ratio_stochastic = (
                    current_iterative_ratio_stochastic * float(stoch_correction_factor)
                )
                stoch_rel_update = float(stoch_correction_factor - 1.0)

        self._steps.append(int(step))
        self._exact_ratio_t.append(exact_ratio)
        self._exact_ratio_next_t.append(exact_ratio_next)
        self._iterative_ratio_t.append(float(current_iterative_ratio))
        self._iterative_ratio_next_recursive_t.append(float(next_iterative_ratio_recursive))
        self._iterative_ratio_stochastic_t.append(float(current_iterative_ratio_stochastic))
        self._iterative_ratio_next_stochastic_t.append(float(next_iterative_ratio_stochastic))
        self._raw_ratio_stochastic_t.append(raw_ratio_stochastic)
        self._relative_update_stochastic_t.append(stoch_rel_update)
        self._ess_t.append(ess_fraction)
        self._q_lin_rel_disc_t.append(q_lin_rel_disc)
        self._p_lin_rel_disc_t.append(p_lin_rel_disc)
        self._fullsum_energy_t.append(fullsum_energy)
        self._fullsum_energy_relerr_t.append(fullsum_relerr)

        self._iterative_ratio_next = float(next_iterative_ratio_recursive)
        self._iterative_ratio_next_stochastic = float(next_iterative_ratio_stochastic)

        if step % self.plot_every != 0:
            return
        self._plot_diagnostics()