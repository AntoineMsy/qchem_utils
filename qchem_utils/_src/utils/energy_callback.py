import os
import jax.numpy as jnp

from advanced_drivers._src.callbacks.base import AbstractCallback
from netket.utils import struct
from netket.stats import Stats


class EnergyPerSiteCallback(AbstractCallback, mutable=True):
    r"""
    Computes the energy per lattice site, estimated from P² (i.e. with `state_p`
    as the sampling distribution).  The number of sites is read directly from
    `driver.state_p.hilbert.n_orbitals`, so nothing extra needs to be passed at
    construction time.
    """

    _compute_every: int = struct.field(pytree_node=False, default=1)
    _n_s: int = struct.field(pytree_node=False, default=2**15)

    def __init__(self, compute_every: int = 1, n_s: int = 2**15):
        super().__init__()
        self._compute_every = compute_every
        self._n_s = n_s

    def on_compute_update_end(self, step, log_data, driver):
        if step % self._compute_every != 0:
            return

        n_sites = driver.state_p.hilbert.n_orbitals  # = Lx * Ly for Hubbard

        n_s_orig = driver.state_p.n_samples
        driver.state_p.n_samples = self._n_s
        stats = driver.state_p.expect(driver._ham)
        driver.state_p.n_samples = n_s_orig

        log_data["energy_per_site"] = Stats(
            mean=stats.mean / n_sites,
            error_of_mean=stats.error_of_mean / n_sites,
            variance=stats.variance / n_sites**2,
            tau_corr=stats.tau_corr,
            R_hat=stats.R_hat,
        )