import netket as nk
import jax.numpy as jnp
import optax
# from qchem_utils import Vit
import optax
import os
import numpy as np
from copy import copy
import json
# from matplotlib import pyplot as plt

import neuralimportancesampling as nis

from neuralimportancesampling._src.driver.ngd_antoine.grad_sample.models import PCMolecule 
from neuralimportancesampling._src.driver.ngd_antoine.grad_sample.ansatz import LogNeuralBackflow

from neuralimportancesampling.driver import NISDriver
from neuralimportancesampling.wrapper import RealWrapper
from neuralimportancesampling.callback import ComputeEnergyCallback, ESSCallback
from neuralimportancesampling._src.network.cisd import CISD_pdf_overdispersed, _raised_exponential
from neuralimportancesampling._src.utils.cisd import get_cisd_important_configs, configs_to_binary_samples
from neuralimportancesampling.driver import KLfwd
from pyscf import ci

# create system and state
# LiH molecule
out_dir = '/leonardo_work/EUHPC_A05_006/NeurIS/qchem_utils/out_test/mlp'
cid = 62714

out_path = os.path.join(out_dir, str(cid))
os.makedirs(out_path, exist_ok=True)
mol, mo_coeff, mf = PCMolecule.molecule_cached(cid=cid, 
                                               dir='/leonardo_work/EUHPC_A05_006/NeurIS/qchem_utils/pubchem_cache', 
                                               basis="STO-3G")#62714)
system = PCMolecule(mol=mol, mo_coeff=mo_coeff)

H = system.hamiltonian.to_jax_operator()
hi = system.hilbert_space

# network and sampler details
n_samples = 2**10
seed = 0
sampler_seed = 0
hidden_units = 256
n_layers = 2
alpha=1

# VMC wavefunction and sampler
model_p = LogNeuralBackflow(hilbert=hi, hidden_units=hidden_units, n_layers=n_layers)
sampler_p = nk.sampler.ExactSampler(hilbert=hi)
psi = nk.vqs.MCState(sampler=sampler_p, model=model_p, n_samples=n_samples, seed=seed, sampler_seed=sampler_seed)

# NIS distribution and sampler
model_q = nk.models.RBM(alpha=alpha, param_dtype=jnp.float64)
sampler_q = nk.sampler.ExactSampler(hilbert=hi, machine_pow=1)
q = nk.vqs.MCState(sampler=sampler_q, model=model_q, n_samples=n_samples, seed=seed, sampler_seed=sampler_seed)

alpha=2
model_q = nk.models.RBM(alpha=alpha, param_dtype=jnp.float64)
sampler_q = nk.sampler.MetropolisExchange(hilbert=system.hilbert_space, graph=system.graph, machine_pow=1)
q = nk.vqs.MCState(sampler=sampler_q, model=model_q, n_samples=n_samples, seed=seed, sampler_seed=sampler_seed)

# network and sampler details
n_samples = 2**10
seed = 0
sampler_seed = 0
hidden_units = 32
n_layers = 2
alpha=1

# VMC wavefunction and sampler
model_p = LogNeuralBackflow(hilbert=hi, hidden_units=hidden_units, n_layers=n_layers)
sampler_p = nk.sampler.ExactSampler(hilbert=hi)
psi = nk.vqs.MCState(sampler=sampler_p, model=model_p, n_samples=n_samples, seed=seed, sampler_seed=sampler_seed)

# NIS distribution and sampler
model_q = nk.models.RBM(alpha=alpha, param_dtype=jnp.float64)
sampler_q = nk.sampler.ExactSampler(hilbert=hi, machine_pow=1)
q = nk.vqs.MCState(sampler=sampler_q, model=model_q, n_samples=n_samples, seed=seed, sampler_seed=sampler_seed)

# VMC-NIS iterations
n_iter_vmc = 10
n_iter_nis = 1

# VMC-NIS diagonal shifts
diag_shift_vmc = 1e-5
diag_shift_nis = 1e-5

# VMC-NIS learning rates
lr_vmc = optax.linear_schedule(init_value=5e-2, transition_steps=n_iter_vmc, end_value=5e-3)
lr_nis =  optax.linear_schedule(init_value=2e-2, transition_steps=n_iter_vmc, end_value=7e-3)

# VMC-NIS optimizers
optimizer_vmc = optax.sgd(learning_rate=lr_vmc)
optimizer_nis = optax.sgd(learning_rate=lr_nis)
# NIS driver (with NGD update)
nis_driver_class = NISDriver
nis_driver_build_parameters = {
    "optimizer": optimizer_nis,
    "update_fn": "snr_mat_sr",
    "debug": False,   
    "diag_shift_nis": diag_shift_nis,        
    "solver_fn": nk.optimizer.solver.solve        
}
nis_driver_run_parameters = {
    "n_iter": n_iter_nis,
}

# VMC driver (with NGD update)
driver = nis.driver.VMC_NG(
    hamiltonian=H,
    optimizer=optimizer_vmc,
    state_p=copy(psi),
    state_q=copy(q),
    diag_shift=diag_shift_vmc,
    linear_solver_fn=nk.optimizer.solver.solve,    
    nis_driver_class=nis_driver_class,
    nis_driver_build_parameters=nis_driver_build_parameters,
    nis_driver_run_parameters=nis_driver_run_parameters,
    do_cache=True,
    use_ntk=True,
    on_the_fly=False,
    debug=False,
    batch_chunk_size=2**10,
    # model_chunk_size=None,
)
logger = nk.logging.RuntimeLog()
driver.run(
    1, 
    out=logger, 
    show_progress=True,
    timeit=True
)
logger = nk.logging.RuntimeLog()
# # optimization run
driver.run(
    n_iter_vmc, 
    out=logger, 
    show_progress=True,
    timeit=True
)
# logger.serialize(os.path.join(out_path, 'vmc_run.log'))
# a, b = driver.compute_loss_and_update()