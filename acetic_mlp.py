import netket as nk
import jax.numpy as jnp
import optax
from qchem_utils import ViT_trans_equi
import optax
import os
import numpy as np
from copy import copy
import json
# from matplotlib import pyplot as plt

import neuralimportancesampling as nis

from neuralimportancesampling._src.driver.ngd_antoine.grad_sample.models import PCMolecule 
from neuralimportancesampling._src.driver.ngd_antoine.grad_sample.ansatz import Backflow_noMF
from qchem_utils._src.nets.bf import Backflow_noMF, MLP

from neuralimportancesampling.driver import NISDriver
from neuralimportancesampling.wrapper import RealWrapper
from neuralimportancesampling.callback import ComputeEnergyCallback, ESSCallback
from neuralimportancesampling._src.network.cisd import CISD_pdf_overdispersed, _raised_exponential
from neuralimportancesampling._src.utils.cisd import get_cisd_important_configs, configs_to_binary_samples
from neuralimportancesampling.driver import KLfwd
from pyscf import ci

# create system and state
# LiH molecule
out_dir = '/leonardo_work/EUHPC_A05_006/NeurIS/qchem_utils/out_acetic/mlp'
cid = 176

out_path = os.path.join(out_dir, str(cid))
os.makedirs(out_path, exist_ok=True)
mol, mo_coeff, mf = PCMolecule.molecule_cached(cid=176, 
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
print(hi.size)
# VMC wavefunction and sampler
from flax import linen as nn
# vit = ViT_trans_equi(n_layers=2,
#     d_model=32,
#     d_output= hi.n_orbitals * hi.n_fermions,
#     d_latent=32,
#     heads=8,
#     b=1,
#     n_patches=system.hilbert_space.size//2,
#     graph = nk.graph.Chain(length=system.hilbert_space.size//2, pbc=False),
#     separate_spin= False,
#     complex = True,)
vit = MLP(n_layers=2,
        n_features = hi.size,
        hidden_activation=nn.gelu,
        n_out = hi.n_orbitals * hi.n_fermions,
            )

model_p = Backflow_noMF(hilbert=hi, 
                        model=vit, 
                        enforce_spin_flip=True)
# sampler_p = nk.sampler.ExactSampler(hilbert=hi)
sampler_p = nk.sampler.MetropolisExchange(hilbert=hi, graph = system.graph)
psi = nk.vqs.MCState(sampler=sampler_p, model=model_p, n_samples=n_samples, seed=seed, sampler_seed=sampler_seed, chunk_size=2**8)

# NIS distribution and sampler
alpha=2
model_q = nk.models.RBM(alpha=alpha, param_dtype=jnp.float64)
sampler_q = nk.sampler.MetropolisExchange(hilbert=hi, graph = system.graph, machine_pow=1)
# sampler_q = nk.sampler.MetropolisExchange(hilbert=system.hilbert_space, graph=system.graph, machine_pow=1)
q = nk.vqs.MCState(sampler=sampler_q, model=model_q, n_samples=n_samples, seed=seed, sampler_seed=sampler_seed)

# network and sampler details
n_samples = 2**10
seed = 0
sampler_seed = 0
hidden_units = 32
n_layers = 2
alpha=1

# NIS distribution and sampler
model_q = nk.models.RBM(alpha=alpha, param_dtype=jnp.float64)
sampler_q = nk.sampler.MetropolisExchange(hilbert=hi, graph = system.graph, machine_pow=1)
q = nk.vqs.MCState(sampler=sampler_q, model=model_q, n_samples=n_samples, seed=seed, sampler_seed=sampler_seed)
# KL optimization hyperparameters
n_iter_preopt = 100
lr_q = optax.linear_schedule(5e-2, 1e-2, n_iter_preopt)
diag_shift_q = 1e-4

# CISD calculations
myci = ci.CISD(mf).run()
coeffs = myci.cisdvec_to_amplitudes(myci.ci)
configs, weights = get_cisd_important_configs(coeffs, mol.nelec)
samples = configs_to_binary_samples(configs, mol.nao)
weights = weights / jnp.sqrt(jnp.sum(weights**2))

# Hartree-Fock state
n_up, n_down = hi.n_fermions_per_spin
hf_state = jnp.zeros(hi.size, dtype=jnp.int8)
hf_state = hf_state.at[:n_up].set(1)
hf_state = hf_state.at[hi.n_orbitals:hi.n_orbitals+n_down].set(1)

# overdispersed CISD state
cisd = CISD_pdf_overdispersed(cisd_dim=samples.shape[0], n_electrons=hi.n_fermions, n_orbitals=hi.n_orbitals, alpha=0.5, beta=0.5, func_args=(1,0), func=_raised_exponential)
model_params = {'samples': jnp.array(samples), 'weights': jnp.array(weights, dtype=jnp.complex128)}
# sampler = nk.sampler.ExactSampler(hilbert=hi, machine_pow=1)
sampler = nk.sampler.MetropolisExchange(hilbert=hi, graph = system.graph, machine_pow=1)
q_ov = nk.vqs.MCState(sampler=sampler, model=cisd, n_samples=n_samples, seed=seed, sampler_seed=sampler_seed)

vars = q_ov.variables
vars['model_params'] = model_params
q_ov.variables = vars
# KL divergence over q(x)
optimizer = optax.sgd(learning_rate=lr_q)

logger = nk.logging.RuntimeLog()
# match importance sampler to CISD
driver = KLfwd(
    optimizer=optimizer,
    state_p=copy(q_ov),
    state_q=copy(q),
    diag_shift=diag_shift_q,
    use_ngd=False
)
# driver.run(n_iter=n_iter_preopt, out=logger)

logger.serialize(os.path.join(out_path, 'kl_q.log'))

# P_ov = q_ov.to_array()
# idx = jnp.argsort(P_ov, descending=True)
# P_ov = P_ov[idx]

# trained_q = driver.state_q.to_array()
# trained_q = trained_q[idx]

# untrained_q = q.to_array()
# untrained_q = untrained_q[idx]

q = driver.state_q
lr_p = optax.linear_schedule(5e-3, 1e-3, n_iter_preopt)
diag_shift_p = 1e-2
model_p2 = RealWrapper(Backflow_noMF(hilbert=hi, 
                        model=vit, 
                        enforce_spin_flip=True))

sampler_p2 = nk.sampler.MetropolisExchange(hilbert=hi, graph = system.graph, machine_pow=1)
q_nnbf = nk.vqs.MCState(sampler=sampler_p2, model=model_p2, n_samples=n_samples, seed=seed, sampler_seed=sampler_seed, chunk_size=2**8)

logger = nk.logging.RuntimeLog()
# match state amplitudes to cisd
driver = KLfwd(
    optimizer=optimizer,
    state_p=copy(q_ov),
    state_q=copy(q_nnbf),
    diag_shift=diag_shift_p,
    use_ngd=False,
    use_ntk=False
)
# driver.run(n_iter=n_iter_preopt, out=logger)
logger.serialize(os.path.join(out_path, 'kl_nnbf.log'))

# P_ov = q_ov.to_array()
# idx = jnp.argsort(P_ov, descending=True)
# P_ov = P_ov[idx]

# trained_q = driver.state_q.to_array()
# trained_q = trained_q[idx]

# untrained_q = jnp.abs(psi.to_array())**2
# untrained_q = untrained_q[idx]


vars_p = {'params': driver.state_q.variables['params']['network']}
psi.variables = vars_p
psi.sampler_state = driver.state_q.sampler_state
# VMC-NIS iterations
n_iter_vmc = 1000
n_iter_nis = 1

# VMC-NIS diagonal shifts
diag_shift_vmc = 1e-5
diag_shift_nis = 1e-5

# VMC-NIS learning rates
lr_vmc = optax.linear_schedule(init_value=5e-2, transition_steps=n_iter_vmc, end_value=5e-3)
lr_nis =  optax.linear_schedule(init_value=1e-2, transition_steps=n_iter_vmc, end_value=7e-3)

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
print('Running VMC ...')
print(psi.expect(H))
# VMC driver (with NGD update)
driver = nis.driver.VMC_NG(
    hamiltonian=H,
    optimizer=optimizer_vmc,
    state_p=copy(psi),
    state_q=copy(q),
    diag_shift=diag_shift_vmc,
    # linear_solver_fn=nk.optimizer.solver.solve,    
    nis_driver_class=nis_driver_class,
    nis_driver_build_parameters=nis_driver_build_parameters,
    nis_driver_run_parameters=nis_driver_run_parameters,
    do_cache=True,
    use_ntk=True,
    on_the_fly=False,
    debug=False,
    chunk_size_bwd=2**10,
    batch_chunk_size=2**8,
    # model_chunk_size=2**4,
)
logger = nk.logging.RuntimeLog()

# # optimization run
driver.run(
    n_iter_vmc, 
    out=logger, 
    show_progress=True,
    timeit=True
)
logger.serialize(os.path.join(out_path, 'vmc_run.log'))
# a, b = driver.compute_loss_and_update()