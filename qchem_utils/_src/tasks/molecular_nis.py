import hydra
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
import os
import json
import logging
from copy import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# JAX / NetKet
import jax.numpy as jnp
import netket as nk
import optax
import advanced_drivers as advd
from qchem_utils._src.nets.bf import Backflow_noMF, MLP
from qchem_utils.nets import ViT_trans_equi, ViT_SA
# Domain Specific Imports (Based on your script)
from neuralimportancesampling._src.driver.ngd_antoine.grad_sample.models import PCMolecule 
from netket.models import ARNNDense
from neuralimportancesampling._src.driver.ngd_antoine.grad_sample.ansatz import LogNeuralBackflow
from neuralimportancesampling.driver import NISDriver, KLfwd
from neuralimportancesampling._src.autoregressive.constrained_ar_model import ConstrainedFermionARNN
from neuralimportancesampling._src.autoregressive.constrained_ar_sampler import ConstrainedARDirectSampler
from neuralimportancesampling.wrapper import RealWrapper, OverdispersedWrapper
from neuralimportancesampling.callback import ComputeEnergyCallback, ESSCallback
from neuralimportancesampling._src.network.cisd import CISD_pdf_overdispersed, _raised_exponential
from neuralimportancesampling._src.utils.cisd import get_cisd_important_configs, configs_to_binary_samples
from pyscf import ci
import neuralimportancesampling as nis

from qchem_utils.utils import (
    SaveStatesCallback,
    EnergyPerSiteCallback,
    PlotEnergyFromPsiCallback,
    PlotTrainingEnergyCallback,
    PlotAlphaCallback,
    SNRAlphaCallback,
)
from qchem_utils.callbacks import IterativeNormalizationCallback
log = logging.getLogger(__name__)

class MolecularNISRunner:
    # 2. Map config keys to directory name strings
    DIR_NAME_MAP = {
        "ViT": "vit_backflow",
        "MLP": "mlp_backflow",
        "RBM": "rbm_dist",
        "AR": "ar_dist"
    }

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.out_dir = self.cfg.outdir
        self.seed = cfg.training.seed
        self.n_samples = cfg.training.n_s
        
        # Placeholders
        self.mol = None
        self.system = None
        self.hilbert = None
        self.hamiltonian = None
        
        # NetKet States
        self.psi = None # Wavefunction P
        self.q = None   # Distribution Q
        self.q_cisd = None # Pre-training target
        self.exact_sampling = cfg.exact_sampling
        self.pretraining = False
        # exact_energy can be provided in cfg (e.g. from FCI/CCSD(T) reference)
        self.exact_energy = cfg.get('exact_energy', None)

        self.fullsum = cfg.get('fullsum', False)

    def setup_system(self):
        """Initializes the PCMolecule and Hilbert Space."""
        log.info(f"Setting up system for CID: {self.cfg.system.cid}")
        print(self.cfg.system.cache_dir)
        self.mol, mo_coeff, mf = PCMolecule.molecule_cached(
            cid=self.cfg.system.cid, 
            dir=self.cfg.system.cache_dir, 
            basis=self.cfg.system.basis
        )
        self.system = PCMolecule(mol=self.mol, mo_coeff=mo_coeff)
        # self.hamiltonian = self.system.hamiltonian.to_jax_operator()
        self.hamiltonian = self.system.hamiltonian
        self.hilbert = self.system.hilbert_space
        
        # Store mean-field object for CISD calculation later
        self.mf = mf

    def _build_psi(self):
        """Uses the mapped class to build the wavefunction."""
        hi = self.system.hilbert_space
        if self.cfg.ansatz.name == "ViT":
            internal_model = ViT_trans_equi(
                n_layers=self.cfg.ansatz.n_layers,
                d_output=hi.n_orbitals * hi.n_fermions,
                d_model=self.cfg.ansatz.d_model,
                n_patches=hi.size // 2,
                d_latent=self.cfg.ansatz.d_latent,
                heads=self.cfg.ansatz.heads,
                b=1,
                graph=nk.graph.Chain(length=hi.size // 2, pbc=False),
                # ... other params from cfg ...
                complex=self.cfg.ansatz.complex
            )
        elif self.cfg.ansatz.name == 'ViTSA':
            internal_model = ViT_SA(
                n_layers=self.cfg.ansatz.n_layers,
                d_output=hi.n_orbitals * hi.n_fermions,
                d_model=self.cfg.ansatz.d_model,
                n_patches=hi.size // 2,
                d_latent=self.cfg.ansatz.d_latent,
                heads=self.cfg.ansatz.heads,
                b=1,
                graph=nk.graph.Chain(length=hi.size // 2, pbc=False),
                # ... other params from cfg ...
                complex=self.cfg.ansatz.complex
            )
        elif self.cfg.ansatz.name == 'MLP':
            internal_model = MLP(
                n_layers=self.cfg.ansatz.n_layers,
                n_features=self.cfg.ansatz.n_features,
                n_out=hi.n_orbitals * hi.n_fermions
            )
        else:
            raise ValueError('Wave function name unsupported')
        
        if self.exact_sampling:
            sampler_p = nk.sampler.ExactSampler(hilbert=self.system.hilbert_space)
        else:
            sampler_p = nk.sampler.MetropolisExchange(hilbert=self.system.hilbert_space, graph = self.system.graph, n_chains=self.n_samples//2)
        
        model_p =  Backflow_noMF(
            hilbert=hi, 
            model=internal_model, 
            enforce_spin_flip=True
        )
        return model_p, sampler_p

    def _build_sampler_net(self):
        """Uses the mapped class to build the sampling distribution (Q)."""
        self.pretraining = False
        if self.cfg.sampler_net.name == 'RBM':
            self.pretraining = True
            model_q = nk.models.RBM(alpha = self.cfg.sampler_net.alpha, param_dtype=jnp.float64)
            if self.exact_sampling:
                sampler_q = nk.sampler.ExactSampler(hilbert=self.system.hilbert_space, machine_pow=1)
            else:
                sampler_q = nk.sampler.MetropolisExchange(hilbert=self.system.hilbert_space, graph = self.system.graph, n_chains=self.n_samples//2, machine_pow=1)
        elif self.cfg.sampler_net.name == 'FermionAR':
            self.pretraining = True
            base_model = ARNNDense(
                hilbert=self.system.hilbert_space,
                layers=1,
                features=128,
                activation=nk.nn.activation.gelu,
                machine_pow=1,
            )
            model_q = ConstrainedFermionARNN(base_model= base_model, 
                                             target_up = self.system.hilbert_space.n_fermions_per_spin[0], 
                                             target_down = self.system.hilbert_space.n_fermions_per_spin[1], 
                                             target=-1 
                                             )
            sampler_q = ConstrainedARDirectSampler(self.system.hilbert_space)
        elif self.cfg.sampler_net.name == 'Overdispersed':
            model_q = OverdispersedWrapper(self.psi.model, alpha_init=self.cfg.sampler_net.alpha_init)
            if self.exact_sampling:
                sampler_q = nk.sampler.ExactSampler(hilbert=self.system.hilbert_space, machine_pow=1)
            else:
                sampler_q = nk.sampler.MetropolisExchange(
                    hilbert=self.system.hilbert_space,
                    graph=self.system.graph,
                    n_chains=self.n_samples // 2,
                    machine_pow=1,
                )
        elif self.cfg.sampler_net.name == 'VMC':
            # Standard Born sampling – no Q distribution needed
            return None, None
        else:
            raise ValueError('Sampler model name unsupported')

        return model_q, sampler_q

    def setup_networks(self):
        """Initializes the P and Q variational states."""
        log.info("Initializing Networks...")
        
        # 1. Setup Wavefunction (P)
        model_p, sampler_p = self._build_psi()
        # Note: Depending on logic, you might need RealWrapper here immediately or later
        self.psi = nk.vqs.MCState(
            sampler=sampler_p, model=model_p, 
            n_samples=self.n_samples, seed=self.seed, sampler_seed=self.seed, chunk_size=8192
        )

        # 2. Setup Importance Sampler (Q) – skipped for plain VMC
        model_q, sampler_q = self._build_sampler_net()
        if model_q is not None:
            self.q = nk.vqs.MCState(
                sampler=sampler_q, model=model_q, 
                n_samples=self.n_samples, seed=self.seed, sampler_seed=self.seed
            )

    def setup_fullsum(self):
        model_p, sampler_p = self._build_psi()
        self.psi = nk.vqs.FullSumState(model = model_p, hilbert=self.system.hilbert_space)

    def compute_cisd_target(self):
        """Computes CISD to create the 'Overdispersed' target distribution."""
        log.info("Running CISD calculation...")
        myci = ci.CISD(self.mf).run()
        coeffs = myci.cisdvec_to_amplitudes(myci.ci)
        configs, weights = get_cisd_important_configs(coeffs, self.mol.nelec)
        samples = configs_to_binary_samples(configs, self.mol.nao)
        
        # Normalize
        weights = weights / jnp.sqrt(jnp.sum(weights**2))
        
        # Define CISD Model
        cisd_model = CISD_pdf_overdispersed(
            cisd_dim=samples.shape[0], 
            n_electrons=self.hilbert.n_fermions, 
            n_orbitals=self.hilbert.n_orbitals, 
            alpha=0.5, beta=0.5, 
            func_args=(1,0), func=_raised_exponential
        )
        
        model_params = {
            'samples': jnp.array(samples), 
            'weights': jnp.array(weights, dtype=jnp.complex128)
        }
        
        if self.exact_sampling:
            sampler = nk.sampler.ExactSampler(hilbert=self.system.hilbert_space, machine_pow=1)
        else:
            sampler = nk.sampler.MetropolisExchange(hilbert=self.system.hilbert_space, graph = self.system.graph, machine_pow=1)
        self.q_ov = nk.vqs.MCState(
            sampler=sampler, model=cisd_model, 
            n_samples=self.n_samples, seed=self.seed, sampler_seed=self.seed
        )
        
        # Inject calculated parameters
        vars = self.q_ov.variables
        vars['model_params'] = model_params
        self.q_ov.variables = vars

    def run_pretraining(self):
        """Performs KL Divergence minimization for both Q and P against CISD."""
        if self.cfg.training.get("pretrain", None) is None:
            return
        cfg_pretrain = self.cfg.training.pretrain
        log.info("Starting Pre-training (KL Divergence)...")
        n_iter = self.cfg.training.pretrain.iterations
        lr_schedule = optax.linear_schedule(self.cfg.training.pretrain.lr, 1e-2, n_iter)
        optimizer = optax.sgd(learning_rate=lr_schedule)

        diag_shift = self.cfg.training.pretrain.diag_shift
        use_ngd_sampler = cfg_pretrain.get('use_ngd_sampler', False)
        use_ngd_wf = cfg_pretrain.get('use_ngd_pretrain', False)
        # 1. Train Q (Importance Sampler) to match CISD
        log.info("Pre-training Q...")
        driver_q = KLfwd(
            optimizer=optimizer,
            state_p=copy(self.q_ov), # Target
            state_q=copy(self.q),    # Trainable
            diag_shift=diag_shift,
            use_ngd=use_ngd_sampler
        )
        logger_q = nk.logging.RuntimeLog()
        driver_q.run(n_iter=n_iter, out=logger_q)
        # Update self.q with trained weights
        self.q = driver_q.state_q 
        
        # 2. Train P (Wavefunction) to match CISD
        log.info("Pre-training Psi...")
        model_p2 = RealWrapper(self.psi.model)
        
        if self.exact_sampling:
            sampler_p2 = nk.sampler.ExactSampler(hilbert=self.system.hilbert_space, machine_pow=1)
        else:
            sampler_p2 = nk.sampler.MetropolisExchange(hilbert=self.system.hilbert_space, n_chains=self.n_samples//2, graph = self.system.graph, machine_pow=1)

        q_nnbf = nk.vqs.MCState(sampler=sampler_p2, model=model_p2, n_samples=self.n_samples, seed=self.seed, sampler_seed=self.seed)
        if use_ngd_wf:
            use_ntk=True
        else:
            use_ntk=False
        driver_p = KLfwd(
            optimizer=optimizer,
            state_p=copy(self.q_ov),
            state_q=copy(q_nnbf),
            diag_shift=diag_shift,
            use_ngd=use_ngd_wf,
            use_ntk=use_ntk
        )
        logger_p = nk.logging.RuntimeLog()
        driver_p.run(n_iter=n_iter, out=logger_p)
        
        # Transfer learned params to main psi
        vars_p = {'params': driver_p.state_q.variables['params']['network']}
        self.psi.variables = vars_p
        self.psi.sampler_state = driver_p.state_q.sampler_state
        
        log.info("Pre-training complete.")

    def run_vmc_sr(self):
        """Standard Born-sampling VMC with SR preconditioner."""
        log.info("Starting VMC-SR Optimization...")
        t_cfg = self.cfg.training

        lr_vmc = optax.linear_schedule(
            t_cfg.vmc.lr_start, t_cfg.vmc.lr_end,
            transition_steps=t_cfg.vmc.iterations,
        )
        # Support both single diag_shift and start/end scheduling
        diag_shift_start = t_cfg.vmc.get('diag_shift_start', t_cfg.vmc.diag_shift)
        diag_shift_end = t_cfg.vmc.get('diag_shift_end', t_cfg.vmc.diag_shift)
        dshift_steps = t_cfg.vmc.get('dshift_transition_steps', t_cfg.vmc.iterations)
        diagshift_vmc = optax.linear_schedule(
            diag_shift_start, diag_shift_end, transition_steps=dshift_steps,
        )
        use_ntk = self.psi.n_samples < self.psi.n_parameters
        driver = advd.driver.VMC_NG(
            hamiltonian=self.hamiltonian,
            optimizer=optax.sgd(lr_vmc),
            variational_state=copy(self.psi),
            diag_shift=diagshift_vmc,
            use_ntk=use_ntk,
        )

        logger = nk.logging.JsonLog(
            os.path.join(self.out_dir, "vmc_run"), save_params=False
        )
        n_orbs = self.hilbert.n_orbitals
        exact_per_orb = (
            self.exact_energy / n_orbs if self.exact_energy is not None else None
        )

        driver.run(
            t_cfg.vmc.iterations,
            out=logger,
            show_progress=True,
            callback=[
                PlotEnergyFromPsiCallback(
                    out_dir=os.path.join(self.out_dir, "plots"),
                    plot_every=100,
                    exact_energy=exact_per_orb,
                ),
                PlotTrainingEnergyCallback(
                    out_dir=os.path.join(self.out_dir, "plots"),
                    plot_every=100,
                    n_sites=n_orbs,
                    exact_energy=exact_per_orb,
                ),
            ],
            timeit=True,
        )
        log.info(f"Run finished. Results saved to {self.out_dir}")

    def run_vmc_nis(self):
        """Runs the main NIS-VMC optimization loop."""
        log.info("Starting VMC-NIS Optimization...")
        
        t_cfg = self.cfg.training
        
        # Learning Rate Schedules
        lr_vmc = optax.linear_schedule(t_cfg.vmc.lr_start, t_cfg.vmc.lr_end, transition_steps=t_cfg.vmc.iterations)
        lr_nis = optax.linear_schedule(t_cfg.nis.lr_start, t_cfg.nis.lr_end, transition_steps=t_cfg.vmc.iterations)
        
        opt_vmc = optax.sgd(lr_vmc)
        opt_nis = optax.sgd(lr_nis)
        
        # NIS Driver Config
        nis_build_params = {
            "optimizer": opt_nis,
            "update_fn": t_cfg.nis.update_fn,
            "debug": False,   
            "diag_shift_nis": t_cfg.nis.diag_shift,        
            "solver_fn": nk.optimizer.solver.solve        
        }
        
        # VMC Driver
        driver = nis.driver.VMC_NG(
            hamiltonian=self.hamiltonian.to_jax_operator(),
            optimizer=opt_vmc,
            state_p=copy(self.psi),
            state_q=copy(self.q),
            diag_shift=t_cfg.vmc.diag_shift,
            # linear_solver_fn=nk.optimizer.solver.solve,    
            nis_driver_class=NISDriver,
            # momentum=0.9,
            nis_driver_build_parameters=nis_build_params,
            nis_driver_run_parameters={"n_iter": t_cfg.nis.iterations},
            do_cache=True,
            use_ntk=True,
            on_the_fly=False,
            batch_chunk_size=1024,
        )
        
        logger = nk.logging.JsonLog(os.path.join(self.out_dir, 'vmc_run.log'), save_params=False)
      
        n_orbs = self.hilbert.size
        exact_per_orb = (
            self.exact_energy / n_orbs if self.exact_energy is not None else None
        )
        driver.run(
            t_cfg.vmc.iterations, 
            out=logger, 
            show_progress=True,
            callback=[
                        ComputeEnergyCallback(
                            full_sum=False,
                            compute_from_psi2=True,
                            compute_every=50,),
                        ESSCallback(compute_every=25),
                        # EnergyPerSiteCallback(compute_every=100),
                        # PlotEnergyFromPsiCallback(
                        #     out_dir=os.path.join(self.out_dir, "plots"),
                        #     plot_every=100,
                        #     exact_energy=exact_per_orb,
                        # ),
                        IterativeNormalizationCallback(),
                        # PlotTrainingEnergyCallback(
                        #     out_dir=os.path.join(self.out_dir, "plots"),
                        #     plot_every=100,
                        #     n_sites=n_orbs,
                        #     exact_energy=exact_per_orb,
                        # ),
                        *(
                            [
                                PlotAlphaCallback(
                                    out_dir=os.path.join(self.out_dir, "plots"),
                                    plot_every=100,
                                ),
                                SNRAlphaCallback(
                                    out_dir=os.path.join(self.out_dir, "plots"),
                                    H_sp=self.hamiltonian.to_sparse(),
                                    compute_every=50,
                                    plot_every=100,
                                ),
                            ]
                            if self.cfg.sampler_net.name == "Overdispersed" else []
                        ),
                    ],
            timeit=True
        )
        
        # Serialize Final Logs
        # logger.serialize(os.path.join(self.out_dir, 'vmc_run.log'))
        log.info(f"Run finished. Results saved to {self.out_dir}")

    def run_fullsum(self):
        t_cfg = self.cfg.training
    
        # Learning Rate Schedules
        lr_vmc = optax.linear_schedule(t_cfg.vmc.lr_start, t_cfg.vmc.lr_end, transition_steps=t_cfg.vmc.iterations)
        lr_nis = optax.linear_schedule(t_cfg.nis.lr_start, t_cfg.nis.lr_end, transition_steps=t_cfg.vmc.iterations)
        
        opt_vmc = optax.sgd(lr_vmc)
        
        driver = nk.driver.VMC(hamiltonian=self.hamiltonian,
                                optimizer=opt_vmc,
                                variational_state=self.psi)
        
        logger = nk.logging.JsonLog(os.path.join(self.out_dir, 'vmc_run.log'), save_params=False)
        print(type(self.hamiltonian))
        driver.run(
            t_cfg.vmc.iterations, 
            out=logger, 
            show_progress=True,
            timeit=True
        )
            
    def __call__(self):
        self.setup_system()
        if self.fullsum:
            self.setup_fullsum()
            self.run_fullsum()
        else:
            self.setup_networks()
            if self.cfg.sampler_net.name == 'VMC':
                self.run_vmc_sr()
            else:
                if self.pretraining:
                    self.run_pretraining()
                self.run_vmc_nis()