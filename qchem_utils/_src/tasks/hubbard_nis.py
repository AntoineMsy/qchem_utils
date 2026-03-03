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
from flax import linen as nn
from netket_fermions.activations import log_cosh
# JAX / NetKet
import jax.numpy as jnp
import netket as nk
import optax
from qchem_utils._src.nets.bf import MLP
import advanced_drivers as advd
from netket.experimental.operator import FermiHubbardJax
from netket_fermions.jastrows import Fermi_Jastrow_MLP, fermi_Jastrow_exp, T_fermi_Jastrow_exp, Full_Symm_fermi_Jastrow_exp
from netket_fermions.backflows import Backflow_Translational, Backflow
from netket_fermions.networks import ViT_trans_equi, MLP


from netket_fermions._src.utils.prod_module import ProductModule
# Domain Specific Imports (Based on your script)
from netket.models import ARNNDense
from neuralimportancesampling.driver import NISDriver, KLfwd
from neuralimportancesampling.wrapper import OverdispersedWrapper
from neuralimportancesampling._src.autoregressive.constrained_ar_model import ConstrainedFermionARNN
from neuralimportancesampling._src.autoregressive.constrained_ar_sampler import ConstrainedARDirectSampler
from neuralimportancesampling.wrapper import RealWrapper
from neuralimportancesampling.callback import ComputeEnergyCallback, ESSCallback

from qchem_utils.utils import SaveStatesCallback, EnergyPerSiteCallback, PlotEnergyFromPsiCallback, PlotTrainingEnergyCallback, PlotAlphaCallback, SNRAlphaCallback
import neuralimportancesampling as nis

log = logging.getLogger(__name__)

class HubbardNISRunner:
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
        self.exact_energy = None  # set by setup_system() via Lanczos if feasible
        self.exact_sampling = cfg.exact_sampling

        self.fullsum = cfg.get('fullsum', False)

    def setup_system(self):
        """Initializes the PCMolecule and Hilbert Space."""
        if self.cfg.system.graph == 'torus':
            # Periodic boundary conditions along both axes → torus topology
            self.graph = nk.graph.Grid(extent=[self.cfg.system.Lx, self.cfg.system.Ly], pbc=[True, True])
        elif self.cfg.system.graph == 'cylinder':
            # Periodic boundary condition only along the short axis (Ly) → cylinder topology
            self.graph = nk.graph.Grid(extent=[self.cfg.system.Lx, self.cfg.system.Ly], pbc=[False, True])
        total_sites = self.cfg.system.Lx * self.cfg.system.Ly
        n_h = 1/2 #fixed n_holes
        n_fermions_per_spin = (int(total_sites * (1 - n_h) / 2), int(total_sites * (1 - n_h) / 2))
        # n_fermions_per_spin = (5, 5) # for L=4x4
        print(n_fermions_per_spin)
        self.hilbert = nk.hilbert.SpinOrbitalFermions(n_orbitals=self.graph.n_nodes, s=1/2, n_fermions_per_spin=n_fermions_per_spin)
        self.hamiltonian = FermiHubbardJax(
            hilbert= self.hilbert,
            graph= self.graph,
            t= self.cfg.system.t,
            U= self.cfg.system.U,
        )
        self.graph_doubled = nk.graph.disjoint_union(self.graph, self.graph)

        # ── Exact ground-state energy via Lanczos (small systems only) ────
        # Cache file lives next to this source file so all runs share it.
        _cache_path = os.path.join(os.path.dirname(__file__), "exact_energies_cache.json")

        # Build a deterministic key from every physical parameter that
        # affects the spectrum.
        nup, ndown = n_fermions_per_spin
        _cache_key = (
            f"hubbard"
            f"_Lx{self.cfg.system.Lx}"
            f"_Ly{self.cfg.system.Ly}"
            f"_t{self.cfg.system.t}"
            f"_U{self.cfg.system.U}"
            f"_{self.cfg.system.graph}"
            f"_nup{nup}_ndown{ndown}"
        )

        MAX_EXACT_STATES = 2 ** 25  # Lanczos is fast up to a few million states
        if self.hilbert.n_states <= MAX_EXACT_STATES:
            # ── Try to load from cache first ──────────────────────────────
            _cache: dict = {}
            if os.path.isfile(_cache_path):
                try:
                    with open(_cache_path, "r") as _f:
                        _cache = json.load(_f)
                except Exception as exc:
                    log.warning(f"Could not read exact-energy cache ({_cache_path}): {exc}")

            if _cache_key in _cache:
                self.exact_energy = float(_cache[_cache_key])
                log.info(
                    f"Exact ground-state energy loaded from cache: "
                    f"{self.exact_energy:.6f}  (key: {_cache_key})"
                )
            else:
                # ── Compute via Lanczos and persist ───────────────────────
                try:
                    evals = nk.exact.lanczos_ed(
                        self.hamiltonian, k=1, compute_eigenvectors=False
                    )
                    self.exact_energy = float(evals[0])
                    log.info(
                        f"Exact ground-state energy (Lanczos): "
                        f"{self.exact_energy:.6f}  (key: {_cache_key})"
                    )
                    _cache[_cache_key] = self.exact_energy
                    try:
                        with open(_cache_path, "w") as _f:
                            json.dump(_cache, _f, indent=2)
                        log.info(f"Exact energy cached to {_cache_path}")
                    except Exception as exc:
                        log.warning(f"Could not write exact-energy cache: {exc}")
                except Exception as exc:
                    log.warning(f"Lanczos ED failed: {exc}")
        else:
            log.info(
                f"Hilbert space too large ({self.hilbert.n_states:.2e} states) "
                "for exact diagonalisation – skipping."
            )
        
    def _build_psi(self):
        """Uses the mapped class to build the wavefunction."""
        hi = self.hilbert
        if self.cfg.ansatz.name == "ViT":
            b = 2
            internal_model = ViT_trans_equi(
                n_layers=self.cfg.ansatz.n_layers,
                d_output=hi.n_orbitals * hi.n_fermions,
                d_model=self.cfg.ansatz.d_model,
                n_patches=hi.size // b,
                d_latent=self.cfg.ansatz.d_latent,
                heads=self.cfg.ansatz.heads,
                b=b,
                separate_spin= False,
                graph=self.graph,
                is_equivariant=True,
                make_it_invariant=True,
                # roll=True,
                # ... other params from cfg ...
                complex=self.cfg.ansatz.complex
            )
        elif self.cfg.ansatz.name == 'MLP':
            internal_model = MLP(
                n_layers=self.cfg.ansatz.n_layers,
                n_features=self.cfg.ansatz.n_features,
                n_out= hi.n_orbitals * hi.n_fermions,
                param_dtype=jnp.float64
            )
    
        else:
            raise ValueError('Wave function name unsupported')
        
        if self.exact_sampling:
            sampler_p = nk.sampler.ExactSampler(hilbert=hi)
        else:
            sampler_p = nk.sampler.MetropolisExchange(hilbert=hi, graph = self.graph_doubled, sweep_size=self.hilbert.size, n_chains=self.n_samples//2)
        
        # model_p =  Backflow_Translational(
        #     hilbert=hi, 
        #     model=internal_model, 
        #     enforce_spin_flip=True
        # )
        bf = Backflow(model=internal_model,
                            hilbert=hi,
                            graph=self.graph,
                            enforce_spin_flip=True,
                            mean_field_init='default',
                            initializer=nn.initializers.lecun_normal(),
                            param_dtype=jnp.float64,
                            # param_dtype=pars_type,
                            # kernel_init=initializer
                            )
        
        jastrow_mlp = Fermi_Jastrow_MLP(n_layers=2,
                                        d_model=8,
                                        initializer=nn.initializers.lecun_normal(),
                                        param_dtype=jnp.float64,
                                        out_activation=log_cosh)
        model_p = ProductModule(bf, jastrow_mlp)
        return model_p, sampler_p

    def _build_sampler_net(self):
        """Uses the mapped class to build the sampling distribution (Q)."""
        self.pretraining = False
        if self.cfg.sampler_net.name == 'RBM':
            self.pretraining = True
            model_q = nk.models.RBM(alpha = self.cfg.sampler_net.alpha, param_dtype=jnp.float64)
            if self.exact_sampling:
                sampler_q = nk.sampler.ExactSampler(hilbert=self.hilbert, machine_pow=1)
            else:
                sampler_q = nk.sampler.MetropolisExchange(hilbert=self.hilbert, graph = self.graph_doubled, n_chains=self.n_samples//2, sweep_size=self.hilbert.size, machine_pow=1)
        elif self.cfg.sampler_net.name == 'FermionAR':
            self.pretraining = True
            base_model = ARNNDense(
                hilbert=self.hilbert,
                layers=1,
                features=128,
                activation=nk.nn.activation.gelu,
                machine_pow=1,
            )
            model_q = ConstrainedFermionARNN(base_model= base_model, 
                                             target_up = self.hilbert.n_fermions_per_spin[0], 
                                             target_down = self.hilbert.n_fermions_per_spin[1], 
                                             target=-1 
                                             )
            sampler_q = ConstrainedARDirectSampler(self.hilbert)
        elif self.cfg.sampler_net.name == 'Overdispersed':
            model_q = OverdispersedWrapper(self.psi.model, alpha_init=self.cfg.sampler_net.alpha_init)
            if self.exact_sampling:
                sampler_q = nk.sampler.ExactSampler(hilbert=self.hilbert, machine_pow=1)
            else:
                sampler_q = nk.sampler.MetropolisExchange(hilbert=self.hilbert, graph = self.graph_doubled, sweep_size=self.hilbert.size, n_chains=self.n_samples//2, machine_pow=1)
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
            n_samples=self.n_samples, seed=self.seed, sampler_seed=self.seed
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
        self.psi = nk.vqs.FullSumState(model = model_p, hilbert=self.hilbert)

    # def compute_cisd_target(self):
    #     """Computes CISD to create the 'Overdispersed' target distribution."""
    #     log.info("Running CISD calculation...")
    #     myci = ci.CISD(self.mf).run()
    #     coeffs = myci.cisdvec_to_amplitudes(myci.ci)
    #     configs, weights = get_cisd_important_configs(coeffs, self.mol.nelec)
    #     samples = configs_to_binary_samples(configs, self.mol.nao)
        
    #     # Normalize
    #     weights = weights / jnp.sqrt(jnp.sum(weights**2))
        
    #     # Define CISD Model
    #     cisd_model = CISD_pdf_overdispersed(
    #         cisd_dim=samples.shape[0], 
    #         n_electrons=self.hilbert.n_fermions, 
    #         n_orbitals=self.hilbert.n_orbitals, 
    #         alpha=0.5, beta=0.5, 
    #         func_args=(1,0), func=_raised_exponential
    #     )
        
    #     model_params = {
    #         'samples': jnp.array(samples), 
    #         'weights': jnp.array(weights, dtype=jnp.complex128)
    #     }
        
    #     if self.exact_sampling:
    #         sampler = nk.sampler.ExactSampler(hilbert=self.hilbert, machine_pow=1)
    #     else:
    #         sampler = nk.sampler.MetropolisExchange(hilbert=self.hilbert, graph = self.graph, machine_pow=1)
    #     self.q_ov = nk.vqs.MCState(
    #         sampler=sampler, model=cisd_model, 
    #         n_samples=self.n_samples, seed=self.seed, sampler_seed=self.seed
    #     )
        
    #     # Inject calculated parameters
    #     vars = self.q_ov.variables
    #     vars['model_params'] = model_params
    #     self.q_ov.variables = vars

    def run_pretraining(self):
        """Pre-training pipeline:
        1. (Default) Initialize P at the mean-field (U=0, free-fermion) solution via VMC.
        2. Train Q (importance sampler) to match |P|^2 via forward KL divergence.
        """
        if self.cfg.training.get("pretrain", None) is None:
            return
        cfg_pretrain = self.cfg.training.pretrain
        log.info("Starting Pre-training...")

        n_iter = cfg_pretrain.iterations
        diag_shift = cfg_pretrain.diag_shift
        use_ngd_sampler = cfg_pretrain.get('use_ngd_sampler', False)

        # ------------------------------------------------------------------
        # Step 1 – Initialize P at the mean-field (U=0) solution
        # ------------------------------------------------------------------
        # init_mean_field = cfg_pretrain.get('init_mean_field', True)
        # if init_mean_field:
        #     log.info("Initializing P at the mean-field (U=0 free-fermion) solution...")
        #     n_iter_mf = cfg_pretrain.get('n_iter_mean_field', n_iter)
        #     lr_mf = optax.linear_schedule(cfg_pretrain.lr, 1e-4, n_iter_mf)
        #     optimizer_mf = optax.sgd(learning_rate=lr_mf)

        #     mf_hamiltonian = FermiHubbardJax(
        #         hilbert=self.hilbert,
        #         graph=self.graph,
        #         t=self.cfg.system.t,
        #         U=0.0,  # non-interacting limit → mean-field ground state
        #     )
        #     driver_mf = nk.driver.VMC(
        #         hamiltonian=mf_hamiltonian,
        #         optimizer=optimizer_mf,
        #         variational_state=self.psi,
        #         preconditioner=nk.optimizer.SR(diag_shift=diag_shift),
        #     )
        #     logger_mf = nk.logging.RuntimeLog()
        #     driver_mf.run(n_iter=n_iter_mf, out=logger_mf)
        #     log.info("Mean-field initialization of P complete.")

        # ------------------------------------------------------------------
        # Step 2 – Train Q to match P via forward KL divergence
        #          KLfwd expects both states to be probability distributions
        #          (machine_pow=1), so we expose P through RealWrapper.
        # ------------------------------------------------------------------
        log.info("Pre-training Q to match P via forward KL divergence...")
        lr_schedule = optax.linear_schedule(cfg_pretrain.lr, 1e-2, n_iter)
        optimizer = optax.sgd(learning_rate=lr_schedule)

        # Build a machine_pow=1 view of P that shares its parameters
        model_psi_dist = RealWrapper(self.psi.model)
        if self.exact_sampling:
            sampler_psi_dist = nk.sampler.ExactSampler(hilbert=self.hilbert, machine_pow=1)
        else:
            sampler_psi_dist = nk.sampler.MetropolisExchange(
                hilbert=self.hilbert,
                graph=self.graph,
                n_chains=self.n_samples // 2,
                machine_pow=1,
            )
        psi_as_dist = nk.vqs.MCState(
            sampler=sampler_psi_dist,
            model=model_psi_dist,
            n_samples=self.n_samples,
            seed=self.seed,
            sampler_seed=self.seed,
            # variables=self.psi.variables,  # share parameters with original P
        )

        driver_q = KLfwd(
            optimizer=optimizer,
            state_p=psi_as_dist,  # target: P viewed as a probability distribution
            state_q=copy(self.q), # trainable: Q
            diag_shift=diag_shift,
            use_ngd=use_ngd_sampler,
        )
        logger_q = nk.logging.JsonLog(os.path.join(self.out_dir, "pretrain_kl.log"), save_params=False)
        driver_q.run(n_iter=n_iter, out=logger_q)
        self.q = driver_q.state_q

        self._save_pretrain_curve(logger_q)
        log.info("Pre-training complete.")

    def _save_pretrain_curve(self, logger):
        """Plots the KL divergence optimization curve from pre-training and saves the figure."""
        try:
            kl_data = logger.data["KL"]
            steps = np.array(kl_data["iters"])
            means = np.array(kl_data["Mean"])

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(steps, means, color="steelblue", lw=1.5)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("KL divergence")
            ax.set_title("Pre-training: forward KL divergence (Q → P)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            out_path = os.path.join(self.out_dir, "pretrain_kl_curve.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f"Pre-training KL curve figure saved to {out_path}")
        except Exception as e:
            log.warning(f"Could not save pre-training KL curve figure: {e}")

    def run_vmc_sr(self):
        """Standard Born-sampling VMC with SR preconditioner."""
        log.info("Starting VMC-SR Optimization...")
        t_cfg = self.cfg.training

        lr_vmc = optax.linear_schedule(
            t_cfg.vmc.lr_start, t_cfg.vmc.lr_end,
            transition_steps=t_cfg.vmc.iterations,
        )
        diagshift_vmc = optax.linear_schedule(
            t_cfg.vmc.diag_shift_start, t_cfg.vmc.diag_shift_end,
            transition_steps=t_cfg.vmc.dshift_transition_steps,
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
        exact_per_site = (
            self.exact_energy / (self.cfg.system.Lx * self.cfg.system.Ly)
            if self.exact_energy is not None else None
        )

        driver.run(
            t_cfg.vmc.iterations,
            out=logger,
            show_progress=True,
            callback=[
                PlotEnergyFromPsiCallback(
                    out_dir=os.path.join(self.out_dir, "plots"),
                    plot_every=100,
                    exact_energy=exact_per_site,
                ),
                PlotTrainingEnergyCallback(
                    out_dir=os.path.join(self.out_dir, "plots"),
                    plot_every=100,
                    exact_energy=exact_per_site,
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
        lr_vmc = optax.linear_schedule(t_cfg.vmc.lr_start, t_cfg.vmc.lr_end, transition_steps=t_cfg.vmc.dshift_transition_steps)
        lr_nis = optax.linear_schedule(t_cfg.nis.lr_start, t_cfg.nis.lr_end, transition_steps=t_cfg.vmc.dshift_transition_steps)
        
        opt_vmc = optax.sgd(lr_vmc)
        opt_nis = optax.sgd(lr_nis)
        diagshift_vmc = optax.linear_schedule(t_cfg.vmc.diag_shift_start, t_cfg.vmc.diag_shift_end, transition_steps=t_cfg.vmc.dshift_transition_steps)
        
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
            diag_shift=diagshift_vmc,
            linear_solver_fn=nk.optimizer.solver.solve,    
            nis_driver_class=NISDriver,
            # momentum=0.9,
            nis_driver_build_parameters=nis_build_params,
            nis_driver_run_parameters={"n_iter": t_cfg.nis.iterations},
            do_cache=True,
            use_ntk=True,
            on_the_fly=False,
            batch_chunk_size=1024,
        )
        
        logger = nk.logging.JsonLog(os.path.join(self.out_dir, 'vmc_run'), save_params=False)
        
        driver.run(
            t_cfg.vmc.iterations, 
            out=logger, 
            show_progress=True,
            callback=[
                        # ComputeEnergyCallback(
                        #     full_sum=False,
                        #     compute_from_psi2=True,
                        #     compute_every=50,),
                        ESSCallback(compute_every=100),
                        EnergyPerSiteCallback(compute_every=100),
                        # SaveStatesCallback(out_dir=os.path.join(self.out_dir, "checkpoints"), save_every=100, keep_last_n=5),
                        PlotEnergyFromPsiCallback(
                            out_dir=os.path.join(self.out_dir, "plots"),
                            plot_every=100,
                            exact_energy=(
                                self.exact_energy / (self.cfg.system.Lx * self.cfg.system.Ly)
                                if self.exact_energy is not None else None
                            ),
                        ),
                        PlotTrainingEnergyCallback(out_dir=os.path.join(self.out_dir, "plots"), plot_every=100, exact_energy=(
                                self.exact_energy / (self.cfg.system.Lx * self.cfg.system.Ly)
                                if self.exact_energy is not None else None
                            ),),
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
                        )
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
        
        logger = nk.logging.JsonLog(os.path.join(self.out_dir, 'vmc_run'), save_params=False)
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