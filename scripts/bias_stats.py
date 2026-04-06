#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netket as nk
import netket.jax as nkjax
import numpy as np
import nqxpack
from omegaconf import DictConfig, OmegaConf
from netket.optimizer.solver import cholesky

from qchem_utils.tasks import MolecularNISRunner


EPS = 1e-18


@dataclass(frozen=True)
class CheckpointPair:
	step: int
	state_p_path: Path
	state_q_path: Path


def _parse_levels(text: str) -> list[float]:
	levels: list[float] = []
	for token in text.split(","):
		val = float(token.strip())
		if val <= 0.0:
			raise ValueError(f"Levels must be > 0, got {val}.")
		levels.append(val)
	if not levels:
		raise ValueError("No relative-energy levels were provided.")
	return levels


def _find_config_path(out_dir: Path) -> Path:
	candidates = [
		out_dir / ".hydra" / "config.yaml",
		out_dir / "config.yaml",
	]
	for candidate in candidates:
		if candidate.is_file():
			return candidate
	raise FileNotFoundError(
		"Could not find run config. Expected one of: "
		f"{candidates[0]} or {candidates[1]}"
	)


def _find_log_path(out_dir: Path) -> Path | None:
	candidates = [
		out_dir / "vmc_run.log.log",
		out_dir / "vmc_run.log",
		out_dir / "vmc_run",
	]
	for candidate in candidates:
		if candidate.is_file():
			return candidate

	wildcard = sorted(out_dir.glob("vmc_run*"))
	for path in wildcard:
		if path.is_file():
			return path
	return None


def _step_from_filename(path: Path, prefix: str) -> int | None:
	match = re.search(rf"{prefix}_step(\d+)", path.name)
	if match is None:
		return None
	return int(match.group(1))


def _collect_checkpoints(out_dir: Path) -> list[CheckpointPair]:
	p_files = sorted(
		[
			*out_dir.glob("state_p_step*.mpack"),
			*out_dir.glob("state_p_step*.nk"),
		]
	)
	q_files = sorted(
		[
			*out_dir.glob("state_q_step*.mpack"),
			*out_dir.glob("state_q_step*.nk"),
		]
	)

	p_by_step: dict[int, Path] = {}
	q_by_step: dict[int, Path] = {}

	for path in p_files:
		step = _step_from_filename(path, "state_p")
		if step is not None:
			p_by_step[step] = path

	for path in q_files:
		step = _step_from_filename(path, "state_q")
		if step is not None:
			q_by_step[step] = path

	if not p_by_step:
		raise FileNotFoundError(
			f"No state_p checkpoint found in {out_dir}. "
			"Expected files like state_p_step{step}.mpack or .nk"
		)
	if not q_by_step:
		raise FileNotFoundError(
			f"No state_q checkpoint found in {out_dir}. "
			"Expected files like state_q_step{step}.mpack or .nk"
		)

	common_steps = sorted(set(p_by_step).intersection(q_by_step))
	if not common_steps:
		raise RuntimeError(
			"Found state_p and state_q files, but no matching step numbers."
		)

	return [
		CheckpointPair(step=s, state_p_path=p_by_step[s], state_q_path=q_by_step[s])
		for s in common_steps
	]


def _load_json_log(log_path: Path) -> dict[str, Any]:
	txt = log_path.read_text(encoding="utf-8").strip()
	if not txt:
		raise RuntimeError(f"Log file is empty: {log_path}")
	try:
		return json.loads(txt)
	except json.JSONDecodeError as exc:
		raise RuntimeError(f"Could not parse JSON log {log_path}: {exc}") from exc


def _extract_energy_series(
	log_data: dict[str, Any],
	preferred_key: str | None,
) -> tuple[str, np.ndarray, np.ndarray]:
	candidate_keys: list[str] = []
	if preferred_key is not None:
		candidate_keys.append(preferred_key)
	candidate_keys.extend(["Energy", "full_energy_fs", "full_energy_from_p"])

	for key in candidate_keys:
		block = log_data.get(key)
		if not isinstance(block, dict):
			continue
		iters = block.get("iters")
		mean = block.get("Mean")
		if iters is None or mean is None:
			continue

		if isinstance(mean, dict):
			if "real" in mean:
				values = np.asarray(mean["real"], dtype=float)
			elif "value" in mean:
				values = np.asarray(mean["value"], dtype=float)
			else:
				continue
		else:
			values = np.asarray(mean, dtype=float)

		steps = np.asarray(iters, dtype=int)
		if steps.ndim != 1 or values.ndim != 1:
			continue
		n = min(steps.shape[0], values.shape[0])
		if n == 0:
			continue
		return key, steps[:n], values[:n]

	raise RuntimeError(
		"Could not extract an energy series from log. "
		"Tried keys: "
		+ ", ".join(candidate_keys)
	)


def _nearest_energy_by_step(
	target_steps: np.ndarray,
	energy_steps: np.ndarray,
	energies: np.ndarray,
) -> np.ndarray:
	values: list[float] = []
	for step in target_steps:
		idx = int(np.argmin(np.abs(energy_steps - step)))
		values.append(float(energies[idx]))
	return np.asarray(values, dtype=float)


def _select_steps_by_levels(
	all_steps: np.ndarray,
	rel_errors: np.ndarray,
	levels: list[float],
) -> np.ndarray:
	selected: list[int] = []
	valid_mask = np.isfinite(rel_errors) & (rel_errors > 0.0)
	valid_indices = np.where(valid_mask)[0]

	if valid_indices.size == 0:
		return all_steps

	rel_valid = rel_errors[valid_indices]
	log_rel = np.log10(rel_valid)

	for level in levels:
		target = math.log10(level)
		local_idx = int(np.argmin(np.abs(log_rel - target)))
		global_idx = int(valid_indices[local_idx])
		selected.append(int(all_steps[global_idx]))

	# Always include the best currently observed step as a useful anchor.
	best_idx = int(np.argmin(rel_errors[valid_indices]))
	selected.append(int(all_steps[int(valid_indices[best_idx])]))

	selected_unique = np.array(sorted(set(selected)), dtype=int)
	return selected_unique


def _rebuild_hamiltonian(cfg: DictConfig) -> Any:
	runner = MolecularNISRunner(cfg)
	runner.setup_system()
	return runner.hamiltonian


def _prepare_cfg_for_offline(cfg: DictConfig, out_dir: Path) -> DictConfig:
	# Hydra run configs can keep ${now:...} interpolations in outdir, which are
	# not resolvable in this standalone script. Use a concrete path instead.
	cfg_offline = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
	if not isinstance(cfg_offline, DictConfig):
		raise RuntimeError("Could not convert run config into DictConfig for offline usage")
	cfg_offline.outdir = str(out_dir)
	return cfg_offline


def _matrix_quadratic_form(matrix: jax.Array, vec: jax.Array) -> jax.Array:
	v = jnp.asarray(vec)
	return jnp.real(jnp.vdot(v, matrix @ v))


def _compute_bias_terms_from_x(
	mu_x: jax.Array,
	x_terms: jax.Array,
	q_pdf: jax.Array,
	w_norm: jax.Array,
	*,
	n_samples: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
	"""
	Compute bias-only terms from X terms.

	Returns:
	  bias_signed, bias_abs, raw_rel_bias, scaled_rel_bias, ess_inv
	"""
	ess_inv = jnp.sum(q_pdf * (w_norm**2))
	eqw2_x = jnp.sum(q_pdf * (w_norm**2) * x_terms, axis=1)
	bias_num = mu_x * ess_inv - eqw2_x
	bias_signed = bias_num / float(max(int(n_samples), 1))
	bias_abs = jnp.abs(bias_signed)
	raw_rel = bias_num / jnp.where(jnp.abs(mu_x) > EPS, mu_x, 1.0)
	scaled_rel = bias_signed / jnp.where(jnp.abs(mu_x) > EPS, mu_x, 1.0)
	return bias_signed, bias_abs, raw_rel, scaled_rel, ess_inv


def _compute_snr_ess_from_x(
	mu_x: jax.Array,
	x_terms: jax.Array,
	q_pdf: jax.Array,
	w_norm: jax.Array,
) -> tuple[jax.Array, jax.Array]:
	"""Compute SNR/ESS-only terms from X terms."""
	centered_x = x_terms - mu_x[:, None]
	var = jnp.sum(q_pdf * (w_norm**2) * jnp.abs(centered_x) ** 2, axis=1)
	snr = jnp.mean(jnp.abs(mu_x) / jnp.sqrt(jnp.maximum(var, EPS)))
	ess = 1.0 / jnp.sum(q_pdf * (w_norm**2))
	return snr, ess


def _prepare_ol_like_sr_fullsum(
	jacobian: jax.Array,
	*,
	mode: str,
	pdf_eff: jax.Array,
	q_pdf: jax.Array,
	w_norm: jax.Array,
	embed_power: float,
) -> jax.Array:
	"""
	Prepare O_L exactly in the spirit of sr_srt_common._prepare_input:
	- center with effective pdf (pdf_eff)
	- embed sqrt(q) and powers of normalized weights in O_L.

	Centering uses q*w_norm, and factors are embedded as:
	  factor = sqrt(q) * w_norm**embed_power
	"""
	weights_expanded = jnp.expand_dims(pdf_eff, tuple(range(1, jacobian.ndim)))
	O_L = jacobian - jnp.sum(jacobian * weights_expanded, axis=0, keepdims=True)

	factor = jnp.sqrt(q_pdf) * (w_norm ** embed_power)
	O_L = O_L * jnp.expand_dims(factor, tuple(range(1, jacobian.ndim)))

	if mode == "complex":
		# (#ns, 2, np) -> (#ns*2, np)
		O_L = jax.lax.collapse(O_L, 0, 2)
	elif mode == "real":
		pass
	else:
		raise NotImplementedError(f"Unsupported mode: {mode}")
	return O_L


def _compute_x_terms_like_sr_fullsum(
	jacobian: jax.Array,
	local_grad: jax.Array,
	*,
	mode: str,
	pdf_eff: jax.Array,
) -> jax.Array:
	r"""
	Compute X terms with SR-like centering and flattening.

	Returns X_terms with shape (n_params, n_states) corresponding to
	X = 2 Re(\delta \partial_theta log\psi * \delta H_loc).
	"""
	weights_expanded = jnp.expand_dims(pdf_eff, tuple(range(1, jacobian.ndim)))
	O_L = jacobian - jnp.sum(jacobian * weights_expanded, axis=0, keepdims=True)

	de = local_grad.flatten() - jnp.sum(local_grad.flatten() * pdf_eff)

	if mode == "complex":
		de2 = jnp.stack([jnp.real(de), jnp.imag(de)], axis=-1)
		dv = jax.lax.collapse(de2, 0, 2)
		O_L = jax.lax.collapse(O_L, 0, 2)
		X_terms = O_L.T * dv
		X_terms = X_terms[:, ::2] + X_terms[:, 1::2]
	elif mode == "real":
		X_terms = jnp.real(O_L.T * de[None, :])
	else:
		raise NotImplementedError(f"Unsupported mode: {mode}")

	return 2.0 * jnp.real(X_terms)


def _solve_preconditioned_update(
	s_matrix: jax.Array,
	rhs: jax.Array,
	diag_shift: float,
) -> tuple[jax.Array, Any]:
	rhs_c = jnp.asarray(rhs)
	n_params = s_matrix.shape[0]
	s_shifted = s_matrix + diag_shift * jnp.eye(n_params, dtype=s_matrix.dtype)

	sol, info = cholesky(s_shifted, rhs_c)
	return sol, info


def _fullsum_bias_metrics(
	state_p: Any,
	state_q: Any,
	hamiltonian: Any,
	*,
	diag_shift: float,
	n_samples: int | None = None,
) -> dict[str, float]:
	fs_state = nk.vqs.FullSumState(
		hilbert=state_p.hilbert,
		model=state_p.model,
		chunk_size=None,
		seed=0,
	)
	fs_state.variables = state_p.variables

	all_states = fs_state.hilbert.all_states()
	pdf = fs_state.probability_distribution()
	vstate_arr = fs_state.to_array()

	h_sparse = hamiltonian.to_sparse()
	hloc = (h_sparse @ vstate_arr) / vstate_arr

	mode = getattr(state_p, "mode", "complex")
	jacobian_orig = nkjax.jacobian(
		fs_state._apply_fun,
		fs_state.parameters,
		all_states,
		fs_state.model_state,
		mode=mode,
		dense=True,
		center=False,
		chunk_size=None,
	)

	unnorm_pdf_2 = jnp.abs(vstate_arr) ** 2
	logpsi_q = state_q._apply_fun(state_q.variables, all_states)
	# In this project, state_q._apply_fun already returns log q.
	q = jnp.exp(jnp.real(logpsi_q))
	q = jnp.clip(q, a_min=EPS)
	q_pdf = q / jnp.sum(q)

	w = unnorm_pdf_2 / q
	w_mean = jnp.sum(q_pdf * w)
	w_norm = w / jnp.maximum(w_mean, EPS)
	pdf_eff = q_pdf * w_norm
	X_terms = _compute_x_terms_like_sr_fullsum(
		jacobian_orig,
		hloc,
		mode=mode,
		pdf_eff=pdf_eff,
	)

	# Build SR-like O_L for S (same centering/embedding logic as gradient path).
	O_L_mu = _prepare_ol_like_sr_fullsum(
		jacobian_orig,
		mode=mode,
		pdf_eff=pdf_eff,
		q_pdf=q_pdf,
		w_norm=w_norm,
		embed_power=0.5,
	)

	# Full-sum reference force (exact under p through effective weighting in X_terms).
	force_unbiased = jnp.sum(pdf_eff * X_terms, axis=1)

	ns_state = getattr(state_q, "n_samples", None)
	ns = n_samples if n_samples is not None else ns_state
	if ns is None:
		ns = 1
	ns = max(int(ns), 1)

	bias_signed, bias_abs, rho0_vec, rel_bias_vec, ess_inv = _compute_bias_terms_from_x(
		force_unbiased,
		X_terms,
		q_pdf,
		w_norm,
		n_samples=ns,
	)
	snr, ess = _compute_snr_ess_from_x(force_unbiased, X_terms, q_pdf, w_norm)

	bias_l2 = jnp.linalg.norm(bias_signed)
	force_l2 = jnp.linalg.norm(force_unbiased)

	# Use the same SR-like preprocessing for S as used to estimate force/update tensors.
	s_matrix = O_L_mu.T @ O_L_mu
	bias_qgt = jnp.sqrt(jnp.maximum(_matrix_quadratic_form(s_matrix, bias_signed), 0.0))
	force_qgt = jnp.sqrt(jnp.maximum(_matrix_quadratic_form(s_matrix, force_unbiased), 0.0))

	# Optimization-impact metrics in preconditioned space: P = S + diag_shift * I.
	upd_bias, info_bias = _solve_preconditioned_update(
		s_matrix,
		bias_signed,
		diag_shift=diag_shift,
	)
	upd_force, info_force = _solve_preconditioned_update(
		s_matrix,
		force_unbiased,
		diag_shift=diag_shift,
	)

	s_upd_bias = s_matrix @ upd_bias
	s_upd_force = s_matrix @ upd_force

	upd_bias_s2 = jnp.real(jnp.vdot(upd_bias, s_upd_bias))
	upd_force_s2 = jnp.real(jnp.vdot(upd_force, s_upd_force))

	upd_bias_snorm = jnp.sqrt(jnp.maximum(upd_bias_s2, 0.0))
	upd_force_snorm = jnp.sqrt(jnp.maximum(upd_force_s2, 0.0))
	upd_ratio_s = upd_bias_snorm / jnp.maximum(upd_force_snorm, EPS)

	upd_bias_l2 = jnp.linalg.norm(upd_bias)
	upd_force_l2 = jnp.linalg.norm(upd_force)
	upd_ratio_l2 = upd_bias_l2 / jnp.maximum(upd_force_l2, EPS)

	num_align = jnp.real(jnp.vdot(upd_force, s_upd_bias))
	den_align = jnp.maximum(upd_force_snorm * upd_bias_snorm, EPS)
	upd_alignment_s = num_align / den_align

	# Proxy for energy-relevant contamination by bias after preconditioning.
	energy_proxy_num = jnp.abs(jnp.vdot(force_unbiased, upd_bias))
	energy_proxy_den = jnp.maximum(jnp.abs(jnp.vdot(force_unbiased, upd_force)), EPS)
	energy_impact_proxy = energy_proxy_num / energy_proxy_den

	def _normalize_info(info: Any) -> float:
		if info is None:
			return 0.0
		return float(jnp.asarray(info))

	return {
		"FullMeanBias": float(jnp.mean(bias_abs)),
		"FullMedianBias": float(jnp.median(bias_abs)),
		"FullMaxBias": float(jnp.max(bias_abs)),
		"FullMinBias": float(jnp.min(bias_abs)),
		"FullStdBias": float(jnp.std(bias_abs)),
		"FullFracBiasGt1e2": float(jnp.mean(bias_abs > 1e2)),
		"FullSNR": float(snr),
		"FullESS": float(ess),
		"BiasL2": float(bias_l2),
		"ForceL2": float(force_l2),
		"BiasToForceL2": float(bias_l2 / jnp.maximum(force_l2, EPS)),
		"BiasQGTNorm": float(bias_qgt),
		"ForceQGTNorm": float(force_qgt),
		"BiasToForceQGT": float(bias_qgt / jnp.maximum(force_qgt, EPS)),
		"N_samples_bias_scale": float(ns),
		"RawRelativeBiasMean": float(jnp.mean(jnp.abs(rho0_vec))),
		"RawRelativeBiasMedian": float(jnp.median(jnp.abs(rho0_vec))),
		"ScaledRelativeBiasMean": float(jnp.mean(jnp.abs(rel_bias_vec))),
		"ScaledRelativeBiasMedian": float(jnp.median(jnp.abs(rel_bias_vec))),
		"BiasQGTQuad": float(_matrix_quadratic_form(s_matrix, bias_signed)),
		"ForceQGTQuad": float(_matrix_quadratic_form(s_matrix, force_unbiased)),
		"EssInverse": float(ess_inv),
		"DiagShift": float(diag_shift),
		"UpdateBiasSNorm": float(upd_bias_snorm),
		"UpdateForceSNorm": float(upd_force_snorm),
		"UpdateBiasToForceS": float(upd_ratio_s),
		"UpdateBiasL2": float(upd_bias_l2),
		"UpdateForceL2": float(upd_force_l2),
		"UpdateBiasToForceL2": float(upd_ratio_l2),
		"UpdateAlignmentS": float(upd_alignment_s),
		"EnergyImpactProxy": float(energy_impact_proxy),
		"SolveInfoBias": _normalize_info(info_bias),
		"SolveInfoForce": _normalize_info(info_force),
	}


def _print_table(rows: list[dict[str, Any]]) -> None:
	if not rows:
		print("No rows to print.")
		return

	headers = [
		"step",
		"energy",
		"rel_energy",
		"FullMeanBias",
		"FullESS",
		"BiasToForceQGT",
		"UpdateBiasToForceS",
		"UpdateAlignmentS",
		"EnergyImpactProxy",
	]

	print("\nBias diagnostics by selected checkpoint:\n")
	print(" ".join(f"{h:>14}" for h in headers))
	for row in rows:
		fields = [
			f"{int(row['step']):14d}",
			f"{row['energy']:14.8f}",
			f"{row['rel_energy']:14.4e}",
			f"{row['FullMeanBias']:14.4e}",
			f"{row['FullESS']:14.4e}",
			f"{row['BiasToForceQGT']:14.4e}",
			f"{row['UpdateBiasToForceS']:14.4e}",
			f"{row['UpdateAlignmentS']:14.4e}",
			f"{row['EnergyImpactProxy']:14.4e}",
		]
		print(" ".join(fields))


def _save_json(rows: list[dict[str, Any]], out_path: Path) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", encoding="utf-8") as f:
		json.dump(rows, f, indent=2)


def _save_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
	if not rows:
		return
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)


def _make_plot_title(cfg: DictConfig, diag_shift: float) -> str:
	system_cfg = cfg.get("system", {})
	cid = system_cfg.get("cid", "unknown")
	basis = system_cfg.get("basis", "")
	ansatz = cfg.get("ansatz", {}).get("name", "?")
	sampler = cfg.get("sampler_net", {}).get("name", "?")
	return (
		f"Bias impact summary - system {cid} ({basis}) | {ansatz}/{sampler} | "
		f"diag_shift={diag_shift:.1e}"
	)


def _plot_summary(rows: list[dict[str, Any]], cfg: DictConfig, out_path: Path, diag_shift: float) -> None:
	if not rows:
		return

	out_path.parent.mkdir(parents=True, exist_ok=True)
	rows_sorted = sorted(rows, key=lambda r: int(r["step"]))

	steps = np.array([int(r["step"]) for r in rows_sorted], dtype=float)
	ess = np.array([float(r["FullESS"]) for r in rows_sorted], dtype=float)
	rel_e = np.array([max(float(r["rel_energy"]), EPS) for r in rows_sorted], dtype=float)
	bias_force_qgt = np.array([float(r["BiasToForceQGT"]) for r in rows_sorted], dtype=float)
	upd_ratio_s = np.array([float(r["UpdateBiasToForceS"]) for r in rows_sorted], dtype=float)
	upd_ratio_l2 = np.array([float(r["UpdateBiasToForceL2"]) for r in rows_sorted], dtype=float)
	energy_proxy = np.array([float(r["EnergyImpactProxy"]) for r in rows_sorted], dtype=float)

	fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
	fig.suptitle(_make_plot_title(cfg, diag_shift), fontsize=12)

	# Panel 1: impact metrics vs relative energy.
	ax = axs[0, 0]
	ax.plot(rel_e, np.maximum(upd_ratio_s, EPS), "o-", label="UpdateBiasToForceS")
	ax.plot(rel_e, np.maximum(energy_proxy, EPS), "s-", label="EnergyImpactProxy")
	ax.plot(rel_e, np.maximum(bias_force_qgt, EPS), "^-", label="BiasToForceQGT")
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlabel("Relative energy error")
	ax.set_ylabel("Impact metric")
	ax.grid(True, which="both", alpha=0.3)
	ax.legend(fontsize=8)

	# Panel 2: update impact as a function of ESS, colored by rel-energy.
	ax = axs[0, 1]
	sc = ax.scatter(
		ess,
		np.maximum(upd_ratio_s, EPS),
		c=np.log10(rel_e),
		cmap="viridis",
		s=48,
		edgecolor="k",
		linewidth=0.3,
	)
	ax.set_yscale("log")
	ax.set_xlabel("ESS")
	ax.set_ylabel("UpdateBiasToForceS")
	ax.grid(True, which="both", alpha=0.3)
	cbar = fig.colorbar(sc, ax=ax)
	cbar.set_label("log10(relative energy)")

	# Panel 3: preconditioned-vs-raw update ratio comparison.
	ax = axs[1, 0]
	ax.plot(steps, np.maximum(upd_ratio_l2, EPS), "o-", label="UpdateBiasToForceL2")
	ax.plot(steps, np.maximum(upd_ratio_s, EPS), "s-", label="UpdateBiasToForceS")
	ax.set_yscale("log")
	ax.set_xlabel("Optimization step")
	ax.set_ylabel("Update ratio")
	ax.grid(True, which="both", alpha=0.3)
	ax.legend(fontsize=8)

	# Panel 4: ESS and rel-energy evolution across selected steps.
	ax = axs[1, 1]
	ax.plot(steps, ess, "o-", color="tab:blue", label="ESS")
	ax.set_xlabel("Optimization step")
	ax.set_ylabel("ESS", color="tab:blue")
	ax.tick_params(axis="y", labelcolor="tab:blue")
	ax.grid(True, alpha=0.3)

	ax2 = ax.twinx()
	ax2.plot(steps, rel_e, "s--", color="tab:red", label="rel_energy")
	ax2.set_yscale("log")
	ax2.set_ylabel("Relative energy", color="tab:red")
	ax2.tick_params(axis="y", labelcolor="tab:red")

	fig.savefig(out_path, dpi=160)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Offline bias diagnostics from saved NIS checkpoints. "
			"Loads state_p/state_q checkpoints, reconstructs the Hamiltonian from run config, "
			"and computes FullSum-style bias diagnostics plus QGT norm comparisons."
		)
	)
	parser.add_argument("out_dir", type=Path, help="Run output directory containing checkpoints")
	parser.add_argument(
		"--levels",
		type=str,
		default="1e-1,1e-2,1e-3,1e-4,1e-5,1e-6",
		help="Comma-separated target relative-energy levels used to pick representative steps",
	)
	parser.add_argument(
		"--exact-energy",
		type=float,
		default=None,
		help=(
			"Reference energy E_ref for relative error |E-E_ref|/max(|E_ref|, eps). "
			"If omitted, uses cfg.exact_energy when available, otherwise min logged energy."
		),
	)
	parser.add_argument(
		"--energy-key",
		type=str,
		default=None,
		help="Optional exact log key to use for energy extraction",
	)
	parser.add_argument(
		"--all-steps",
		action="store_true",
		help="Compute diagnostics for all checkpoint pairs instead of level-based selection",
	)
	parser.add_argument(
		"--diag-shift",
		type=float,
		default=1e-4,
		help="Diagonal shift used in the optimization preconditioner P = S + shift I",
	)
	parser.add_argument(
		"--plot-out",
		type=Path,
		default=None,
		help="Optional output path for the summary plot (defaults to out_dir/bias_stats_summary.png)",
	)
	parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
	parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path")

	args = parser.parse_args()

	out_dir = args.out_dir.expanduser().resolve()
	if not out_dir.is_dir():
		raise NotADirectoryError(f"Not a directory: {out_dir}")

	levels = _parse_levels(args.levels)
	checkpoints = _collect_checkpoints(out_dir)

	cfg_path = _find_config_path(out_dir)
	cfg = OmegaConf.load(cfg_path)
	if not isinstance(cfg, DictConfig):
		raise RuntimeError(f"Could not parse DictConfig from {cfg_path}")
	cfg = _prepare_cfg_for_offline(cfg, out_dir)

	hamiltonian = _rebuild_hamiltonian(cfg)

	log_path = _find_log_path(out_dir)
	log_data: dict[str, Any] | None = None
	step_energy = np.full(len(checkpoints), np.nan, dtype=float)
	rel_energy = np.full(len(checkpoints), np.nan, dtype=float)

	if log_path is not None:
		log_data = _load_json_log(log_path)
		energy_key, energy_steps, energy_vals = _extract_energy_series(log_data, args.energy_key)

		cp_steps = np.asarray([cp.step for cp in checkpoints], dtype=int)
		step_energy = _nearest_energy_by_step(cp_steps, energy_steps, energy_vals)

		ref_energy = args.exact_energy
		if ref_energy is None:
			ref_energy = cfg.get("exact_energy", None)
		if ref_energy is None:
			ref_energy = float(np.min(energy_vals))

		denom = max(abs(float(ref_energy)), EPS)
		rel_energy = np.abs(step_energy - float(ref_energy)) / denom

		print(f"Using energy log key: {energy_key}")
		print(f"Reference energy for relative error: {float(ref_energy):.12f}")
	elif not args.all_steps:
		raise RuntimeError(
			"Could not find vmc_run log in out_dir. "
			"Use --all-steps to bypass energy-level selection."
		)

	cp_steps = np.asarray([cp.step for cp in checkpoints], dtype=int)
	if args.all_steps:
		selected_steps = cp_steps
	else:
		selected_steps = _select_steps_by_levels(cp_steps, rel_energy, levels)

	step_to_cp = {cp.step: cp for cp in checkpoints}
	selected_checkpoints = [step_to_cp[s] for s in selected_steps]

	rows: list[dict[str, Any]] = []
	for cp in selected_checkpoints:
		state_p = nqxpack.load(cp.state_p_path)
		state_q = nqxpack.load(cp.state_q_path)
		training_cfg = cfg.get("training", None)
		n_samples_cfg = training_cfg.get("n_s", None) if training_cfg is not None else None
		metrics = _fullsum_bias_metrics(
			state_p,
			state_q,
			hamiltonian,
			diag_shift=args.diag_shift,
			n_samples=n_samples_cfg,
		)

		idx = int(np.where(cp_steps == cp.step)[0][0])
		row: dict[str, Any] = {
			"step": int(cp.step),
			"state_p": str(cp.state_p_path),
			"state_q": str(cp.state_q_path),
			"energy": float(step_energy[idx]),
			"rel_energy": float(rel_energy[idx]),
		}
		row.update(metrics)
		rows.append(row)

	rows.sort(key=lambda r: r["step"])
	_print_table(rows)

	if args.json_out is not None:
		_save_json(rows, args.json_out.expanduser().resolve())
	if args.csv_out is not None:
		_save_csv(rows, args.csv_out.expanduser().resolve())

	plot_out = args.plot_out.expanduser().resolve() if args.plot_out is not None else (out_dir / "bias_stats_summary.png")
	_plot_summary(rows, cfg, plot_out, args.diag_shift)
	print(f"Saved summary plot: {plot_out}")


if __name__ == "__main__":
	main()
