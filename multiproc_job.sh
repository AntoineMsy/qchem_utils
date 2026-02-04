#!/bin/bash
#SBATCH --partition=boost_usr_prod 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out 
#SBATCH --time=24:00:00
#SBATCH --account=euhpc_a05_006 
#SBATCH --qos=boost_qos_lprod

# ----------------------------
# Argument handling
# ----------------------------
ANSATZ="$1"
NS="$2"
if [[ -z "$ANSATZ" ]]; then
  echo "ERROR: No ansatz specified."
  echo "Usage: sbatch $0 <ansatz>"
  echo "Example: sbatch $0 vit"
  exit 1
fi

echo "Running with ansatz = ${ANSATZ}"

# ----------------------------
# Environment
# ----------------------------
export UV_PROJECT_ENVIRONMENT=$SCRATCH/uv_envs/neuris_env

export NETKET_EXPERIMENTAL_SHARDING=1
export JAX_PLATFORM_NAME=gpu

# (recommended for ROCm stability, optional)
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket

# ----------------------------
# Run
# ----------------------------
cd "$WORK/NeurIS/qchem_utils/" || exit 1

source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

srun python main.py --config-name mol_nis_n2_test ansatz="${ANSATZ}" training.n_s=${NS}