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
# ANSATZ="$1"
# NS="$2"
# if [[ -z "$ANSATZ" ]]; then
#   echo "ERROR: No ansatz specified."
#   echo "Usage: sbatch $0 <ansatz>"
#   echo "Example: sbatch $0 vit"
#   exit 1
# fi

# echo "Running with ansatz = ${ANSATZ}"

# ----------------------------
# Environment
# ----------------------------
export UV_PROJECT_ENVIRONMENT=$SCRATCH/uv_envs/is_env

export NETKET_EXPERIMENTAL_SHARDING=1
export JAX_PLATFORM_NAME=gpu

# (recommended for ROCm stability, optional)
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket

# ----------------------------
# Run
# ----------------------------
cd "$WORK/NeurIS/importance_sampling_nqs/" || exit 1

source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
srun python main.py --config-name li2o_gs
# srun python main.py --config-name li2o_gs ansatz="${ANSATZ}" training.n_s=${NS}