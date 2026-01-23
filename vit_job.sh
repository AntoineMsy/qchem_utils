#!/bin/bash
#SBATCH --partition=boost_usr_prod 
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=%j.out 
#SBATCH --time 24:00:00
#SBATCH --account=euhpc_a05_006 
#SBATCH --qos=boost_qos_lprod

export UV_PROJECT_ENVIRONMENT=$SCRATCH/uv_envs/neuris_env

export NETKET_EXPERIMENTAL_SHARDING=1
export JAX_PLATFORM_NAME=gpu

cd $WORK/NeurIS/acetic_acid/
# srun uv --offline --project $WORK/uv_envs/netket_env run 
# you can also do
source ${UV_PROJECT_ENVIRONMENT}/bin/activate
srun python main.py --config-name j1j2_10ckpt_fine is_distrib.alpha=2.0 auto_is=true