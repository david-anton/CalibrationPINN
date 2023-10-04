#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=PINN2D
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere:1

## Build command
## singularity build --fakeroot --force parametricpinn.sif app/.devcontainer/container.def

SCRIPT=mains_others/pinn_2d_linearelasticity.py

srun singularity exec \
 --cleanenv \
 --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
 --nv \
 parametricpinn.sif \
 python3 /home/davanton/development/ParametricPINN/app/${SCRIPT}


