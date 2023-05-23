#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=PPINN2D
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

## Build command
## singularity build --fakeroot --force parametricpinn.sif app/.devcontainer/container.def

SCRIPT=parametric_pinn_2d.py

srun singularity exec \
 --cleanenv \
 --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
 --nv \
 parametricpinn.sif \
 python3 -u /home/davanton/development/ParametricPINN/app/${SCRIPT}


