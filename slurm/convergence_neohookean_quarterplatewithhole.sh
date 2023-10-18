#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=ConvStudy
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere:1

## Build command
## singularity build --fakeroot --force parametricpinn.sif app/.devcontainer/container.def

SCRIPT=parametricpinn/fem/tests/convergence_neohookean_quarterplatewithhole.py

srun singularity exec \
 --cleanenv \
 --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
 --nv \
 parametricpinn.sif \
 python3 /home/davanton/development/ParametricPINN/app/${SCRIPT}