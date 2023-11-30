#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=Test
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere:1

## Build command
## singularity build --fakeroot --force parametricpinn.sif app/.devcontainer/container_conda.def

srun nvidia-smi

SCRIPT=cuda_test.py

srun singularity run \
 --nv \
 parametricpinn.sif \
 python3 /home/davanton/development/ParametricPINN/app/${SCRIPT}

##srun singularity run \
## --cleanenv \
## --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
## --nv \
## --nvccli \
## parametricpinn.sif \
## python3 /home/davanton/development/ParametricPINN/app/${SCRIPT}


