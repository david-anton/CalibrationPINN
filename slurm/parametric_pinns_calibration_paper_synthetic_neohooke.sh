#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=336:00:00
#SBATCH --job-name=QPWHNH
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere:1

## Build command
## singularity build --fakeroot --force parametricpinn.sif app/.devcontainer/container_conda.def

SCRIPT=parametric_pinns_calibration_paper_synthetic_neohooke.py

srun singularity run \
 --cleanenv \
 --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
 --nv \
 --nvccli \
 parametricpinn.sif \
 python3 /home/davanton/development/ParametricPINN/app/${SCRIPT}


