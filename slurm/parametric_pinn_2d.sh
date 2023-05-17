#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=PPINN2D
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

##singularity build --fakeroot --force parametricpinn.sif app/.devcontainer/container.def

SCRIPT=parametric_pinn_2d.py

srun singularity exec --cleanenv --nv --bind output:/data/output,input:/data/input parametricpinn.sif python3 /data/app/${SCRIPT}
