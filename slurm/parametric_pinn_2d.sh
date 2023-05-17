#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=PPINN2D
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

##singularity build --fakeroot --force parametricpinn.sif app/.devcontainer/container.def

SCRIPT=parametric_pinn_2d.py

# srun singularity exec --cleanenv --nv --no-mount /home/davanton/.bashrc,/home/davanton/.bash_profile parametricpinn.sif python3 app/${SCRIPT}
srun singularity exec \
 --cleanenv \
 --no-home \
 --bind output:/data/output,input:/data/input,app:/data/app \
 parametricpinn.sif \
 python3 app/${SCRIPT}


