#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=ConvStudy
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere:1

srun singularity run \
 --cleanenv \
 --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
 --nv \
 --nvccli \
 parametricpinn.sif \
 /bin/bash
