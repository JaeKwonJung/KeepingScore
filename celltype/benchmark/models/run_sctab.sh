#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --job-name=scTab
#SBATCH --output=logs/scTab/slurm-%j.out
#SBATCH --error=logs/scTab/slurm-%j.err

module purge all
module load mamba

export TF_ENABLE_ONEDNN_OPTS=0

# Run script
/projects/b1042/GoyalLab/jaekj/envs/KS_celltype/bin/python -u model_sctab.py --run_id 1 --seed 1
