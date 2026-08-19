#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --job-name=XGBoost-gpu
#SBATCH --output=logs/xgb/slurm-%j.out
#SBATCH --error=logs/xgb/slurm-%j.err

module purge all
module load mamba

export TF_ENABLE_ONEDNN_OPTS=0

# Run script
/projects/b1042/GoyalLab/jaekj/envs/KS_celltype/bin/python -u model_xgboost.py --run_id 1