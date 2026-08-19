#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1 
#SBATCH --time=48:00:00 
#SBATCH --mem=64G 
#SBATCH --job-name="linear_model"  
#SBATCH --output=logs/linear_model/slurm-%j.out
#SBATCH --error=logs/linear_model/slurm-%j.err

module purge all
module load mamba

# Call Python directly from your env
/projects/b1042/GoyalLab/jaekj/envs/KS_celltype/bin/python -u model_linear.py --run_id 1 --seed 1



