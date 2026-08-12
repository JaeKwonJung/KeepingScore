#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu 
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1 
#SBATCH --job-name=vanilla_diffusion
#SBATCH --output=logs/diffusion_%j.out
#SBATCH --error=logs/diffusion_%j.err
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Load modules
module load mamba

# Make log directory
mkdir -p logs

# Run Python script
/projects/b1042/GoyalLab/jaekj/envs/KS_celltype/bin/python diffusion_model.py \
    --emb_pred_path ../embeddings \
    --epochs 500 \
    --batch_size 2048 \
    --n_devices 1 \
    --n_workers 1 \
    --state_dir ./state \
    --umap_save_dir ./umap_plots
