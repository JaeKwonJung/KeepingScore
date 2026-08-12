#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=normal 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --job-name=samlping
#SBATCH --output=logs/sampling_%j.out
#SBATCH --error=logs/sampling_%j.err
#SBATCH --mem=32G
#SBATCH --time=10:00:00

# Load modules
module load mamba

# Make log directory
mkdir -p logs

# Run Python script
/projects/b1042/GoyalLab/jaekj/envs/KS_celltype/bin/python sampling.py \
    --data_path ../embeddings \
    --save_path ./sample_mean/mean.npz \
    --mapping_path ../../scTab_files/merlin_cxg_2023_05_15_sf-log1p/categorical_lookup/cell_type.parquet \
    --sample_size 300 \
    --seed 42
