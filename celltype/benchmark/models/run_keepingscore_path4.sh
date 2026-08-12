#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --job-name=KS_path4
#SBATCH --output=logs/KS_rand_300_T_1000_Path_4/slurm-%j.out
#SBATCH --error=logs/KS_rand_300_T_1000_Path_4/slurm-%j.err

module purge all
module load mamba

export TF_ENABLE_ONEDNN_OPTS=0

# Run script
/projects/b1042/GoyalLab/jaekj/envs/KS_celltype/bin/python -u model_keepingscore_path4.py \
    --data_path "../../embeddings" \
    --checkpoint "../../diffusion_model/tb_logs/Vanilla Diffusion Model/version_0/checkpoints/epoch=285-step=2128412.ckpt" \
    --T 1000 \
    --n_paths 4 \
    --sample_size 300 \
    --save_dir "./KS_300_T_1000_Path_4" \
    