#!/bin/bash
#SBATCH --job-name=profile
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --output=/home/c-zitong/cs336-assignment2-systems/log/profile_%j.out
#SBATCH --error=/home/c-zitong/cs336-assignment2-systems/log/profile_%j.err

cd ~/cs336-assignment2-systems

# python cs336-systems/cs336_systems/profiling_script.py \
#     --d_model 1600 \
#     --d_ff 6400 \
#     --num_layers 48 \
#     --num_heads 25

python cs336-systems/cs336_systems/profiling_script.py \
    --wandb_run_name "profiling_xl" \
    --d_model 1600 \
    --d_ff 6400 \
    --num_layers 48 \
    --num_heads 25

python cs336-systems/cs336_systems/profiling_script.py \
    --wandb_run_name "profiling_xl_only_forward" \
    --only_forward True \
    --d_model 1600 \
    --d_ff 6400 \
    --num_layers 48 \
    --num_heads 25