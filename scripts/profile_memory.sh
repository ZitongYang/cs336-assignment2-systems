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

# forward and backward
# python cs336-systems/cs336_systems/profiling_memory.py \
#     --wandb_run_name "profiling_small_memory" \
#     --d_model 768 \
#     --d_ff 3072 \
#     --num_layers 12 \
#     --num_heads 12

# python cs336-systems/cs336_systems/profiling_memory.py \
#     --wandb_run_name "profiling_medium_memory" \
#     --d_model 1024 \
#     --d_ff 4096 \
#     --num_layers 24 \
#     --num_heads 16

# python cs336-systems/cs336_systems/profiling_memory.py \
#     --wandb_run_name "profiling_large_memory" \
#     --d_model 1280 \
#     --d_ff 5120 \
#     --num_layers 36 \
#     --num_heads 20

# python cs336-systems/cs336_systems/profiling_memory.py \
#     --wandb_run_name "profiling_xl_memory" \
#     --d_model 1600 \
#     --d_ff 6400 \
#     --num_layers 48 \
#     --num_heads 25

# python cs336-systems/cs336_systems/profiling_memory.py \
#     --wandb_run_name "profiling_2p7b_memory" \
#     --d_model 2560 \
#     --d_ff 10240 \
#     --num_layers 32 \
#     --num_heads 32

python cs336-systems/cs336_systems/profiling_memory.py \
    --wandb_run_name "profiling_2p7b_memory_mixed" \
    --d_model 2560 \
    --d_ff 10240 \
    --num_layers 32 \
    --num_heads 32 \
    --mixed_precision=True

# forward only
# python cs336-systems/cs336_systems/profiling_memory.py \
#     --wandb_run_name "profiling_small_memory_forward" \
#     --d_model 768 \
#     --d_ff 3072 \
#     --num_layers 12 \
#     --num_heads 12 \
#     --only_forward=True

# python cs336-systems/cs336_systems/profiling_memory.py \
#     --wandb_run_name "profiling_medium_memory_forward" \
#     --d_model 1024 \
#     --d_ff 4096 \
#     --num_layers 24 \
#     --num_heads 16 \
#     --only_forward=True

# python cs336-systems/cs336_systems/profiling_memory.py \
#     --wandb_run_name "profiling_large_memory_forward" \
#     --d_model 1280 \
#     --d_ff 5120 \
#     --num_layers 36 \
#     --num_heads 20 \
#     --only_forward=True

# python cs336-systems/cs336_systems/profiling_memory.py \
#     --wandb_run_name "profiling_xl_memory_forward" \
#     --d_model 1600 \
#     --d_ff 6400 \
#     --num_layers 48 \
#     --num_heads 25 \
#     --only_forward=True

# python cs336-systems/cs336_systems/profiling_memory.py \
#     --wandb_run_name "profiling_2p7b_memory_forward" \
#     --d_model 2560 \
#     --d_ff 10240 \
#     --num_layers 32 \
#     --num_heads 32 \
#     --only_forward=True
    
python cs336-systems/cs336_systems/profiling_memory.py \
    --wandb_run_name "profiling_2p7b_memory_forward_mixed" \
    --d_model 2560 \
    --d_ff 10240 \
    --num_layers 32 \
    --num_heads 32 \
    --only_forward=True \
    --mixed_precision=True
