#!/bin/bash
#SBATCH --job-name=mixedprecision
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --output=/home/c-zitong/cs336-assignment2-systems/log/mixed_%j.out
#SBATCH --error=/home/c-zitong/cs336-assignment2-systems/log/mixed_%j.err

cd ~/cs336-assignment2-systems

python cs336-systems/cs336_systems/benchmarking_lm.py \
    --warmup_iters=5 \
    --wandb_run_name "benchmarking_small_mixed" \
    --mixed_precision=True \
    --d_model 768 \
    --d_ff 3072 \
    --num_layers 12 \
    --num_heads 12

python cs336-systems/cs336_systems/benchmarking_lm.py \
    --warmup_iters=5 \
    --wandb_run_name "benchmarking_medium_mixed" \
    --mixed_precision=True \
    --d_model 1024 \
    --d_ff 4096 \
    --num_layers 24 \
    --num_heads 16

python cs336-systems/cs336_systems/benchmarking_lm.py \
    --warmup_iters=5 \
    --wandb_run_name "benchmarking_large_mixed" \
    --mixed_precision=True \
    --d_model 1280 \
    --d_ff 5120 \
    --num_layers 36 \
    --num_heads 20

python cs336-systems/cs336_systems/benchmarking_lm.py \
    --warmup_iters=5 \
    --wandb_run_name "benchmarking_xl_mixed" \
    --mixed_precision=True \
    --d_model 1600 \
    --d_ff 6400 \
    --num_layers 48 \
    --num_heads 25

python cs336-systems/cs336_systems/benchmarking_lm.py \
    --warmup_iters=5 \
    --wandb_run_name "benchmarking_2p7b_mixed" \
    --mixed_precision=True \
    --d_model 2560 \
    --d_ff 10240 \
    --num_layers 32 \
    --num_heads 32