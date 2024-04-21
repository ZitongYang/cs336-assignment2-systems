#!/bin/bash
#SBATCH --job-name=benchmarking
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --output=/home/c-zitong/cs336-assignment2-systems/log/benchmarking_%j.out
#SBATCH --error=/home/c-zitong/cs336-assignment2-systems/log/benchmarking_%j.err

cd ~/cs336-assignment2-systems

python cs336-systems/cs336_systems/benchmarking_script.py \
    --wandb_run_name "benchmarking_small" \
    --d_model 768 \
    --d_ff 3072 \
    --num_layers 12 \
    --num_heads 12

python cs336-systems/cs336_systems/benchmarking_script.py \
    --wandb_run_name "benchmarking_medium" \
    --d_model 1024 \
    --d_ff 4096 \
    --num_layers 24 \
    --num_heads 16

python cs336-systems/cs336_systems/benchmarking_script.py \
    --wandb_run_name "benchmarking_large" \
    --d_model 1280 \
    --d_ff 5120 \
    --num_layers 36 \
    --num_heads 20

python cs336-systems/cs336_systems/benchmarking_script.py \
    --wandb_run_name "benchmarking_xl" \
    --d_model 1600 \
    --d_ff 6400 \
    --num_layers 48 \
    --num_heads 25

python cs336-systems/cs336_systems/benchmarking_script.py \
    --wandb_run_name "benchmarking_2p7b" \
    --d_model 2560 \
    --d_ff 10240 \
    --num_layers 32 \
    --num_heads 32

python cs336-systems/cs336_systems/benchmarking_script.py \
    --wandb_run_name "benchmarking_small_no_warmup" \
    --warmup_iters 0 \
    --d_model 768 \
    --d_ff 3072 \
    --num_layers 12 \
    --num_heads 12

python cs336-systems/cs336_systems/benchmarking_script.py \
    --wandb_run_name "benchmarking_medium_no_warmup" \
    --warmup_iters 0 \
    --d_model 1024 \
    --d_ff 4096 \
    --num_layers 24 \
    --num_heads 16

python cs336-systems/cs336_systems/benchmarking_script.py \
    --wandb_run_name "benchmarking_large_no_warmup" \
    --warmup_iters 0 \
    --d_model 1280 \
    --d_ff 5120 \
    --num_layers 36 \
    --num_heads 20

python cs336-systems/cs336_systems/benchmarking_script.py \
    --wandb_run_name "benchmarking_xl_no_warmup" \
    --warmup_iters 0 \
    --d_model 1600 \
    --d_ff 6400 \
    --num_layers 48 \
    --num_heads 25

python cs336-systems/cs336_systems/benchmarking_script.py \
    --wandb_run_name "benchmarking_2p7b_no_warmup" \
    --warmup_iters 0 \
    --d_model 2560 \
    --d_ff 10240 \
    --num_layers 32 \
    --num_heads 32