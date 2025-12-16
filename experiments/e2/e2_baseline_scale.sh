#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --exclude=sphinx1,sphinx2
#SBATCH --gres=gpu:h200:4
#SBATCH --job-name=houjun-forking-e2_baseline_scale
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=./logs/e2_baseline_scale.log
#SBATCH --partition=sphinx
#SBATCH --time=14-0

cd .

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=8316
export NCCL_SOCKET_IFNAME=$(ifconfig | awk '/^en/{print $1; exit}' | sed 's/://')

uv python install 3.11 --force

./experiments/scripts/develop "source .venv/bin/activate && torchrun --rdzv-backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --nnodes 1 --nproc-per-node 4 main.py e2_baseline_scale --warm_start /sphinx/u/houjun/checkpoints/fork/e2_baseline_scale/best --data_dir /sphinx/u/houjun/dataset/openwebtext --plan regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular --flops_promised 312e12 --block_size 512 --max_block_size 2048 --per_device_batch_size 16 --n_head 16 --n_embd 1024 --out_dir /sphinx/u/houjun/checkpoints/fork --wandb"

