#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:h200:2
#SBATCH --job-name=houjun-forking-e0_fork_mergekilled
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=./logs/e0_fork_mergekilled.log
#SBATCH --partition=sphinx
#SBATCH --time=14-0

cd .

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=8313

NCCL_SOCKET_IFNAME=ens3f0 ./experiments/scripts/develop "source .venv/bin/activate && torchrun --rdzv-backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --nnodes 1 --nproc-per-node 2 main.py e0_fork_mergekilled --warm_start /juice2/scr2/houjun/fork/output/e0_fork_mergekilled/best --data_dir /scr/biggest/houjun/dataset_cache/openwebtext --plan regular regular regular fork regular regular regular fork regular regular regular fork regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular --flops_promised 989e12 --block_size 512 --max_block_size 2048 --per_device_batch_size 18 --n_head 16 --n_embd 1024 --out_dir /juice2/scr2/houjun/fork/output --merge_killed_tokens --wandb"

