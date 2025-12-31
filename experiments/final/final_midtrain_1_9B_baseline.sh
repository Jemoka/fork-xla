#!/bin/bash

#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --job-name=houjun-forking-final_midtrain_1_9b_baseline
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --requeue
#SBATCH --mem=128G
#SBATCH --time=14-0
#SBATCH --nodelist=sphinx11,sphinx10
#SBATCH --open-mode=append
#SBATCH --output=./logs/final_midtrain_1_9b_baseline.log

set -euo pipefail

cd .

MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
MASTER_PORT=8316
WORLD_SIZE="$SLURM_NTASKS"

export MASTER_ADDR MASTER_PORT WORLD_SIZE

srun --export=ALL bash -lc '
  set -euo pipefail

  # Pick first Ethernet NIC (en*)
  export NCCL_SOCKET_IFNAME="$(
    ip -o link show | awk -F": " '"'"'$2 ~ /^en/ { print $2; exit }'"'"'
  )"

  export RANK="$SLURM_PROCID"
  export LOCAL_RANK="$SLURM_LOCALID"

  echo "Hello from $(hostname): NCCL_IFACE=$NCCL_SOCKET_IFNAME LOCAL_RANK=$LOCAL_RANK"

  uv python install 3.11 --force

  ./experiments/scripts/develop "
    source .venv/bin/activate && \
    uv run python \
      main.py \
      final_midtrain_1_9b_baseline \
      --warm_start /sphinx/u/houjun/checkpoints/fork/final_midtrain_1_9b_baseline/recovery \
      --midtrain /sphinx/u/houjun/checkpoints/fork/jax/pretrain/final_pretrain_1_9b_baseline/checkpoint/184320 \
      --data_file /juice2/scr2/houjun/fork-xla/experiments/data/midtrain.toml \
      --plan $(printf "regular %.0s" {1..36}) \
      --flops_promised 989e12 \
      --block_size 512 \
      --max_block_size 1024 \
      --n_head 16 \
      --n_embd 2048 \
      --out_dir /sphinx/u/houjun/checkpoints/fork/jax/midtrain \
      --per_device_batch_size 32 \
      --validation_interval 256 \
      --checkpoint_interval 2048 \
      --validation_steps 64 \
      --report_interval 1 \
      --total_steps 8500 \
      --lr 2.5e-4 \
      --warmup_pct 0 \
      --decay_pct 1.0 \
      --shard_into 1 \
      --wandb
  "
'
