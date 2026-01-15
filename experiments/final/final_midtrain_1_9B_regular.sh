#!/bin/bash

#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --job-name=houjun-forking-final_midtrain_1_9b_regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --requeue
#SBATCH --mem=128G
#SBATCH --time=14-0
#SBATCH --nodelist=sphinx11,sphinx10
#SBATCH --open-mode=append
#SBATCH --output=./logs/final_midtrain_1_9b_regular.log

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
      test_445100_midtrain_regular \
      --warm_start /sphinx/u/houjun/checkpoints/regular/midtrain/test_445100_midtrain_regular/recovery \
      --midtrain /sphinx/u/houjun/checkpoints/fork/jax/pretrain/final_pretrain_1_9b_regular/best \
      --data_file /juice2/scr2/houjun/regular-xla/experiments/data/midtrain.toml \
      --flops_promised 989e12 \
      --out_dir /sphinx/u/houjun/checkpoints/regular/jax/midtrain \
      --per_device_batch_size 4 \
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
