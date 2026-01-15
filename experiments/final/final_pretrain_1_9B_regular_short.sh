#!/bin/bash

#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --job-name=houjun-forking-final_pretrain_1_9b_regular_short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --requeue
#SBATCH --mem=256G
#SBATCH --time=14-0
#SBATCH --nodelist=sphinx9,sphinx10,sphinx11
#SBATCH --open-mode=append
#SBATCH --output=./logs/final_pretrain_1_9b_regular_short.log

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

  ./experiments/scripts/develop "
    source .venv/bin/activate && \
    uv run --no-group tpu --group cuda python \
      main.py \
      final_pretrain_1_9b_regular_short \
      --warm_start /sphinx/u/houjun/checkpoints/fork/final_pretrain_1_9b_regular_short/recovery \
      --data_file /home/houjun/data/recipes/pretrain.toml \
      --data_file /juice2/scr2/houjun/fork-xla/experiments/data/owt.toml \
      --plan regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular \
      --flops_promised 989e12 \
      --block_size 512 \
      --n_head 16 \
      --out_dir /sphinx/u/houjun/checkpoints/fork/jax/pretrain \
      --per_device_batch_size 4 \
      --validation_interval 512 \
      --checkpoint_interval 10240 \
      --validation_steps 2048 \
      --report_interval 8 \
      --lr 2.5e-4 \
      --shard_into 1 \
      --total_steps 10000 \
      --wandb
  "
'
