#!/bin/bash

#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --job-name=houjun-forking-final_pretrain_1_9b_baseline
#SBATCH --nodes=1-2
#SBATCH --gpus-per-node=6
#SBATCH --ntasks-per-node=1
#SBATCH --requeue
#SBATCH --mem=64G
#SBATCH --time=14-0
#SBATCH --nodelist=sphinx11,sphinx10
#SBATCH --open-mode=append
#SBATCH --output=./logs/final_pretrain_1_9b_baseline.log

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
    torchrun --rdzv-backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      --nnodes $WORLD_SIZE \
      --nproc-per-node 2 \
      main.py \
      final_pretrain_1_9b_baseline \
      --warm_start /sphinx/u/houjun/checkpoints/fork/final_pretrain_1_9b_baseline/recovery \
      --data_file /juice2/scr2/houjun/fork/experiments/data/pretrain.toml \
      --plan $(printf "regular %.0s" {1..36}) \
      --flops_promised 312e12 \
      --block_size 512 \
      --max_block_size 1024 \
      --per_device_batch_size 18 \
      --n_head 16 \
      --n_embd 2048 \
      --out_dir /sphinx/u/houjun/checkpoints/fork \
      --per_device_batch_size 20 \
      --validation_interval 2048 \
      --checkpoint_interval 40960 \
      --plot_interval 10240 \
      --wandb
  "
'
