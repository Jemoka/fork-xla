#!/bin/bash

#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --job-name=houjun-forking-eval_pretrain_1_9B_regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --requeue
#SBATCH --mem=128G
#SBATCH --time=14-0
#SBATCH --nodelist=sphinx3,sphinx4,sphinx5,sphinx6,sphinx7,sphinx8,sphinx9,sphinx10,sphinx11
#SBATCH --open-mode=append
#SBATCH --output=./logs/eval_pretrain_1_9B_regular.log

set -euo pipefail

cd .

MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
MASTER_PORT=8316
WORLD_SIZE="$SLURM_NTASKS"

export MASTER_ADDR MASTER_PORT WORLD_SIZE

# Default checkpoint path - override with environment variable CHECKPOINT_PATH
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/sphinx/u/houjun/checkpoints/fork/jax/pretrain/final_pretrain_1_9b_regular/best}"

# Optional: specify evals (space-separated), or leave empty for all
EVALS="${EVALS:-}"

# Output file for results
OUTPUT_FILE="${OUTPUT_FILE:-/juice2/scr2/houjun/fork-xla/output/eval_pretrain_1_9B_regular_results.json}"

srun --export=ALL bash -lc '
  set -euo pipefail

  # Pick first Ethernet NIC (en*)
  export NCCL_SOCKET_IFNAME="$(
    ip -o link show | awk -F": " '"'"'$2 ~ /^en/ { print $2; exit }'"'"'
  )"

  export RANK="$SLURM_PROCID"
  export LOCAL_RANK="$SLURM_LOCALID"

  echo "Hello from $(hostname): NCCL_IFACE=$NCCL_SOCKET_IFNAME LOCAL_RANK=$LOCAL_RANK"

  # Build eval flags if EVALS is set
  EVAL_FLAGS=""
  if [ -n "'"$EVALS"'" ]; then
    for eval in '"$EVALS"'; do
      EVAL_FLAGS="$EVAL_FLAGS --evals $eval"
    done
  fi

  ./experiments/scripts/develop "
    source .venv/bin/activate && \
    uv run python \
      evaluate.py \
      --checkpoint '"$CHECKPOINT_PATH"' \
      --encoding gpt2 \
      --truncate \
      --output '"$OUTPUT_FILE"' \
      --shard-into 1 \
      --per-device-batch-size 2 \
      $EVAL_FLAGS
  "
'
