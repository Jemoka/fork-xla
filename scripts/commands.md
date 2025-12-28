# setup

```bash
gcloud compute tpus tpu-vm ssh worker0 \
  --zone=us-east1-d \
  --worker=all \
  --command="curl -fsSL https://raw.githubusercontent.com/Jemoka/fork-xla/refs/heads/master/scripts/setup.sh | bash"
```

# run

```bash
gcloud compute tpus tpu-vm ssh node-1 \
  --zone us-central2-b \
  --worker=all \
  --command="bash -lc '
set -euo pipefail

cd ~/fork-xla

nohup uv run main.py final_pretrain_1_9b_baseline \
  --distributed \
  --warm_start /home/houjun/checkpoints/final_pretrain_1_9b_baseline/recovery \
  --data_file /home/houjun/data/recipes/pretrain.toml \
  --plan regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular \
  --flops_promised 275e12 \
  --block_size 512 \
  --n_head 16 \
  --out_dir /home/houjun/checkpoints \
  --validation_interval 2048 \
  --checkpoint_interval 40960 \
  --per_device_batch_size 128 \
  --validation_steps 2048 \
  --shard_into 4 \
  --report_interval 32 \
  --wandb \
  > ~/final_pretrain_1_9b_baseline.log 2>&1 &

disown
'"
```

