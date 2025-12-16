#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --partition={partition}
#SBATCH --output={logdir}/%j_%t_log.out
#SBATCH --error={logdir}/%j_%t_log.err
#SBATCH --signal=SIGUSR2@300
#SBATCH --propagate=NONE
#SBATCH --no-requeue
#SBATCH --open-mode=append
{account_line}{qos_line}{constraint_line}{mem_line}

cd /juice2/scr2/houjun/fork/

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT={master_port}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export UV_PATH=/sailhome/houjun/.cargo/bin/uv

cd .

echo "HOSTNAME: $HOSTNAME"
srun --ntasks=$SLURM_NNODES --nodes=$SLURM_NNODES --exact \
    --output={logdir}/%j_%N_%t.out \
    --error={logdir}/%j_%N_%t.err \
    --ntasks-per-node=1 ls /sphinx/u/houjun/stages/fork && \
    cd /sphinx/u/houjun/stages/fork && \
    source .venv/bin/activate && \
    torchrun  \
    --nproc_per_node={gpus_per_node}  \
    --nnodes=$SLURM_NNODES  \
    --node_rank=$SLURM_PROCID  \
    --rdzv_backend=c10d  \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT  \
    --max_restarts=0 \
    {script_path} {script_args}
 
