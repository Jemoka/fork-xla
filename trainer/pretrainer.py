# common standard library utilities
import os
from pathlib import Path
import sys
import time
import json
import math
import shutil
import random
from random import Random

from argparse import Namespace
from contextlib import contextmanager

# machine learning and data utilities
import numpy as np
import pandas as pd

import jax
from jax import config
config.update("jax_default_matmul_precision", "bfloat16")
import jax.numpy as jnp
from jax import random as jax_random
import flax
from flax.training import train_state, checkpoints, orbax_utils
from flax import struct
import optax
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# MLOps
import wandb

# logging
from loguru import logger

# our stuff
from model import *
from utils import plot_logger, parse_dataset_spec

from jax.experimental import multihost_utils

R = Random(7)

class Pretrainer:
    def device_count(self):
        return jax.device_count()

    def main_process(self):
        return jax.process_index() == 0

    @staticmethod
    def find_accumulation_steps(
            global_batch_size: int,
            max_batch_size: int,
            dp_replicate: int
    ):
        for batch_size in reversed(range(1, max_batch_size + 1)):
            if global_batch_size % (batch_size * dp_replicate) == 0:
                return batch_size, global_batch_size // (batch_size * dp_replicate)
        raise ValueError(f"No grad_acc found for global_batch_size {global_batch_size} and max_batch_size {max_batch_size} and dp_replicate {dp_replicate}")

    def __init__(self, args, run_id=None, distributed=False):
        # set up the trainer
        self.args = args

        # If we are distribtued, initialize
        if distributed:
            self.distributed = True
        
        self.per_device_batch_size = args.per_device_batch_size

        # Compute replication mesh setup
        devs = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
        local = jax.local_device_count()
        devices = np.array(devs).reshape(-1, local)
        devices = devices.reshape(-1, args.shard_into)

        self.mesh = Mesh(devices, axis_names=("batch", "shard")) # sharded data parallel

        logger.info("DEVICES | host: {}/{} | topology: {}, mesh: {}, count: {}", jax.process_index(), jax.process_count(), devices.shape, self.mesh, jax.device_count())

        self.replicas = jax.device_count() // args.shard_into
        self.local_replicas = local // args.shard_into
        assert self.replicas == self.mesh.shape["batch"]
        assert self.local_replicas * jax.process_count() == self.replicas

        # compute how much to accumulate
        self.per_device_batch_size, self.accumulate_steps = self.find_accumulation_steps(
            args.batch_size,
            self.per_device_batch_size,
            self.replicas
        )

        # scale *up* the total steps as a function of how many steps to accumulate
        self.total_batches = args.total_steps * self.accumulate_steps

        # print statistics
        if self.main_process():
            logger.info(
                "BATCHING | {} batchsize/node * ({} local * {} prox = {} dp) * {} accumulation = {} batchsize",
                self.per_device_batch_size,
                self.local_replicas,
                jax.process_count(),
                self.replicas,
                self.accumulate_steps,
                args.batch_size,
            )
            logger.info(
                "STEPS | {} micro batches // {} accumulation = {} steps",
                self.total_batches,
                self.accumulate_steps,
                self.total_batches // self.accumulate_steps
            )
            logger.info(
                "TOKENS | {} steps * {} batchsize * {} blocksize = {} tokens",
                self.total_batches // self.accumulate_steps,
                args.batch_size,
                args.block_size,
                (self.total_batches // self.accumulate_steps)*
                args.batch_size*
                args.block_size
            )

        # initialize wandb
        if self.main_process():
            wandb.init(
                project="fork-xla",
                config=vars(args),
                mode=None if args.wandb else "disabled",
                name=args.experiment,
                resume="allow",
                id=run_id,
            )

        # set up logger that's noop on non-main process
        def log(*args, **kwargs):
            if self.main_process():
                wandb.log(*args, **kwargs)

        self.plot, self.get_plots = plot_logger(logger=log, args=self.args)

        # ...and the output path
        save_dir = Path(args.out_dir) / args.experiment
        save_dir.mkdir(parents=True, exist_ok=True)

        self.save_dir = save_dir / "checkpoint"
        self.recovery_dir = str(save_dir / "recovery")
        self.best_dir = str(save_dir / "best")

        # Initialize model
        self.model = Thoughtbubbles(args)

        # Initialize random keys
        self.key = jax_random.PRNGKey(0)

        # raise if we are applying dropout, pretty sure the semantics of
        # dropout key is not implemented correctly
        if args.dropout > 0.0:
            logger.warning("Dropout stochacisity may not have been implemented correctly...")

        # Initialize model parameters with dummy input
        self.key, init_key, self.dropout_key = jax_random.split(self.key, num=3)
        dummy_input = jnp.ones((1, args.block_size), dtype=jnp.int32)
        variables = self.model.init(init_key, dummy_input, deterministic=True)
        params = variables['params']

        # Create optimizer
        if self.args.optimizer == "adamw":
            self.tx = self.model.configure_optimizers_adamw(
                weight_decay=args.weight_decay,
                learning_rate=args.lr,
                betas=(args.beta1, args.beta2),
                device_type="gpu" if jax.devices()[0].platform == "gpu" else "tpu",
            )
            if self.main_process():
                logger.info(f"OPTIMIZER | using AdamW")
        else:
            raise RuntimeError("Sadly I haven't ported muon yet mmmmm...")

        # Create learning rate schedule (WSD: Warmup-Stable-Decay)
        warmup_steps = int((self.total_batches // self.accumulate_steps) * self.args.warmup_pct)
        decay_steps = int((self.total_batches // self.accumulate_steps) * self.args.decay_pct)
        stable_steps = (self.total_batches // self.accumulate_steps) - warmup_steps - decay_steps

        warmup_schedule = optax.linear_schedule(
            init_value=args.lr * 0.01,
            end_value=args.lr,
            transition_steps=warmup_steps
        )
        stable_schedule = optax.constant_schedule(args.lr)
        decay_schedule = optax.linear_schedule(
            init_value=args.lr,
            end_value=args.lr * 0.01,
            transition_steps=decay_steps
        )

        self.schedule = optax.join_schedules(
            schedules=[warmup_schedule, stable_schedule, decay_schedule],
            boundaries=[warmup_steps, warmup_steps + stable_steps]
        )

        # Update optimizer with schedule
        self.tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_schedule(lambda count: self.schedule(count)),
            self.tx
        )

        # Create training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.tx
        )

        # define sharding spces for data and parameters
        # (accumulate, batch, seq_len)
        # replicate accumulate steps, shard along "batch" mesh axes, replicate seq length
        self.data_sharding = NamedSharding(self.mesh, P(None, "batch", None))
        self.state_sharding = flax.linen.logical_to_mesh_sharding(
            flax.linen.get_partition_spec(self.state), self.mesh, rules=SHARDING_PLAN
        )

        # Parameters goes to TPUs
        self.state = jax.device_put(self.state, self.state_sharding)

        self.total_params = sum(x.size for x in jax.tree_util.tree_leaves(self.state.params)) / 1e6
        if self.main_process():
            logger.info(f"MODEL | Total Parameters: {self.total_params:.2f}m")

        # compute training size + the counter (useful for mid-checkpoint recovery)
        self.global_step_counter_ = 0
        self.best_val_score_ = float("-inf")  # "score" means higher is better

        # configure dataset
        if vars(args).get("data_file") is None:
            from flywheel import Sampling, Strategy, MemmapDataset
            # legacy data format, single non-mixture data
            self.data_strategy = Strategy(args, [
                Sampling(MemmapDataset(args, args.data_dir), 1.0)
            ])
        else:
            self.data_strategy = parse_dataset_spec(args.data_file, args)

        # weeeeeeeeeeee
        # print the model
        if self.main_process():
            logger.info(self.model)

        # dataloader cache
        self.async_dl_cache = {}

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # Chowdhery et al., 2022
        PER_MLP_COST = 6
        PER_ATTN_COST = 12

        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        # Count parameters
        N = self.total_params
        cfg = self.args
        L, H, Q, T = len(cfg.plan), cfg.n_head, cfg.n_embd // cfg.n_head, (cfg.max_block_size
                                                                           if "fork" in cfg.plan
                                                                           else cfg.block_size)
        flops_per_token = PER_MLP_COST * N + PER_ATTN_COST * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = self.args.flops_promised * self.device_count()
        mfu = flops_achieved / flops_promised
        return mfu, flops_achieved

    def train(self):
        if not Path(self.recovery_dir).exists():
            if self.main_process():
                logger.info("Building recovery checkpoint...")
            self.save(self.recovery_dir)
        if self.main_process():
            logger.info("Beginning training...")
        # try:
        self.epoch()
        # except Exception as e:
        #     raise e
        #     if self.main_process():
        #         logger.info(f"TRAIN | FAILURE | building recovery checkpoint for {e}")
        #     try:
        #         # move recovery checkpoint to self.recovery_dir+"_last_good"
        #         if self.main_process():
        #             shutil.rmtree(self.recovery_dir.removesuffix("/")+"_last_good", ignore_errors=True)
        #             shutil.move(self.recovery_dir, self.recovery_dir.removesuffix("/")+"_last_good")
        #         self.save(self.recovery_dir)
        #         if self.main_process():
        #             logger.error(f"TRAIN | FAILURE | Encountered exception {str(e)}; safely checkpointed, so we're blowing up now...")
        #     except Exception as es:
        #         if self.main_process():
        #             logger.error(f"TRAIN | FAILURE | Encountered exception {str(e)}; CHECKPOINT FAILED with '{str(es)}', but eh, so we're blowing up anyways...")
        #         raise e
        #     raise e
        self.finish()

    def finish(self):
        pass  # noop

    def batch(self, slice="train", deterministic_key=None):
        if slice == "train":
            strategy = self.async_dl_cache.get("train")
            if not strategy:
                self.async_dl_cache["train"] = self.data_strategy.get_async_batches(
                    self.per_device_batch_size *
                    self.local_replicas *
                    self.accumulate_steps,
                    slice
                )
                strategy = self.async_dl_cache["train"]

            x, y, padding_mask = strategy.get_batch()
        else:
            # we will load a number of samples divisble by per_device_batch_size
            # so that we can reshape it so + enable batched loads
            x, y, padding_mask = self.data_strategy.get_batch(
                (self.args.validation_steps//(self.per_device_batch_size*self.local_replicas))*
                (self.per_device_batch_size*self.local_replicas),
                slice,
                deterministic_key=deterministic_key,
            )

        return x, y, padding_mask


    def make_train_step(self):

        def train_step_inner(state, batch, dropout_key, accumulate_steps):

            def train_eval(state, batch, dropout_key, accumulate_steps):
                """batch -> gradient"""
                x, y, padding_mask = batch

                def loss_fn(params):
                    logits, loss = state.apply_fn(
                        {'params': params},
                        x, y,
                        deterministic=False,
                        rngs={'dropout': dropout_key}
                    )
                    return loss / accumulate_steps

                loss, grads = jax.value_and_grad(loss_fn)(state.params)

                return loss, grads

            def reduce(carry, batch):
                """(gradient, loss) -> batch -> ((gradient_acc, loss_acc), None)"""

                grad, loss = carry
                loss_single, grad_single = train_eval(state, batch, dropout_key, accumulate_steps)

                grad_acc = jax.tree_util.tree_map(lambda a, g: a + g, grad, grad_single)
                loss_acc = loss + loss_single

                return (grad_acc, loss_acc), None

            grad_zero = jax.tree_util.tree_map(jnp.zeros_like, state.params)
            (grad_sum, loss_sum), _ = jax.lax.scan(reduce, (grad_zero, 0.0), batch)

            state = state.apply_gradients(grads=grad_sum)

            return state, loss_sum

        return jax.jit(
            train_step_inner,
            in_shardings=(self.state_sharding, self.data_sharding, None, None),
            out_shardings=(self.state_sharding, None),
        )

    def make_valid_step(self):
        x, y, padding_mask = self.batch("val", deterministic_key=32)

        # reshape into (steps, per_device_bs*replicas, seq_len) and shard the middle axis
        x = x.reshape(-1, self.per_device_batch_size*self.local_replicas, x.shape[-1])
        y = y.reshape(-1, self.per_device_batch_size*self.local_replicas, y.shape[-1])
        padding_mask = padding_mask.reshape(-1, self.per_device_batch_size*self.local_replicas, padding_mask.shape[-1])
        x, y, padding_mask = jax.device_put(
            (x, y, padding_mask),
            self.data_sharding
        )

        # because batch is fixed we should be jitting the inner function

        def valid_step_inner(state, batch):
            x, y, padding_mask = batch  # x: (S, B_local, T)

            def reduce(carry, xb):
                loss_sum, count = carry
                x_i, y_i, mask_i = xb  # (B_local, T)

                _, loss_i = state.apply_fn({'params': state.params}, x_i, y_i, deterministic=True)
                n = x_i.shape[0] * x_i.shape[1]
                return (loss_sum + loss_i * n, count + n), None

            (loss_sum, count), _ = jax.lax.scan(reduce, (0.0, 0), (x, y, padding_mask))

            return loss_sum, count

        valid_step_inner_jit = jax.jit(
            valid_step_inner,
            in_shardings=(self.state_sharding, self.data_sharding),
            out_shardings=(None, None),
        )


        def valid_step_wrapper(state):
            loss_sum, count = valid_step_inner_jit(state, (x, y, padding_mask))
            loss_sum = jax.device_get(loss_sum)
            count    = jax.device_get(count)

            # if these come back as per-device arrays, reduce them here
            loss_sum_local = float(jnp.sum(loss_sum))
            count_local    = float(jnp.sum(count))

            loss_sum_g = multihost_utils.process_allgather(jnp.asarray(loss_sum_local))
            count_g    = multihost_utils.process_allgather(jnp.asarray(count_local))

            loss = float(jnp.sum(loss_sum_g) / jnp.sum(count_g))

            score = 1/loss
            metrics = {"val/loss": loss, "val/score": score}

            return score, metrics

        return valid_step_wrapper

    def epoch(self):
        if self.main_process():
            logger.info("BEGIN EPOCH")

        mfu_measurement_step_counter = 0
        mfu_start = time.time()

        # JIT compile + shard train step across workers 
        train_step = self.make_train_step()
        valid_step = self.make_valid_step()

        grads_accum = None

        # because sometimes the load function may skip some epochs
        for indx in range(
                self.global_step_counter_,
                self.total_batches + 1,
                self.accumulate_steps
        ):
            logger.debug("DATA | {} | START", indx)
            x, y, padding_mask = self.batch()
            x = x.reshape(
                -1,
                self.local_replicas*self.per_device_batch_size,
                self.args.block_size
            )
            y = y.reshape(
                -1,
                self.local_replicas*self.per_device_batch_size,
                self.args.block_size
            )
            padding_mask = padding_mask.reshape(
                -1,
                self.local_replicas*self.per_device_batch_size,
                self.args.block_size
            )
            logger.debug("DATA | {} | PLACING", indx)

            batch = jax.device_put(
                (x, y, padding_mask),
                self.data_sharding

            )
            logger.debug("DATA | {} | PLACED", indx)

            self.state, loss = train_step(self.state, batch, self.dropout_key, self.accumulate_steps)
            logger.debug("COMPUTATION | {} | FINISHED", indx)
            train_metrics = {}

            # Get current learning rate
            mfu_measurement_step_counter += self.accumulate_steps

            # perform logging, and then increment
            if (
                (indx % self.accumulate_steps == 0)
                and (indx // self.accumulate_steps)
                % self.args.report_interval
                == 0
                and indx != 0
            ):
                multihost_utils.sync_global_devices("report:pre")
                train_metrics["train/lr"] = float(self.schedule(self.state.step))
                loss_val = float(loss)
                if mfu_measurement_step_counter > 0 and self.args.estimate_mfu:
                    mfu, flops = self.estimate_mfu(
                        mfu_measurement_step_counter
                        * self.per_device_batch_size
                        * self.replicas,
                        time.time() - mfu_start,
                    )
                    mfu_start = time.time()
                    mfu_measurement_step_counter = 0
                    train_metrics["train/mfu"] = mfu
                    train_metrics["train/flops"] = flops

                if self.main_process():
                    train_metrics["train/tokens"] = (
                        (((indx+1) // self.accumulate_steps)*
                        self.args.batch_size*self.args.block_size)
                    )
                    train_metrics["train/loss"] = loss_val

                    wandb.log(
                        train_metrics,
                        step=indx // self.accumulate_steps,
                    )
                    logger.info(
                        "TRAIN | {}/{} | loss {}",
                        indx // self.accumulate_steps,
                        self.total_batches // self.accumulate_steps,
                        loss_val,
                    )
                multihost_utils.sync_global_devices("report:post")

            if (indx % self.accumulate_steps == 0):
                self.global_step_counter_ += self.accumulate_steps

            if self.main_process():
                logger.debug("STEP | {} | {}", indx, train_metrics)

            # save a checkpoint, if needed
            if (
                    indx != 0 and
                    indx % self.accumulate_steps == 0 and
                    (indx // self.accumulate_steps)
                    % self.args.checkpoint_interval
                    == (self.args.checkpoint_interval//2) # offset checkpoint to not crash with val
            ):
                self.save(str(self.save_dir / str(indx // self.accumulate_steps)))

            # perform validation and save a checkpoint, if needed
            if (
                    indx != 0 and
                    indx % self.accumulate_steps == 0 and
                    (indx // self.accumulate_steps)
                    % self.args.validation_interval
                    == (self.args.validation_interval//3) # so we don't ovelap with checkpoint
            ):
                score, val_metrics = valid_step(self.state)
                val_metrics["train/tokens"] = (
                    (((indx+1) // self.accumulate_steps)*
                     self.args.batch_size*self.args.block_size)
                )
                if self.main_process():
                    wandb.log(
                        val_metrics,
                        step=indx // self.accumulate_steps,
                    )
                    logger.info(
                        "VAL | {} | score {}",
                        indx // self.accumulate_steps,
                        score,
                    )

                if score > self.best_val_score_:
                    if self.main_process():
                        logger.info("VAL | BEST SCORE | score {}", score)
                    self.best_val_score_ = score
                    self.save(self.best_dir)

    def load(self, path):
        logger.debug("CHECKPOINT | loading checkpoint from {}", path)

        # Load random state
        rng_state = np.load(os.path.join(path, "rng.npy"), allow_pickle=True).item()
        random.setstate(rng_state["python_random"])
        np.random.set_state(rng_state["numpy_random"])
        self.key = jax_random.PRNGKey(rng_state["jax_random"])

        # Load checkpoint using Orbax
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(os.path.join(path, "checkpoint"), target=jax.device_get(self.state))
        self.state = jax.device_put(restored, self.state_sharding)

        # Load config
        with open(os.path.join(path, "config.json"), "r") as df:
            data = json.load(df)

        self.args = Namespace(**data.get("config", {}))
        self.global_step_counter_ = data.get("steps", 0)
        self.best_val_score_ = data.get("score", 0)

    def save(self, path):
        logger.debug("CHECKPOINT | saving checkpoint at {}", path)

        multihost_utils.sync_global_devices("save:pre")

        # Write to fast local temp, then move to final (possibly slow) path
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), f"checkpoint_{os.path.basename(path)}_{time.time()}")

        if self.main_process():
            os.makedirs(temp_path, exist_ok=True)
            logger.debug("CHECKPOINT | created checkpoint directory")

            # Save random state
            rng_state = {
                "python_random": random.getstate(),
                "numpy_random": np.random.get_state(),
                "jax_random": int(self.key[0]),  # Save seed
            }
            np.save(os.path.join(temp_path, "rng.npy"), rng_state)
            logger.debug("CHECKPOINT | saved random state")

        # Save checkpoint
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(
            os.path.join(temp_path, "checkpoint"),
            jax.device_get(self.state),
            force=True
        )
        checkpointer.wait_until_finished()
        logger.debug("CHECKPOINT | saved training state")

        if self.main_process():
            # Save config
            with open(os.path.join(temp_path, "config.json"), "w") as df:
                json.dump(
                    {
                        "config": vars(self.args),
                        "steps": self.global_step_counter_,
                        "score": self.best_val_score_,
                        "wandb": wandb.run.id if self.args.wandb else None,
                    },
                    df,
                )
            logger.debug("CHECKPOINT | saved configuration")

        multihost_utils.sync_global_devices("save:post")

        # Main process copies from temp to final location (cross-device safe)
        if self.main_process():
            shutil.rmtree(path, ignore_errors=True)
            shutil.copytree(temp_path, path, dirs_exist_ok=True, ignore_dangling_symlinks=True)
            shutil.rmtree(temp_path, ignore_errors=True)
            logger.debug("CHECKPOINT | moved to {}", path)

    @classmethod
    def from_pretrained(cls, path, disable_wandb=True, distributed=False):
        with open(os.path.join(path, "config.json"), "r") as df:
            data = json.load(df)
        args = Namespace(**data.get("config", {}))
        args.wandb = False if disable_wandb else args.wandb
        new = cls(args, run_id=data.get("wandb"), distributed=distributed)
        new.load(path)

        if disable_wandb:
            new.args.wandb = False

        return new
