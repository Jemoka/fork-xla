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

R = Random(7)

class Trainer:
    def device_count(self):
        return jax.device_count()

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
            jax.distributed.initialize()
        
        self.per_device_batch_size = args.per_device_batch_size

        # Compute replication mesh setup
        devices = np.array(jax.devices())

        self.ddp_mesh = Mesh(devices, axis_names=("batch",)) # DDP sharding, so batch dim goes to each device
        self.replicas = devices.shape[0]

        # compute how much to accumulate
        self.per_device_batch_size, self.accumulate_steps = self.find_accumulation_steps(
            args.batch_size,
            self.per_device_batch_size,
            self.replicas
        )

        # scale *up* the total steps as a function of how many steps to accumulate
        self.total_batches = args.total_steps * self.accumulate_steps

        # print statistics
        logger.info(
            "BATCHING | {} batchsize/node * {} chips * {} accumulation = {} batchsize",
            self.per_device_batch_size,
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
        wandb.init(
            project="fork",
            config=vars(args),
            mode=None if args.wandb else "disabled",
            name=args.experiment,
            resume="allow",
            id=run_id,
        )

        # set up logger that's noop on main process
        def log(*args, **kwargs):
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

        total = sum(x.size for x in jax.tree_util.tree_leaves(self.state.params)) / 1e6
        logger.info(f"MODEL | Total Parameters: {total:.2f}m")

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
        logger.info(self.model)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # Chowdhery et al., 2022
        PER_MLP_COST = 6
        PER_ATTN_COST = 12

        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        # Count parameters
        N = sum(x.size for x in jax.tree_leaves(self.state.params))
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
            logger.info("Building recovery checkpoint...")
            self.save(self.recovery_dir)
        logger.info("Beginning training...")
        try:
            self.epoch()
        except Exception as e:
            logger.info(f"TRAIN | FAILURE | building recovery checkpoint")
            try:
                # move recovery checkpoint to self.recovery_dir+"_last_good"
                shutil.rmtree(self.recovery_dir.removesuffix("/")+"_last_good", ignore_errors=True)
                shutil.move(self.recovery_dir, self.recovery_dir.removesuffix("/")+"_last_good")
                self.save(self.recovery_dir)
                logger.error(f"TRAIN | FAILURE | Encountered exception {str(e)}; safely checkpointed, so we're blowing up now...")
            except Exception as es:
                logger.error(f"TRAIN | FAILURE | Encountered exception {str(e)}; CHECKPOINT FAILED with '{str(es)}', but eh, so we're blowing up anyways...")
                raise e
            raise e
        self.finish()

    def finish(self):
        pass  # noop

    def val(self):
        losses = []
        scores = []
        count = 0
        for i in range(
            0, vars(self.args).get("validation_steps", 256), self.per_device_batch_size
        ):
            # get a batch and inference
            x, y = self.batch("val", deterministic_key=i)

            # Run validation step
            logits, loss = self.state.apply_fn(
                {'params': self.state.params},
                x, y,
                deterministic=True
            )

            # for pretraining, loss is just the inverse of the loss
            losses.append(float(loss))
            scores.append((1 / float(loss)) * x.shape[0])
            count += x.shape[0]

        loss = sum(losses) / len(losses)
        score = sum(scores) / count
        metrics = {"val/loss": loss, "val/score": score}

        return score, metrics

    def batch(self, slice="train", deterministic_key=None, accumulate=False):
        accumulate_multiplier = self.accumulate_steps if accumulate else 1
        x, y = self.data_strategy.get_batch(
            (self.per_device_batch_size*
             self.device_count()* # because we will then shard it
             accumulate_multiplier), # because we will then accumulate it
            slice,
            deterministic_key=deterministic_key,
        )

        return x, y


    @staticmethod
    def train_step(state, batch, dropout_key, accumulate_steps):

        def train_eval(state, batch, dropout_key, accumulate_steps):
            """batch -> gradient"""
            x, y = batch

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
       
    
    def epoch(self):
        logger.info("BEGIN EPOCH")

        mfu_measurement_step_counter = 0
        mfu_start = time.time()

        # JIT compile + shard train step across workers 
        train_step = jax.jit(
            self.train_step,
            in_shardings=(None, NamedSharding(self.ddp_mesh, P(None, "batch", None)), None, None),
            out_shardings=(None, None),
        )

        grads_accum = None

        # because sometimes the load function may skip some epochs
        for indx in range(
            self.global_step_counter_*self.accumulate_steps,
            self.total_batches + 1,
            self.accumulate_steps
        ):
            x,y = self.batch(accumulate=True)
            x = x.reshape(
                -1,
                self.device_count()*self.per_device_batch_size,
                self.args.block_size
            )
            y = y.reshape(
                -1,
                self.device_count()*self.per_device_batch_size,
                self.args.block_size
            )

            batch = jax.device_put(
                (x,y),
                # accumulate steps, batchwise, seq length
                NamedSharding(self.ddp_mesh, P(None, "batch", None))
            )

            state, loss = train_step(self.state, batch, self.dropout_key, self.accumulate_steps)
            train_metrics = {"train/loss": float(loss)}

            # Get current learning rate
            train_metrics["train/lr"] = float(self.schedule(self.state.step))
            mfu_measurement_step_counter += 1

            # perform logging, and then increment
            if (
                (indx % self.accumulate_steps == 0)
                and (indx // self.accumulate_steps)
                % self.args.report_interval
                == 0
                and indx != 0
            ):
                if mfu_measurement_step_counter > 0:
                    mfu, flops = self.estimate_mfu(
                        mfu_measurement_step_counter
                        * self.per_device_batch_size
                        * self.device_count(),
                        time.time() - mfu_start,
                    )
                    mfu_start = time.time()
                    mfu_measurement_step_counter = 0
                    train_metrics["train/mfu"] = mfu
                    train_metrics["train/flops"] = flops

                train_metrics["train/tokens"] = (
                    (((indx+1) // self.accumulate_steps)*
                     self.args.batch_size*self.args.block_size)
                )

                wandb.log(
                    train_metrics,
                    step=indx // self.accumulate_steps,
                )
                logger.info(
                    "TRAIN | {}/{} | loss {}",
                    indx // self.accumulate_steps,
                    self.total_batches // self.accumulate_steps,
                    float(loss),
                )
            if (indx % self.accumulate_steps == 0):
                self.global_step_counter_ += 1

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
                    == 0
            ):
                score, val_metrics = self.val()
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
        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(os.path.join(path, "checkpoint"))
        self.state = restored["state"]

        # Load config
        with open(os.path.join(path, "config.json"), "r") as df:
            data = json.load(df)

        self.args = Namespace(**data.get("config", {}))
        self.global_step_counter_ = data.get("steps", 0)
        self.best_val_score_ = data.get("score", 0)

    def save(self, path):
        logger.debug("CHECKPOINT | saving checkpoint at {}", path)

        os.makedirs(path, exist_ok=True)

        # Save random state
        rng_state = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "jax_random": int(self.key[0]),  # Save seed
        }
        np.save(os.path.join(path, "rng.npy"), rng_state)

        # Save checkpoint 
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(
            os.path.join(path, "checkpoint"),
            {'state': jax.device_get(self.state)},
            force=True
        )

        # Save config
        with open(os.path.join(path, "config.json"), "w") as df:
            json.dump(
                {
                    "config": vars(self.args),
                    "steps": self.global_step_counter_,
                    "score": self.best_val_score_,
                    "wandb": wandb.run.id if self.args.wandb else None,
                },
                df,
            )

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
