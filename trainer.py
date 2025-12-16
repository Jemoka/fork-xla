# common standard library utilities
import os
from pathlib import Path
PTXAS_PATH = "/usr/local/cuda/bin/ptxas"

if os.path.exists(PTXAS_PATH):
    os.environ["TRITON_PTXAS_PATH"] = PTXAS_PATH

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.filesystem import FileSystemWriter

from torch.optim.lr_scheduler import SequentialLR, LinearLR

# huggingface
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

# MLOps
import wandb

# logging
from loguru import logger

# our stuff
from model import *
from data import *
from utils import plot_logger, parse_dataset_spec

R = Random(7)

class Checkpoint(Stateful):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self):
        model, optim = get_state_dict(self.model, self.optimizer)
        return {
            "model": model,
            "optimizer": optim,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"]
        )
        self.scheduler.load_state_dict(state_dict["scheduler"])


class Trainer:
    def main_process(self):
        if not self.distributed:
            return True
        else:
            return (
                torch.distributed.get_rank() == 0
                and int(os.environ.get("LOCAL_RANK", -1)) == 0
            )

    def world_size(self):
        if self.distributed:
            return dist.get_world_size()
        else:
            return 1

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

    def __init__(self, args, run_id=None, compile=False):
        # set up the trainer
        self.args = args

        # Flag to check if we are running in a distributed environment
        self.distributed = False
        # self.accumulate_steps = args.accumulate_steps
        self.per_device_batch_size = args.per_device_batch_size
        replicas = 1

        # scale down effective batch size to accumulate size per device
        if os.environ.get("LOCAL_RANK"):
            dist.init_process_group("cpu:gloo,cuda:nccl")
            self.distributed = True
            replicas = dist.get_world_size()

        # compute how much to accumulate
        self.per_device_batch_size, self.accumulate_steps = self.find_accumulation_steps(
            args.batch_size,
            self.per_device_batch_size,
            replicas
        )

        # scale *up* the total steps as a function of how many steps to accumulate
        self.total_batches = args.total_steps * self.accumulate_steps

        # print statistics
        if self.main_process():
            logger.info(
                "BATCHING | {} batchsize/node * {} nodes * {} accumulation = {} batchsize",
                self.per_device_batch_size,
                replicas,
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
                project="fork",
                config=vars(args),
                mode=None if args.wandb else "disabled",
                name=args.experiment,
                resume="allow",
                id=run_id,
            )

        # set up logger that's noop on main process
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

        self.model = Thoughtbubbles(args)

        if self.args.optimizer == "adamw":
            self.optim = self.model.configure_optimizers_adamw(
                weight_decay=args.weight_decay,
                learning_rate=args.lr,
                betas=(args.beta1, args.beta2),
                device_type="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            logger.warning("The current muon optimizer has a DPP distributed checkpointing bug in it, I haven't fixed it yet but beware you can't load the optimizer back under DDP for some reason.")
            self.optim = self.model.configure_optimizers_muon(self.distributed)

        if self.distributed:
            # could switch to FSDP, as needed
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model.to(self.device),
                find_unused_parameters=False
            )
        else:
            self.model = self.model.to(self.device)

        # scheduler: WSD (Warmup-Stable-Decay)
        warmup_steps = int((self.total_batches // self.accumulate_steps) * self.args.warmup_pct)
        decay_steps = int((self.total_batches // self.accumulate_steps) * self.args.decay_pct)
        stable_steps = (self.total_batches // self.accumulate_steps) - warmup_steps - decay_steps

        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optim,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        stable = torch.optim.lr_scheduler.ConstantLR(
            self.optim,
            factor=1.0,
            total_iters=stable_steps,
        )
        decay = torch.optim.lr_scheduler.LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=decay_steps,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optim,
            schedulers=[warmup, stable, decay],
            milestones=[warmup_steps, warmup_steps + stable_steps],
        )

        # compute training size + the counter (useful for mid-checkpoint recovery)
        self.global_step_counter_ = 0
        self.best_val_score_ = float("-inf")  # "score" means higher is better
        self.__checkpoint_op = None

        if compile:
            # jit the model
            self.model.compile()
            self.args.compiled = True
            logger.info("MODEL | Initialized model with torch.compile!")
        else:
            self.args.compiled = False

        # configure dataset
        if vars(args).get("data_file") is None:
            from flywheel import Sampling, Strategy, MemmapDataset
            # legacy data format, single non-mixure data
            self.data_strategy = Strategy(args, [
                Sampling(MemmapDataset(args, args.data_dir), 1.0)
            ])
        else:
            self.data_strategy = parse_dataset_spec(args.data_file, args)

        # weeeeeeeeeeee
        if self.main_process():
            wandb.watch(self.model)
            # print the model
            logger.info(self.model)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # Chowdhery et al., 2022
        PER_MLP_COST = 6
        PER_ATTN_COST = 12

        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        if self.distributed:
            N = self.model.module.get_num_params()
        else:
            N = self.model.get_num_params()
        cfg = self.args
        L, H, Q, T = len(cfg.plan), cfg.n_head, cfg.n_embd // cfg.n_head, (cfg.max_block_size 
                                                                           if "fork" in cfg.plan 
                                                                           else cfg.block_size)
        flops_per_token = PER_MLP_COST * N + PER_ATTN_COST * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = self.args.flops_promised * self.world_size()
        mfu = flops_achieved / flops_promised
        return mfu, flops_achieved

    def _vomit(self):
        """recursively vomit every method and attribute of self into a namespace

        only useful if you are Jack and has a weird Jupyter setup. I apologise
        for the this abuse of all that's good about Python.
        """

        from types import SimpleNamespace

        ns = SimpleNamespace()

        from torch.nn import ModuleList

        buffer = [(i, self) for i in dir(self) if i[0] != "_"]
        names = []
        while len(buffer) > 0:
            member, head = buffer.pop(-1)
            attr = getattr(head, member)
            ns.__dict__[member] = attr

            def include(attr):
                for j in dir(attr):
                    if j not in names and j[0] != "_":
                        buffer.append((j, attr))
                        names.append(j)

            # some special rules for including things
            if isinstance(attr, ModuleList):
                for component in attr:
                    include(component)
            else:
                include(attr)

        return ns

    def train(self):
        logger.info("Waiting for everyone...")
        if self.distributed:
            dist.barrier()
        if not Path(self.recovery_dir).exists():
            logger.info("Building recovery checkpoint...")
            self.save(self.recovery_dir)
        logger.info("Beginning training...")
        try:
            self.epoch()
        except Exception as e:
            logger.info(f"TRAIN | FAILURE | building recovery checkpoint")
            # move recovery checkpoint to self.recovery_dir+"_last_good"
            shutil.rmtree(self.recovery_dir.removesuffix("/")+"_last_good", ignore_errors=True)
            shutil.move(self.recovery_dir, self.recovery_dir.removesuffix("/")+"_last_good")
            self.save(self.recovery_dir)
            logger.error(f"TRAIN | FAILURE | Encountered exception {str(e)}; safely checkpointed, so we're blowing up now...")
            raise e
        self.finish()

    def finish(self):
        pass  # noop

    def val(self):
        self.model.eval()
        losses = []
        scores = []
        count = 0
        for i in range(
            0, vars(self.args).get("validation_steps", 256), self.per_device_batch_size
        ):
            with torch.inference_mode():
                # get a batch and inference
                x, y = self.batch("val", deterministic_key=i)
                output, loss = self.model(x, y)

                # for pretraining, loss is just the inverse of the loss
                losses.append(loss.cpu().item())
                scores.append(((1 / loss) * x.size(0)).item())

            count += x.size(0)

        loss = sum(losses) / len(losses)
        score = sum(scores) / count
        metrics = {"val/loss": loss, "val/score": score}
        self.model.train()

        return score, metrics

    def batch(self, slice="train", deterministic_key=None):
        x, y = self.data_strategy.get_batch(
            self.per_device_batch_size,
            slice,
            self.device,
            deterministic_key=deterministic_key,
        )
        return x.long(), y.long()

    def epoch(self):
        if self.main_process():
            logger.info("BEGIN EPOCH")

        mfu_measurement_step_counter = 0
        mfu_start = time.time()

        # because sometimes the load function may skip some epochs
        for indx in range(
            self.global_step_counter_*self.accumulate_steps,
            self.total_batches + 1
        ):
            i = self.batch()

            # take a step, optionally with plotting
            if (indx % self.accumulate_steps == 0) and (
                    indx // self.accumulate_steps
            ) % self.args.plot_interval == 0:
                with self.plot(
                    indx // self.accumulate_steps,
                    debug=(not self.args.wandb),
                ):
                    loss, train_metrics = self.step(i, indx)
            else:
                loss, train_metrics = self.step(i, indx)
            # because the first one is *20 for embeddings
            train_metrics["train/lr"] = self.optim.param_groups[-1]["lr"]
            mfu_measurement_step_counter += 1

            # perform logging, and then increment
            # (we do this because global_step_counter_
            #  is useful not as the # of steps but how
            #  many we need to skip for warm start)
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
                        * self.world_size(),
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

                if self.main_process():
                    wandb.log(
                        train_metrics,
                        step=indx // self.accumulate_steps,
                    )
                    logger.info(
                        "TRAIN | {}/{} | loss {}",
                        indx // self.accumulate_steps,
                        self.total_batches // self.accumulate_steps,
                        loss,
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
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    score, val_metrics = self.val()
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

    def gradients(self, batch):
        x, y = batch

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            _, loss = self.model(x, y)

        (loss / self.accumulate_steps).backward()
        loss = loss.cpu().item()
        metrics = {"train/loss": loss}

        return loss, metrics

    def step(self, batch, indx):
        if indx % self.accumulate_steps == 0:
            loss, metrics = self.gradients(batch)
            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()
        else:
            if self.distributed:
                with self.model.no_sync():
                    loss, metrics = self.gradients(batch)
            else:
                loss, metrics = self.gradients(batch)

        return loss, metrics

    def load(self, path):
        logger.debug("CHECKPOINT | loading checkpoint from {}", path)

        if self.distributed:
            dist.barrier()

        state = torch.load(os.path.join(path, "rng.pt"), weights_only=False)
        random.setstate(state["python_random"])
        np.random.set_state(state["numpy_random"])
        torch.set_rng_state(state["torch_random"])

        ckpt = {
            "state":
            Checkpoint(
                self.model,
                self.optim,
                self.scheduler,
            )
        }
        dcp.load(
            state_dict=ckpt,
            checkpoint_id=os.path.join(path, "distcp_ckpt")
        )

        with open(os.path.join(path, "config.json"), "r") as df:
            data = json.load(df)

        self.args = Namespace(**data.get("config", {}))
        self.global_step_counter_ = data.get("steps", 0)
        self.best_val_score_ = data.get("score", 0)

    def save(self, path):
        logger.debug("CHECKPOINT | saving checkpoint at {}", path)

        os.makedirs(path, exist_ok=True)

        if self.__checkpoint_op is not None:
            self.__checkpoint_op.result()
        self.__checkpoint_op = dcp.async_save(
            {
                "state":
                Checkpoint(
                    self.model,
                    self.optim,
                    self.scheduler,
                )
            }, 
            storage_writer=FileSystemWriter(
                path=os.path.join(path, "distcp_ckpt"),
                overwrite=True
            )
        )
        if self.main_process():
            torch.save(
                {
                    "python_random": random.getstate(),
                    "numpy_random": np.random.get_state(),
                    "torch_random": torch.get_rng_state(),
                    "torch_cuda_random": (torch.cuda.get_rng_state_all()
                                              if torch.cuda.is_available()
                                              else None),
                },
                os.path.join(path, "rng.pt")
            )

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
    def from_pretrained(cls, path, disable_wandb=True, compile=False):
        with open(os.path.join(path, "config.json"), "r") as df:
            data = json.load(df)
        args = Namespace(**data.get("config", {}))
        args.wandb = False if disable_wandb else args.wandb
        new = cls(args, run_id=data.get("wandb"), compile=compile)
        new.load(path)

        if disable_wandb:
            new.args.wandb = False

        return new

    @property
    def device(self):
        local_rank = 0 if not self.distributed else int(os.environ["LOCAL_RANK"])
        return (
            torch.device(f"cuda:{local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
