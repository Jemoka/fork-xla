# common standard library utilities
import os
import sys
import glob
import time
import json
import math
import random
from random import Random

from pathlib import Path
from argparse import Namespace

# machine learning and data utilities
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR

from torch.utils.data import DataLoader, Dataset

# huggingface
from transformers import AutoConfig, AutoModel, AutoTokenizer

# MLOps
import wandb

# logging
from loguru import logger

# data utilities
import datasets
from datasets import Dataset, DatasetDict
from tqdm import tqdm

tqdm.pandas()

cached_memmap_train = None
cached_memmap_val = None


def get_batch_pmd(
    args, batch_size, split="train", device="cpu", deterministic_key=None
):
    """get batches based on the "poor man's dataloader" strategy"""
    global cached_memmap_train, cached_memmap_val

    # args is the run configuration + config is the GPT config
    data_dir = args.data_dir
    block_size = args.block_size

    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        if cached_memmap_train is not None:
            data = cached_memmap_train
        else:
            data = np.memmap(
                os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
            )
            cached_memmap_train = data
    else:
        if cached_memmap_val is not None:
            data = cached_memmap_val
        else:
            data = np.memmap(
                os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
            )
            cached_memmap_val = data
    if deterministic_key:
        portion = batch_size * block_size
        ix = torch.arange(
            deterministic_key * portion, (deterministic_key + 1) * portion, block_size
        )
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device != "cpu":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_batch(
    args,
    batch_size,
    split="train",
    device="cpu",
    deterministic_key=None,
    return_label=False,
):
    if "openwebtext" in args.data_dir or "pes2o" in args.data_dir:
        return get_batch_pmd(args, batch_size, split, device, deterministic_key)
    else:
        logger.warning(
            "We can't infer what the data format is supposed to be; assuming a poor man's dataloader!"
        )
        return get_batch_pmd(args, batch_size, split, device, deterministic_key)
