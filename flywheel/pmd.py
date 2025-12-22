"""
pmd.py
"Poor Man's Dataloader" Datasets

Converted to JAX for TPU/XLA training.
"""
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

# logging
from loguru import logger

# data utilities
import datasets
from datasets import Dataset, DatasetDict
from tqdm import tqdm
tqdm.pandas()

from flywheel.strategy import Dataset


class MemmapDataset(Dataset):

    def __init__(self, args, path, has_val=True):
        self.cache_train = None
        self.cache_val = None
        self.has_val = has_val
        super().__init__(args, path)

    def get_batch(self, batch_size, split="train", deterministic_key=None):
        """get batches based on the "poor man's dataloader" strategy"""

        if not self.has_val and split == "val":
            split = "train"

        args = self.args

        # args is the run configuration + config is the GPT config
        data_dir = self.path
        block_size = args.block_size

        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == "train":
            if self.cache_train is not None:
                data = self.cache_train
            else:
                data = np.memmap(
                    os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
                )
                self.cache_train = data
        else:
            if self.cache_val is not None:
                data = self.cache_val
            else:
                data = np.memmap(
                    os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
                )
                self.cache_val = data

        if deterministic_key:
            portion = batch_size * block_size
            ix = np.arange(
                deterministic_key * portion, (deterministic_key + 1) * portion, block_size
            )
        else:
            ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))

        x = np.stack(
            [data[i : i + block_size].astype(np.int64) for i in ix]
        )
        y = np.stack(
            [
                data[i + 1 : i + 1 + block_size].astype(np.int64)
                for i in ix
            ]
        )

        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)

        return x, y
