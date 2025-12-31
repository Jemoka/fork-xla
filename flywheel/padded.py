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


class PaddedDataset(Dataset):

    def __init__(self, args, path, has_val=True):
        self.cache_train = None
        self.cache_val = None
        self.cache_train_mask = None
        self.cache_val_mask = None

        with open(Path(path) / "shape.json", "r") as f:
            self.shape = json.load(f)

        self.has_val = has_val
        super().__init__(args, path)

    def get_batch(self, batch_size, split="train", deterministic_key=None):
        """get batches based on the "poor man's dataloader" strategy"""

        if not self.has_val and split == "val":
            split = "train"

        args = self.args
        shape = self.shape[split]

        # args is the run configuration + config is the GPT config
        data_dir = self.path
        block_size = args.block_size

        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == "train":
            if self.cache_train is not None:
                data = self.cache_train
                mask = self.cache_train_mask
            else:
                data = np.memmap(
                    os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r", shape=shape
                )
                self.cache_train = data
                mask = np.memmap(
                    os.path.join(data_dir, "train.bin.mask"), dtype=np.bool_, mode="r", shape=shape
                )
                self.cache_train_mask = mask

        else:
            if self.cache_val is not None:
                data = self.cache_val
                mask = self.cache_val_mask
            else:
                data = np.memmap(
                    os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r", shape=shape
                )
                self.cache_val = data
                mask = np.memmap(
                    os.path.join(data_dir, "val.bin.mask"), dtype=np.bool_, mode="r", shape=shape
                )
                self.cache_val_mask = mask

        # check that the dataset is at least as long as the block size
        assert data.shape[1] >= block_size, "Dataset is smaller than block size."
        data = data[:, -(block_size+1):]
        mask = mask[:, -(block_size+1):]

        if deterministic_key:
            ix = np.arange(
                min(deterministic_key, data.shape[0] - batch_size),
                min(deterministic_key + batch_size, data.shape[0] - batch_size),
            )
        else:
            ix = np.random.randint(0, len(data), size=(batch_size,))

        x = data[ix][:,:-1].astype(np.int64)
        y = data[ix][:,1:].astype(np.int64)
        padding = mask[ix][:,:-1].astype(np.bool)

        # left-crop to the nearest multiple of 16
        crop_len = (x.shape[1] // 16) * 16
        x = x[:,-crop_len:]
        y = y[:,-crop_len:]
        padding = padding[:,-crop_len:]

        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)
        padding_mask = np.array(padding)

        # set padding tokens to -1
        y[~(mask[ix][:,1:])[:,-crop_len:]] = -1

        return x, y, padding_mask
