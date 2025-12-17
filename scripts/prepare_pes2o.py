# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset # huggingface datasets
import click
import itertools

import json
import wandb
import random

from utils import read_lines_from_zst
from collections import defaultdict

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

cached_streams = {}
cached_val_set = defaultdict(list)

def format_pes2o_url(id, split="train"):
    return ("https://huggingface.co/datasets/allenai/peS2o/resolve/"
            f"main/data/v3/{split}-{id:04d}-of"
            f"-{'0136' if split == 'train' else '0060'}.zst")

def get_stream(id, split="train", reset=False):
    global cached_streams
    if reset or not cached_streams.get((id, split)):
        cached_streams[(id, split)] = read_lines_from_zst(
            format_pes2o_url(id, split)
        )

    return cached_streams[(id, split)]

def get_stream_iterator(split):
    i = 0
    while True:
        stream = get_stream(i, split)
        for j in stream:
            yield json.loads(j)
        i += 1 
        if (split == "train" and i > 136) or (split == "valid" and i > 60):
            raise StopIteration


@click.command()
@click.argument("output_path", type=click.Path(exists=True))
@click.option("--max_samples", type=int, default=None)
@click.option("--max_val_samples", type=int, default=100_000)
def prepare_pes2o(output_path, max_samples, max_val_samples):
    """
    Encoding pes2o with the GPT2 encoding and mmaping the output to OUTPUT_PATH
    """
    # output_path ="/sphinx/u/houjun/dataset/pes2o_new"
    # max_samples = 8_000_000
    # max_val_samples = 8_000_000

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # create streaming generators for train and val
    def train_generator():
        if max_samples is not None:
            return itertools.islice(get_stream_iterator("train"), max_samples)
        else:
            return get_stream_iterator("train")
    def val_generator():
        if max_samples is not None:
            return itertools.islice(get_stream_iterator("valid"), max_samples)
        else:
            return get_stream_iterator("valid")

    # create streaming datasets
    train_dataset = IterableDataset.from_generator(train_generator)
    val_dataset = IterableDataset.from_generator(val_generator)
    # tokenize the dataset
    train_tokenized = train_dataset.map(
        process,
        remove_columns=['text']
    )
    val_tokenized = val_dataset.map(
        process,
        remove_columns=['text']
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in zip(["train", "val"], [train_tokenized, val_tokenized]):
        filename = os.path.join(output_path, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        
        # Start with initial size and grow as needed
        initial_size = 1_000_000  # Start with 1M tokens
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(initial_size,))
        
        idx = 0
        for sample in tqdm(dset, desc=f'writing {filename}'):
            sample_ids = np.array(sample['ids'], dtype=dtype)
            sample_len = len(sample_ids)
            
            # Resize memmap if needed
            if idx + sample_len > arr.shape[0]:
                arr.flush()
                new_size = max(arr.shape[0] * 2, idx + sample_len)
                arr = np.memmap(filename, dtype=dtype, mode='r+', shape=(new_size,))
            
            # Write sample to memmap
            arr[idx : idx + sample_len] = sample_ids
            idx += sample_len
            
            if idx % 100_000 == 0:  # Flush periodically
                arr.flush()
        
        # Trim file to actual size
        arr.flush()
        arr._mmap.close()
        del arr
        
        # Truncate the file to remove trailing zeros
        with open(filename, 'r+b') as f:
            f.truncate(idx * dtype().itemsize)

if __name__ == "__main__":
    prepare_pes2o()

