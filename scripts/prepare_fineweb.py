# saves the FineWeb dataset to a binary file for training

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, IterableDataset
import click
import random

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

enc = tiktoken.get_encoding("gpt2")

@click.command()
@click.argument("output_path", type=click.Path(exists=True))
@click.option("--snapshot", type=str, default="CC-MAIN-2022-21", help="FineWeb snapshot to download")
@click.option("--max_samples", type=int, default=None, help="Maximum number of samples to process (None for all)")
@click.option("--val_ratio", type=float, default=0.0005, help="Ratio of samples to use for validation")
@click.option("--seed", type=int, default=2357, help="Random seed for train/val split")
def prepare_fineweb(output_path, snapshot, max_samples, val_ratio, seed):
    """
    Encoding FineWeb with the GPT2 encoding and mmaping the output to OUTPUT_PATH

    Example: python prepare_fineweb.py /path/to/output --snapshot CC-MAIN-2022-21
    """

    # Set random seed for reproducibility
    random.seed(seed)

    # Load FineWeb dataset in streaming mode to avoid memory issues
    # Dataset is ~200GB but system has 32GB memory, so streaming is essential
    fw = load_dataset(
        "HuggingFaceFW/fineweb",
        name=snapshot,
        split="train",
        streaming=True
    )

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the dataset
    tokenized = fw.map(
        process,
        remove_columns=['text', 'id', 'dump', 'url', 'date', 'file_path', 'language',
                        'language_score', 'token_count']
    )

    # Limit samples if specified
    if max_samples is not None:
        import itertools
        tokenized = itertools.islice(tokenized, max_samples)

    # Create memmaps for train and val splits
    train_filename = os.path.join(output_path, 'train.bin')
    val_filename = os.path.join(output_path, 'val.bin')
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)

    # Start with initial size and grow as needed
    initial_size = 1_000_000  # Start with 1M tokens
    train_arr = np.memmap(train_filename, dtype=dtype, mode='w+', shape=(initial_size,))
    val_arr = np.memmap(val_filename, dtype=dtype, mode='w+', shape=(initial_size,))

    train_idx = 0
    val_idx = 0

    for sample in tqdm(tokenized, desc='processing FineWeb'):
        sample_ids = np.array(sample['ids'], dtype=dtype)
        sample_len = len(sample_ids)

        # Randomly assign to train or val based on val_ratio
        is_val = random.random() < val_ratio

        if is_val:
            # Resize val memmap if needed
            if val_idx + sample_len > val_arr.shape[0]:
                val_arr.flush()
                new_size = max(val_arr.shape[0] * 2, val_idx + sample_len)
                val_arr = np.memmap(val_filename, dtype=dtype, mode='r+', shape=(new_size,))

            # Write sample to val memmap
            val_arr[val_idx : val_idx + sample_len] = sample_ids
            val_idx += sample_len

            if val_idx % 100_000 == 0:  # Flush periodically
                val_arr.flush()
        else:
            # Resize train memmap if needed
            if train_idx + sample_len > train_arr.shape[0]:
                train_arr.flush()
                new_size = max(train_arr.shape[0] * 2, train_idx + sample_len)
                train_arr = np.memmap(train_filename, dtype=dtype, mode='r+', shape=(new_size,))

            # Write sample to train memmap
            train_arr[train_idx : train_idx + sample_len] = sample_ids
            train_idx += sample_len

            if train_idx % 100_000 == 0:  # Flush periodically
                train_arr.flush()

    # Trim train file to actual size
    train_arr.flush()
    train_arr._mmap.close()
    del train_arr

    with open(train_filename, 'r+b') as f:
        f.truncate(train_idx * dtype().itemsize)

    # Trim val file to actual size
    val_arr.flush()
    val_arr._mmap.close()
    del val_arr

    with open(val_filename, 'r+b') as f:
        f.truncate(val_idx * dtype().itemsize)

    print(f"\nDone! Created:")
    print(f"  train.bin: {train_idx:,} tokens")
    print(f"  val.bin: {val_idx:,} tokens")

if __name__ == "__main__":
    prepare_fineweb()
