# saves the MMLU dataset to a binary file for training
# MMLU contains multi-task language understanding questions across various subjects

import os
import json
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import click

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

@click.command()
@click.argument("output_path", type=click.Path(exists=True))
@click.option("--block_size", default=512, help="Block size for padding/truncation")
@click.option("--pad_token", default=0, help="Token ID to use for padding")
@click.option("--val_pct", default=0.05, help="Percentage of data to use for validation")
def prepare_mmlu(output_path, block_size, pad_token, val_pct):
    """
    Encoding MMLU with the GPT2 encoding and mmaping the output to OUTPUT_PATH.
    Creates two files per split: {split}.bin (tokens) and {split}.bin.mask (padding mask).
    """

    # Load MMLU dataset from HuggingFace
    # Using auxiliary_train split which contains training data
    dataset = load_dataset("cais/mmlu", "auxiliary_train", num_proc=num_proc_load_dataset)["train"]

    # Create train/val split
    split_dataset = dataset.train_test_split(test_size=val_pct, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val

    # Define the encoding function
    def process(example):
        # Format: question + choices + answer
        example = example["train"]
        text = (
            example["question"] + "\n" +
            "choices: " + ", ".join(example["choices"]) + "\n" +
            "answer: " + example["choices"][example["answer"]]
        )

        ids = enc.encode_ordinary(text)  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['train'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Track shapes for shape.json
    shapes = {}

    # Write each split to binary files with fixed block size
    for split, dset in tokenized.items():
        num_samples = len(dset)

        # Statistics tracking
        num_truncated = 0
        num_padded = 0
        total_length = 0
        max_length = 0
        min_length = float('inf')

        # Create memmap files
        tokens_filename = os.path.join(output_path, f'{split}.bin')
        mask_filename = os.path.join(output_path, f'{split}.bin.mask')

        dtype = np.uint16  # can do since enc.max_token_value == 50256 is < 2**16
        tokens_arr = np.memmap(tokens_filename, dtype=dtype, mode='w+', shape=(num_samples, block_size))
        mask_arr = np.memmap(mask_filename, dtype=np.bool_, mode='w+', shape=(num_samples, block_size))

        # Process each example with padding/truncation
        for idx in tqdm(range(num_samples), desc=f'writing {tokens_filename}'):
            ids = dset[idx]['ids']
            seq_len = len(ids)

            # Update statistics
            total_length += seq_len
            max_length = max(max_length, seq_len)
            min_length = min(min_length, seq_len)

            if seq_len > block_size:
                # Left truncate: keep the rightmost block_size tokens
                num_truncated += 1
                ids = ids[-block_size:]
                tokens_arr[idx] = np.array(ids, dtype=dtype)
                mask_arr[idx] = True  # All tokens are real (no padding)
            else:
                # Left pad with pad_token
                if seq_len < block_size:
                    num_padded += 1
                padding_len = block_size - seq_len
                padded = [pad_token] * padding_len + ids
                tokens_arr[idx] = np.array(padded, dtype=dtype)

                # Mask: False for padding, True for real tokens
                mask = [False] * padding_len + [True] * seq_len
                mask_arr[idx] = np.array(mask, dtype=np.bool_)

        # Flush to disk
        tokens_arr.flush()
        mask_arr.flush()

        # Track shape
        shapes[split] = [num_samples, block_size]

        # Print statistics
        avg_length = total_length / num_samples
        print(f"\n{split.upper()} STATISTICS:")
        print(f"  Total samples: {num_samples}")
        print(f"  Shape: ({num_samples}, {block_size})")
        print(f"  Truncated: {num_truncated} ({100*num_truncated/num_samples:.2f}%)")
        print(f"  Padded: {num_padded} ({100*num_padded/num_samples:.2f}%)")
        print(f"  Exact fit: {num_samples - num_truncated - num_padded}")
        print(f"  Sequence lengths - Min: {min_length}, Max: {max_length}, Avg: {avg_length:.2f}")

    # Write shape.json
    with open(os.path.join(output_path, "shape.json"), "w") as f:
        json.dump(shapes, f, indent=4)
    print(f"\nWrote shape.json: {shapes}")

    # Example usage:
    # tokens = np.memmap('train.bin', dtype=np.uint16, mode='r').reshape(-1, block_size)
    # mask = np.memmap('train.bin.mask', dtype=np.bool_, mode='r').reshape(-1, block_size)

if __name__ == '__main__':
    prepare_mmlu()
