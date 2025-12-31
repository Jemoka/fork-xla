# saves the GSM8K-Aug dataset to a binary file for training
# GSM8K-Aug contains grade school math problems with step-by-step solutions

import os
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
def prepare_gsm8k_aug(output_path, block_size, pad_token):
    """
    Encoding GSM8K-Aug with the GPT2 encoding and mmaping the output to OUTPUT_PATH.
    Creates two files per split: {split}.bin (tokens) and {split}.bin.mask (padding mask).
    """

    # Load GSM8K-Aug dataset from HuggingFace
    # Contains train split with augmented math problems
    dataset = load_dataset("whynlp/gsm8k-aug", num_proc=num_proc_load_dataset, trust_remote_code=True)

    # Define the encoding function
    def process(example):
        # Format: question + steps (cleaned) + answer
        text = (
            example["question"] + "\n" +
            " ".join([step.replace("<<", "").replace(">>", "") for step in example["steps"]]) + "\n" +
            "answer: " +
            example["answer"]
        )

        ids = enc.encode_ordinary(text)  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['question', 'steps', 'answer'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Write each split to binary files with fixed block size
    for split, dset in tokenized.items():
        num_samples = len(dset)

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

            if seq_len > block_size:
                # Left truncate: keep the rightmost block_size tokens
                ids = ids[-block_size:]
                tokens_arr[idx] = np.array(ids, dtype=dtype)
                mask_arr[idx] = True  # All tokens are real (no padding)
            else:
                # Left pad with pad_token
                padding_len = block_size - seq_len
                padded = [pad_token] * padding_len + ids
                tokens_arr[idx] = np.array(padded, dtype=dtype)

                # Mask: False for padding, True for real tokens
                mask = [False] * padding_len + [True] * seq_len
                mask_arr[idx] = np.array(mask, dtype=np.bool_)

        # Flush to disk
        tokens_arr.flush()
        mask_arr.flush()

        print(f"{split}: {num_samples} samples, shape ({num_samples}, {block_size})")

    # Example usage:
    # tokens = np.memmap('train.bin', dtype=np.uint16, mode='r').reshape(-1, block_size)
    # mask = np.memmap('train.bin.mask', dtype=np.bool_, mode='r').reshape(-1, block_size)

if __name__ == '__main__':
    prepare_gsm8k_aug()
