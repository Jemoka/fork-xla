# saves the FineWeb dataset to a binary file for training

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, IterableDataset
import click
import random
from tqdm import tqdm

MEMMAP = "/home/houjun/bubbles/datasets/pes2o/train.bin"

data = np.memmap(MEMMAP, dtype=np.uint16, mode="r")

# scan from back to front for zero
print("finding zero...")
lo = 0
hi = data.shape[-1]-1

while lo < hi:
    print("BST", hi, lo)
    mid = ((hi-lo)//2)+lo
    if (data[mid] != 0 and data[mid-1] != 0):
        lo = mid+1
    elif (data[mid] == 0 and data[mid-1] == 0):
        hi = mid
    else:
        hi -= 1 # linear search, since our bst terminates when "its a run of 2 zeros"

# truncate the file up to that index 
print("truncating data...")
del data
with open(MEMMAP, 'r+b') as f:
    f.truncate(lo * np.uint16().itemsize)

print("done!")
