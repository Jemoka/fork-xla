# common standard library utilities
import os
import sys
import time
import json
import math
import random
from random import Random
from collections import defaultdict

from pathlib import Path
from argparse import Namespace

# machine learning and data utilities
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

# huggingface
import logging
from loguru import logger

# to contextualize plotting
from contextlib import contextmanager

# plotting tools
import wandb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# setting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette(
    [
        "#8C1515",  # Red
        "#175E54",  # Green
        "#E98300",  # Orange
        "#007C92",  # Teal
        "#DAD7CB",  # Light Gray
        "#B83A4B",  # Cardinal Red
        "#4D4F53",
    ]
)  # Dark Gray

R = Random(7)


def sort_by_key(data, reverse=False):
    """small utility to sort data by key, which is a usual usecase"""
    sorted_list = [
        i[1] for i in sorted(list(data.items()), reverse=reverse, key=lambda a: a[0])
    ]
    final_dict = defaultdict(list)

    for i in sorted_list:
        for k, v in i.items():
            final_dict[k].append(v)

    return dict(final_dict)


###############################################

# global fig size
FIGSIZE = (12, 8)


# each of the plot functions should decide
# what its going to do, and return the correspdoing
# wandb object. For instance, if you are plotting
# text, wandb.Html. If you are plotting image,
# return wandb.Image, etc.
#
# you should return both the *PRE WRAP* object
# as well as the *POST WRAP* image; the former
# will be for saving, the latter will be for emitting
def plot_forking(data):
    # fig1, ax = plt.subplots(figsize=FIGSIZE)

    # extract a sorted ordering of layers' forking scores
    data = sort_by_key(data)

    # Convert JAX arrays to numpy for plotting
    def to_numpy(x):
        if isinstance(x, jnp.ndarray):
            return np.array(x)
        return x

    pad_max = max([to_numpy(i[0]).shape[-1] for i in data["cumulative_scores"]])
    cum_scores_logits_unnorm = [
        to_numpy(i) for i,j in zip(data["cumulative_scores"],data["token_index"])
    ]
    cum_scores_unnorm = np.stack([np.concatenate([np.exp(i[0]), np.zeros(pad_max-i[0].shape[-1])]) for i in cum_scores_logits_unnorm])

    # we copy down the token index to calculate soft and large scores
    token_index = np.stack([np.concatenate([to_numpy(i[0]), np.zeros(pad_max-to_numpy(i[0]).shape[-1], dtype=np.int32)])
                               for i in data["token_index"]])
    token_index_ignore = np.stack([np.concatenate([np.zeros(to_numpy(i[0]).shape[-1], dtype=bool), np.ones(pad_max-to_numpy(i[0]).shape[-1], dtype=bool)])
                               for i in data["token_index"]])


    fig2, ax = plt.subplots(figsize=FIGSIZE)
    scores_unnorm = cum_scores_unnorm
    sns.heatmap(scores_unnorm.astype(np.float32), 
                cmap='viridis',
                xticklabels='auto',
                yticklabels='auto',
                ax=ax)
    plt.xlabel('Sequence Position')
    plt.ylabel('Layer')
    plt.tight_layout()
    plt.title('Cumulative Scores (Unnormalized)')
    plt.close(fig2) # remember to call close against the figure!


    fig3, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap(token_index.astype(np.float32), 
                cmap='viridis',
                xticklabels='auto',
                yticklabels='auto',
                ax=ax)
    plt.xlabel('Sequence Position')
    plt.ylabel('Layer')
    plt.tight_layout()
    plt.title('Token ID')
    plt.close(fig3) # remember to call close against the figure!

    # compute the number of tokens / forks that exist
    sum_per_token = np.ones_like(token_index)
    sum_per_token[token_index_ignore] = 0
    block = int(token_index.max()) + 1

    # Manual scatter-add implementation for numpy
    sum_per_token_result = np.zeros(
        (*token_index.shape[:-1], block),
        dtype=token_index.dtype
    )
    for b in range(token_index.shape[0]):
        for i in range(token_index.shape[1]):
            if not token_index_ignore[b, i]:
                idx = token_index[b, i]
                sum_per_token_result[b, idx] += sum_per_token[b, i]
    sum_per_token = sum_per_token_result

    soft_score_per_token_result = np.zeros(
        (*token_index.shape[:-1], block),
        dtype=cum_scores_unnorm.dtype
    )
    for b in range(token_index.shape[0]):
        for i in range(token_index.shape[1]):
            if not token_index_ignore[b, i]:
                idx = token_index[b, i]
                soft_score_per_token_result[b, idx] += cum_scores_unnorm[b, i]
    soft_score_per_token = soft_score_per_token_result

    fig4, ax = plt.subplots(figsize=FIGSIZE)

    # otherwise something really funny indeed happens
    # which is that all the padding gets placed at the beginning
    sum_per_token[0][0] = 0

    sns.heatmap(sum_per_token.astype(np.float32), 
                cmap='viridis',
                xticklabels='auto',
                yticklabels='auto',
                ax=ax)


    plt.xticks(rotation=90)
    plt.xlabel('Token')
    plt.ylabel('Layer')
    plt.tight_layout()
    plt.title('Number of Forks')
    plt.close(fig4) # remember to call close against the figure!

    fig5, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap(soft_score_per_token.astype(np.float32), 
                cmap='viridis',
                xticklabels='auto',
                yticklabels='auto',
                ax=ax)
    plt.xticks(rotation=90)
    plt.xlabel('Token')
    plt.ylabel('Layer')
    plt.tight_layout()
    plt.title('Soft Number of forks')
    plt.close(fig5) # remember to call close against the figure!

    return {
        # "forking/cum_scores": (fig1, wandb.Image(fig1)),
        "forking/cum_scores_unnorm": (fig2, wandb.Image(fig2)),
        "forking/token_index": (fig3, wandb.Image(fig3)),
        "forking/num_forks": (fig4, wandb.Image(fig4)),
        "forking/num_forks_soft": (fig5, wandb.Image(fig5)),
    }

# what function to run for what function?
PLOTS = {
    "forking": plot_forking
}

###############################################


def plot(name, data):
    plot_func = PLOTS.get(name)
    if plot_func == None:
        return None
    else:
        return plot_func(data)
