"""
strategy.py
Data Loading Strategy

Converted to JAX for TPU/XLA training.
"""

import numpy as np
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, args, path):
        self.args = args
        self.path = path

    @abstractmethod
    def get_batch(self, batch_size, split="train", deterministic_key=None):
        pass


@dataclass
class Sampling:
    dataset: Dataset
    rate: float


class Strategy:
    def __init__(self, args, mixture: list[Sampling]):
        self.args = args
        self.mixture = mixture

        # validate that rates sum to 1
        total_rate = sum(sampling.rate for sampling in mixture)
        if not abs(total_rate - 1.0) < 1e-6:
            raise ValueError("Sampling rates must sum to 1.")

    def get_batch(self, batch_size, split="train", deterministic_key=None):
        # Choose a dataset based on the sampling rates
        r = random if deterministic_key is None else random.Random(deterministic_key)
        rand_val = r.random()
        cumulative_rate = 0.0

        batch = None
        for sampling in self.mixture:
            cumulative_rate += sampling.rate
            if rand_val < cumulative_rate:
                batch = sampling.dataset.get_batch(batch_size, split, deterministic_key)
                break
        else:
            # Fallback (should not reach here if rates sum to 1)
            batch = self.mixture[-1].dataset.get_batch(batch_size, split, deterministic_key)

        x, y = batch

        # validate batch, if any rows have zeros then resample
        cut_batch_x = x[(~(x == 0).all(axis=-1))]
        cut_batch_y = y[(~(x == 0).all(axis=-1))]
        if cut_batch_x.shape[0] < batch_size:
            x_addn, y_addn = self.get_batch(
                batch_size - cut_batch_x.shape[0], split,
                deterministic_key+1 if deterministic_key is not None else None
            )
            x = np.concatenate([cut_batch_x, x_addn], axis=0)
            y = np.concatenate([cut_batch_y, y_addn], axis=0)

        return (x, y)
