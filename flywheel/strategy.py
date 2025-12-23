"""
strategy.py
Data Loading Strategy

Converted to JAX for TPU/XLA training.
"""

import numpy as np
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod
from threading import Thread
from queue import Queue


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


class AsyncStrategy:
    def __init__(self, strategy, kwargs):
        """Asynchronous data loading strategy using threading.

        Args:
            strategy: The underlying strategy to fetch batches from
            kwargs: Arguments to pass to strategy.get_batch()
            prefetch_buffer: Number of batches to prefetch (default 2)
        """

        self.strategy = strategy
        self.args = kwargs
        self.queue = Queue()
        self.stop_flag = False
        self.error = None

        self.thread = Thread(
            target=self._fetch_worker,
            daemon=True
        )
        self.thread.start()

    def get_batch(self):
        """Get the next batch. Blocks until a batch is available.

        Returns:
            Tuple of numpy arrays (x, y)

        Raises:
            Exception if the worker thread encountered an error
        """
        # Check if worker thread died with an error
        if self.error is not None:
            raise self.error

        # Blocking get - waits until a batch is available
        item = self.queue.get()

        # Check if item is an exception from worker thread
        if isinstance(item, Exception):
            self.error = item
            raise item

        return item

    def _fetch_worker(self):
        """Worker thread that continuously fetches batches."""
        try:
            while not self.stop_flag:
                batch = self.strategy.get_batch(**self.args)
                self.queue.put(batch)  # Blocks if queue is full (backpressure)
        except Exception as e:
            # Put exception in queue so main thread can handle it
            self.queue.put(e)

    def close(self):
        """Stop the worker thread and clean up."""
        self.stop_flag = True
        # Clear the queue to unblock the worker if it's waiting on a full queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break
        # Wait for thread to finish
        self.thread.join(timeout=1.0)

    def __del__(self):
        """Cleanup when object is garbage collected."""
        self.close()

class Strategy:
    def __init__(self, args, mixture: list[Sampling]):
        self.args = args
        self.mixture = mixture

        # validate that rates sum to 1
        total_rate = sum(sampling.rate for sampling in mixture)
        if not abs(total_rate - 1.0) < 1e-6:
            raise ValueError("Sampling rates must sum to 1.")

    def get_async_batches(self, batch_size, split="train", deterministic_key=None):
        return AsyncStrategy(self, {
            "batch_size": batch_size,
            "split": split,
            "deterministic_key": deterministic_key
        })

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


