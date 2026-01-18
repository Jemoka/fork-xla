from abc import ABC, abstractmethod, abstractproperty
from typing import Any
from tiktoken import get_encoding

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

from loguru import logger

class Evaluation(ABC):

    @abstractproperty
    def name(self) -> str:
        ...

    def prefix(self) -> str:
        return self.name+"_"

    @abstractmethod
    def __len__(self) -> int:
        ...

    @staticmethod
    def find_accumulation_steps(
            dataset_size: int,
            max_batch_size: int,
            dp_replicate: int
    ):
        for batch_size in reversed(range(1, max_batch_size + 1)):
            if dataset_size % (batch_size * dp_replicate) == 0:
                return batch_size, dataset_size // (batch_size * dp_replicate)

        logger.warning(f"We coludn't find a good size for dataset_size {dataset_size} and max_batch_size {max_batch_size} and dp_replicate {dp_replicate}. Will chop off the end of this evaluation.")

class RolloutEvaluation(Evaluation):


    def score(self, ys: str, y_hats: str) -> float:
        """From generated results, compute score

        Args:
            ys (str): ground truth
            y_hats (str): generated results
        Returns:
            float: computed score, higher is better
        """

        results = [
            self.check(i,j)
            for i, j in zip(ys, y_hats)
        ]
        return sum(results) / len(results)

    def check(self, y: str, y_hat: str) -> bool:
        """Check if y_hat matches y

        Args:
            y (str): ground truth
            y_hat (str): generated result
        Returns:
            bool: whether y_hat matches y
        """

        raise NotImplementedError("Please override this method or self.score, not neither!")

    @abstractmethod
    def clean(self, y_hat: str) -> str:
        """Clean generated result before checking

        Args:
            y_hat (str): generated result, which *can include* the prompt
        Returns:
            str: cleaned/normalized result available for comparison
        """

        ...

    @abstractmethod
    def get(self, indx) -> (str, str):
        """return (x,y)"""
        ...

    @abstractproperty
    def num_tokens(self) -> int:
        """maximum rollout length BEYOND prompt"""
        ...

    def __call__(self, trainer, encoding, truncate=False, temperature=0.0, top_p=1.0, **kwargs):
        eval = self

        # encoding everything
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            x,y = zip(*[eval.get(i) for i in range(len(eval))])
            xs, masks = trainer.pad(encoding.encode_batch(x))
        else:
            xs, masks = None, None
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # divide into pieces such that each host gets a batch with
        if not truncate:
            per_device_batch_size, accumulate_steps = self.find_accumulation_steps(
                xs.shape[0],
                trainer.per_device_batch_size,
                trainer.replicas
            )
        else:
            xs = xs[:(xs.shape[0]//(trainer.replicas*trainer.per_device_batch_size))*
                    (trainer.replicas*trainer.per_device_batch_size)]
            masks = masks[:(xs.shape[0]//(trainer.replicas*trainer.per_device_batch_size))*
                          (trainer.replicas*trainer.per_device_batch_size)]
            
        # divide into pieces
        pieces_xs = jnp.array_split(xs, jax.process_count(), axis=0)
        pieces_masks = jnp.array_split(masks, jax.process_count(), axis=0)
        xs = pieces_xs[jax.process_index()]
        masks = pieces_masks[jax.process_index()]

        # reshape into (accumulate_steps, per_device_batch_size, T)
        xs = xs.reshape(-1, trainer.local_replicas*trainer.per_device_batch_size, xs.shape[-1])
        masks = masks.reshape(-1, trainer.local_replicas*trainer.per_device_batch_size, xs.shape[-1])

        # and finally make fake tensor across hosts
        xs = multihost_utils.host_local_array_to_global_array(
            xs, trainer.mesh, trainer.data_pspec
        )
        masks = multihost_utils.host_local_array_to_global_array(
            masks, trainer.mesh, trainer.data_pspec
        )

        # create a subkey
        trainer.key, key = jax.random.split(trainer.key)

        def evaluate(state, xs, masks, key):
            def reduce(_, batch):
                results = trainer._autoregress(state, key, batch[0], batch[1], self.num_tokens, temperature, top_p)
                return None, results

            _, rollouts = jax.lax.scan(
                reduce,
                None,
                (xs, masks)
            )

            # concanenate results across batches
            results = jnp.reshape(rollouts, (-1, rollouts.shape[-1]))

            return results

        # jit!
        wrapped_evaluate = jax.jit(
            evaluate,
            in_shardings=(
                trainer.state_sharding,
                trainer.data_sharding,
                trainer.data_sharding,
                None
            ),
            out_shardings=None
        )

        results = wrapped_evaluate(trainer.state, xs[:2], masks[:2], key)

        # collect across hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        results = multihost_utils.process_allgather(results)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # flatten
        results = jnp.reshape(results, (-1, results.shape[-1]))

        # decode
        decoded_results = encoding.decode_batch(results.tolist())




        


class PerplexityEvaluation(Evaluation):

    @abstractmethod
    def get(self, indx) -> (str, list[str], int):
        """return prefix (could be empty)+suffixes; int is the correct index"""
        ...



