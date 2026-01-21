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
            if per_device_batch_size is None:
                # Truncate if we can't find a good batch size
                truncate = True

        if truncate:
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
                results = trainer._autoregress(state, key, batch[0], batch[1], trainer.args.block_size, temperature, top_p, **kwargs)
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

        results = wrapped_evaluate(trainer.state, xs, masks, key)

        # collect across hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        results = multihost_utils.process_allgather(results)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # flatten
        results = jnp.reshape(results, (-1, results.shape[-1]))

        # decode
        decoded_results = encoding.decode_batch(results.tolist())

        # score
        score = eval.score(y, [eval.clean(i) for i in decoded_results])

        return score

class EncodingEvaluation(Evaluation):

    def score(self, xs: list[str], y_hats: list[str]) -> float:
        """From input and model predictions, compute score

        Args:
            xs (list[str]): input strings
            y_hats (list[str]): model predictions (argmax of logits, shifted by 1)
        Returns:
            float: computed score, higher is better
        """
        results = [
            self.check(x, y_hat)
            for x, y_hat in zip(xs, y_hats)
        ]
        return sum(results) / len(results)

    def check(self, x: str, y_hat: str) -> bool:
        """Check if prediction is correct given input

        Args:
            x (str): input string
            y_hat (str): model prediction (cleaned, decoded argmax)
        Returns:
            bool: whether prediction is correct
        """
        raise NotImplementedError("Please override this method or self.score!")

    @abstractmethod
    def clean(self, y_hat: str) -> str:
        """Clean model prediction before checking

        Args:
            y_hat (str): raw decoded model prediction
        Returns:
            str: cleaned/normalized result available for comparison
        """
        ...

    @abstractmethod
    def get(self, indx) -> str:
        """return input string x"""
        ...

    def __call__(self, trainer, encoding, truncate=False):
        eval = self

        # encoding everything
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            x = [eval.get(i) for i in range(len(eval))]
            xs, masks = trainer.pad(encoding.encode_batch(x))
        else:
            x = None
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
            if per_device_batch_size is None:
                # Truncate if we can't find a good batch size
                truncate = True

        if truncate:
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

        def evaluate(state, xs, masks):
            def reduce(_, batch):
                x_batch, mask_batch = batch
                # Single forward pass through the model
                logits, _ = state.apply_fn(
                    {'params': state.params},
                    x_batch,
                    padding_mask=mask_batch,
                    deterministic=True
                )
                # Take argmax to get predicted tokens, exclude last position
                # (logits at position i predict token i+1)
                predictions = jnp.argmax(logits[:, :-1, :], axis=-1)
                return None, predictions

            _, results = jax.lax.scan(
                reduce,
                None,
                (xs, masks)
            )

            # concatenate results across batches
            results = jnp.reshape(results, (-1, results.shape[-1]))
            return results

        # jit!
        wrapped_evaluate = jax.jit(
            evaluate,
            in_shardings=(
                trainer.state_sharding,
                trainer.data_sharding,
                trainer.data_sharding,
            ),
            out_shardings=None
        )

        results = wrapped_evaluate(trainer.state, xs, masks)

        # collect across hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        results = multihost_utils.process_allgather(results)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # flatten
        results = jnp.reshape(results, (-1, results.shape[-1]))

        # decode predictions
        decoded_outputs = encoding.decode_batch(results.tolist())

        # score
        score = eval.score(x, [eval.clean(i) for i in decoded_outputs])

        return score


class PerplexityEvaluation(Evaluation):

    @abstractmethod
    def get(self, indx) -> (str, list[str], int):
        """return prefix (could be empty)+suffixes; int is the correct index"""
        ...

    def __call__(self, trainer, encoding, truncate=False):
        eval = self

        # Gather all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            all_data = [eval.get(i) for i in range(len(eval))]
            prefixes, continuations_list, correct_indices = zip(*all_data)

            # Flatten: create (prefix+continuation, sample_idx, continuation_idx, prefix_len) for all combinations
            flattened_inputs = []
            prefix_lengths = []
            metadata = []  # (sample_idx, continuation_idx, num_continuations)
            for sample_idx, (prefix, continuations, correct_idx) in enumerate(all_data):
                # Encode prefix to get its length
                prefix_encoded = encoding.encode(prefix)
                prefix_len = len(prefix_encoded)

                for cont_idx, continuation in enumerate(continuations):
                    full_text = prefix + continuation
                    flattened_inputs.append(full_text)
                    prefix_lengths.append(prefix_len)
                    metadata.append((sample_idx, cont_idx, len(continuations)))

            # Encode all inputs
            encoded_inputs = encoding.encode_batch(flattened_inputs)
            xs, masks = trainer.pad(encoded_inputs)
            prefix_lengths_array = jnp.array(prefix_lengths, dtype=jnp.int32)

            # Convert metadata to arrays for broadcasting
            metadata_array = jnp.array(metadata, dtype=jnp.int32)
            correct_indices_array = jnp.array(correct_indices, dtype=jnp.int32)
        else:
            xs, masks, prefix_lengths_array, metadata_array, correct_indices_array = None, None, None, None, None

        # Broadcast to all hosts
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        prefix_lengths_array = multihost_utils.broadcast_one_to_all(prefix_lengths_array)
        metadata_array = multihost_utils.broadcast_one_to_all(metadata_array)
        correct_indices_array = multihost_utils.broadcast_one_to_all(correct_indices_array)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Handle truncation or find appropriate batch sizes
        if not truncate:
            per_device_batch_size, accumulate_steps = self.find_accumulation_steps(
                xs.shape[0],
                trainer.per_device_batch_size,
                trainer.replicas
            )
            if per_device_batch_size is None:
                # Truncate if we can't find a good batch size
                truncate = True

        if truncate:
            valid_size = (xs.shape[0] // (trainer.replicas * trainer.per_device_batch_size)) * \
                (trainer.replicas * trainer.per_device_batch_size)
            xs = xs[:valid_size]
            masks = masks[:valid_size]
            prefix_lengths_array = prefix_lengths_array[:valid_size]
            metadata_array = metadata_array[:valid_size]

        # Divide into pieces for each process
        pieces_xs = jnp.array_split(xs, jax.process_count(), axis=0)
        pieces_masks = jnp.array_split(masks, jax.process_count(), axis=0)
        pieces_prefix_lens = jnp.array_split(prefix_lengths_array, jax.process_count(), axis=0)
        pieces_metadata = jnp.array_split(metadata_array, jax.process_count(), axis=0)

        xs = pieces_xs[jax.process_index()]
        masks = pieces_masks[jax.process_index()]
        prefix_lens_local = pieces_prefix_lens[jax.process_index()]
        metadata_local = pieces_metadata[jax.process_index()]

        # Reshape into (accumulate_steps, batch_size, T)
        batch_size = trainer.local_replicas * trainer.per_device_batch_size
        xs = xs.reshape(-1, batch_size, xs.shape[-1])
        masks = masks.reshape(-1, batch_size, masks.shape[-1])
        prefix_lens_local = prefix_lens_local.reshape(-1, batch_size)

        # Create global arrays across hosts
        xs = multihost_utils.host_local_array_to_global_array(
            xs, trainer.mesh, trainer.data_pspec
        )
        masks = multihost_utils.host_local_array_to_global_array(
            masks, trainer.mesh, trainer.data_pspec
        )
        # prefix_lens: (accumulate, batch) - replicate accumulate, shard batch
        prefix_lens_pspec = jax.sharding.PartitionSpec(None, "batch")
        prefix_lens_local = multihost_utils.host_local_array_to_global_array(
            prefix_lens_local, trainer.mesh, prefix_lens_pspec
        )

        def evaluate(state, xs, masks, prefix_lens):
            """Compute per-sample loss only on continuation tokens (prefix tokens set to -1)"""
            def reduce(_, batch):
                x_batch, mask_batch, prefix_len_batch = batch
                # Shift x to create y for next token prediction
                y_batch = jnp.roll(x_batch, -1, axis=-1)
                y_batch = y_batch.at[:, -1].set(0)  # Last position doesn't matter

                # Set prefix positions to -1 so they're ignored in loss
                seq_positions = jnp.arange(x_batch.shape[-1])[None, :]  # (1, seq_len)
                is_prefix = seq_positions < prefix_len_batch[:, None]  # (batch, seq_len)
                y_batch = jnp.where(is_prefix, -1, y_batch)

                # Get logits from model (no loss)
                logits, _ = state.apply_fn(
                    {'params': state.params},
                    x_batch,
                    padding_mask=mask_batch,
                    deterministic=True
                )

                # Cast logits to float32 for numerical stability
                logits_f32 = logits.astype(jnp.float32)
                logits_flat = logits_f32.reshape(-1, logits_f32.shape[-1])  # (batch * seq_len, vocab)
                targets_flat = y_batch.reshape(-1)  # (batch * seq_len,)

                # Mask out ignore index (-1) AND padding tokens
                mask = targets_flat != -1
                targets_masked = jnp.where(mask, targets_flat, 0)

                # Compute cross-entropy loss per token
                log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
                token_losses = -jnp.take_along_axis(log_probs, targets_masked[:, None], axis=-1).squeeze(-1)
                token_losses = jnp.where(mask, token_losses, 0.0)

                # Reshape back to (batch, seq_len) and average per sample
                token_losses = token_losses.reshape(x_batch.shape[0], x_batch.shape[1])
                mask = mask.reshape(x_batch.shape[0], x_batch.shape[1])
                per_sample_loss = jnp.sum(token_losses, axis=-1) / jnp.maximum(jnp.sum(mask, axis=-1), 1.0)

                return None, per_sample_loss

            _, losses = jax.lax.scan(
                reduce,
                None,
                (xs, masks, prefix_lens)
            )

            # Flatten losses across batches
            losses = jnp.reshape(losses, (-1,))
            return losses

        # JIT the evaluation function
        wrapped_evaluate = jax.jit(
            evaluate,
            in_shardings=(
                trainer.state_sharding,
                trainer.data_sharding,
                trainer.data_sharding,
                jax.sharding.NamedSharding(trainer.mesh, prefix_lens_pspec)
            ),
            out_shardings=None
        )

        losses = wrapped_evaluate(trainer.state, xs, masks, prefix_lens_local)

        # Gather results across all hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        losses = multihost_utils.process_allgather(losses)
        metadata_gathered = multihost_utils.process_allgather(metadata_local)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Flatten
        losses = jnp.reshape(losses, (-1,))
        metadata_gathered = jnp.reshape(metadata_gathered, (-1, 3))

        # Convert to numpy for easier processing
        losses = jax.device_get(losses)
        metadata_gathered = jax.device_get(metadata_gathered)
        correct_indices_array = jax.device_get(correct_indices_array)

        # Group losses by sample_idx using metadata
        # This handles: (1) variable continuations per sample, (2) truncation removing partial samples
        from collections import defaultdict
        sample_losses = defaultdict(list)
        sample_num_continuations = {}

        for i, (sample_idx, cont_idx, num_conts) in enumerate(metadata_gathered):
            sample_idx = int(sample_idx)
            sample_losses[sample_idx].append(losses[i])
            sample_num_continuations[sample_idx] = int(num_conts)

        # Only evaluate samples that have all their continuations (complete samples)
        correct = 0
        total = 0

        for sample_idx in sorted(sample_losses.keys()):
            expected_conts = sample_num_continuations[sample_idx]
            actual_conts = len(sample_losses[sample_idx])

            # Skip incomplete samples (truncated)
            if actual_conts != expected_conts:
                continue

            losses_for_sample = jnp.array(sample_losses[sample_idx])
            pred = int(jnp.argmin(losses_for_sample))
            correct_idx = int(correct_indices_array[sample_idx])

            correct += int(pred == correct_idx)
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return float(accuracy)



