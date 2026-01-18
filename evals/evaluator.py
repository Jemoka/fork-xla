import jax
from tiktoken import get_encoding
from evals.prototypes import PerplexityEvaluation, RolloutEvaluation
from loguru import logger

class Evaluator:

    def __init__(self, evaluations: list[Evaluation]):
        self.evaluations = evaluations

    def __call__(self, encoding, trainer):
        """Run evaluations

        Args:
            encoding: tiktoken encoding or str name of encoding
            rollout_fn: function that takes in list of tokenized prompts and returns list of tokenized predictions
            batch_size (int, optional): batch size for processing. Defaults to 4.
            logger (_type_, optional): logger function. Defaults to None.
        """

        if isinstance(encoding, str):
            encoding = get_encoding(encoding)

        results = {}

        for evaluation in self.evaluations:
            logger.info(f"EVAL | Running evaluation: {evaluation.name}")

            # Collect all prompts and ground truths
            prompts = []
            ground_truths = []
            for i in range(len(evaluation)):
                item = evaluation.get(i)
                # Handle both tuple (prompt, gt) and single prompt formats
                if isinstance(item, tuple):
                    prompt, gt = item
                else:
                    prompt = item
                    gt = item  # Fallback if get only returns string
                prompts.append(prompt)
                ground_truths.append(gt)

            # Process in batches
            all_predictions = []
            num_batches = (len(prompts) + batch_size - 1) // batch_size
            for i in range(0, len(prompts), batch_size):
                if i > 0 and debug__truncate is not None and i >= debug__truncate:
                    break
                batch_prompts = prompts[i:i+batch_size]
                logger.debug(f"EVAL | Processing batch {i//batch_size + 1}/{num_batches}")

                # Encode prompts to tokens
                encoded_prompts = encoding.encode_batch(batch_prompts)

                # Generate predictions using rollout_fn (returns numpy-like tokens)
                batch_token_predictions = rollout_fn(encoded_prompts)

                # Decode tokens back to text
                batch_predictions = encoding.decode_batch(batch_token_predictions)
                all_predictions.extend(batch_predictions)

            # Clean predictions
            cleaned_predictions = [evaluation.clean(pred) for pred in all_predictions]

            # Score
            score = evaluation.score(ground_truths, cleaned_predictions)

            logger.info(f"EVAL | {evaluation.name} score: {score:.4f}")

            results[f"{evaluation.prefix()}score"] = score

        return results
