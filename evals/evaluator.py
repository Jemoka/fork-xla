import jax
from tiktoken import get_encoding
from evals.evaluations import PerplexityEvaluation, RolloutEvaluation, Evaluation
from loguru import logger

class Evaluator:

    def __init__(self, evaluations: list[Evaluation]):
        self.evaluations = evaluations

    def __call__(self, encoding, trainer, truncate=False):
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
            score = evaluation(trainer, encoding, truncate=truncate)
            logger.info(f"EVAL | {evaluation.name} score: {score:.4f}")

            results[f"{evaluation.prefix()}score"] = score

        return results
