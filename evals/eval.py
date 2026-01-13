from abc import ABC, abstractmethod, abstractproperty
from tiktoken import get_encoding

class Evaluation(ABC):

    @abstractproperty
    def name(self) -> str:
        ...

    def prefix(self) -> str:
        return self.name+"_"

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
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

class Evaluator:
    def __init__(self, evaluations: list[Evaluation]):
        self.evaluations = evaluations

    def __call__(self, encoding, rollout_fn, batch_size: int = 4, logger=None):
        if isinstance(encoding, str):
            encoding = get_encoding(encoding)

        results = {}

        for evaluation in self.evaluations:
            if logger:
                logger(f"EVAL | Running evaluation: {evaluation.name}")

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
                batch_prompts = prompts[i:i+batch_size]
                if logger:
                    logger(f"EVAL | Processing batch {i//batch_size + 1}/{num_batches}")

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

            if logger:
                logger(f"EVAL | {evaluation.name} score: {score:.4f}")

            results[f"{evaluation.prefix()}score"] = score

        return results
