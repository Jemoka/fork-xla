from evals.evaluations import PerplexityEvaluation

from datasets import load_dataset


class HellaSwag(PerplexityEvaluation):
    def __init__(self, split="validation"):
        self.ds = load_dataset("Rowan/hellaswag", split=split)

    @property
    def name(self) -> str:
        return "hellaswag"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx) -> (str, list[str], int):
        """return (prefix, continuations, correct_index)"""
        sample = self.ds[indx]
        prefix = sample["ctx"]
        continuations = [" " + ending for ending in sample["endings"]]
        correct_idx = int(sample["label"])
        return (prefix, continuations, correct_idx)
