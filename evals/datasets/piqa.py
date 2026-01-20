from evals.evaluations import PerplexityEvaluation

from datasets import load_dataset


class PIQA(PerplexityEvaluation):
    def __init__(self, split="validation"):
        self.ds = load_dataset("ybisk/piqa", split=split, trust_remote_code=True)

    @property
    def name(self) -> str:
        return "piqa"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx) -> (str, list[str], int):
        """return (prefix, continuations, correct_index)"""
        sample = self.ds[indx]
        prefix = sample["goal"]
        continuations = [" " + sample["sol1"], " " + sample["sol2"]]
        correct_idx = int(sample["label"])
        return (prefix, continuations, correct_idx)
