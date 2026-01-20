from evals.evaluations import EncodingEvaluation

from datasets import load_dataset


class Lambada(EncodingEvaluation):
    def __init__(self, split="test"):
        self.ds = load_dataset("cimec/lambada", split=split)

    @property
    def name(self) -> str:
        return "lambada"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx) -> str:
        """return input string"""
        return self.ds[indx]["text"]

    def clean(self, y_hat: str) -> str:
        """Return decoded output as-is"""
        return y_hat

    def check(self, x: str, y_hat: str) -> bool:
        """Check if last word of prediction matches last word of input"""
        pred_last_word = y_hat.split(" ")[-1].strip()
        target_last_word = x.split(" ")[-1].strip()
        return pred_last_word == target_last_word
