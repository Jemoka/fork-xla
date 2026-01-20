from evals.evaluations import PerplexityEvaluation

from datasets import load_dataset


class ARCEasy(PerplexityEvaluation):
    def __init__(self, split="validation"):
        self.ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split, trust_remote_code=True)

    @property
    def name(self) -> str:
        return "arc_easy"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx) -> (str, list[str], int):
        """return (prefix, continuations, correct_index)"""
        sample = self.ds[indx]
        prefix = sample["question"]
        continuations = [" " + text for text in sample["choices"]["text"]]
        correct_idx = sample["choices"]["label"].index(sample["answerKey"])
        return (prefix, continuations, correct_idx)


class ARCChallenge(PerplexityEvaluation):
    def __init__(self, split="validation"):
        self.ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, trust_remote_code=True)

    @property
    def name(self) -> str:
        return "arc_challenge"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx) -> (str, list[str], int):
        """return (prefix, continuations, correct_index)"""
        sample = self.ds[indx]
        prefix = sample["question"]
        continuations = [" " + text for text in sample["choices"]["text"]]
        correct_idx = sample["choices"]["label"].index(sample["answerKey"])
        return (prefix, continuations, correct_idx)
