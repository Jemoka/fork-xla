from evals.evaluations import PerplexityEvaluation

import random
import datasets
from datasets import load_dataset, get_dataset_config_names

class Blimp(PerplexityEvaluation):
    def __init__(self, subset=None):
        if subset is not None:
            self.ds = load_dataset("nyu-mll/blimp", subset)["train"]
        else:
            configs = get_dataset_config_names("nyu-mll/blimp")
            all = [load_dataset("nyu-mll/blimp", subset)  for subset in configs]
            self.ds = datasets.combine.concatenate_datasets([i["train"] for i in all])
        self.R = random.Random(7)

    @property
    def name(self) -> str:
        return "blimp"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx) -> (str, list[str], int):
        """return (x,y)"""

        rev = self.R.choice([True, False])
        sample = self.ds[indx]
        continuations = [sample["sentence_good"], sample["sentence_bad"]]

        if rev:
            return ("", list(reversed(continuations)), 1)
        else:
            return ("", list(continuations), 0)
        
