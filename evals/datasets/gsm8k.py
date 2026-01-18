from evals.evaluations import RolloutEvaluation
from datasets import load_dataset

class GSM8k(RolloutEvaluation):

    def __init__(self, split="test"):
        self.ds = load_dataset("openai/gsm8k", "main")[split]

    @property
    def num_tokens(self):
        return 128

    @property
    def name(self) -> str:
        return "gsm8k"

    def get(self, indx):
        sample = self.ds[indx]
        return sample["question"], sample["answer"].split("####")[-1].strip()

    def __len__(self) -> int:
        return len(self.ds)

    def clean(self, y_hat: str) -> str:
        clean = y_hat.split("####")[-1].strip().split("answer:")[-1].strip()
        # truncate answer at <|endoftext|>
        if "<|endoftext|>" in clean:
            clean = clean.split("<|endoftext|>")[0].strip()
        # only keep numerical component / decimal point
        clean = "".join([c for c in clean if c in "0123456789.-"]).strip()

        return clean

    def check(self, y: str, y_hat: str) -> bool:
        try:
            if float(y.strip()) == float(y_hat.strip()):
                return True
            return False
        except:
            return False



