import json
from dataclasses import dataclass

@dataclass
class SAETrainConfig:
    input_size: int
    hidden_size: int
    batch_size: int = 4096
    batch_shuffle: bool = True
    steps: int = 5000
    lr: float = 5e-5
    l1_penalty: float = 1e-1

    # Adam
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    #
    warmup_steps: int = None
    sparsity_warmup_steps: int = None
    decay_start: int = None

    seed: int = 42
    device: str = None
    wandb_name: str = None

    @classmethod
    def from_json(cls, path) -> "SAETrainConfig":
        with open(path, "r") as f:
            return cls(**json.load(f))
