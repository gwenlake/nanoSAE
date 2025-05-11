import json
from dataclasses import dataclass

@dataclass
class TrainConfig:
    input_size: int
    hidden_size: int
    batch_size: int = 4096
    batch_shuffle: bool = True
    steps: int = 5000
    lr: float = 5e-5
    l1_penalty: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = None
    sparsity_warmup_steps: int = None
    decay_start: int = None
    topk: int = None
    topk_threshold: float = None
    seed: int = 42
    device: str = None
    wandb_name: str = None

    def from_json(path):
        with open(path, "r") as f:
            config = TrainConfig(**json.load(f))
        return config
