import os
import json
from typing import Any, Literal
from dataclasses import dataclass

@dataclass
class SAEConfig:

    # SAE parameters
    input_size: int
    hidden_size: int
    model_name: str = None

    # Dataset
    context_size: int = 128
    dataset_path: str = None
    dataset_normalization: Literal["zscore"] = None

    # Training
    steps: int = 5000

    # Batch size
    batch_size: int = 4096
    batch_shuffle: bool = True

    # Learning rate schedule
    lr: float = 5e-5
    lr_warmup_steps: int = None
    lr_decay_start: int = None

    # Sparsity
    l1_penalty: float = 1e-1
    l1_warmup_steps: int = None

    # Adam
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # misc
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"  # type: ignore

    # Wandb
    wandb_entity: str = None
    wandb_project: str = None
    wandb_run_name: str = None

    @classmethod
    def from_json(cls, path) -> "SAEConfig":
        with open(path, "r") as f:
            config = json.load(f)

        return cls(**config)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "dtype": str(self.dtype),
            "device": str(self.device),
        }

    def to_json(self, filepath: str) -> None:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
