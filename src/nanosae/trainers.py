import os
from typing import Optional, Callable
from collections import namedtuple
import wandb
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import SAEConfig
from .models import SAE
from .utils import zscore_normalize_rows


def get_lr_schedule(total_steps: int, decay_start: int, warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    """
    Creates a learning rate schedule function with linear warmup followed by an optional decay phase.

    Note: resample_steps creates a repeating warmup pattern instead of the standard phases, but
    is rarely used in practice.

    Args:
        total_steps: Total number of training steps
        warmup_steps: Steps for linear warmup from 0 to 1
        decay_start: Optional step to begin linear decay to 0
        sparsity_warmup_steps: Used for validation with decay_start

    Returns:
        Function that computes LR scale factor for a given step
    """
    if decay_start:
        assert 0 <= decay_start < total_steps, "decay_start must be >= 0 and < steps."
        if warmup_steps:
            assert decay_start > warmup_steps, "decay_start must be > warmup_steps."
    if warmup_steps:
        assert 0 <= warmup_steps < total_steps, "warmup_steps must be >= 0 and < steps."

    def lr_schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            # Warm-up phase
            return step / warmup_steps

        if step >= decay_start:
            # Decay phase
            return (total_steps - step) / (total_steps - decay_start)

        # Constant phase
        return 1.0

    return lr_schedule


class SAETrainer:
    """
    Standard SAE training scheme following the Anthropic April update.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    Decoder column norms are NOT constrained to 1.
    This trainer does not support resampling or ghost gradients.
    This trainer will have fewer dead neurons than the standard trainer.
    """
    def __init__(self, config: SAEConfig):

        self.config = config

        # initialize SAE
        self.model = SAE(config=self.config)

        if self.config.device is None:
            self.config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.config.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, betas=(self.config.adam_beta1, self.config.adam_beta2))

        # learning rate linear decay (Anthropic Apr 2024)
        self.scheduler = None
        if self.config.lr_decay_start:
            lr_fn = get_lr_schedule(total_steps=self.config.steps, decay_start=self.config.lr_decay_start, warmup_steps=self.config.lr_warmup_steps)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
    def loss(self, x, step: int, **kwargs):

        # sparsity scale warmup (Anthropic Apr 2024)
        sparsity_scale = 1.0
        if self.config.l1_warmup_steps:
            sparsity_scale = min(step / self.config.l1_warmup_steps, 1.0)

        # loss
        x_hat, f = self.model(x)
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        mse = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = (f * self.model.decoder.weight.norm(p=2, dim=0)).sum(dim=-1).mean()

        loss = mse + self.config.l1_penalty * sparsity_scale * l1_loss

        if self.wandb:
            self.wandb.log(
                {
                    'loss' : loss.item(),
                    'mse' : mse.item(),
                    'l1_loss' : l1_loss.item(),
                    'l2_loss' : l2_loss.item(),
                    'l1_penalty' : loss.item(),
                    'sparsity_scale' : loss.item(),
                }
            )

        return loss


    def update(self, step, data):
        data = data.to(self.config.device)
        self.optimizer.zero_grad()
        loss = self.loss(data, step=step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def train(self, data) -> SAE:

        self.wandb = None
        if self.config.wandb_project and self.config.wandb_entity:
            self.wandb = wandb.init(
                entity=self.config.wandb_entity,
                project=self.config.wandb_project,
                config=self.config.to_dict(),
                # name=self.config.run_name,
            )

        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        if self.config.dataset_normalization == "zscore":
            print("Normalizing data (zscore)")
            data = zscore_normalize_rows(data)

        dataset = torch.FloatTensor(data).to(self.config.device)

        assert(dataset.shape[1] == self.config.input_size)
        print("Dataset size:", len(dataset))

        train_dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=self.config.batch_shuffle)
        for step in tqdm(range(self.config.steps)):
            data = next(iter(train_dataloader))
            self.update(step=step, data=data)

        if self.wandb:
            self.wandb.finish()

        return self.model
    
    def save(self, model: SAE, filepath: str):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        torch.save(model, filepath)
