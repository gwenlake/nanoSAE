from typing import Optional, Callable
from collections import namedtuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .models import SAE


def get_lr_schedule(
    total_steps: int,
    warmup_steps: int,
    decay_start: Optional[int] = None,
    resample_steps: Optional[int] = None,
    sparsity_warmup_steps: Optional[int] = None,
) -> Callable[[int], float]:
    """
    Creates a learning rate schedule function with linear warmup followed by an optional decay phase.

    Note: resample_steps creates a repeating warmup pattern instead of the standard phases, but
    is rarely used in practice.

    Args:
        total_steps: Total number of training steps
        warmup_steps: Steps for linear warmup from 0 to 1
        decay_start: Optional step to begin linear decay to 0
        resample_steps: Optional period for repeating warmup pattern
        sparsity_warmup_steps: Used for validation with decay_start

    Returns:
        Function that computes LR scale factor for a given step
    """
    if decay_start is not None:
        assert resample_steps is None, (
            "decay_start and resample_steps are currently mutually exclusive."
        )
        assert 0 <= decay_start < total_steps, "decay_start must be >= 0 and < steps."
        assert decay_start > warmup_steps, "decay_start must be > warmup_steps."
        if sparsity_warmup_steps is not None:
            assert decay_start > sparsity_warmup_steps, (
                "decay_start must be > sparsity_warmup_steps."
            )

    assert 0 <= warmup_steps < total_steps, "warmup_steps must be >= 0 and < steps."

    if resample_steps is None:

        def lr_schedule(step: int) -> float:
            if step < warmup_steps:
                # Warm-up phase
                return step / warmup_steps

            if decay_start is not None and step >= decay_start:
                # Decay phase
                return (total_steps - step) / (total_steps - decay_start)

            # Constant phase
            return 1.0
    else:
        assert 0 < resample_steps < total_steps, "resample_steps must be > 0 and < steps."

        def lr_schedule(step: int) -> float:
            return min((step % resample_steps) / warmup_steps, 1.0)

    return lr_schedule


def get_sparsity_warmup_fn(
    total_steps: int, sparsity_warmup_steps: Optional[int] = None
) -> Callable[[int], float]:
    """
    Return a function that computes a scale factor for sparsity penalty at a given step.

    If `sparsity_warmup_steps` is None or 0, returns 1.0 for all steps.
    Otherwise, scales from 0.0 up to 1.0 across `sparsity_warmup_steps`.
    """

    if sparsity_warmup_steps is not None:
        assert 0 <= sparsity_warmup_steps < total_steps, (
            "sparsity_warmup_steps must be >= 0 and < steps."
        )

    def scale_fn(step: int) -> float:
        if not sparsity_warmup_steps:
            # If it's None or zero, we just return 1.0
            return 1.0
        else:
            # Gradually increase from 0.0 -> 1.0 as step goes from 0 -> sparsity_warmup_steps
            return min(step / sparsity_warmup_steps, 1.0)

    return scale_fn

class SAETrainer:
    """
    Standard SAE training scheme following the Anthropic April update.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    Decoder column norms are NOT constrained to 1.
    This trainer does not support resampling or ghost gradients.
    This trainer will have fewer dead neurons than the standard trainer.
    """
    def __init__(self, config: TrainConfig):

        self.config = config

        # self.seed = seed
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)

        # initialize dictionary
        self.model = SAE(input_size=self.config.input_size, hidden_size=self.config.hidden_size)

        if self.config.device is None:
            self.config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.config.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))

        lr_fn = get_lr_schedule(self.config.steps, self.config.warmup_steps, self.config.decay_start, resample_steps=None, sparsity_warmup_steps=self.config.sparsity_warmup_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(self.config.steps, self.config.sparsity_warmup_steps)

    def loss(self, x, step: int, logging=False, **kwargs):

        sparsity_scale = 1.0
        if self.sparsity_warmup_fn:
            sparsity_scale = self.sparsity_warmup_fn(step)

        x_hat, f = self.model(x, output_features=True)
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = (f * self.model.decoder.weight.norm(p=2, dim=0)).sum(dim=-1).mean()

        loss = recon_loss + self.config.l1_penalty * sparsity_scale * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss' : l2_loss.item(),
                    'mse_loss' : recon_loss.item(),
                    'sparsity_loss' : l1_loss.item(),
                    'loss' : loss.item()
                }
            )


    def update(self, step, data):
        data = data.to(self.config.device)
        self.optimizer.zero_grad()
        loss = self.loss(data, step=step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

    def train(self, data):
        dataset = torch.FloatTensor(data).to(self.config.device)

        assert(dataset.shape[1] == self.config.input_size)
        print("Dataset size:", len(dataset))

        train_dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=self.config.batch_shuffle)
        for step in tqdm(range(self.config.steps)):
            data = next(iter(train_dataloader))
            self.update(step=step, data=data)

        return self.model
    