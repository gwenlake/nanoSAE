import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import pandas as pd

from .config import SAEConfig


class SAE(nn.Module):
    """
    The SAE architecture and initialization used in
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.encoder = nn.Linear(config.input_size, config.hidden_size, bias=True)
        self.decoder = nn.Linear(config.hidden_size, config.input_size, bias=True)

        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        self.initialize_weights()
        self.initialize_biases()

    def initialize_biases(self):
        # initialize biases to zeros
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

    def initialize_weights(self):
        # init and normalize columns of w
        w = torch.randn(self.input_size, self.hidden_size)
        w = w / w.norm(dim=0, keepdim=True) * 0.1

        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

    def encode(self, x):
        return nn.ReLU()(self.encoder(x))

    def decode(self, f):
        return self.decoder(f)

    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def features(self, x):
        f = self.encode(x)
        f = (f * self.decoder.weight.norm(p=2, dim=0)).sum(dim=-1).mean()
        return f.detach().numpy()
    
    def features_df(self, x, top_k: int = None):
        if isinstance(x, list):
            x = torch.FloatTensor(np.array(x)).to(self.config.device)
        f = self.features(x)
        f = pd.DataFrame({"feature_id": range(self.hidden_size), "feature_value": f[0]})
        if top_k is not None:
            f = f.sort_values("feature_value", ascending=False).head(top_k)
        return f
    
    @staticmethod
    def from_pretrained(path: str, device="cpu") -> "SAE":
        model = torch.load(path, weights_only=False)
        if device is not None:
            model.to(device)
        return model
