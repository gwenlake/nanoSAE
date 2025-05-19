import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import pandas as pd

from .config import SAEConfig


class SAE(nn.Module):

    def __init__(self, config: SAEConfig):
        super().__init__()

        self.config = config
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        if self.config.architecture == "standard":
            self.encoder = nn.Linear(self.config.input_size, self.config.hidden_size, bias=True)
            self.decoder = nn.Linear(self.config.hidden_size, self.config.input_size, bias=False)
            self.initialize_weights_basic()
            self.bias = nn.Parameter(torch.zeros(self.config.input_size))
        
        elif self.config.architecture == "anthropic":
            self.encoder = nn.Linear(self.config.input_size, self.config.hidden_size, bias=True)
            self.decoder = nn.Linear(self.config.hidden_size, self.config.input_size, bias=True)
            self.initialize_weights_basic()
            init.zeros_(self.encoder.bias)
            init.zeros_(self.decoder.bias)

    def initialize_weights_basic(self):
        # init and normalize columns of w
        w = torch.randn(self.config.input_size, self.config.hidden_size)
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

    # def scale_biases(self, scale: float):
    #     self.b_dec.data *= scale
    #     self.b_enc.data *= scale
    #     self.threshold.data *= scale

    def encode(self, x):
        if self.config.architecture == "anthropic":
            return nn.ReLU()(self.encoder(x))

        # elif self.config.architecture == "jumprelu":
        #     if self.apply_b_dec_to_input:
        #         x = x - self.b_dec
        #     pre_jump = x @ self.W_enc + self.b_enc
        #     f = nn.ReLU()(pre_jump * (pre_jump > self.threshold))
        #     return f

        return nn.ReLU()(self.encoder(x - self.bias))

    def decode(self, f):

        if self.config.architecture == "anthropic":
            return self.decoder(f)

        # elif self.config.architecture == "jumprelu":
        #     return f @ self.W_dec + self.b_dec

        return self.decoder(f) + self.bias

    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def features(self, x):
        f = self.encode(x)

        if self.config.architecture == "anthropic":
            f = (f * self.decoder.weight.norm(p=2, dim=0)).sum(dim=-1).mean()

        return f.detach().numpy()
    
    def features_df(self, x, top_k: int = None):
        if isinstance(x, list):
            x = torch.FloatTensor(np.array(x)).to(self.config.device)
        f = self.features(x)
        f = pd.DataFrame({"feature_id": range(self.config.hidden_size), "feature_value": f[0]})
        if top_k is not None:
            f = f.sort_values("feature_value", ascending=False).head(top_k)
        return f
    
    @staticmethod
    def from_pretrained(path: str, device="cpu") -> "SAE":
        model = torch.load(path, weights_only=False)
        if device is not None:
            model.to(device)
        return model