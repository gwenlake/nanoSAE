import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class SAE(nn.Module):
    """
    The SAE architecture and initialization used in
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.Linear(input_size, hidden_size, bias=True)
        self.decoder = nn.Linear(hidden_size, input_size, bias=True)

        # initialize encoder and decoder weights
        w = torch.randn(input_size, hidden_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

        # initialize biases to zeros
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

    def encode(self, x):
        return nn.ReLU()(self.encoder(x))

    def decode(self, f):
        return self.decoder(f)

    def forward(self, x, output_features=False):
        f     = self.encode(x)
        x_hat = self.decode(f)
        # multiply f by decoder column norms
        f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    @property
    def dictionary(self):
        """Dictionary elements are simply the normalized decoder weights."""
        return F.normalize(self.decoder.weight, dim=0)
    
    @staticmethod
    def from_pretrained(path: str, device=None):
        model = torch.load(path, weights_only=False)
        if device is not None:
            model.to(device)
        return model
