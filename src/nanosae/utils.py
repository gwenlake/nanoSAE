import re
import tiktoken
import numpy as np
import torch


def get_device() -> torch.device:
    """
    Helper function to return the correct device (cuda, mps, or cpu).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def clean_text(text: str):
    text = text.replace("\n", " ")
    text = re.sub(' +', ' ', text)
    text = text.strip()
    text = text.strip("\"")
    text = text.strip("'")
    return text

def zscore_normalize(data: list[float], eps: float = 1e-5):
    mu = np.mean(data)
    std = np.std(data)
    return (data - mu) / (std + eps)

def zscore_normalize_rows(data: np.ndarray):
    return np.apply_along_axis(zscore_normalize, axis=1, arr=data)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def tokensize(text: str, encoding_name: str = "cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    token_integers = encoding.encode(text)
    return [encoding.decode_single_token_bytes(token).decode("utf8") for token in token_integers]
