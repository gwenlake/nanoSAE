from typing import (
    Iterable,
    TypeVar,
    Callable,
)

import re
import tiktoken
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm


T = TypeVar("T")


EXCLUDED_TOKENS = [
    "<s>",
    "</s>",
    "<unk>",
    "<mask>",
    "<pad>",
    "[UNK]",
    "[SEP]",
    "[PAD]",
    "[CLS]",
    "[MASK]"
]


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


def to_str_tokens(text: str, tokenizer: Callable, prefix: str = None):
    if prefix:
        text = prefix + text
    output = tokenizer(
        text,
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=False
    )    
    return tokenizer.convert_ids_to_tokens(output["input_ids"])

def sample_unique_indices(large_number: int, small_number: int):
    """Samples a small number of unique indices from a large number of indices."""
    weights = torch.ones(large_number)  # Equal weights for all indices
    sampled_indices = torch.multinomial(weights, small_number, replacement=False)
    return sampled_indices

def merge_lists(*lists: Iterable[T]) -> list[T]:
    """Merges a bunch of lists into a single list."""
    return [item for sublist in lists for item in sublist]

def print_gpu_mem(step_name=""):
    print(f"{step_name} ~ {np.round(torch.cuda.memory_allocated()/2e30, 2)} GiB allocated on GPU.")

def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")
    
def nested_list_to_numpy(data):
    """
    Helper function to convert a pandas nested list to numpy.
    """
    new_data = np.array(data).ravel()
    new_data = np.reshape(new_data, (len(data),-1)) 
    return new_data

def nested_list_torch(data):
    data = nested_list_to_numpy(data)
    data = torch.FloatTensor(data).to("cpu")
    return data
