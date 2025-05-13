import re
import tiktoken
import numpy as np
import pandas as pd

def clean_text(text: str):
    text = text.replace("\n", " ")
    text = re.sub(' +', ' ', text)
    text = text.strip()
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

def get_top_tokens(data: list, n: int):
    top_tokens = []
    for d in data:
        tokens = tokensize(d)
        tokens = [t for t in tokens if len(t)>2]
        top_tokens.extend(tokens)
    df = pd.DataFrame({"token": top_tokens})
    df = df.groupby("token").size().reset_index(name='count')
    return df.sort_values("count", ascending=False).head(n)

def make_token_df(text: str):
    data = []
    tokens = tokensize(text)
    for pos, token in enumerate(tokens):
        x = {}
        x["token"] = token
        x["pos"] = pos
        x["context"] = text
        data.append(x)
    return pd.DataFrame(data)