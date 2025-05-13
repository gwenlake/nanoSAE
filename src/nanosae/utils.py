import numpy as np

def zscore_normalize(data: list[float], eps: float = 1e-5):
    mu = np.mean(data)
    std = np.std(data)
    return (data - mu) / (std + eps)

def zscore_normalize_rows(data: np.ndarray):
    return np.apply_along_axis(zscore_normalize, axis=1, arr=data)
