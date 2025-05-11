import numpy as np


def zscore(v: list[float]):
    return (v - np.mean(v)) / np.std(v)

def zscore_pd(data):
    ndata = []
    for i, r in data.iterrows():
        n = zscore(r)
        ndata.append(n)
    return np.array(ndata)

