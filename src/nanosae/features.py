import pandas as pd

from .utils import tokensize

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

def get_topfreq_tokens(data: list, n: int):
    top_tokens = []
    for d in data:
        tokens = tokensize(d)
        tokens = [t for t in tokens if len(t)>2]
        top_tokens.extend(tokens)
    df = pd.DataFrame({"token": top_tokens})
    df = df.groupby("token").size().reset_index(name='count')
    return df.sort_values("count", ascending=False).head(n)

def get_topk_features(data, num_features: int, k: int = 10):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    top_features = []
    for id in range(num_features):
        topk = data[data.feature_id == id].sort_values("feature_value", ascending=False).head(k)
        top_features.extend(topk.to_dict(orient="records"))
    top_features = pd.DataFrame(top_features)
    top_features = top_features.groupby(by=["token", "feature_id"])["feature_value"].mean().reset_index(name="average_feature_value")
    return top_features

