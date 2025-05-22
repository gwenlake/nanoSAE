import pandas as pd
import numpy as np
import torch

from .utils import tokensize


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


class FeatureAnalyzer:


    def __init__(self, model):
        self.model = model

    def get_topk_tokens(self, *, features: torch.Tensor, token_df: pd.DataFrame, feature_id: int, k: int = 20):
        
        # get the feature column
        feature_values = features[:, :feature_id].flatten()
        # feature_mean = feature_values.mean()        

        top_values, top_indices = torch.topk(feature_values, k=10*k)
        bot_values, bot_indices = torch.topk(feature_values, k=10*k, largest=False)

        top_df = token_df.iloc[top_indices]
        bot_df = token_df.iloc[bot_indices]

        top_df["feature"] = top_values.tolist()
        bot_df["feature"] = bot_values.tolist()

        top_df = top_df.sort_values('feature', ascending=False).drop_duplicates(['token'])
        bot_df = bot_df.sort_values('feature', ascending=True).drop_duplicates(['token'])
        bot_df = bot_df.sort_values('feature', ascending=False)

        return top_df[:k], bot_df[:k]
    
