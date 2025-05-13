import pandas as pd

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
