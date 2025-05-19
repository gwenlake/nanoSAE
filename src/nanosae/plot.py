import numpy as np
import pandas as pd

from umap import UMAP
import hdbscan
import plotly.express as px

from .sae import SAE


def plot_umap(data):
    hdb = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=3).fit(data)
    umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.05, metric="cosine")

    data_reduced = umap.fit_transform(data)

    df_umap = (
        pd.DataFrame(data_reduced, columns=['x', 'y'])
        .assign(cluster=lambda df: hdb.labels_.astype(str))
        .query('cluster != "-1"')
        .sort_values(by='cluster')
    )

    fig = px.scatter(df_umap, x='x', y='y', color='cluster')
    fig.show()


def plot_dictionary(model: SAE):
    data = pd.DataFrame(model.decoder.weight.T.cpu().detach().numpy())
    plot_umap(data)
