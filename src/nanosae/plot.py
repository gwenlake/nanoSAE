import numpy as np
import pandas as pd

from umap import UMAP
import hdbscan
import plotly.express as px
import matplotlib.pyplot as plt

from .models import SAE


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


def plot_neurons(activations, num_neurons=50, neurons_per_row=10, save_path=None):
    num_rows = (num_neurons + neurons_per_row - 1) // neurons_per_row  
    fig, axes = plt.subplots(num_rows, neurons_per_row, figsize=(neurons_per_row * 2, num_rows * 2))
    axes = axes.flatten()

    for i in range(num_neurons):
        if i >= activations.shape[1]:
            break
        ax = axes[i]
        ax.imshow(activations[:, i].reshape(-1, 1), aspect='auto', cmap='hot')
        ax.set_title(f'Neuron {i+1}', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600)

    # plt.show()

    return fig
