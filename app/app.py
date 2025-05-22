import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px 
import seaborn as sns
import altair as alt


from nanosae.models import SAE
from nanosae.features import FeatureAnalyzer
from nanosae.plot import plot_neurons
from nanosae.utils import nested_list_to_numpy, nested_list_torch


LOCAL_DIR = "/Users/sylvain/Library/CloudStorage/OneDrive-Bibliothèquespartagées-Gwenlake/Research - Documents/Models/sae-semantics"


def load_data():
    data = pd.read_parquet(os.path.join(LOCAL_DIR, "data", "sp500-financial-news", "sp500-financial-news-embeddings-01.parquet"))
    return data

def load_model():
    model = SAE.from_pretrained(os.path.join(LOCAL_DIR, "models", "sp500-financial-news", f"H10000-S100000-B4096.pt"))
    return model

def plot_topk_tokens(subplot, data, color: str):
    plt.subplot(1, 2, subplot)
    ax = sns.barplot(y='token', x='activation', data=data, color=color)
    # sns.despine(left=True)
    # ax.grid(False)
    # ax.tick_params(bottom=True, left=False)
    return None

def get_features(model, data):
    data = nested_list_torch(data)
    features, _ = model(data)
    return features


def main():

    st.set_page_config(page_icon="📥", page_title="nanoSAE", layout="wide")

    st.title("nanoSAE")
    st.write("A Simple and Fast Library for Training Sparse Autoencoders for Semantic Analysis")

    model = load_model()
    token_df = load_data()
    token_df = token_df[:500]

    # uniq_token_df = token_df.drop_duplicates(subset="token")
    # token_df.sort_values('count', ascending=False).drop_duplicates(['Sp','Mt'])
    # uniq_token_df = token_df.groupby("token")

    features = get_features(model, token_df["embeddings"].tolist())

    analyzer = FeatureAnalyzer(model)
    top_tokens, bot_tokens = analyzer.get_topk_tokens(features=features, token_df=token_df, feature_id=1)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Top Features")
        st.write(alt.Chart(top_tokens).mark_bar().encode(
            x="feature",
            y=alt.X('token', sort=None),
        ).properties(height=600))

    with col2:
        st.header("Bottom Features")
        st.write(alt.Chart(bot_tokens).mark_bar().encode(
            x="feature",
            y=alt.X('token', sort=None),
        ).properties(height=600))

    st.header("Neurons")
    fig = plot_neurons(features.detach().numpy(), num_neurons=50, neurons_per_row=10)
    st.pyplot(fig)

    st.header("Top Context")
    context_list = list(set(top_tokens["context"].tolist()))
    for context in context_list:

        rows = top_tokens[top_tokens.context == context]
        rows = rows.sort_values('pos')

        max_f = rows.feature.max()
        min_f = rows.feature.min()

        text = ""
        for i, row in rows.iterrows():
            txt = row["token"]
            transparency = 1 - (row["feature"] - min_f) / (max_f-min_f)
            if not transparency:
                transparency = 1.0
            txt = txt.replace("▁", " ")
            text += f"""<span style="background: rgba(51, 170, 51, {transparency}); padding: 1px;">{txt}</span>"""

        st.markdown(text, unsafe_allow_html=True)

    # top_context = list(set(top_tokens["context"].tolist()))

main()
