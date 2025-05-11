# nanoSAE

nanoSAE is a a simple and fast Python library for training sparse autoencoders for semantic analysis.

This repository presents the outcome of work carried out by the Gwenlake team (Sylvain Barthélémy, Guillaume Béguec, Antoine de Parthenay, and Elsa Doyen) as part of our investigations into the use of Sparse Autoencoders (SAEs) for semantic analysis.

These efforts build directly on the research by Anthropic, as detailed in the “Towards Monosemanticity” article series, which explores the application of SAEs to the semantic interpretation of large language models via a dictionary-learning methodology.

The code is adapted from the excellent Dictionary Learning library (https://github.com/saprmarks/dictionary_learning), notably incorporating Anthropic’s SAE implementation from their April 2024 update (https://transformer-circuits.pub/2024/april-update/index.html#training-saes), as well as elements from “Disentangling Dense Embeddings with Sparse Autoencoders” (https://arxiv.org/abs/2408.00657), notably the methodology to analyze features.

# Running the demo

1. Run data.py to generate a parquet file with embedding vectors on 56k financial headlines.
2. Run train.py to estimate an SAE using dictionnary learning on this dataset (adjust the parameters using config/anthropic-april2024.json).
3. Plot umap and analyse the results using analysis.py
4. To be done: analyse features using an Interpretor and a Predictor, using llama 3 (or another small model).
