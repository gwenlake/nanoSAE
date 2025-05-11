# nanoSAE

nanoSAE is a a simple and fast Python library for training sparse autoencoders for semantic analysis.

This repository presents the results of work carried out by the Gwenlake team (Sylvain Barthélémy, Guillaume Béguec, Antoine de Parthenay, and Elsa Doyen) as part of our investigation into the use of **Sparse Autoencoders (SAEs)** for semantic analysis.

These efforts build directly upon Anthropic’s research, as detailed in the *“Towards Monosemanticity”* article series, which explores applying SAEs to the semantic interpretation of large language models via a dictionary-learning methodology.

The code is adapted from the excellent [Dictionary Learning library](https://github.com/saprmarks/dictionary_learning), notably incorporating Anthropic’s SAE implementation from their April 2024 update ([Training SAEs](https://transformer-circuits.pub/2024/april-update/index.html#training-saes)), as well as elements from [“Disentangling Dense Embeddings with Sparse Autoencoders”](https://arxiv.org/abs/2408.00657), particularly the feature-analysis methodology.

## Install

```bash
pip install git+https://github.com/gwenlake/nanoSAE
```

## Training a model

You can train a Sparse Autoencoder as follows
```python
import pandas as pd
from nanosae import SAETrainer, TrainConfig

data = pd.read_csv("dataset.csv")

train_cfg = TrainConfig(input_size=1024)
trainer = SAETrainer(config=train_cfg)
model = trainer.train(data=data)

torch.save(model, "sae.pt")
```

## Citation

Please cite the package as follows:

```
@misc{gwenlakenanosae2025codebase,
   title = {nanoSAE},
   author = {Barthélémy, Sylvain and Béguec, Guillaume and De Parthenay, Antoine and Doyen, Elsa},
   year = {2025},
   howpublished = {\url{https://github.com/gwenlake/nanoSAE}},
}
```