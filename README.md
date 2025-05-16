# nanoSAE

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/gwenlake/nanoSAE)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/gwenlake/nanoSAE/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A simple and fast Python library for training Sparse Autoencoders (SAEs) for semantic analysis.


## 📋 Overview

This repository presents the results of work carried out by the Gwenlake team (Sylvain Barthélémy, Guillaume Béguec, Antoine de Parthenay, and Elsa Doyen) as part of our investigation into the use of **Sparse Autoencoders (SAEs)** for semantic analysis.

These efforts build directly upon Anthropic's research, as detailed in the *"Towards Monosemanticity"* article series, which explores applying SAEs to the semantic interpretation of large language models via a dictionary-learning methodology.

The code is adapted from the excellent [Dictionary Learning library](https://github.com/saprmarks/dictionary_learning), notably incorporating Anthropic's SAE implementation from their April 2024 update ([Training SAEs](https://transformer-circuits.pub/2024/april-update/index.html#training-saes)), as well as elements from ["Disentangling Dense Embeddings with Sparse Autoencoders"](https://arxiv.org/abs/2408.00657), particularly the feature-analysis methodology.

## ✨ Features

- Simple API for training and using Sparse Autoencoders
- Fast implementation optimized for performance
- Comprehensive tools for semantic analysis
- Integration with popular ML frameworks
- Visualization utilities for feature analysis

## 🔧 Installation

### Requirements

- Python 3.12 or higher
- PyTorch 2.7.0 or higher

### Install from GitHub

```bash
pip install git+https://github.com/gwenlake/nanoSAE
```

## 📚 Usage

### Training a Sparse Autoencoder

```python
import pandas as pd
import torch
from nanosae import SAETrainer, TrainConfig

# Load your dataset
data = pd.read_csv("dataset.csv")

# Configure the training parameters
train_cfg = TrainConfig(
    input_size=1024,
    hidden_size=4096,  # Number of features to learn
    l1_coefficient=0.001  # Sparsity coefficient
)

# Initialize the trainer
trainer = SAETrainer(config=train_cfg)

# Train the model
model = trainer.train(data=data)

# Save the trained model
torch.save(model, "sae.pt")
```

### Using a Trained Model

```python
import torch
from nanosae import SAEModel

# Load the trained model
model = torch.load("sae.pt")

# Process new data
activations = model.encode(new_data)
reconstructed = model.decode(activations)

# Analyze feature activations
feature_stats = model.analyze_features(test_data)
```

## 📊 Examples

For more detailed examples, check out the [examples directory](https://github.com/gwenlake/nanoSAE/tree/main/examples).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you use nanoSAE in your research, please cite it as follows:

```bibtex
@misc{gwenlakenanosae2025codebase,
   title = {nanoSAE},
   author = {Sylvain Barthélémy, Guillaume Béguec, Antoine de Parthenay, and Elsa Doyen},
   year = {2025},
   howpublished = {\url{https://github.com/gwenlake/nanoSAE}},
}
```

## 🙏 Acknowledgements

- [Anthropic](https://www.anthropic.com/) for their groundbreaking research on Sparse Autoencoders
- [Dictionary Learning library](https://github.com/saprmarks/dictionary_learning) for the foundational implementation
- All contributors who have helped improve this project
