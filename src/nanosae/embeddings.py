import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class Embeddings:

    def __init__(self, model: str):
        self.model = SentenceTransformer(model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def _embed(self, text: str, prefix: str = None, normalize_embeddings: bool = True):
        if prefix:
            text = prefix + text
        return self.model.encode(text, normalize_embeddings=normalize_embeddings).tolist()

    def embed(self, input_text: list[str], prefix: str = None, normalize_embeddings: bool = True):
        embeddings = []
        if len(input_text)==1:
            text = input_text[0]
            embeddings.append(self._embed(text, prefix=prefix, normalize_embeddings=normalize_embeddings))
        else:
            for text in tqdm(input_text):
                embeddings.append(self._embed(text, prefix=prefix, normalize_embeddings=normalize_embeddings))
        return embeddings
