import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class Embeddings:

    def __init__(self, model: str):
        self.model = SentenceTransformer(model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.tokenizer = self.model.tokenizer

    def _embed(self, text: str, prefix: str = None, output_value: str = None):
        if prefix:
            text = prefix + text
        if output_value == "token_embeddings":
            return [list(e) for e in self.model.encode(text, output_value=output_value)]
        return self.model.encode(text).tolist()

    def embed(self, input_text: list[str], prefix: str = None, output_value: str = None):
        embeddings = []
        if output_value == "token_embeddings":
            for text in input_text:
                embeddings += self._embed(text, prefix=prefix, output_value=output_value)
            return embeddings
        
        if len(input_text) == 1:
            text = input_text[0]
            embeddings.append(self._embed(text, prefix=prefix))
        else:
            for text in tqdm(input_text):
                embeddings.append(self._embed(text, prefix=prefix))
        return embeddings
    
    def get_tokens(self, text: str):
        return self.tokenizer.tokenize(text)
