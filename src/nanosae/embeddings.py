import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EXCLUDED_TOKENS = ["<s>", "</s>", "<unk>", "<mask>", "<pad>"]

class Embeddings:

    def __init__(self, model: str):
        self.model = SentenceTransformer(model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.tokenizer = self.model.tokenizer

    def _embed(self, text: str, prefix: str = None, normalize_embeddings: bool = True, output_value: str = None):
        if prefix:
            text = prefix + text
        if output_value == "token_embeddings":
            return self.model.encode(text, output_value=output_value).tolist()
        return [self.model.encode(text, normalize_embeddings=normalize_embeddings).tolist()]

    def embed(self, input_text: list[str], prefix: str = None, normalize_embeddings: bool = True, output_value: str = None):
        embeddings = []
        for text in tqdm(input_text):
            embeddings += self._embed(text, prefix=prefix, normalize_embeddings=normalize_embeddings, output_value=output_value)
        return embeddings
    
    def get_tokens(self, text: str, prefix: str = None):
        if prefix:
            text = prefix + text
        output = self.tokenizer(
            text,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=False
        )
        ids = output["input_ids"]
        return self.tokenizer.convert_ids_to_tokens(ids)

    def get_token_df(self, input_text: list[str], prefix: str = None, normalize_embeddings: bool = True):
        data = []
        for text in tqdm(input_text):
            embeddings = self._embed(text, prefix=prefix, normalize_embeddings=normalize_embeddings, output_value="token_embeddings")
            tokens = self.get_tokens(text, prefix=prefix)
            for i, token in enumerate(tokens):
                if prefix and i < 4:
                    continue
                if token not in EXCLUDED_TOKENS:
                    data.append({
                        "token": token,
                        "pos": i,
                        "context": text,
                        "embeddings": embeddings[i],
                    })
        return pd.DataFrame(data)
