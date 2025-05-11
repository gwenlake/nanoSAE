
import re
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


class Embeddings:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
    def clean_text(self, text: str):
        text = text.replace("\n", " ")
        text = re.sub(' +', ' ', text)
        text = text.strip()
        return text

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _embed(self, input_text: list[str], normalize_embeddings: bool = False):
        # Tokenize the input texts
        batch_dict = self.tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors='pt')
 
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
 
        # normalize embeddings
        if normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()

    def embed(self, input_text: list[str], normalize_embeddings: bool = False, clean_text: bool = False, prefix: str = None):
        embeddings = []
        batch_size = 100
        for i in range(0, len(input_text), batch_size):
            i_end = min(len(input_text), i+batch_size)
            batch = input_text[i:i_end]
            for i, _ in enumerate(batch):
                if clean_text:
                    batch[i] = self.clean_text(batch[i])
                if prefix:
                    batch[i] = prefix + batch[i]
            embeddings += self._embed(batch, normalize_embeddings=normalize_embeddings)
        return embeddings
