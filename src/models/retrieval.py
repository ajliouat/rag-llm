import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from src.utils.faiss_utils import build_faiss_index

class DenseRetriever:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.index = build_faiss_index()

    def retrieve_documents(self, query: str, top_k: int = 5):
        query_embedding = self.model(**self.tokenizer(query, return_tensors="pt")).last_hidden_state.mean(dim=1).detach().numpy()
        distances, indices = self.index.search(query_embedding, top_k)
        return indices