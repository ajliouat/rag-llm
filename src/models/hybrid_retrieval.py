from src.models.retrieval import DenseRetriever
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self):
        self.dense_retriever = DenseRetriever()
        self.bm25 = BM25Okapi([])  # Initialize with your corpus

    def retrieve_documents(self, query: str, top_k: int = 5):
        dense_results = self.dense_retriever.retrieve_documents(query, top_k)
        bm25_results = self.bm25.get_top_n(query.split(), self.bm25.corpus, n=top_k)
        return list(set(dense_results + bm25_results))