import unittest
from src.models.retrieval import DenseRetriever

class TestRetrieval(unittest.TestCase):
    def test_retrieve_documents(self):
        retriever = DenseRetriever()
        results = retriever.retrieve_documents("What is the capital of France?", top_k=5)
        self.assertEqual(len(results), 5)

if __name__ == "__main__":
    unittest.main()