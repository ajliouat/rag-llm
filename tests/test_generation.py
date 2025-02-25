import unittest
from src.models.generation import GenerativeModel

class TestGeneration(unittest.TestCase):
    def test_generate_answer(self):
        model = GenerativeModel()
        answer = model.generate_answer("What is the capital of France?", ["Paris is the capital of France."])
        self.assertIn("Paris", answer)

if __name__ == "__main__":
    unittest.main()