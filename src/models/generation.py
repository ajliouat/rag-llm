from transformers import AutoModelForCausalLM, AutoTokenizer

class GenerativeModel:
    def __init__(self, model_name="google/flan-t5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_answer(self, question: str, documents: list):
        context = " ".join(documents)
        input_text = f"Question: {question}\nContext: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)