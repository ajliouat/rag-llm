import json

def preprocess_data(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # Preprocess data (e.g., tokenization, cleaning)
    processed_data = [{"question": item["question"], "context": item["context"], "answer": item["answer"]} for item in data]
    
    with open(output_path, "w") as f:
        json.dump(processed_data, f)