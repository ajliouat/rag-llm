from evaluate import load

bleu = load("bleu")
rouge = load("rouge")

def evaluate(dataset):
    results = []
    for item in dataset:
        prediction = "Paris"  # Replace with model prediction
        reference = item["answer"]
        results.append({
            "bleu": bleu.compute(predictions=[prediction], references=[reference])["bleu"],
            "rouge": rouge.compute(predictions=[prediction], references=[reference])["rougeL"]
        })
    return results