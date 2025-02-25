from src.evaluation.metrics import evaluate
from src.data.dataloader import load_dataset

def run_benchmark():
    _, val_dataset = load_dataset()
    results = evaluate(val_dataset)
    print(results)

if __name__ == "__main__":
    run_benchmark()