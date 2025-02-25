from datasets import load_dataset

def load_dataset():
    train_dataset = load_dataset("json", data_files="data/processed/train.json")
    val_dataset = load_dataset("json", data_files="data/processed/val.json")
    return train_dataset, val_dataset