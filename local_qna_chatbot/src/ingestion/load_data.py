import json
import os

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_dataset(raw_path="../data/raw/dataset.json"):
    data = load_json(raw_path)
    records = []
    for item in data:
        text = ""
        if "question" in item:
            text += item["question"] + " "
        if "answer" in item:
            text += item["answer"]
        records.append({"id": item.get("id"), "text": text})
    return records
