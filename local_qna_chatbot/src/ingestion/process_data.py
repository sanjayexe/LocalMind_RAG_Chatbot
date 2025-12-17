import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.load_data import load_dataset
from src.ingestion.cleaner import clean_text
from src.ingestion.chunker import process_records

def main():
    # Ensure the processed directory exists
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process the dataset
    print("Loading dataset...")
    records = load_dataset()
    
    # Clean the text
    print("Cleaning text...")
    for record in records:
        record["text"] = clean_text(record["text"])
    
    # Chunk the text
    print("Chunking text...")
    processed_data = process_records(records)
    
    # Save the processed data
    output_path = processed_dir / "processed_data.json"
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    main()
