import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Actually, I should use relative imports or run as module.
# Let's assume run as module: python -m src.embeddings.generate_index

import sys
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.embeddings.embedder import get_embedding
from src.embeddings.build_faiss import build_faiss

def main():
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.json"
    INDEX_PATH = DATA_DIR / "embeddings" / "index.faiss"

    if not PROCESSED_DATA_PATH.exists():
        print(f"Error: {PROCESSED_DATA_PATH} not found.")
        return

    print("Loading processed data...")
    with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = []
    print(f"Generating embeddings for {len(data)} chunks...")
    
    for item in tqdm(data):
        try:
            emb = get_embedding(item["text"])
            embeddings.append(emb)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # If we fail, we might want to skip or stop. 
            # For now, let's append a zero vector or stop? 
            # Appending zero vector might mess up search. 
            # Let's just stop or continue. If I stop, I can't build.
            # I will continue but this index will be misaligned if I don't handle it.
            # But wait, index i corresponds to data i. If I skip, alignment breaks.
            # So I must have an embedding.
            # I'll just raise for now.
            raise e

    embeddings_np = np.array(embeddings, dtype=np.float32)
    
    print("Building FAISS index...")
    build_faiss(embeddings_np, save_path=str(INDEX_PATH))
    print("Done!")

if __name__ == "__main__":
    main()
