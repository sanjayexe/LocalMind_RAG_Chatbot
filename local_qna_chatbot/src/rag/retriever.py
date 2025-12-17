import os
import json
import numpy as np
from typing import List
from pathlib import Path
from ..embeddings.search_faiss import search
from ..embeddings.build_faiss import load_faiss_index
from ..embeddings.embedder import get_embedding

# Define paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
INDEX_PATH = DATA_DIR / "embeddings" / "index.faiss"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.json"

def retrieve_relevant_context(
    query: str, 
    k: int = 3, 
    index=None, 
    dataset: List[dict] = None
) -> List[str]:
    """
    Retrieve relevant context for a given query using FAISS semantic search.
    
    Args:
        query (str): The user's question or query
        k (int): Number of relevant chunks to retrieve
        index: FAISS index object (optional)
        dataset: List of data dicts with "text" key (optional)
        
    Returns:
        List[str]: List of relevant text chunks
    """
    try:
        # Load index if not provided
        if index is None:
            if not INDEX_PATH.exists():
                print(f"Warning: FAISS index not found at {INDEX_PATH}. Please run the ingestion script.")
                return []
            index = load_faiss_index(str(INDEX_PATH))
        
        # Load dataset if not provided
        processed_data = dataset
        if processed_data is None:
            if not PROCESSED_DATA_PATH.exists():
                 print(f"Warning: Processed data not found at {PROCESSED_DATA_PATH}.")
                 return []
            with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
                processed_data = json.load(f)
            
        # Generate embedding for the query
        query_embedding = get_embedding(query)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search
        distances, indices = search(index, query_embedding, k=k, return_distances=True)
        
        results = []
        for idx in indices:
            if 0 <= idx < len(processed_data):
                results.append(processed_data[idx]["text"])
                
        return results
        
    except Exception as e:
        print(f"Error in retrieval: {str(e)}")
        return []
