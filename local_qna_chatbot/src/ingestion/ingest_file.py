import os
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import faiss
from ..ingestion.cleaner import clean_text
from ..ingestion.chunker import chunk_text
from ..embeddings.embedder import get_embedding

def process_uploaded_file(file_path: str) -> Tuple[faiss.Index, List[Dict]]:
    """
    Process an uploaded file: read, clean, chunk, embed, and build FAISS index.
    
    Args:
        file_path: Path to the uploaded file
        
    Returns:
        Tuple containing:
            - faiss.Index: The built FAISS index
            - List[Dict]: List of chunks with metadata [{"id": 1, "text": "..."}]
    """
    path = Path(file_path)
    text = ""
    
    # Read file based on extension
    try:
        if path.suffix.lower() == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        elif path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        else:
             # Try reading as text
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        raise ValueError(f"Could not read file {path.name}: {e}")

    if not text.strip():
        raise ValueError("File is empty")

    # Clean
    cleaned_text = clean_text(text)
    
    # Chunk
    chunks = chunk_text(cleaned_text)
    
    # Create dataset structure
    dataset = []
    embeddings = []
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        try:
            emb = get_embedding(chunk)
            embeddings.append(emb)
            dataset.append({"id": i + 1, "text": chunk})
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")
            
    if not embeddings:
        raise ValueError("No embeddings were generated")
        
    # Build FAISS Index
    embeddings_np = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    return index, dataset
