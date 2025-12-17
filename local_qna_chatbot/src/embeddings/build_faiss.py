import os
import numpy as np
import faiss
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def build_faiss(embeddings: np.ndarray, ids: Optional[List[int]] = None, save_path: Optional[str] = None):
    """
    Build a FAISS index from the given embeddings.
    
    Args:
        embeddings: Numpy array of shape (n_samples, embedding_dim)
        ids: Optional list of IDs corresponding to the embeddings
        save_path: Optional path to save the FAISS index
        
    Returns:
        faiss.Index: The built FAISS index
    """
    try:
        # Convert embeddings to float32 if they aren't already
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        # Get the dimension of the embeddings
        dimension = embeddings.shape[1]
        
        # Create the FAISS index
        index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        index.add(embeddings)
        
        # Save the index if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            faiss.write_index(index, save_path)
            logger.info(f"FAISS index saved to {save_path}")
            
        return index
        
    except Exception as e:
        logger.error(f"Error building FAISS index: {str(e)}")
        raise

def load_faiss_index(load_path: str):
    """
    Load a FAISS index from disk.
    
    Args:
        load_path: Path to the saved FAISS index
        
    Returns:
        faiss.Index: The loaded FAISS index
    """
    try:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"FAISS index not found at {load_path}")
            
        index = faiss.read_index(load_path)
        logger.info(f"FAISS index loaded from {load_path}")
        return index
        
    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}")
        raise

def search_faiss(index, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for similar vectors.
    
    Args:
        index: The FAISS index to search in
        query_embedding: Query embedding to search with
        k: Number of nearest neighbors to return
        
    Returns:
        Tuple containing:
            - Distances to the nearest neighbors
            - Indices of the nearest neighbors
    """
    try:
        # Ensure the query embedding is in the correct format
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
            
        # Reshape if necessary
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search the index
        distances, indices = index.search(query_embedding, k)
        
        return distances[0], indices[0]
        
    except Exception as e:
        logger.error(f"Error searching FAISS index: {str(e)}")
        raise
