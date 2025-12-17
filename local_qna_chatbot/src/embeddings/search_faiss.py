import numpy as np
import faiss
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def search(
    index,
    query_embedding: np.ndarray,
    k: int = 5,
    return_distances: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Search the FAISS index for similar vectors.
    
    Args:
        index: The FAISS index to search in
        query_embedding: Query embedding to search with
        k: Number of nearest neighbors to return
        return_distances: Whether to return distances along with indices
        
    Returns:
        If return_distances is True, returns a tuple of (distances, indices)
        If return_distances is False, returns only the indices
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
        
        if return_distances:
            return distances[0], indices[0]
        return indices[0]
        
    except Exception as e:
        logger.error(f"Error searching FAISS index: {str(e)}")
        raise
