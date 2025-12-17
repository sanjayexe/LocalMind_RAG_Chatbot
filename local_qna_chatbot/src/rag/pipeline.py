from typing import Dict, Any, Optional
from .retriever import retrieve_relevant_context
from .generator import generate_response
from ..ingestion.ingest_file import process_uploaded_file

class RAGPipeline:
    def __init__(self, k_context: int = 2, temperature: float = 0.7):
        """
        Initialize the RAG pipeline.
        
        Args:
            k_context (int): Number of context chunks to retrieve
            temperature (float): Controls randomness in generation (0.0 to 1.0)
        """
        self.k_context = k_context
        self.temperature = temperature
        self.initialized = False
        self.index = None
        self.dataset = None
    
    def initialize(self, documents_path: Optional[str] = None):
        """
        Initialize the pipeline with documents.
        
        Args:
            documents_path (str, optional): Path to documents for initialization
        """
        if documents_path:
             try:
                 print(f"Initializing RAG pipeline with document: {documents_path}")
                 self.index, self.dataset = process_uploaded_file(documents_path)
                 self.initialized = True
                 print(f"RAG pipeline initialized successfully with {len(self.dataset)} chunks.")
             except Exception as e:
                 print(f"Error initializing RAG pipeline: {e}")
                 self.initialized = False
                 raise e
        else:
             # If no document provided, we might be in "default" mode using disk index
             # But for this specific user request, we want to rely on upload.
             # State remains uninitialized if we strictly require upload.
             # However, let's allow "default" only if we want to fallback, but previously code didn't load anything.
             # Let's say we are initialized ONLY if we have data.
             pass
    
    def process_query(self, query: str, temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query (str): The user's question or query
            temperature (float, optional): Controls randomness in generation (0.0 to 1.0)
            
        Returns:
            Dict[str, Any]: Dictionary containing the response and metadata
        """
        if not self.initialized:
            return {"error": "RAG pipeline not initialized. Please upload a document first."}
            
        try:
            # Use provided temperature or instance temperature
            temp = temperature if temperature is not None else self.temperature
            
            # Retrieve relevant context
            # Pass the in-memory index and dataset
            context = retrieve_relevant_context(
                query, 
                k=self.k_context,
                index=self.index,
                dataset=self.dataset
            )
            
            # Generate response using the context
            response = generate_response(
                prompt=query,
                context=context,
                temperature=temp
            )
            
            return {
                "response": response,
                "context": context,
                "query": query
            }
            
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}
