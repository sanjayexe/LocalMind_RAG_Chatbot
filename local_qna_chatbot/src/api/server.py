from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, Union, List
from src.rag.pipeline import RAGPipeline
import uvicorn
import os
import tempfile
import shutil

app = FastAPI()

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    question: str
    temperature: Optional[float] = 0.7

class QueryResponse(BaseModel):
    response: str
    context: List[str] = Field(default_factory=list)
    query: str = ""

class ErrorResponse(BaseModel):
    error: str
    status: str = "error"

@app.get("/")
async def root():
    return {
        "message": "Local QnA Chatbot API is running",
        "status": "healthy"
    }

@app.post("/ask", response_model=Union[QueryResponse, ErrorResponse])
async def ask_question(request: QueryRequest):
    """
    Ask a question to the QnA chatbot.
    
    Args:
        request (QueryRequest): Contains the question and optional temperature
    
    Returns:
        Union[QueryResponse, ErrorResponse]: Response containing either the answer or an error
    """
    if not rag_pipeline.initialized:
        return ErrorResponse(error="RAG pipeline not initialized. Please upload a document first.")
    
    try:
        # Process the query
        result = rag_pipeline.process_query(
            query=request.question,
            temperature=request.temperature
        )
        
        # Check if there was an error in processing
        if "error" in result:
            return ErrorResponse(error=result["error"])
            
        # Ensure all required fields are present
        if not all(k in result for k in ["response", "context", "query"]):
            return ErrorResponse(error="Invalid response format from RAG pipeline")
            
        return QueryResponse(**result)
        
    except Exception as e:
        return ErrorResponse(error=f"Error processing your request: {str(e)}")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document to be processed by the RAG system.
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        try:
            # Initialize the RAG pipeline with the uploaded document
            rag_pipeline.initialize(documents_path=temp_path)
            return {
                "status": "success",
                "message": "Document uploaded and processed successfully",
                "filename": file.filename
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)