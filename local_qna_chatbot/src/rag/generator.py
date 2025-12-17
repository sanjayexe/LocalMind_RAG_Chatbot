import requests
from typing import List

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "phi3"

def generate_response(prompt: str, context: List[str] = None, temperature: float = 0.1) -> str:
    """
    Generate a response using Ollama's Chat API.
    """
    system_instruction = (
        "You are a helpful assistant. Read the following context and answer the user's question directly and concisely. "
        "Do not start with 'The context provided...' or similar phrases. Just state the answer based on the context. "
        "Generate only 2-3 lines unless the user asked to explain it in detail."
    )

    if context:
        context_text = "\n\n".join(context)
        user_content = f"Context:\n{context_text}\n\nQuestion: {prompt}"
    else:
        user_content = prompt

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        
        # Parse the chat response
        result = response.json()
        if "message" in result and "content" in result["message"]:
            answer = result["message"]["content"].strip()
            return answer if answer else "I couldn't generate a response (empty output)."
        else:
            return "Unexpected response format from Ollama."
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"I'm sorry, I encountered an error while generating a response: {str(e)}"
