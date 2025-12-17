import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text"

def get_embedding(text):
    res = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": text}
    ).json()
    return res["embedding"]
