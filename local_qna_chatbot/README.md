# 🧠 LocalMind

**LocalMind** is a privacy-focused, AI-powered document assistant. It uses **Retrieval-Augmented Generation (RAG)** to allow you to chat with your PDF and text documents securely on your own machine. 

## ✨ Key Features

*   **100% Local & Private**: All processing happens on your device using [Ollama](https://ollama.ai/). No data leaves your machine.
*   **Smart Intelligence**: Powered by the **Phi-3** model for high-quality, concise reasoning.
*   **Context Aware**: Vector search (FAISS) retrieves only the relevant parts of your document to answer questions.
*   **Modern UI**: Sleek Streamlit interface with auto-hiding notifications and session recovery.
*   **Auto-Healing**: Automatically restores your session if the backend restarts, ensuring a seamless experience.
*   **File Support**: 
    *   📄 **PDF** (with intelligent text extraction)
    *   📝 **TXT**

## 🏗️ Architecture

```mermaid
graph LR
    User[User] -->|Uploads PDF/TXT| Ingestion[Ingestion Engine]
    Ingestion -->|Text Extraction| Chunks[Text Chunks]
    Chunks -->|Embed (nomic-embed-text)| VectorDB[(FAISS Index)]
    
    User -->|Asks Question| API[FastAPI Backend]
    API -->|Embed Query| VectorDB
    VectorDB -->|Retrieve Context| Context[Relevant Chunks]
    
    Context -->|Prompt + Question| LLM[Ollama (Phi-3)]
    LLM -->|Answer| App[LocalMind UI]
```

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.10+**
2.  **Ollama**: Installed and running.
3.  **Models**:
    ```bash
    ollama pull phi3
    ollama pull nomic-embed-text
    ```

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd local_qna_chatbot
    ```

2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### ▶️ Running LocalMind

**Step 1: Start the Backend (API)**
Open a terminal and run:
```powershell
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```
*Wait until you see "Application startup complete".*

**Step 2: Start the Frontend (UI)**
Open a **new** terminal window and run:
```powershell
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

## 🛠️ Configuration

*   **Speed vs. Accuracy**: 
    *   Edit `src/rag/pipeline.py` and change `k_context` (Default: 2). Lower = Faster, Higher = More Context.
*   **Model**:
    *   Edit `src/rag/generator.py` to switch models (e.g., to `tinyllama` for speed or `mistral` for power).

## 📂 Project Structure

```
local_qna_chatbot/
├── src/
│   ├── api/          # FastAPI Server (server.py)
│   ├── ingestion/    # PDF/Text processing (ingest_file.py)
│   ├── embeddings/   # FAISS Vector Storage build logic
│   └── rag/          # RAG Brain (pipeline.py, generator.py)
├── app.py            # Streamlit Frontend UI
├── requirements.txt  # Project Dependencies
└── README.md         # Documentation
```
