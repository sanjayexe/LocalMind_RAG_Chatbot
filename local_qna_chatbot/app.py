import streamlit as st
import requests
import os
import time
import json
import re
from typing import Optional

# Set page config
st.set_page_config(
    page_title="LocalMind",
    page_icon="💲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stTextInput>div>div>input {
        padding: 12px;
    }
    .stButton>button {
        width: 100%;
        padding: 0.5rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
    }
    .bot-message {
        background-color: #f1f1f1;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000"

def upload_file(file):
    """Upload a file to the backend"""
    files = {"file": (file.name, file, file.type)}
    try:
        response = requests.post(f"{API_URL}/upload/", files=files, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_msg = f"Error uploading file: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}

def ask_question(question: str, temperature: float = 0.7) -> dict:
    """Send a question to the RAG model with better error handling"""
    try:
        print(f"Sending request to {API_URL}/ask with question: {question}")
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "temperature": temperature},
            timeout=300
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        response.raise_for_status()
        result = response.json()
        
        print("Full API Response:", result)
        
        if isinstance(result, dict):
            if "error" in result:
                return {"answer": f"Error: {result['error']}"}
            elif "response" in result:
                return {"answer": result["response"]}
            elif "detail" in result:
                return {"answer": f"Error: {result['detail']}"}
            elif "answer" in result:
                return {"answer": result["answer"]}
        
        return {"answer": f"Unexpected response format: {result}"}
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error getting response: {str(e)}"
        print(error_msg)
        return {"answer": error_msg}
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing response: {str(e)}"
        print(error_msg)
        return {"answer": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        return {"answer": error_msg}

def main():
    st.title("💲 LocalMind")
    st.markdown("Upload a document and ask questions about its content.")


    # Sidebar for settings and status
    with st.sidebar:
        st.header("Settings")
        
        # API Configuration
        api_url = st.text_input(
            "Backend API URL",
            value="https://nonconfidentially-preallowable-jerri.ngrok-free.dev ",
            help="For local use: http://localhost:8000\nFor Ngrok: https://nonconfidentially-preallowable-jerri.ngrok-free.dev "
        )
        
        temperature = st.slider(
            "Temperature", 
            0.0, 1.0, 0.7, 0.1,
            help="Higher values make output more random, lower values more focused"
        )
    
    # Update global API URL
    global API_URL
    API_URL = api_url
        


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, DOCX, or TXT)", 
        type=["pdf", "docx", "txt"]
    )
    
    # Initialize upload state
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "upload_status" not in st.session_state:
        st.session_state.upload_status = None

    if uploaded_file is not None:
        # Check if it's a new file
        if uploaded_file.name != st.session_state.current_file:
            # Place status container right here
            status_container = st.empty()
            with status_container:
                with st.spinner("Uploading and indexing document..."):
                    upload_result = upload_file(uploaded_file)
                    if upload_result and upload_result.get("status") == "success":
                        st.success("✅ Document uploaded successfully!")
                        st.session_state.current_file = uploaded_file.name
                        time.sleep(3) # Show for 3 seconds
                        st.rerun()
                    else:
                        error_msg = upload_result.get("message", "Unknown error") if upload_result else "No response from server"
                        st.error(f"❌ Upload failed: {error_msg}")
                        st.session_state.current_file = None # Allow retry
                        time.sleep(5)
                        st.rerun()
    else:
        # Reset if file removed
        st.session_state.current_file = None
        st.session_state.upload_status = None

    # Chat input
    if prompt := st.chat_input("Ask a question about the document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                with st.spinner("Thinking..."):
                    response = ask_question(prompt, temperature)
                    full_response = response.get("answer", "No answer provided")
                    
                    # Auto-recover if backend lost the session (restarted)
                    if "RAG pipeline not initialized" in full_response and uploaded_file:
                        st.toast("Restoring connection...", icon="🔄")
                        uploaded_file.seek(0)
                        upload_file(uploaded_file)
                        # Retry the question
                        response = ask_question(prompt, temperature)
                        full_response = response.get("answer", "No answer provided")
                    
            except Exception as e:
                full_response = f"An error occurred: {str(e)}"
                print(f"Error in main chat loop: {str(e)}")
            
            # Display the response immediately to ensure visibility
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()