import streamlit as st

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🏠",
)

st.title("FullstackGPT Home")

st.subheader("Now we are at Assignment 6!")

st.markdown(
        """
        Welcome to this project! This application serves as a practical demonstration of 
        **Retrieval-Augmented Generation (RAG)**, built using the powerful **LangChain framework** and a Large Language Model (LLM). Its core function is simple: to allow you to **chat directly 
        with the content of your uploaded documents.**
        
        ### 💡 Core RAG Pipeline (What happens inside):
        
        1.  **File Input:** You upload documents (PDF, TXT, etc.).
        2.  **Text Preparation:** The app automatically breaks the large document into small, manageable pieces.
        3.  **Memory Creation (Vector Store):** These pieces are converted into numerical data (embeddings) 
            and stored in an index, acting as the document's 'searchable memory.'
        4.  **Grounded Q\&A:** When you ask a question, the app first searches this memory for relevant facts, 
            and then sends those facts to the LLM to generate an accurate, **non-hallucinatory** answer.
        
        ### 🚀 Simple Steps to Run
        
        1.  **Upload:** Use the file uploader (usually on the sidebar) to add a PDF or text document.
        2.  **Process:** Click the main button to prepare the document and build the memory index.
        3.  **Chat:** Once ready, type your questions into the chat box below to test the RAG system!
        
        ---
        
        > ✅ **Project Goal:** This demo successfully shows how an AI can retrieve specific knowledge 
        > from an external source (your document) before generating a response.
        
        """
)