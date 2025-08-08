# RAG Chat Application

An interactive Retrieval-Augmented Generation (RAG) application built with Streamlit, Cohere, and FAISS.
Upload multiple documents (PDF, DOCX, TXT) and chat with them — powered by semantic search + LLMs.

## Features
```
📂 Multiple File Upload — PDF, DOCX, and TXT support
📝 Automatic Text Extraction — Extracts and processes document content
🧠 Semantic Search with FAISS — Retrieves the most relevant chunks
🤖 LLM-powered Responses — Uses Cohere's command-r-plus model for answers
💬 Chat Interface — Ask questions, get context-based answers instantly
🎨 Styled UI — Custom CSS for clean sections and collapsible previews
```
## File Structure

```
rag-chat-with-your-docs/
│── app.py               # Main Streamlit app
│── requirements.txt     # Python dependencies
│── .env                 # API keys (not pushed to GitHub)
│── README.md   
```

## How to run locally
1. Install dependencies:
`pip install -r requirements.txt`
2. Set up environment variables
`COHERE_API_KEY=your_api_key_here`
3. Run the Streamlit app:
`streamlit run app.py`