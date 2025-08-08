import streamlit as st
import fitz  # PyMuPDF
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
import cohere

# Load environment and initialize Cohere
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Custom CSS
st.markdown("""
    <style>
        .section-title {
            font-size: 22px;
            font-weight: bold;
            margin-top: 30px;
            margin-bottom: 10px;
            color: #333;
        }
        .file-list {
            margin-bottom: 15px;
        }
        .preview-box {
            border: 1px solid #ccc;
            padding: 10px;
            background-color: #f9f9f9;
            font-family: monospace;
            white-space: pre-wrap;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📚 RAG Document Chat App")

# Section 1: File Upload
st.markdown('<div class="section-title">1. Upload Your Files</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Choose PDF, DOCX, or TXT files",
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True
)

# Text extraction
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

all_texts = []
if uploaded_files:
    st.markdown('<div class="section-title">2. Files Uploaded</div>', unsafe_allow_html=True)
    for uploaded_file in uploaded_files:
        st.markdown(f"<div class='file-list'>📄 {uploaded_file.name}</div>", unsafe_allow_html=True)
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = extract_text_from_txt(uploaded_file)
        else:
            text = f"Unsupported file type: {uploaded_file.type}"
        all_texts.append(text)

    st.markdown('<div class="section-title">3. Extracted Text Preview</div>', unsafe_allow_html=True)
    for idx, text in enumerate(all_texts):
        with st.expander(f"📄 Preview: {uploaded_files[idx].name}"):
            st.markdown(f"<div class='preview-box'>{text[:700]}</div>", unsafe_allow_html=True)

# Embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, max_length=500):
    sentences = text.split('. ')
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        length = len(sentence.split())
        if current_length + length > max_length:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentence]
            current_length = length
        else:
            current_chunk.append(sentence)
            current_length += length
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    return chunks

def embed_documents(texts):
    all_chunks = []
    for text in texts:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    embeddings = embedder.encode(all_chunks)
    return all_chunks, embeddings

def retrieve_chunks(question, index, chunks, top_k=5):
    question_embedding = embedder.encode([question])
    distances, indices = index.search(np.array(question_embedding).astype("float32"), top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(question, context):
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    try:
        response = co.chat(
            model="command-r-plus",
            message=prompt,
            max_tokens=200,
            temperature=0.0,
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"

if uploaded_files:
    chunks, embeddings = embed_documents(all_texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    st.markdown('<div class="section-title">4. Ask Questions About the Documents</div>', unsafe_allow_html=True)
    question = st.text_input("🔍 Ask a question:")
    if question:
        relevant_chunks = retrieve_chunks(question, index, chunks)
        context = "\n\n".join(relevant_chunks)
        answer = generate_answer(question, context)
        st.markdown("### 💬 Answer:")
        st.write(answer)
