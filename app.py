"""Streamlit interface for the beginner-friendly RAG document Q&A application."""

import hashlib
from pathlib import Path

import streamlit as st

from rag import (
    create_embeddings,
    create_vector_store,
    generate_answer,
    load_pdf,
    retrieve_documents,
    split_documents,
)


st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG-Based Document Q&A")
st.caption("Upload a PDF, then ask questions answered only from its contents.")


@st.cache_resource(show_spinner="Loading the Sentence Transformer model...")
def get_embeddings():
    """Load the local embedding model once per Streamlit server."""
    return create_embeddings()


def save_uploaded_pdf(uploaded_file) -> Path:
    """Save an uploaded PDF locally so PyPDFLoader can read it."""
    data_directory = Path("data")
    data_directory.mkdir(exist_ok=True)
    file_bytes = uploaded_file.getvalue()
    file_id = hashlib.sha256(file_bytes).hexdigest()[:12]
    pdf_path = data_directory / f"{file_id}_{Path(uploaded_file.name).name}"
    pdf_path.write_bytes(file_bytes)
    return pdf_path


with st.sidebar:
    st.header("How it works")
    st.write("1. Extract PDF text")
    st.write("2. Split it into chunks")
    st.write("3. Embed chunks with all-MiniLM-L6-v2")
    st.write("4. Search FAISS for the best 3 chunks")
    st.write("5. Ask OpenAI using only those chunks")
    st.divider()
    model_name = st.text_input("OpenAI model", value="gpt-4o-mini")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    upload_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("upload_id") != upload_id:
        try:
            with st.spinner("Extracting text, creating chunks, and building the FAISS index..."):
                pdf_path = save_uploaded_pdf(uploaded_file)
                pages = load_pdf(pdf_path)
                chunks = split_documents(pages)
                vector_store = create_vector_store(chunks, get_embeddings())
            st.session_state.upload_id = upload_id
            st.session_state.vector_store = vector_store
            st.session_state.chunk_count = len(chunks)
            st.success(f"Ready: indexed {len(chunks)} chunks from {len(pages)} PDF pages.")
        except Exception as error:
            st.error(f"Could not process this PDF: {error}")

if "vector_store" in st.session_state:
    question = st.text_input("Ask a question about the PDF")
    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving the top 3 chunks and generating an answer..."):
                    source_chunks = retrieve_documents(st.session_state.vector_store, question)
                    answer = generate_answer(source_chunks, question, model_name)

                st.subheader("Answer")
                st.write(answer)
                st.subheader("Retrieved source chunks")
                for index, chunk in enumerate(source_chunks, start=1):
                    page = chunk.metadata.get("page", 0) + 1
                    with st.expander(f"Source {index} — page {page}"):
                        st.write(chunk.page_content)
            except Exception as error:
                st.error(f"Could not answer the question: {error}")
else:
    st.info("Upload a PDF to begin.")
