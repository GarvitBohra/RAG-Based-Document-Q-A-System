"""Small, reusable functions for the document question-answering pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load OPENAI_API_KEY from .env when the app starts. The key is never hard-coded.
load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NOT_FOUND_MESSAGE = "I don't have enough information in the provided context to answer that."


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Read a PDF into LangChain Document objects, one document per page."""
    documents = PyPDFLoader(str(pdf_path)).load()
    if not documents:
        raise ValueError("No text could be extracted from this PDF.")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Break pages into overlapping, retrieval-friendly text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The PDF did not produce any text chunks.")
    return chunks


def create_embeddings() -> HuggingFaceEmbeddings:
    """Create local Sentence Transformer embeddings (no API key required)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(
    chunks: list[Document], embeddings: HuggingFaceEmbeddings
) -> FAISS:
    """Embed chunks and keep them in an in-memory FAISS vector store."""
    return FAISS.from_documents(chunks, embeddings)


def retrieve_documents(vector_store: FAISS, question: str) -> list[Document]:
    """Return the three chunks most semantically similar to a question."""
    return vector_store.similarity_search(question, k=3)


def generate_answer(documents: list[Document], question: str, model: str = "gpt-4o-mini") -> str:
    """Ask OpenAI to answer strictly from the retrieved chunks."""
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_api_key_here":
        raise ValueError("Add a valid OPENAI_API_KEY to .env before asking a question.")

    context = "\n\n".join(
        f"[Source {index}]\n{document.page_content}"
        for index, document in enumerate(documents, start=1)
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise document question-answering assistant. Use only the supplied context. "
                "If the context contains an answer, answer the question directly and concisely. "
                "A sentence in the context that states the requested fact is sufficient evidence. "
                "Do not reject an answer that is explicitly present in the context. "
                f"Only when the context does not contain the answer, say exactly: "
                f"'{NOT_FOUND_MESSAGE}' Do not use outside knowledge.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    response = (prompt | ChatOpenAI(model=model, temperature=0)).invoke(
        {"context": context, "question": question}
    )
    return str(response.content)
