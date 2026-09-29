# RAG-Based Document Q&A System

A simple Streamlit application that allows users to upload a PDF and ask questions about its content.

The project uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant parts of the document and provide them as context to an OpenAI model before generating an answer.

## How RAG Works

The application follows these steps:

1. Upload a PDF.
2. Extract the text from the PDF.
3. Split the text into smaller chunks.
4. Convert the chunks into embeddings using Sentence Transformers.
5. Store the embeddings in FAISS.
6. When a question is asked, search FAISS for the most relevant chunks.
7. Send the retrieved chunks and question to OpenAI.
8. Display the generated answer along with the retrieved source chunks.

```text
PDF
 ↓
PyPDFLoader
 ↓
Text Splitting
 ↓
Sentence Transformer Embeddings
 ↓
FAISS
 ↓
Similarity Search
 ↓
Relevant Chunks
 ↓
OpenAI
 ↓
Answer
```
