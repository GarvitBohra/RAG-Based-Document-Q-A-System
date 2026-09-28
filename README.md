# RAG-Based Document Q&A System

A simple Streamlit application that lets you upload a PDF and ask questions about it. It uses Retrieval-Augmented Generation (RAG) so the OpenAI model answers from the most relevant parts of the uploaded document rather than relying on general knowledge.

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. Instead of sending an entire document to an LLM for every question:

1. The document is split into small chunks.
2. Each chunk is converted into a numerical embedding (a semantic meaning vector).
3. The embeddings are stored in a vector database/index.
4. A question is also embedded and compared to the stored chunks.
5. The most relevant chunks are sent to the LLM as context with the question.

This makes answers more relevant, reduces prompt size, and helps ground the LLM in the source document.

## How this project works

```text
PDF upload
   -> PyPDFLoader extracts page text
   -> RecursiveCharacterTextSplitter creates overlapping chunks
   -> all-MiniLM-L6-v2 creates embeddings
   -> FAISS stores and searches the embeddings
   -> Top 3 relevant chunks + question go to OpenAI
   -> Streamlit shows the answer and source chunks
```

The prompt tells the LLM to use only the retrieved context. If the context does not support an answer, it must reply: `I don't have enough information in the provided context to answer that.`

## Architecture

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI: PDF upload, index creation, question input, answer/source display |
| `rag.py` | Reusable RAG functions: loading, chunking, embeddings, FAISS search, LLM answer |
| `data/` | Local uploaded PDFs (ignored by Git) |
| `.env` | Your local OpenAI API key (ignored by Git) |

The FAISS index lives in memory while the app is running. Uploading a new PDF creates a new index, keeping the example straightforward.

## Installation

### 1. Create and activate a virtual environment

From the `rag-document-qa` folder:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`all-MiniLM-L6-v2` downloads automatically the first time it is used. This is the local Sentence Transformers model used to create embeddings.

### 3. Configure the OpenAI API key

Copy the example file:

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```text
OPENAI_API_KEY=your_api_key_here
```

Never commit `.env`. It is already listed in `.gitignore`.

## Run the application

From the `rag-document-qa` folder, with the virtual environment active:

```bash
streamlit run app.py
```

Then open the local URL Streamlit displays in your browser. Upload a text-based PDF, wait for the success message, and ask a question. The sidebar lets you change the OpenAI model if your account uses a different supported model.

## Example questions

- “What is the main purpose of this document?”
- “What are the three key recommendations?”
- “What deadline is mentioned?”
- “Who is responsible for approval?”
- “Does the document mention a refund policy?”

For the last type of question, the app should state that it lacks enough information if the retrieved PDF context does not contain the answer.

## Technologies used

- **Python** — application language
- **Streamlit** — simple web interface
- **LangChain** — document loader, splitter, embeddings integration, vector store, and LLM integration
- **Sentence Transformers** — local `all-MiniLM-L6-v2` embedding model
- **FAISS** — fast semantic similarity search over vectors
- **OpenAI API** — generates the final answer from retrieved context
- **python-dotenv** — loads the API key safely from `.env`

## Explaining this in an AI/ML interview

You can describe the project like this:

> “I built a document Q&A application using a RAG pipeline. When a user uploads a PDF, I extract its text and split it into overlapping chunks with LangChain’s recursive splitter. I create semantic embeddings with the lightweight all-MiniLM-L6-v2 Sentence Transformer and index them in FAISS. For each question, I retrieve the three most semantically relevant chunks and pass only those chunks, along with a strict grounding prompt, to an OpenAI model. The UI shows both the answer and the retrieved source chunks, which makes the system easier to inspect and reduces hallucinations.”

Useful follow-up points:

- **Why chunking?** LLMs have finite context windows; smaller chunks improve retrieval precision.
- **Why overlap?** Important sentences near a boundary are less likely to lose their surrounding context.
- **Why embeddings?** They retrieve by semantic similarity, not only exact keyword matches.
- **Why FAISS?** It performs efficient nearest-neighbor search over vectors.
- **How is hallucination reduced?** The model is instructed to use only retrieved context and return a fixed “not available” message otherwise; displaying chunks makes the result auditable.
- **What would you improve for production?** Add citations with chunk IDs, persistent FAISS storage, document-level access controls, better PDF/OCR handling, evaluation datasets, and retrieval-quality monitoring.

## Project structure

```text
rag-document-qa/
├── app.py
├── rag.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── README.md
└── data/
```
