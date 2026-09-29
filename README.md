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

The application is designed to answer questions using the retrieved document content instead of relying on the model's general knowledge.

## Features

* Upload PDF documents
* Extract text from PDFs
* Split documents into overlapping chunks
* Generate local embeddings
* Semantic search using FAISS
* Generate answers using OpenAI
* Display retrieved source chunks
* Simple Streamlit interface
* API key stored in `.env`

## Project Structure

```text
rag-document-qa/
├── app.py
├── rag.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── data/
```

### `app.py`

Contains the Streamlit application.

It handles:

* PDF upload
* Creating the document index
* User questions
* Calling the RAG pipeline
* Displaying answers
* Displaying retrieved source chunks

### `rag.py`

Contains the main RAG functionality:

* PDF loading
* Text splitting
* Embedding creation
* FAISS vector store
* Similarity search
* OpenAI response generation

### `data/`

Used for storing uploaded PDFs locally while the application is running. Uploaded files are ignored by Git.

## Technologies Used

* **Python**
* **Streamlit**
* **LangChain**
* **Sentence Transformers**
* **FAISS**
* **OpenAI API**
* **PyPDF**
* **python-dotenv**

## LangChain Usage

LangChain is used as the main framework for connecting different parts of the RAG pipeline.

In this project I use:

* `PyPDFLoader` for PDF loading
* `RecursiveCharacterTextSplitter` for chunking
* `HuggingFaceEmbeddings` for embeddings
* `FAISS` for vector search
* `ChatPromptTemplate` for creating the LLM prompt
* `ChatOpenAI` for connecting to OpenAI

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/GarvitBohra/RAG-Based-Document-Q-A-System.git
cd RAG-Based-Document-Q-A-System
```

### 2. Create a virtual environment

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

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The `all-MiniLM-L6-v2` model will be downloaded automatically the first time it is used.

### 4. Add your OpenAI API key

Create a `.env` file based on `.env.example`.

```text
OPENAI_API_KEY=your_api_key_here
```

Do not commit your `.env` file to GitHub.

## Run the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL. Open it in your browser, upload a PDF, and start asking questions.

## Example Questions

You can ask questions such as:

* What is the main purpose of this document?
* What are the key points mentioned?
* What deadline is mentioned?
* Who is responsible for approval?
* Does the document mention a refund policy?

If the retrieved context does not contain enough information to answer the question, the application returns:

```text
I don't have enough information in the provided context to answer that.
```

## Why I Built This Project

I built this project to understand how a practical **RAG pipeline** works from document ingestion to retrieval and LLM-based answer generation.

It helped me work with:

* Document processing
* Embeddings
* Vector search
* LangChain
* FAISS
* Prompt engineering
* OpenAI APIs
* Streamlit

## Future Improvements

Some improvements I would like to add:

* Persistent vector storage
* Better PDF and scanned-document support
* OCR support
* Source citations
* Multiple document support
* Retrieval evaluation
* Improved chunking strategies
* Document-level access control
* Deployment to a cloud platform

## Author

**Garvit Bohra**

AI/ML Engineer | Python | Machine Learning | GenAI | RAG
