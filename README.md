# Resume/Transcript RAG Agent

A Retrieval-Augmented Generation (RAG) system designed to intelligently answer questions about student transcripts, resumes, and academic documents. This tool processes PDF documents, creates vector embeddings for semantic search, and uses a local language model to provide accurate, context-aware responses.

## Features

- **PDF Document Processing**: Automatically ingests PDF files from a data directory
- **Vector Database Storage**: Uses ChromaDB for efficient document retrieval
- **Local LLM Integration**: Powered by Ollama with Llama 3.2:1b model
- **Semantic Search**: HuggingFace embeddings for intelligent document matching
- **Source Attribution**: Returns source documents for transparency
- **Command-Line Interface**: Simple CLI for document ingestion and querying

## Architecture

The system consists of three main components:

1. **Document Ingestion (`ingest.py`)**: Processes PDF files and stores them in a vector database
2. **Query Processing (`response.py`)**: Handles user questions and retrieves relevant information
3. **Main Interface (`main.py`)**: Command-line interface for user interactions

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running
- Sufficient disk space for vector database and model storage

## Setup

### 1. Environment Setup

Initialize and activate a Python virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
source .venv/Scripts/activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Ollama Setup

Install and configure Ollama with the required model:

```bash
# Start Ollama service (keep this running in a separate terminal)
ollama serve

# Pull the Llama 3.2:1b model
ollama pull llama3.2:1b
```

### 4. Environment Configuration (Optional)

Create a `.env` file in the project root to customize directories:

```bash
DATA_DIR=data
CHROMA_DIR=chroma_store
OLLAMA_HOST=http://localhost:11434
```

### 5. Prepare Your Documents

Place your PDF files (transcripts, resumes, etc.) in the `data/` directory:

```
data/
├── transcript1.pdf
├── resume.pdf
└── academic_records.pdf
```

## Usage

### Document Ingestion

Before querying, you must ingest your documents into the vector database:

```bash
python main.py --ingest
```

This command will:
- Scan the `data/` directory for PDF files
- Extract text and split into chunks
- Generate embeddings using HuggingFace transformers
- Store everything in ChromaDB

### Querying Documents

Ask questions about your documents:

```bash
python main.py --query "How many credits does this student have?"
python main.py --query "What is the student's cumulative GPA?"
python main.py --query "List all internship experiences"
```

## Sample Usage

```bash
# First, ingest your documents
$ python main.py --ingest
Ingested 45 chunks into ChromaDB at 'chroma_store'

# Then ask questions
$ python main.py --query "What is the cumulative GPA?"

Answer:
The student has a cumulative GPA of 3.75 based on the academic transcript.

Sources:
- data/transcript.pdf
- data/academic_summary.pdf
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `data` | Directory containing PDF documents |
| `CHROMA_DIR` | `chroma_store` | ChromaDB storage location |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

### Text Splitting Parameters

The system uses the following default parameters for document chunking:
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 150 characters

These can be modified in `ingest.py` if needed for your specific documents.

### Retrieval Parameters

The query system retrieves the top 4 most relevant document chunks by default. This can be adjusted in `response.py` by modifying the `search_kwargs` parameter.

### Performance Optimization

- **For large document collections**: Consider increasing chunk size to reduce the number of embeddings
- **For better accuracy**: Decrease chunk size for more granular retrieval
- **For faster responses**: Use a smaller embedding model or reduce retrieval count

## Dependencies

The project relies on the following key libraries:

- **LangChain**: Framework for building LLM applications
- **ChromaDB**: Vector database for document storage and retrieval
- **HuggingFace Transformers**: Sentence embedding models
- **Ollama**: Local LLM inference
- **PyPDF**: PDF document processing
