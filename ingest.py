import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_store")

# Ánh xạ phần mở rộng file với Document Loader tương ứng
LOADER_MAPPING = {
    ".pdf": (PyPDFLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".md": (UnstructuredMarkdownLoader, {}),
}

def ingest_documents():
    docs = []
    for file in os.listdir(DATA_DIR):
        ext = "." + file.rsplit(".", 1)[-1].lower()
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(os.path.join(DATA_DIR, file), **loader_args)
            docs.extend(loader.load())

    if not docs:
        print("Không tìm thấy tài liệu để xử lý.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    print(f"Ingested {len(chunks)} chunks into ChromaDB at '{CHROMA_DIR}'")