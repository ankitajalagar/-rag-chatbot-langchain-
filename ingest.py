"""
ingest.py
---------
Step 1-4 of the assignment:
  1. Load a PDF document.
  2. Split the document into meaningful chunks.
  3. Generate embeddings for the chunks (Sentence Transformers: all-MiniLM-L6-v2).
  4. Store the embeddings in ChromaDB (persisted to disk).

Run this once (or whenever the source PDF changes) before using chatbot.py.

Usage:
    python ingest.py --pdf data/source.pdf
"""

import argparse
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "chroma_db"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_pdf(pdf_path: str):
    """Step 1: Load a PDF document into LangChain Document objects (one per page)."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[1/4] Loaded PDF '{pdf_path}' -> {len(documents)} page(s).")
    return documents


def split_documents(documents):
    """Step 2: Split the document into meaningful, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[2/4] Split into {len(chunks)} chunk(s).")
    return chunks


def build_embeddings():
    """Step 3: Create the Sentence-Transformers embedding function."""
    print(f"[3/4] Loading embedding model '{EMBED_MODEL_NAME}'...")
    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{EMBED_MODEL_NAME}")
    return embeddings


def store_in_chroma(chunks, embeddings, persist_dir: str = CHROMA_DIR):
    """Step 4: Embed chunks and persist them in a local ChromaDB collection."""
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)  # rebuild fresh each time ingest.py runs

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="rag_chatbot_collection",
    )
    vectordb.persist()
    print(f"[4/4] Stored {len(chunks)} embedded chunk(s) in ChromaDB at '{persist_dir}/'.")
    return vectordb


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into ChromaDB for RAG.")
    parser.add_argument(
        "--pdf",
        type=str,
        default="data/source.pdf",
        help="Path to the source PDF document (default: data/source.pdf)",
    )
    args = parser.parse_args()

    documents = load_pdf(args.pdf)
    chunks = split_documents(documents)
    embeddings = build_embeddings()
    store_in_chroma(chunks, embeddings)

    print("\nIngestion complete. You can now run: python chatbot.py")


if __name__ == "__main__":
    main()
