import os
import time
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import RagQASettings

def load_sources(settings: RagQASettings) -> list[Document]:
    """
    Loads the sources from the document directory.

    Args:
    - settings: RagQASettings object that contains the document directory path.

    Returns:
    - A list of documents.

    Raises:
    - FileNotFoundError: If the document directory does not exist.
    - ValueError: If no PDF files are found in the document directory.
    """
    document_dir = settings.document_dir
    
    if not os.path.exists(document_dir):
        raise FileNotFoundError(f"Document directory {document_dir} does not exist")

    res = []
    for file in os.listdir(document_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(document_dir, file))
            documents = loader.load()
            res.extend(documents)
    
    if len(res) == 0:
        raise ValueError(f"No PDF files found in {document_dir}")

    return res

def split_documents(documents: list[Document], settings: RagQASettings) -> list[Document]:
    """
    Splits the documents into chunks.

    Args:
    - documents: The documents to split.
    - settings: RagQASettings object that contains the chunk size and chunk overlap.

    Returns:
    - A list of documents.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    return splitter.split_documents(documents)

def embed_and_persist_chunks(chunks: list[Document], settings: RagQASettings) -> FAISS:
    """
    Embeds the chunks using the OpenAI API and persists the FAISS index to the vector store path.

    Args:
        chunks: The chunks to embed.
        settings: RagQASettings object that contains the api key and the vector store path.

    Returns:
        FAISS index.

    Raises:
        ValueError: If no chunks are provided.
    """
    if len(chunks) == 0:
        raise ValueError("No chunks to embed")

    embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.api_key)
    faiss_index = FAISS.from_documents(chunks, embedder)
    
    # Create directory if it doesn't exist
    os.makedirs(settings.vector_store_path, exist_ok=True)
    faiss_index.save_local(settings.vector_store_path)
    
    return faiss_index

def main(settings: RagQASettings):

    start_time = time.time()

    documents = load_sources(settings)
    print(f"Number of documents: {len(documents)}")

    chunks = split_documents(documents, settings)
    print(f"Number of chunks: {len(chunks)}")
    print(f"Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks)}")

    store = embed_and_persist_chunks(chunks, settings)

    end_time = time.time()

    print(f"Ingestion completed in {end_time - start_time} seconds")
    return store