from src.ingest import document_loading
from typing import Optional
import os

def ingest_documents(document_dir: str, chunk_size: int = 400, chunk_overlap: int = 40, embedding_model: str = "text-embedding-3-small", vectorstore_dir: Optional[str] = None, save_vectorstore: bool = False):

    try:
        documents = document_loading.load_pdfs(document_dir)
        print("Documents loaded successfully.")
    except Exception as e:
        assert False, "Error loading documents: " + str(e)

    try:
        chunks = document_loading.split_documents(documents, chunk_size, chunk_overlap)
        print("Chunks created successfully. Total chunks: ", len(chunks))
    except Exception as e:
        print("Error creating chunks: ", e)
        assert False, "Error creating chunks: " + str(e)

    # If vectorstore_dir is provided and exists, try to load it
    if vectorstore_dir and os.path.exists(vectorstore_dir):
        # Check if vectorstore files exist
        faiss_index = os.path.join(vectorstore_dir, "index.faiss")
        faiss_pkl = os.path.join(vectorstore_dir, "index.pkl")
        if os.path.exists(faiss_index) and os.path.exists(faiss_pkl):
            try:
                vector_store = document_loading.load_vectorstore(vectorstore_dir, embedding_model)
                print("Vectorstore loaded successfully.")
                return vector_store
            except Exception as e:
                print(f"Warning: Could not load vectorstore from {vectorstore_dir}: {e}")
                print("Creating new vectorstore instead.")
    
    # Create new vectorstore (either no vectorstore_dir provided, or loading failed)
    try:
        save_dir = vectorstore_dir if vectorstore_dir else "./"
        vector_store = document_loading.create_vectorstore(chunks, embedding_model, save_dir, save=save_vectorstore)
        print("Vectorstore created successfully.")
    except Exception as e:
        assert False, "Error creating vectorstore: " + str(e)

    print("Ingestion complete.")
    return vector_store
