from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import Optional

from src.config import RagQASettings

def retrieve(query: str, settings: RagQASettings, store: FAISS) -> Optional[list[Document]]:

    """
    Retrieves the top k chunks from the vector store for a given query (question).

    Args:
    - query: The query to retrieve the most relevant chunks for.
    - settings: RagQASettings object that contains the top k chunks to retrieve.
    - store: The FAISS object that contains the vector store.

    Returns:
    - A list of documents that are the most relevant to the query.
    - None if no chunks are retrieved for the query.
    """

    retriever = store.as_retriever(k=settings.top_k)
    retrieved_chunks = retriever.invoke(query)
    if len(retrieved_chunks) == 0:
        return None
    return retrieved_chunks