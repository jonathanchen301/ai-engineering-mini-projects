from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import Optional, Iterable

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

def build_prompt(system_message: str, question: str, chunks: Optional[list[Document]]) -> tuple[str, Optional[list[Document]]]:

    """
    Formats and builds a prompt for the RAG QA pipeline. Includes context, system message, and question. No memory is used. It is completely stateless.

    Args:
    - system_message: The system message to use for the prompt.
    - question: The question to answer.
    - chunks: The chunks to use for the context. Can be None if no relevant chunks were retrieved.

    Returns:
    - A tuple containing the prompt template and the chunks. The chunks are returned to allow for easy retrieval of the source metadata to display to the user at the final step.
    """

    def format_chunks(chunks: Iterable[Document]) -> str:
        formatted = []
        for idx, d in enumerate(chunks, 1):
            meta = d.metadata if hasattr(d, "metadata") and d.metadata else {}
            title = meta.get("title", "Unknown Title")
            page = meta.get("page", "N/A")
            page_label = meta.get("page_label", "N/A")
            citation_line = f"[{idx}] {title} (page {page})\n"
            formatted.append(f"{citation_line}{d.page_content}")
        return "\n\n".join(formatted)

    if not chunks:
        formatted_chunks = "No chunks retrieved for the query."
    else:
        formatted_chunks = format_chunks(chunks)

    template = f"""
    {system_message}

    CONTEXT:
    {formatted_chunks}

    QUESTION:
    {question}

    Using only the information in the context, answer the question. 
    Include citation numbers like [1], [2] in your answer when referencing sources.
    If you don't know the answer, say "I don't know the answer."
    Return JSON in this format:
    {{
        "answer": "...",
        "citations": [1, 2, ...]
    }}
    """

    return template, chunks