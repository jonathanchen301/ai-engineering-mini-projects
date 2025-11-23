from src.multi_query import multi_query_retrieve, generate_query_variants
from src.chain import retrieve
from src.compression import compress_chunks
from src.config import RagQASettings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import time

def retrieve_optimized(query: str, settings: RagQASettings, store: FAISS) -> list[Document]:

    """
    Retrieve chunks from the vector store using either multi-query or standard retrieval, and optionally compress the chunks.

    Args:
    - query: The query to retrieve the chunks for
    - settings: RagQASettings object that contains the enable_multi_query and enable_compression fields
    - store: The FAISS object that contains the vector store

    Returns:
    - list[Document]: A list of documents that are most relevant to the query
    """

    start_time = time.time()
    if settings.enable_multi_query:
        queries = generate_query_variants(query, settings)
        chunks = multi_query_retrieve(queries, settings, store)
    else:
        chunks = retrieve(query, settings, store)
        if not chunks:
            chunks = []
    
    if settings.enable_compression:
        print("Chunk count before compression: ", len(chunks))
        print("Token count before compression: ", sum(len(chunk.page_content.split()) for chunk in chunks))
        chunks = compress_chunks(chunks, settings)
        print("Chunk count after compression: ", len(chunks))
        print("Token count after compression: ", sum(len(chunk.page_content.split()) for chunk in chunks))
    print("Latency: ", time.time() - start_time)

    return chunks