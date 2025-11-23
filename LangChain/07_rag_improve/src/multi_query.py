from src.config import RagQASettings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.chain import retrieve

def generate_query_variants(original_query: str, settings: RagQASettings) -> list[str]:

    """
    Generate multiple query variants for a given question.

    Args:
    - original_query: The original question to generate query variants for.
    - settings: RagQASettings object that contains the multi_query_count field.

    Returns:
    - list[str]: A list of query variants.
    """

    multi_query_count = settings.multi_query_count
    
    model = ChatOpenAI(model=settings.model,
    api_key=settings.api_key)

    template = """
    Generate {multi_query_count} query variants for the following question:

    QUESTION:
    {original_query}

    Return a list of {multi_query_count} query variants in a JSON array like this:
    {{
    "queries": [
        "query_variant_1",
        "query_variant_2",
        "query_variant_3",
        ...
    ]
    }}

    Return the JSON array only, no other text or comments.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["multi_query_count", "original_query"]
    )

    output_parser = JsonOutputParser()

    chain = prompt | model | output_parser

    try:
        response = chain.invoke({"multi_query_count": multi_query_count, "original_query": original_query})
        return response["queries"]
    except Exception as e:
        print(f"Error generating query variants: {e}")
        return [original_query]

def multi_query_retrieve(queries: list[str], settings: RagQASettings, store: FAISS) -> list[Document]:

    """
    Uses a list of queries to retrieve chunks from the vector store, deduplicates chunks, and returns a list of unique chunks.

    Args:
    - queries: A list of queries to retrieve chunks for
    - settings: RagQASettings object that contains the top_k field
    - store: The FAISS object that contains the vector store

    Returns:
    - list[Document]: A list of documents that are the most relevant to the queries
    """

    top_k = settings.top_k

    chunk_scores = {}
    
    for query in queries:
        results = store.similarity_search_with_score(query, k=top_k)
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                doc, score = result
            else:
                doc = result
                score = 0.0
            
            if not hasattr(doc, 'page_content') or not isinstance(doc.page_content, str):
                continue
                
            content_hash = hash(doc.page_content)
            if content_hash not in chunk_scores or score > chunk_scores[content_hash][1]:
                chunk_scores[content_hash] = (doc, score)
    
    sorted_chunks = sorted(chunk_scores.values(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_chunks][:top_k]