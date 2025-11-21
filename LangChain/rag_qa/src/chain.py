from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from typing import Optional, Iterable
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.messages.ai import AIMessage
from time import time
import json
from langchain_core.output_parsers import PydanticOutputParser

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

def build_prompt(system_message: str, question: str, chunks: Optional[list[Document]]) -> str:

    """
    Formats and builds a prompt for the RAG QA pipeline. Includes context, system message, and question. No memory is used. It is completely stateless.

    Args:
    - system_message: The system message to use for the prompt.
    - question: The question to answer.
    - chunks: The chunks to use for the context. Can be None if no relevant chunks were retrieved.

    Returns:
    - A string containing the prompt template.
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

    template = """
    {system_message}

    CONTEXT:
    {formatted_chunks}

    QUESTION:
    {question}

    Using only the information in the context, answer the question. If you use a citation, include the citation number at the end of the sentence like [1]. At the end of your answer, include all the citations that you used in the following format:

    CITATIONS:
    [1] Document title | Page number | Page label
    [2] Document title | Page number | Page label
    ...

    If you don't know the answer, say "I don't know the answer."
    Return JSON in exactly this format:
    {{
        "answer": "...",
        "citations": [
            {{
                "id": 1,
                "title": "...",
                "page": "...",
                "page_label": "..."
            }}
        ]
    }}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["system_message", "formatted_chunks", "question"]
    )

    return prompt.format(system_message=system_message, formatted_chunks=formatted_chunks, question=question)

class Citation(BaseModel):
    id: int
    title: str
    page: str
    page_label: str

class LLMResponseModel(BaseModel):
    answer: str
    citations: list[Citation]

def call_model(prompt: str, settings: RagQASettings) -> AIMessage:

    """
    Calls the OpenAI model with the prompt and returns the response.

    Records and logs token usage and latency for monitoring.

    Args:
        prompt: The formatted prompt string to send to the model.
        settings: RagQASettings containing model name and API key.

    Returns:
        AIMessage response from the model.
    """

    start_time = time()
    model = ChatOpenAI(model=settings.model, api_key=settings.api_key)
    response = model.invoke(prompt)
    end_time = time()

    usage = response.response_metadata["token_usage"]
    print(f"Latency: {end_time - start_time} seconds")
    print("Prompt tokens:", usage["prompt_tokens"])
    print("Completion tokens:", usage["completion_tokens"])
    print("Total tokens:", usage["total_tokens"])

    return response

def parse_response(response: AIMessage) -> LLMResponseModel:

    """
    Parses the response from the model and returns a LLMResponseModel object.

    Args:
        response: The AIMessage response from the model.

    Returns:
        LLMResponseModel object.
    """

    parser = PydanticOutputParser(
        pydantic_object=LLMResponseModel
    )

    return parser.parse(response.content)

def run_chat(query: str, settings: RagQASettings, vector_store: FAISS) -> LLMResponseModel:

    """
    Runs the chat pipeline for a given query from retrieval to building the prompt to calling the model and parsing the response.

    Args:
        query: The query to answer.
        settings: RagQASettings containing model name and API key.
        vector_store: The FAISS object that contains the vector store.

    Returns:
        LLMResponseModel object.
    """

    chunks = retrieve(query, settings, vector_store)

    if not chunks:
        return LLMResponseModel(
            answer="I couldn't find relevant info in the provided docs.",
            citations=[]
        )
    
    prompt = build_prompt("You are a helpful assistant that can answer questions about the document.", query, chunks)
    raw_response = call_model(prompt, settings)
    response = parse_response(raw_response)
    return response