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

    Instructions:
    1. Answer the question using ONLY the information from the context above.
    2. When you reference information from a source, include the citation number IN THE ANSWER TEXT at the end of the relevant sentence, like this: "The government restarted [1]."
    3. IMPORTANT: Renumber citations sequentially starting from [1] based on the order you reference them, regardless of their original numbers in the context. For example, if you use sources [1] and [4] from the context, label them as [1] and [2] in your answer and citations array.
    4. If you don't know the answer, say "I don't know the answer."

    IMPORTANT:
    - The "answer" field must contain your answer WITH in-text citations like [1], [2] embedded in the text.
    - The "citations" array must ONLY include citations that are actually referenced in your answer (by their numbers).
    - Do NOT include duplicate citations.
    - Citation IDs must match the numbers you used in the answer text (e.g., if you use [1] in the answer, include citation with id: 1).
    - When you renumber citations, use the metadata (title, page, page_label) and a text snippet from the ORIGINAL source in the context. For example, if you reference source [4] from context and renumber it as [1], use the title, page, page_label, and a short excerpt (1-2 sentences) from the text content of source [4] in the context.

    Return JSON in exactly this format:
    {{
        "answer": "Your answer with citations like [1] and [2] embedded in the text.",
        "citations": [
            {{
                "id": 1,
                "title": "Document title from context",
                "page": "Page number from context",
                "page_label": "Page label from context",
                "text_snippet": "A short excerpt (1-2 sentences) from the cited chunk that supports your answer"
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
    text_snippet: str = ""  # Short excerpt from the cited chunk

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