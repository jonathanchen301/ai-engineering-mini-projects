import os
import time
import sys
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import LongDocSummarizerSettings

"""
Copied from my RAG QA Pipeline with slight changes:
- Changed QASettings to LongDocSummarizerSettings
"""

def load_sources(settings: LongDocSummarizerSettings) -> list[Document]:
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

def split_documents(documents: list[Document], settings: LongDocSummarizerSettings) -> list[Document]:
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

def embed_and_persist_chunks(chunks: list[Document], settings: LongDocSummarizerSettings) -> FAISS:
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

def ingest_pipeline(settings: LongDocSummarizerSettings):

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

"""
New functions for map reduce.
"""

def map_chunks(chunks: list[Document], settings: LongDocSummarizerSettings) -> list[dict]:

    """
    Documentation after implmementation.
    """

    model = settings.model
    temperature = settings.temperature

    model = ChatOpenAI(model=model, temperature=temperature)

    template = """
    Summarize the following text, preserving key facts relevant to answering questions.

    TEXT:
    {text}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["text"]
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    res = []
    curr_char = 0

    for i, chunk in enumerate(chunks):
        try:
            response = chain.invoke({"text": chunk.page_content})

            start_char = curr_char
            end_char = start_char + len(chunk.page_content)
            curr_char = end_char

            res.append(
                {
                    "chunk_id": i + 1,
                    "summary": response,
                    "start_char": start_char,
                    "end_char": end_char
                }
            )
        except Exception as e:
            print(f"Error summarizing chunk {chunk.metadata.get('title', 'unknown')}: {e}, using full chunk content as summary.")

            start_char = curr_char
            end_char = start_char + len(chunk.page_content)
            curr_char = end_char

            res.append({
                "chunk_id": i + 1,
                "summary": chunk.page_content,
                "start_char": start_char,
                "end_char": end_char
            })

    if settings.map_mode:
        save_map_outputs(res, settings)

    return res

def save_map_outputs(summaries: list[dict], settings: LongDocSummarizerSettings) -> None:

    """
    Saves map outputs to a JSON file.

    Args:
    - summaries: list[dict] - The summaries to save.
    - settings: LongDocSummarizerSettings - The settings object.

    Returns:
    - None
    """

    dir_path = os.path.dirname(settings.map_output_path)
    if dir_path:  # Only create if there's a directory path
        os.makedirs(dir_path, exist_ok=True)

    save_location = settings.map_output_path
    
    with open(save_location, "w") as f:
        json.dump(summaries, f)

    print(f"Map output saved to {save_location}")

def load_map_outputs(settings: LongDocSummarizerSettings) -> list[dict]:
    
    """
    Load map outputs from a JSON file that was previously saved.

    Args:
    - settings: LongDocSummarizerSettings - The settings object that contains the path to the map output file.

    Returns:
    - list[dict] - The list of summaries.

    Raises:
    - FileNotFoundError: If the map output file does not exist.
    """

    if not os.path.exists(settings.map_output_path):
        raise FileNotFoundError(f"Map output file {settings.map_output_path} does not exist")

    with open(settings.map_output_path, "r") as f:
        return json.load(f)

def reduce_summaries(summaries: list[dict], settings: LongDocSummarizerSettings) -> str:

    """
    Takes a list of chunk summaries from the map phase and reduces them into a single summary.

    Args:
    - summaries: list[dict] - the list of chunk summaries from the map phase.
    - settings: LongDocSummarizerSettings - The settings object.

    Returns:
    - str - A single summary of the document.
    """

    model = settings.model
    temperature = settings.temperature

    model = ChatOpenAI(model=model, temperature=temperature)

    template = """
    Combine the following summaries into one cohesive document summary.

    SUMMARIES:
    {summaries}

    Return the summary only, no other text or comments.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables = ["summaries"]
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    return chain.invoke({
        "summaries": "\n\n".join([f"Summary {summary['chunk_id']}:\n{summary['summary']}" for summary in summaries])
    })