from langchain_core.documents import Document
from src.config import RagQASettings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def compress_chunks(chunks: list[Document], settings: RagQASettings) -> list[Document]:

    """
    Compresses retrieved chunks to a defined smaller number of tokens per chunk.

    Args:
    - chunks: list[Document] - The chunks to compress.
    - settings: RagQASettings - The settings for the compression.
    
    Returns:
    - list[Document] - The compressed chunks.
    """

    max_tokens = settings.compression_max_tokens
    model = ChatOpenAI(model=settings.model,
    api_key=settings.api_key)

    template = """
    Summarize the following text into {max_tokens} tokens or less, preserving key facts relevant to answering questions.

    TEXT:
    {text}

    Return the summary only, no other text or comments.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["max_tokens", "text"]
    )

    parser = StrOutputParser()

    chain = prompt | model | parser
    
    res = []
    for chunk in chunks:
        try:
            response = chain.invoke({"max_tokens": max_tokens, "text": chunk.page_content})
            res.append(Document(page_content=response, metadata=chunk.metadata))
        except Exception as e:
            print(f"Error compressing chunk {chunk.metadata.get('title', 'unknown')}: {e}")
            res.append(chunk)

    return res