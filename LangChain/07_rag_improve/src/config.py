from pydantic import BaseModel, model_validator
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class RagQASettings(BaseModel):

    """
    Settings for the RAG QA Pipeline.

    Attributes:
    - model: The model to use for the RAG QA Pipeline. Currently only support OpenAI models.
    - temperature: The temperature to use for the RAG QA Pipeline. Range from 0 to 1. 0 is the most deterministic and 1 is the most creative.
    - chunk_size: The chunk size to use for the RAG QA Pipeline. Number of tokens within each chunk of documents.
    - chunk_overlap: The chunk overlap to use for the RAG QA Pipeline. Number of tokens to overlap between chunks.
    - top_k: The top k chunks to use for the RAG QA Pipeline. Number of chunks to retrieve with each query from the vector store.
    - document_dir: The directory to use for the RAG QA Pipeline. Contains only PDF files. This directory must exist.
    - vector_store_path: The path to use for the FAISS vector store.

    Raises:
    - ValueError: If chunk_overlap is greater than or equal to chunk_size.
    """

    model: str = "gpt-4o-mini"
    api_key: str = openai_api_key
    temperature: float = 0.0
    chunk_size: int = 400
    chunk_overlap: int = 20
    top_k: int = 3
    vector_store_path: str = "./vector_store"
    document_dir: str = "documents"

    # Upgrades
    enable_compression: bool = False # Toggles compression step
    compression_max_tokens: int = 200 # Maximum number of tokens per compressed chunk
    enable_multi_query: bool = False # Toggles multi-query step
    multi_query_count: int = 3 # Number of queries to run for multi-query step
    eval_mode: bool = False # Toggles evaluation mode

    @model_validator(mode='after')
    def validate_chunk_params(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")
        return self

    @classmethod
    def from_cli(cls, args):
        config = {}
        
        if hasattr(args, "model") and args.model is not None:
            config["model"] = args.model
        if hasattr(args, "api_key") and args.api_key is not None:
            config["api_key"] = args.api_key
        if hasattr(args, "temperature") and args.temperature is not None:
            config["temperature"] = args.temperature
        if hasattr(args, "chunk_size") and args.chunk_size is not None:
            config["chunk_size"] = args.chunk_size
        if hasattr(args, "chunk_overlap") and args.chunk_overlap is not None:
            config["chunk_overlap"] = args.chunk_overlap
        if hasattr(args, "top_k") and args.top_k is not None:
            config["top_k"] = args.top_k
        if hasattr(args, "document_dir") and args.document_dir is not None:
            config["document_dir"] = args.document_dir
        if hasattr(args, "vector_store_path") and args.vector_store_path is not None:
            config["vector_store_path"] = args.vector_store_path

        if hasattr(args, "enable_compression") and args.enable_compression is not None:
            config["enable_compression"] = args.enable_compression
        if hasattr(args, "compression_max_tokens") and args.compression_max_tokens is not None:
            config["compression_max_tokens"] = args.compression_max_tokens
        if hasattr(args, "enable_multi_query") and args.enable_multi_query is not None:
            config["enable_multi_query"] = args.enable_multi_query
        if hasattr(args, "multi_query_count") and args.multi_query_count is not None:
            config["multi_query_count"] = args.multi_query_count
        if hasattr(args, "eval_mode") and args.eval_mode is not None:
            config["eval_mode"] = args.eval_mode

        return cls(**config)