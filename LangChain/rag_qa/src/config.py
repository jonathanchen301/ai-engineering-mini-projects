from pydantic import BaseModel, model_validator

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

    Raises:
    - ValueError: If chunk_overlap is greater than or equal to chunk_size.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    chunk_size: int = 400
    chunk_overlap: int = 20
    top_k: int = 3
    document_dir: str = "documents"

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
        
        return cls(**config)