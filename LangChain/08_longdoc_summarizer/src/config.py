from pydantic import BaseModel, model_validator
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class LongDocSummarizerSettings(BaseModel):

    """
    Setting for the Long Document Summarizer.

    Attributes:
    - model: The model to use
    - api_key: The API key to use
    - temperature: The temperature to use for the model (0 is deterministc, 1 is creative)
    - chunk_size: The chunk size to use
    - chunk_overlap: The chunk overlap to use
    - map_mode: Runs map phase only for testing purposes
    - reduce_mode: Runs reduce phase only for testing purposes
    - map_output_path: Path to save map phase outputs
    """

    model: str = "gpt-4o-mini"
    api_key: str = openai_api_key
    temperature: float = 0.0
    chunk_size: int = 2000
    chunk_overlap: int = 200
    map_mode: bool = False # Runs map phase only for testing purposes
    reduce_mode: bool = False # Runs reduce phase only for testing purposes
    map_output_path: str = "./map_outputs.json" # Path to save map phase outputs
    document_path: str = "./documents" # Path to the documents to summarize

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
        if hasattr(args, "map_mode") and args.map_mode is not None:
            config["map_mode"] = args.map_mode
        if hasattr(args, "reduce_mode") and args.reduce_mode is not None:
            config["reduce_mode"] = args.reduce_mode
        if hasattr(args, "map_output_path") and args.map_output_path is not None:
            config["map_output_path"] = args.map_output_path
        if hasattr(args, "document_path") and args.document_path is not None:
            config["document_path"] = args.document_path

        return cls(**config)