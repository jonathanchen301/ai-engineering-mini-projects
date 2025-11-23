import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RagQASettings
from src.ingest import ingest_pipeline
from langchain_community.vectorstores import FAISS
from src.chain import run_chat as run_chat_chain, retrieve, build_prompt
from src.retrieval import retrieve_optimized
from langchain_openai import OpenAIEmbeddings

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_qa",
        description="CLI for the Single-User RAG QA Pipeline",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run the ingestion pipeline to build and update"
    )

    # Add relevant arguments to ingest parser

    ingest_parser.add_argument(
        "--document-dir",
        type=str,
        default=None,
        help="The directory containing the documents to ingest"
    )
    ingest_parser.add_argument(
        "--vector-store-path",
        type=str,
        default=None,
        help="The path to the vector store"
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for the text splitter.",
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap for the text splitter.",
    )
    ingest_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override API key if not set via environment/.env.",
    )

    chat_parser = subparsers.add_parser(
        "chat",
        help="Run the chat pipeline for a given query."
    )

    # Add relevant arguments to chat parser

    chat_parser.add_argument(
        "query",
        type=str,
        help="The question to ask"
    )
    chat_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for answering"
    )
    chat_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for model response"
    )
    chat_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve"
    )
    chat_parser.add_argument(
        "--vector-store-path",
        type=str,
        default=None,
        help="Path to the vector store"
    )
    chat_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override API key if not set via environment/.env"
    )
    chat_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Output the assembled prompt and metadata without calling the model"
    )
    chat_parser.add_argument(
        "--enable-compression",
        action="store_true",
        help="Enable compression for retrieved chunks"
    )
    chat_parser.add_argument(
        "--compression-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per compressed chunk (default: 200)"
    )
    chat_parser.add_argument(
        "--enable-multi-query",
        action="store_true",
        help="Enable multi-query retrieval"
    )
    chat_parser.add_argument(
        "--multi-query-count",
        type=int,
        default=None,
        help="Number of query variants to generate (default: 3)"
    )

    return parser

def run_ingest(args: argparse.Namespace) -> None:
    settings = RagQASettings.from_cli(args)
    ingest_pipeline(settings)

def run_chat(args: argparse.Namespace) -> None:
    settings = RagQASettings.from_cli(args)
    
    # Override optimization settings from CLI flags
    if hasattr(args, 'enable_compression') and args.enable_compression:
        settings.enable_compression = True
    if hasattr(args, 'compression_max_tokens') and args.compression_max_tokens is not None:
        settings.compression_max_tokens = args.compression_max_tokens
    if hasattr(args, 'enable_multi_query') and args.enable_multi_query:
        settings.enable_multi_query = True
    if hasattr(args, 'multi_query_count') and args.multi_query_count is not None:
        settings.multi_query_count = args.multi_query_count

    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=settings.api_key)
    
    # Load the vector store
    vector_store = FAISS.load_local(
        settings.vector_store_path,
        allow_dangerous_deserialization=True,
        embeddings=embedder
    )
    
    # Handle dry-run mode
    if args.dry_run:
        # Use retrieve_optimized if optimizations are enabled, otherwise use retrieve
        if settings.enable_multi_query or settings.enable_compression:
            chunks = retrieve_optimized(args.query, settings, vector_store)
        else:
            chunks = retrieve(args.query, settings, vector_store)
        
        if not chunks:
            print("No chunks retrieved for the query.")
            return
        
        prompt = build_prompt(
            "You are a helpful assistant that can answer questions about the document.",
            args.query,
            chunks
        )
        
        # Print metadata
        print("=" * 80)
        print("DRY RUN MODE - Prompt and Metadata")
        print("=" * 80)
        print(f"\nQuery: {args.query}")
        print(f"Number of chunks retrieved: {len(chunks)}")
        print(f"Top-k setting: {settings.top_k}")
        print(f"Model: {settings.model}")
        print(f"Temperature: {settings.temperature}")
        print(f"Prompt length: {len(prompt)} characters")
        print("\n" + "=" * 80)
        print("ASSEMBLED PROMPT:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        return
    
    # Run the chat pipeline
    # Use retrieve_optimized if optimizations are enabled
    if settings.enable_multi_query or settings.enable_compression:
        # We need to modify run_chat_chain to use retrieve_optimized, or create a wrapper
        # For now, let's retrieve chunks first, then build prompt and call model
        chunks = retrieve_optimized(args.query, settings, vector_store)
        if not chunks:
            print("I couldn't find relevant info in the provided docs.")
            return
        
        from src.chain import build_prompt, call_model, parse_response
        prompt = build_prompt(
            "You are a helpful assistant that can answer questions about the document.",
            args.query,
            chunks
        )
        raw_response = call_model(prompt, settings)
        result = parse_response(raw_response)
    else:
        result = run_chat_chain(args.query, settings, vector_store)
    
    # Print the answer
    print("\nAnswer:")
    print(result.answer)
    
    # Print citations if any
    if result.citations:
        print("\nCitations:")
        for citation in result.citations:
            page_display = citation.page if citation.page.startswith("Page") else f"Page {citation.page}"
            # Only show page_label if it's not empty
            page_label_display = f" | {citation.page_label}" if citation.page_label and citation.page_label != "N/A" else ""
            print(f"[{citation.id}] {citation.title} | {page_display}{page_label_display}")
            # Show text snippet if available
            if citation.text_snippet:
                print(f"    \"{citation.text_snippet}\"")

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "chat":
        run_chat(args)
    else:
        parser.print_help()
        sys.exit(1) 

if __name__ == "__main__":
    main()