import argparse
import sys

from src.config import RagQASettings
from src.ingest import ingest_pipeline

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

    return parser

def run_ingest(args: argparse.Namespace) -> None:
    settings = RagQASettings.from_cli(args)
    ingest_pipeline(settings)

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest":
        run_ingest(args)
    else:
        parser.print_help()
        sys.exit(1) 

if __name__ == "__main__":
    main()