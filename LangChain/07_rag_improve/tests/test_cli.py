import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli import build_parser, run_ingest, run_chat, main
from src.config import RagQASettings

class TestBuildParser:
    def test_build_parser_creates_parser(self):
        """Test that build_parser returns an ArgumentParser."""
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == "rag_qa"

    def test_build_parser_has_ingest_subcommand(self):
        """Test that parser has ingest subcommand."""
        parser = build_parser()
        args = parser.parse_args(["ingest", "--document-dir", "test_dir"])
        assert args.command == "ingest"
        assert args.document_dir == "test_dir"

    def test_build_parser_has_chat_subcommand(self):
        """Test that parser has chat subcommand."""
        parser = build_parser()
        args = parser.parse_args(["chat", "test query"])
        assert args.command == "chat"
        assert args.query == "test query"

    def test_chat_parser_has_dry_run_flag(self):
        """Test that chat parser has --dry-run flag."""
        parser = build_parser()
        args = parser.parse_args(["chat", "test query", "--dry-run"])
        assert args.dry_run is True

class TestRunIngest:
    def test_run_ingest_calls_ingest_pipeline(self):
        """Test that run_ingest calls ingest_pipeline with settings."""
        mock_args = argparse.Namespace(
            document_dir="test_dir",
            vector_store_path=None,
            chunk_size=None,
            chunk_overlap=None,
            api_key=None
        )
        
        with patch('src.cli.ingest_pipeline') as mock_pipeline:
            run_ingest(mock_args)
            mock_pipeline.assert_called_once()
            # Check that settings were created
            call_args = mock_pipeline.call_args[0][0]
            assert isinstance(call_args, RagQASettings)

class TestRunChat:
    def test_run_chat_calls_run_chat_chain(self):
        """Test that run_chat calls run_chat_chain in normal mode."""
        mock_args = argparse.Namespace(
            query="test question",
            model=None,
            temperature=None,
            top_k=None,
            vector_store_path="./vector_store",
            api_key="test-key",
            dry_run=False
        )
        
        mock_store = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "Test answer"
        mock_result.citations = []
        
        with patch('src.cli.FAISS.load_local', return_value=mock_store), \
             patch('src.cli.OpenAIEmbeddings'), \
             patch('src.cli.run_chat_chain', return_value=mock_result):
            run_chat(mock_args)
            # Verify run_chat_chain was called
            from src.cli import run_chat_chain
            run_chat_chain.assert_called_once()

    def test_run_chat_dry_run_prints_prompt(self):
        """Test that run_chat prints prompt in dry-run mode."""
        mock_args = argparse.Namespace(
            query="test question",
            model=None,
            temperature=None,
            top_k=None,
            vector_store_path="./vector_store",
            api_key="test-key",
            dry_run=True
        )
        
        mock_store = MagicMock()
        mock_chunks = [MagicMock()]
        
        with patch('src.cli.FAISS.load_local', return_value=mock_store), \
             patch('src.cli.OpenAIEmbeddings'), \
             patch('src.cli.retrieve', return_value=mock_chunks), \
             patch('src.cli.build_prompt', return_value="test prompt"):
            with patch('builtins.print') as mock_print:
                run_chat(mock_args)
                # Verify print was called (for dry-run output)
                assert mock_print.called
                # Check that prompt is in output
                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any("test prompt" in str(call) for call in mock_print.call_args_list)

class TestMain:
    def test_main_routes_to_ingest(self):
        """Test that main routes ingest command correctly."""
        with patch('src.cli.build_parser') as mock_build, \
             patch('src.cli.run_ingest') as mock_ingest:
            mock_parser = MagicMock()
            mock_args = argparse.Namespace(command="ingest")
            mock_parser.parse_args.return_value = mock_args
            mock_build.return_value = mock_parser
            
            main()
            mock_ingest.assert_called_once_with(mock_args)

    def test_main_routes_to_chat(self):
        """Test that main routes chat command correctly."""
        with patch('src.cli.build_parser') as mock_build, \
             patch('src.cli.run_chat') as mock_chat:
            mock_parser = MagicMock()
            mock_args = argparse.Namespace(command="chat")
            mock_parser.parse_args.return_value = mock_args
            mock_build.return_value = mock_parser
            
            main()
            mock_chat.assert_called_once_with(mock_args)