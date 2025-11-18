import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document
from src.chain import retrieve, build_prompt
from src.config import RagQASettings

class TestRetrieve:

    def test_retrieve_returns_chunks(self):
        """Test that retrieve returns a list of documents."""
        query = "What is the main topic?"
        settings = RagQASettings(top_k=3)
        
        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_chunks = [
            Document(page_content="Chunk 1"),
            Document(page_content="Chunk 2")
        ]
        mock_retriever.invoke.return_value = mock_chunks
        mock_store.as_retriever.return_value = mock_retriever
        
        result = retrieve(query, settings, mock_store)
        
        assert result == mock_chunks
        assert len(result) == 2

    def test_retrieve_uses_top_k_from_settings(self):
        """Test that retrieve uses settings.top_k correctly."""
        query = "Test query"
        settings = RagQASettings(top_k=5)
        
        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [Document(page_content="test")]
        mock_store.as_retriever.return_value = mock_retriever
        
        retrieve(query, settings, mock_store)
        
        mock_store.as_retriever.assert_called_once_with(k=5)

    def test_retrieve_returns_none_when_empty(self):
        """Test that retrieve returns None when no chunks are found."""
        query = "Unrelated query"
        settings = RagQASettings(top_k=3)
        
        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_store.as_retriever.return_value = mock_retriever
        
        result = retrieve(query, settings, mock_store)
        
        assert result is None

class TestBuildPrompt:

    def test_build_prompt_returns_tuple(self):
        """Test that build_prompt returns a tuple of (prompt, chunks)."""
        system_msg = "You are a helpful assistant."
        question = "What is the topic?"
        chunks = [Document(page_content="Test content")]
        
        prompt, returned_chunks = build_prompt(system_msg, question, chunks)
        
        assert isinstance(prompt, str)
        assert returned_chunks == chunks
        assert system_msg in prompt
        assert question in prompt

    def test_build_prompt_formats_chunks_with_citations(self):
        """Test that chunks are formatted with numbered citations."""
        system_msg = "Answer questions."
        question = "What is X?"
        chunks = [
            Document(page_content="Content 1", metadata={"title": "Doc1", "page": 1}),
            Document(page_content="Content 2", metadata={"title": "Doc2", "page": 2})
        ]
        
        prompt, _ = build_prompt(system_msg, question, chunks)
        
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "Content 1" in prompt
        assert "Content 2" in prompt

    def test_build_prompt_handles_none_chunks(self):
        """Test that build_prompt handles None chunks gracefully."""
        system_msg = "Answer questions."
        question = "What is X?"
        
        prompt, returned_chunks = build_prompt(system_msg, question, None)
        
        assert "No chunks retrieved" in prompt
        assert returned_chunks is None
        assert question in prompt