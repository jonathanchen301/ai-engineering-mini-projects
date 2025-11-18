import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document
from src.chain import retrieve
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