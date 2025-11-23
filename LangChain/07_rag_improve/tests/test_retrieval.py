import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import retrieve_optimized
from src.config import RagQASettings

class TestRetrieveOptimized:
    def test_retrieve_optimized_standard_retrieval(self):
        """Test standard retrieval without multi-query or compression."""
        settings = RagQASettings(enable_multi_query=False, enable_compression=False)
        query = "test query"
        mock_store = MagicMock()
        
        with patch('src.retrieval.retrieve') as mock_retrieve:
            mock_retrieve.return_value = [
                Document(page_content="test content", metadata={"title": "Test"})
            ]
            
            result = retrieve_optimized(query, settings, mock_store)
            
            assert len(result) == 1
            mock_retrieve.assert_called_once_with(query, settings, mock_store)
    
    def test_retrieve_optimized_with_compression(self):
        """Test retrieval with compression enabled."""
        settings = RagQASettings(enable_multi_query=False, enable_compression=True)
        query = "test query"
        mock_store = MagicMock()
        
        with patch('src.retrieval.retrieve') as mock_retrieve, \
             patch('src.retrieval.compress_chunks') as mock_compress:
            mock_retrieve.return_value = [
                Document(page_content="test content", metadata={"title": "Test"})
            ]
            mock_compress.return_value = [
                Document(page_content="compressed", metadata={"title": "Test"})
            ]
            
            result = retrieve_optimized(query, settings, mock_store)
            
            assert len(result) == 1
            mock_compress.assert_called_once()
    
    def test_retrieve_optimized_with_multi_query(self):
        """Test retrieval with multi-query enabled."""
        settings = RagQASettings(enable_multi_query=True, enable_compression=False)
        query = "test query"
        mock_store = MagicMock()
        
        with patch('src.retrieval.generate_query_variants') as mock_generate, \
             patch('src.retrieval.multi_query_retrieve') as mock_multi:
            mock_generate.return_value = ["query 1", "query 2"]
            mock_multi.return_value = [
                Document(page_content="test", metadata={"title": "Test"})
            ]
            
            result = retrieve_optimized(query, settings, mock_store)
            
            assert len(result) == 1
            mock_generate.assert_called_once_with(query, settings)
            mock_multi.assert_called_once_with(["query 1", "query 2"], settings, mock_store)
    
    def test_retrieve_optimized_handles_none(self):
        """Test that retrieve_optimized handles None from retrieve()."""
        settings = RagQASettings(enable_multi_query=False, enable_compression=False)
        query = "test query"
        mock_store = MagicMock()
        
        with patch('src.retrieval.retrieve') as mock_retrieve:
            mock_retrieve.return_value = None
            
            result = retrieve_optimized(query, settings, mock_store)
            
            assert result == []