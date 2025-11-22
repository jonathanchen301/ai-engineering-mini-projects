import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compression import compress_chunks
from src.config import RagQASettings

class TestCompressChunks:
    def test_compress_chunks_success(self):
        """Test that compress_chunks returns a list of documents."""
        settings = RagQASettings(compression_max_tokens=200, model="gpt-4o-mini", api_key="test-key")
        chunks = [
            Document(page_content="This is a long text that needs compression.", metadata={"title": "Doc1", "page": 1})
        ]
        
        with patch('src.compression.ChatOpenAI') as mock_model, \
             patch('src.compression.PromptTemplate') as mock_prompt, \
             patch('src.compression.StrOutputParser') as mock_parser:
            
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Compressed summary"
            mock_chain.__or__ = lambda self, other: mock_chain
            mock_prompt.return_value.__or__ = lambda self, other: mock_chain
            
            result = compress_chunks(chunks, settings)
            
            assert len(result) == 1
            assert result[0].page_content == "Compressed summary"
            assert result[0].metadata == chunks[0].metadata
    
    def test_compress_chunks_preserves_metadata(self):
        """Test that compress_chunks preserves metadata."""
        settings = RagQASettings(compression_max_tokens=200, model="gpt-4o-mini", api_key="test-key")
        chunks = [
            Document(page_content="Text", metadata={"title": "Test", "page": 2, "page_label": "Page 2"})
        ]
        
        with patch('src.compression.ChatOpenAI'), \
             patch('src.compression.PromptTemplate') as mock_prompt, \
             patch('src.compression.StrOutputParser'):
            
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Summary"
            mock_prompt.return_value.__or__ = lambda self, other: mock_chain
            
            result = compress_chunks(chunks, settings)
            
            assert result[0].metadata["title"] == "Test"
            assert result[0].metadata["page"] == 2
            assert result[0].metadata["page_label"] == "Page 2"
    
    def test_compress_chunks_error_handling(self):
        """Test that compress_chunks handles errors gracefully."""
        settings = RagQASettings(compression_max_tokens=200, model="gpt-4o-mini", api_key="test-key")
        chunks = [
            Document(page_content="Text", metadata={"title": "Test"})
        ]
        
        with patch('src.compression.ChatOpenAI'), \
             patch('src.compression.PromptTemplate') as mock_prompt, \
             patch('src.compression.StrOutputParser'):
            
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("API error")
            mock_prompt.return_value.__or__ = lambda self, other: mock_chain
            
            result = compress_chunks(chunks, settings)
            
            assert len(result) == 1
            assert result[0].page_content == "Text"
            assert result[0].metadata == chunks[0].metadata