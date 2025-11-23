import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multi_query import generate_query_variants, multi_query_retrieve
from src.config import RagQASettings

class TestGenerateQueryVariants:
    def test_generate_query_variants_success(self):
        """Test that generate_query_variants returns a list of query strings."""
        settings = RagQASettings(multi_query_count=3, model="gpt-4o-mini", api_key="test-key")
        
        with patch('src.multi_query.ChatOpenAI'), \
             patch('src.multi_query.PromptTemplate') as mock_prompt, \
             patch('src.multi_query.JsonOutputParser'):
            
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "queries": ["variant 1", "variant 2", "variant 3"]
            }
            mock_chain.__or__ = lambda self, other: mock_chain
            mock_prompt.return_value.__or__ = lambda self, other: mock_chain
            
            result = generate_query_variants("What is AI?", settings)
            
            assert len(result) == 3
            assert result == ["variant 1", "variant 2", "variant 3"]
    
    def test_generate_query_variants_error_handling(self):
        """Test that generate_query_variants handles errors gracefully."""
        settings = RagQASettings(multi_query_count=3, model="gpt-4o-mini", api_key="test-key")
        
        with patch('src.multi_query.ChatOpenAI'), \
             patch('src.multi_query.PromptTemplate') as mock_prompt, \
             patch('src.multi_query.JsonOutputParser'):
            
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("API error")
            mock_chain.__or__ = lambda self, other: mock_chain
            mock_prompt.return_value.__or__ = lambda self, other: mock_chain
            
            result = generate_query_variants("What is AI?", settings)
            
            assert len(result) == 1
            assert result == ["What is AI?"]

class TestMultiQueryRetrieve:
    def test_multi_query_retrieve_success(self):
        """Test that multi_query_retrieve returns a list of documents."""
        settings = RagQASettings(top_k=2, model="gpt-4o-mini", api_key="test-key")
        queries = ["query 1", "query 2"]
        
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (Document(page_content="Content 1", metadata={"title": "Doc1"}), 0.9),
            (Document(page_content="Content 2", metadata={"title": "Doc2"}), 0.8)
        ]
        
        result = multi_query_retrieve(queries, settings, mock_store)
        
        assert len(result) == 2
        assert isinstance(result[0], Document)
        assert mock_store.similarity_search_with_score.call_count == 2
    
    def test_multi_query_retrieve_deduplicates(self):
        """Test that multi_query_retrieve deduplicates chunks."""
        settings = RagQASettings(top_k=2, model="gpt-4o-mini", api_key="test-key")
        queries = ["query 1", "query 2"]
        
        doc1 = Document(page_content="Same content", metadata={"title": "Doc1"})
        doc2 = Document(page_content="Different content", metadata={"title": "Doc2"})
        
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.side_effect = [
            [(doc1, 0.9), (doc2, 0.8)],  # First query
            [(doc1, 0.85), (doc2, 0.7)]  # Second query (duplicate doc1)
        ]
        
        result = multi_query_retrieve(queries, settings, mock_store)
        
        assert len(result) == 2  # Should deduplicate doc1
        assert result[0].page_content == "Same content"  # Higher score (0.9)
        assert result[1].page_content == "Different content"
    
    def test_multi_query_retrieve_sorts_by_score(self):
        """Test that multi_query_retrieve sorts by score descending."""
        settings = RagQASettings(top_k=2, model="gpt-4o-mini", api_key="test-key")
        queries = ["query 1"]
        
        doc1 = Document(page_content="Content 1", metadata={"title": "Doc1"})
        doc2 = Document(page_content="Content 2", metadata={"title": "Doc2"})
        
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (doc1, 0.7),  # Lower score
            (doc2, 0.9)   # Higher score
        ]
        
        result = multi_query_retrieve(queries, settings, mock_store)
        
        assert len(result) == 2
        assert result[0].page_content == "Content 2"  # Higher score first
        assert result[1].page_content == "Content 1"   # Lower score second