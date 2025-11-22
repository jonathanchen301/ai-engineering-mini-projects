import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chain import run_chat, LLMResponseModel
from src.config import RagQASettings

class TestGolden:
    def test_golden_query_returns_json_with_citations(self):
        """Golden test: known query returns JSON with citations field."""
        query = "What is the main topic?"
        settings = RagQASettings(api_key="test-key")
        
        # Mock vector store
        mock_store = MagicMock()
        mock_chunks = [
            MagicMock(page_content="Test content", metadata={"title": "Doc1", "page": "1"})
        ]
        
        # Mock model response
        mock_json_response = '{"answer": "The main topic is testing.", "citations": [{"id": 1, "title": "Doc1", "page": "1", "page_label": "1", "text_snippet": "Test content"}]}'
        
        with patch('src.chain.retrieve', return_value=mock_chunks), \
             patch('src.chain.call_model') as mock_call:
            from langchain_core.messages.ai import AIMessage
            mock_call.return_value = AIMessage(content=mock_json_response)
            
            result = run_chat(query, settings, mock_store)
            
            # Verify structure
            assert isinstance(result, LLMResponseModel)
            assert hasattr(result, 'answer')
            assert hasattr(result, 'citations')
            assert isinstance(result.citations, list)
            assert len(result.citations) > 0
            assert result.citations[0].id == 1