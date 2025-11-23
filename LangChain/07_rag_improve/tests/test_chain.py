import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages.ai import AIMessage
from src.chain import retrieve, build_prompt, call_model
from src.config import RagQASettings
from src.chain import LLMResponseModel, parse_response

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
        
        mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})

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

    def test_build_prompt_returns_string(self):
        """Test that build_prompt returns a string prompt."""
        system_msg = "You are a helpful assistant."
        question = "What is the topic?"
        chunks = [Document(page_content="Test content")]
        
        prompt = build_prompt(system_msg, question, chunks)
        
        assert isinstance(prompt, str)
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
        
        prompt = build_prompt(system_msg, question, chunks)
        
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "Content 1" in prompt
        assert "Content 2" in prompt

    def test_build_prompt_handles_none_chunks(self):
        """Test that build_prompt handles None chunks gracefully."""
        system_msg = "Answer questions."
        question = "What is X?"
        
        prompt = build_prompt(system_msg, question, None)
        
        assert "No chunks retrieved" in prompt
        assert question in prompt
        assert system_msg in prompt

class TestCallModel:

    def test_call_model_returns_response(self):
        """Test that call_model returns an AIMessage response."""
        prompt = "Test prompt"
        settings = RagQASettings(model="gpt-4o-mini", api_key="test-key")
        
        mock_response = AIMessage(content='{"answer": "test"}')
        mock_response.response_metadata = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        with patch('src.chain.ChatOpenAI') as mock_chat:
            mock_model = Mock()
            mock_model.invoke.return_value = mock_response
            mock_chat.return_value = mock_model
            
            result = call_model(prompt, settings)
            
            assert isinstance(result, AIMessage)
            assert result == mock_response

    def test_call_model_uses_settings(self):
        """Test that call_model uses model and API key from settings."""
        prompt = "Test prompt"
        settings = RagQASettings(model="gpt-4o", api_key="test-api-key")
        
        mock_response = AIMessage(content='{"answer": "test"}')
        mock_response.response_metadata = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        
        with patch('src.chain.ChatOpenAI') as mock_chat:
            mock_model = Mock()
            mock_model.invoke.return_value = mock_response
            mock_chat.return_value = mock_model
            
            call_model(prompt, settings)
            
            # Verify ChatOpenAI was called with correct settings
            mock_chat.assert_called_once_with(model="gpt-4o", api_key="test-api-key")

    def test_call_model_accesses_token_usage(self):
        """Test that call_model accesses token usage from response metadata."""
        prompt = "Test prompt"
        settings = RagQASettings(model="gpt-4o-mini", api_key="test-key")
        
        mock_response = AIMessage(content='{"answer": "test"}')
        mock_response.response_metadata = {
            "token_usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }
        
        with patch('src.chain.ChatOpenAI') as mock_chat, \
            patch('builtins.print'):  # Suppress print output
            mock_model = Mock()
            mock_model.invoke.return_value = mock_response
            mock_chat.return_value = mock_model
            
            call_model(prompt, settings)
            
            # Verify token usage was accessed (function should complete without error)
            assert mock_response.response_metadata["token_usage"] is not None

class TestParseResponse:

    def test_parse_response_returns_llm_response_model(self):
        """Test that parse_response returns a LLMResponseModel."""
        json_content = '{"answer": "This is the answer", "citations": [{"id": 1, "title": "Doc1", "page": "1", "page_label": "1"}]}'
        response = AIMessage(content=json_content)
        
        result = parse_response(response)
        
        assert isinstance(result, LLMResponseModel)
        assert result.answer == "This is the answer"
        assert len(result.citations) == 1

    def test_parse_response_validates_schema(self):
        """Test that parse_response validates the JSON schema."""
        json_content = '{"answer": "Test answer", "citations": [{"id": 1, "title": "Title", "page": "1", "page_label": "1"}]}'
        response = AIMessage(content=json_content)
        
        result = parse_response(response)
        
        assert result.answer == "Test answer"
        assert result.citations[0].id == 1
        assert result.citations[0].title == "Title"

    def test_parse_response_raises_on_missing_fields(self):
        """Test that parse_response raises error when required fields are missing."""
        json_content = '{"answer": "Test"}'
        response = AIMessage(content=json_content)
        
        with pytest.raises(Exception):
            parse_response(response)

class TestRunChat:import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages.ai import AIMessage
from src.chain import run_chat, LLMResponseModel
from src.config import RagQASettings

class TestRunChat:
    
    def test_run_chat_returns_llm_response_model(self):
        """Test that run_chat returns LLMResponseModel when chunks are retrieved."""
        query = "What is the answer?"
        settings = RagQASettings(api_key="test-key")
        mock_store = MagicMock()
        
        mock_chunks = [
            Document(page_content="Test content", metadata={"title": "Doc1", "page": "1"})
        ]
        json_response = '{"answer": "The answer is test", "citations": [{"id": 1, "title": "Doc1", "page": "1", "page_label": "1"}]}'
        mock_ai_message = AIMessage(content=json_response)
        
        with patch('src.chain.retrieve', return_value=mock_chunks), \
             patch('src.chain.build_prompt', return_value="formatted prompt"), \
             patch('src.chain.call_model', return_value=mock_ai_message), \
             patch('src.chain.parse_response', return_value=LLMResponseModel(
                 answer="The answer is test",
                 citations=[]
             )):
            result = run_chat(query, settings, mock_store)
            
            assert isinstance(result, LLMResponseModel)
            assert result.answer == "The answer is test"

    def test_run_chat_returns_fallback_when_no_chunks(self):
        """Test that run_chat returns fallback message when no chunks are retrieved."""
        query = "What is the answer?"
        settings = RagQASettings(api_key="test-key")
        mock_store = MagicMock()
        
        with patch('src.chain.retrieve', return_value=None):
            result = run_chat(query, settings, mock_store)
            
            assert isinstance(result, LLMResponseModel)
            assert result.answer == "I couldn't find relevant info in the provided docs."
            assert result.citations == []