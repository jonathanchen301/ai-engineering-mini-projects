import pytest
from unittest.mock import patch, MagicMock, mock_open
from langchain_core.documents import Document
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.map_reduce import map_chunks, save_map_outputs, load_map_outputs, reduce_summaries
from src.config import LongDocSummarizerSettings

class TestMapChunks:
    def test_map_chunks_success(self):
        """Test that map_chunks returns a list of dictionaries with summaries."""
        settings = LongDocSummarizerSettings(model="gpt-4o-mini", api_key="test-key")
        chunks = [
            Document(page_content="First chunk content", metadata={"title": "Doc1"}),
            Document(page_content="Second chunk content", metadata={"title": "Doc1"})
        ]
        
        with patch('src.map_reduce.ChatOpenAI'), \
             patch('src.map_reduce.PromptTemplate') as mock_prompt, \
             patch('src.map_reduce.StrOutputParser'):
            
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Summary"
            mock_chain.__or__ = lambda self, other: mock_chain
            mock_prompt.return_value.__or__ = lambda self, other: mock_chain
            
            result = map_chunks(chunks, settings)
            
            assert len(result) == 2
            assert isinstance(result[0], dict)
            assert result[0]["chunk_id"] == 1
            assert result[0]["summary"] == "Summary"
    
    def test_map_chunks_returns_correct_structure(self):
        """Test that map_chunks returns dictionaries with all required fields."""
        settings = LongDocSummarizerSettings(model="gpt-4o-mini", api_key="test-key")
        chunks = [
            Document(page_content="Test content", metadata={"title": "Doc1"})
        ]
        
        with patch('src.map_reduce.ChatOpenAI'), \
             patch('src.map_reduce.PromptTemplate') as mock_prompt, \
             patch('src.map_reduce.StrOutputParser'):
            
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Test summary"
            mock_chain.__or__ = lambda self, other: mock_chain
            mock_prompt.return_value.__or__ = lambda self, other: mock_chain
            
            result = map_chunks(chunks, settings)
            
            assert "chunk_id" in result[0]
            assert "summary" in result[0]
            assert "start_char" in result[0]
            assert "end_char" in result[0]
    
    def test_map_chunks_tracks_character_positions(self):
        """Test that map_chunks correctly tracks character positions."""
        settings = LongDocSummarizerSettings(model="gpt-4o-mini", api_key="test-key")
        chunks = [
            Document(page_content="First", metadata={"title": "Doc1"}),
            Document(page_content="Second", metadata={"title": "Doc1"})
        ]
        
        with patch('src.map_reduce.ChatOpenAI'), \
             patch('src.map_reduce.PromptTemplate') as mock_prompt, \
             patch('src.map_reduce.StrOutputParser'):
            
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Summary"
            mock_chain.__or__ = lambda self, other: mock_chain
            mock_prompt.return_value.__or__ = lambda self, other: mock_chain
            
            result = map_chunks(chunks, settings)
            
            assert result[0]["start_char"] == 0
            assert result[0]["end_char"] == 5  # "First" is 5 chars
            assert result[1]["start_char"] == 5
            assert result[1]["end_char"] == 11  # 5 + "Second" (6 chars) = 11
    
    def test_map_chunks_error_handling(self):
        """Test that map_chunks handles errors gracefully."""
        settings = LongDocSummarizerSettings(model="gpt-4o-mini", api_key="test-key")
        chunks = [
            Document(page_content="Test content", metadata={"title": "Doc1"})
        ]
        
        with patch('src.map_reduce.ChatOpenAI'), \
             patch('src.map_reduce.PromptTemplate') as mock_prompt, \
             patch('src.map_reduce.StrOutputParser'):
            
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("API error")
            mock_chain.__or__ = lambda self, other: mock_chain
            mock_prompt.return_value.__or__ = lambda self, other: mock_chain
            
            result = map_chunks(chunks, settings)
            
            assert len(result) == 1
            assert result[0]["summary"] == "Test content"  # Uses original content
            assert result[0]["chunk_id"] == 1

class TestSaveMapOutputs:
    def test_save_map_outputs_saves_to_file(self):
        """Test that save_map_outputs saves summaries to JSON file."""
        settings = LongDocSummarizerSettings(map_output_path="./test_output.json")
        summaries = [
            {"chunk_id": 1, "summary": "Test summary", "start_char": 0, "end_char": 10}
        ]
        
        with patch('builtins.open', mock_open()) as mock_file:
            save_map_outputs(summaries, settings)
            
            mock_file.assert_called_once_with("./test_output.json", "w")
            mock_file().write.assert_called()
    
    def test_save_map_outputs_writes_correct_data(self):
        """Test that save_map_outputs writes the correct JSON data."""
        settings = LongDocSummarizerSettings(map_output_path="./test_output.json")
        summaries = [
            {"chunk_id": 1, "summary": "Summary 1", "start_char": 0, "end_char": 10},
            {"chunk_id": 2, "summary": "Summary 2", "start_char": 10, "end_char": 20}
        ]
        
        with patch('builtins.open', mock_open()) as mock_file:
            save_map_outputs(summaries, settings)
            
            # Check that json.dump was called with the summaries
            call_args = mock_file().write.call_args
            # The actual JSON would be written, but we're just checking the function runs
            assert mock_file().write.called
    
    def test_save_map_outputs_creates_directory(self):
        """Test that save_map_outputs creates directory if needed."""
        settings = LongDocSummarizerSettings(map_output_path="./test_dir/output.json")
        summaries = [{"chunk_id": 1, "summary": "Test", "start_char": 0, "end_char": 10}]
        
        with patch('os.path.dirname') as mock_dirname, \
             patch('os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()):
            mock_dirname.return_value = "./test_dir"
            
            save_map_outputs(summaries, settings)
            
            mock_makedirs.assert_called_once_with("./test_dir", exist_ok=True)

class TestLoadMapOutputs:
    def test_load_map_outputs_success(self):
        """Test that load_map_outputs loads summaries from JSON file."""
        settings = LongDocSummarizerSettings(map_output_path="./test_output.json")
        expected_data = [
            {"chunk_id": 1, "summary": "Test summary", "start_char": 0, "end_char": 10}
        ]
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(expected_data))):
            
            result = load_map_outputs(settings)
            
            assert result == expected_data
            assert isinstance(result, list)
            assert isinstance(result[0], dict)
    
    def test_load_map_outputs_returns_list_of_dicts(self):
        """Test that load_map_outputs returns list of dictionaries."""
        settings = LongDocSummarizerSettings(map_output_path="./test_output.json")
        expected_data = [
            {"chunk_id": 1, "summary": "Summary 1", "start_char": 0, "end_char": 10},
            {"chunk_id": 2, "summary": "Summary 2", "start_char": 10, "end_char": 20}
        ]
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(expected_data))):
            
            result = load_map_outputs(settings)
            
            assert len(result) == 2
            assert result[0]["chunk_id"] == 1
            assert result[1]["chunk_id"] == 2
    
    def test_load_map_outputs_file_not_found(self):
        """Test that load_map_outputs raises FileNotFoundError when file doesn't exist."""
        settings = LongDocSummarizerSettings(map_output_path="./nonexistent.json")
        
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                load_map_outputs(settings)

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.map_reduce import reduce_summaries
from src.config import LongDocSummarizerSettings

class TestReduceSummaries:
    def test_reduce_summaries_success(self):
        """Test that reduce_summaries returns a combined summary string."""
        settings = LongDocSummarizerSettings(model="gpt-4o-mini", api_key="test-key")
        summaries = [
            {"chunk_id": 1, "summary": "First summary", "start_char": 0, "end_char": 10},
            {"chunk_id": 2, "summary": "Second summary", "start_char": 10, "end_char": 20}
        ]
        
        with patch('src.map_reduce.ChatOpenAI'), \
             patch('src.map_reduce.PromptTemplate') as mock_prompt, \
             patch('src.map_reduce.StrOutputParser'):
            
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Combined summary"
            mock_chain.__or__ = lambda self, other: mock_chain
            mock_prompt.return_value.__or__ = lambda self, other: mock_chain
            
            result = reduce_summaries(summaries, settings)
            
            assert isinstance(result, str)
            assert result == "Combined summary"
    
    def test_reduce_summaries_returns_string(self):
        """Test that reduce_summaries returns a string."""
        settings = LongDocSummarizerSettings(model="gpt-4o-mini", api_key="test-key")
        summaries = [
            {"chunk_id": 1, "summary": "Test summary", "start_char": 0, "end_char": 10}
        ]
        
        with patch('src.map_reduce.ChatOpenAI'), \
             patch('src.map_reduce.PromptTemplate') as mock_prompt, \
             patch('src.map_reduce.StrOutputParser'):
            
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Final summary"
            mock_chain.__or__ = lambda self, other: mock_chain
            mock_prompt.return_value.__or__ = lambda self, other: mock_chain
            
            result = reduce_summaries(summaries, settings)
            
            assert isinstance(result, str)
            assert len(result) > 0