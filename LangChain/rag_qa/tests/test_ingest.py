import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.config import RagQASettings
from src.ingest import split_documents, load_sources, embed_and_persist_chunks
from unittest.mock import Mock

class TestLoadSources:
    def test_load_sources_success(self, tmp_path):
        """Test loading PDFs from a directory with valid PDFs."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n")
        
        mock_doc = MagicMock()
        mock_doc.page_content = "test content"
        
        settings = RagQASettings(document_dir=str(tmp_path))
        
        with patch('src.ingest.PyPDFLoader') as mock_loader:
            mock_loader.return_value.load.return_value = [mock_doc]
            documents = load_sources(settings)
            assert len(documents) > 0

    def test_load_sources_directory_not_found(self):
        """Test that FileNotFoundError is raised when directory doesn't exist."""
        settings = RagQASettings(document_dir="nonexistent_directory_12345")
        
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_sources(settings)

    def test_load_sources_no_pdfs(self, tmp_path):
        """Test that ValueError is raised when no PDFs are found."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("not a pdf")
        
        settings = RagQASettings(document_dir=str(tmp_path))
        
        with pytest.raises(ValueError, match="No PDF files found"):
            load_sources(settings)

class TestSplitDocuments:

    def test_split_documents_creates_chunks(self):
        """Test that documents are split into multiple chunks."""
        long_text = "This is a test document. " * 100
        doc = Document(page_content=long_text)
        
        settings = RagQASettings(chunk_size=50, chunk_overlap=10)
        chunks = split_documents([doc], settings)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_split_documents_respects_chunk_size(self):
        """Test that chunks don't exceed chunk_size."""
        long_text = "word " * 200
        doc = Document(page_content=long_text)
        
        settings = RagQASettings(chunk_size=100, chunk_overlap=0)
        chunks = split_documents([doc], settings)
    
        assert all(len(chunk.page_content) <= 150 for chunk in chunks)

    def test_split_documents_uses_settings(self):
        """Test that different settings produce different chunk counts."""
        long_text = "word " * 200
        doc = Document(page_content=long_text)
        
        settings_small = RagQASettings(chunk_size=50, chunk_overlap=0)
        settings_large = RagQASettings(chunk_size=200, chunk_overlap=0)
        
        chunks_small = split_documents([doc], settings_small)
        chunks_large = split_documents([doc], settings_large)
        
        assert len(chunks_small) > len(chunks_large)

class TestEmbedAndPersistChunks:
    def test_embed_and_persist_chunks_creates_faiss_index(self):
        """Test that embed_and_persist_chunks creates and returns a FAISS index."""
        chunks = [
            Document(page_content="Test document 1"),
            Document(page_content="Test document 2")
        ]
        settings = RagQASettings(api_key="test-key", vector_store_path="./test_store")
        
        with patch('src.ingest.OpenAIEmbeddings') as mock_embeddings, \
            patch('src.ingest.FAISS') as mock_faiss, \
            patch('os.makedirs'), \
            patch('os.path.exists', return_value=False):
            
            mock_embedder = Mock()
            mock_embeddings.return_value = mock_embedder
            
            mock_index = MagicMock()
            mock_faiss.from_documents.return_value = mock_index
            
            result = embed_and_persist_chunks(chunks, settings)
            
            assert result == mock_index
            mock_faiss.from_documents.assert_called_once()

    def test_embed_and_persist_chunks_uses_correct_embedding_model(self):
        """Test that embed_and_persist_chunks uses the correct embedding model."""
        chunks = [Document(page_content="Test")]
        settings = RagQASettings(api_key="test-key", vector_store_path="./test_store")
        
        with patch('src.ingest.OpenAIEmbeddings') as mock_embeddings, \
            patch('src.ingest.FAISS') as mock_faiss, \
            patch('os.makedirs'), \
            patch('os.path.exists', return_value=False):
            
            mock_embedder = Mock()
            mock_embeddings.return_value = mock_embedder
            mock_faiss.from_documents.return_value = MagicMock()
            
            embed_and_persist_chunks(chunks, settings)
            
            mock_embeddings.assert_called_once_with(
                model="text-embedding-3-small",
                api_key="test-key"
            )

    def test_embed_and_persist_chunks_saves_to_disk(self):
        """Test that embed_and_persist_chunks saves the index to the specified path."""
        chunks = [Document(page_content="Test")]
        settings = RagQASettings(api_key="test-key", vector_store_path="./test_store")
        
        with patch('src.ingest.OpenAIEmbeddings') as mock_embeddings, \
            patch('src.ingest.FAISS') as mock_faiss, \
            patch('os.makedirs') as mock_makedirs, \
            patch('os.path.exists', return_value=False):
            
            mock_embedder = Mock()
            mock_embeddings.return_value = mock_embedder
            
            mock_index = MagicMock()
            mock_faiss.from_documents.return_value = mock_index
            
            embed_and_persist_chunks(chunks, settings)
            
            mock_index.save_local.assert_called_once_with("./test_store")
            mock_makedirs.assert_called_once_with("./test_store", exist_ok=True)