import pytest
import os
import sys
import warnings
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingest.document_loading import load_pdfs, split_documents
from src.ingest.pipeline import ingest_documents
from src.chain.chain_helpers import retrieve_documents
from langchain_core.documents import Document

class TestLoadPDFs:
    
    def test_load_pdfs_returns_list(self):
        """Test that load_pdfs returns a list of documents."""
        documents_dir = os.path.join(project_root, 'src', 'ingest', 'documents')
        documents = load_pdfs(documents_dir)
        
        assert isinstance(documents, list)
        assert len(documents) > 0
    
    def test_documents_have_correct_structure(self):
        """Test that returned documents have page_content and metadata."""
        documents_dir = os.path.join(project_root, 'src', 'ingest', 'documents')
        documents = load_pdfs(documents_dir)
        
        for doc in documents:
            assert isinstance(doc, Document)
            assert hasattr(doc, 'page_content')
            assert hasattr(doc, 'metadata')
    
    def test_metadata_contains_source(self):
        """Test that document metadata contains source filename."""
        documents_dir = os.path.join(project_root, 'src', 'ingest', 'documents')
        documents = load_pdfs(documents_dir)
        
        # Check that at least one document has source in metadata
        has_source = any(
            hasattr(doc, 'metadata') and 
            doc.metadata and 
            'source' in doc.metadata 
            for doc in documents
        )
        assert has_source

class TestSplitDocuments:
    
    def test_split_documents_returns_list(self):
        """Test that split_documents returns a list of chunks."""
        documents_dir = os.path.join(project_root, 'src', 'ingest', 'documents')
        documents = load_pdfs(documents_dir)
        
        chunks = split_documents(documents)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
    
    def test_chunks_have_correct_structure(self):
        """Test that chunks have page_content and metadata."""
        documents_dir = os.path.join(project_root, 'src', 'ingest', 'documents')
        documents = load_pdfs(documents_dir)
        
        chunks = split_documents(documents)
        
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert hasattr(chunk, 'page_content')
            assert hasattr(chunk, 'metadata')
    
    def test_metadata_preserved_after_splitting(self):
        """Test that metadata is preserved in chunks."""
        documents_dir = os.path.join(project_root, 'src', 'ingest', 'documents')
        documents = load_pdfs(documents_dir)
        
        chunks = split_documents(documents)
        
        # Check that chunks have metadata
        has_metadata = any(
            hasattr(chunk, 'metadata') and chunk.metadata
            for chunk in chunks
        )
        assert has_metadata
    
    def test_custom_chunk_size(self):
        """Test that custom chunk_size parameter works."""
        documents_dir = os.path.join(project_root, 'src', 'ingest', 'documents')
        documents = load_pdfs(documents_dir)
        
        chunks = split_documents(documents, chunk_size=200)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0    

class TestIngestion:
    
    def test_ingest_documents_returns_vectorstore(self):
        """Test that ingest_documents returns a vectorstore."""
        documents_dir = os.path.join(project_root, 'src', 'ingest', 'documents')
        
        vectorstore = ingest_documents(document_dir=documents_dir, save_vectorstore=False)
        
        # Should return a vectorstore object
        assert vectorstore is not None
        assert hasattr(vectorstore, 'as_retriever')
    
    def test_ingest_documents_creates_chunks(self):
        """Test that ingestion creates chunks and vectorstore."""
        documents_dir = os.path.join(project_root, 'src', 'ingest', 'documents')
        
        vectorstore = ingest_documents(document_dir=documents_dir, save_vectorstore=False)
        
        # Should be able to create a retriever
        retriever = vectorstore.as_retriever()
        assert retriever is not None
    
class TestRetrieval:
    
    def test_retrieve_documents_returns_list(self):
        """Test that retrieve_documents returns a list of documents."""
        documents_dir = os.path.join(project_root, 'src', 'ingest', 'documents')
        vectorstore = ingest_documents(document_dir=documents_dir, save_vectorstore=False)
        
        results = retrieve_documents(vectorstore, "test query", k=3)
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    def test_retrieve_documents_respects_k_parameter(self):
        """Test that retrieve_documents respects the k parameter."""
        documents_dir = os.path.join(project_root, 'src', 'ingest', 'documents')
        vectorstore = ingest_documents(document_dir=documents_dir, save_vectorstore=False)
        
        results = retrieve_documents(vectorstore, "test query", k=2)
        
        assert len(results) <= 2
    
    def test_retrieve_documents_returns_documents(self):
        """Test that retrieved items are Document objects."""
        documents_dir = os.path.join(project_root, 'src', 'ingest', 'documents')
        vectorstore = ingest_documents(document_dir=documents_dir, save_vectorstore=False)
        
        results = retrieve_documents(vectorstore, "test query", k=3)
        
        for doc in results:
            assert isinstance(doc, Document)
            assert hasattr(doc, 'page_content')