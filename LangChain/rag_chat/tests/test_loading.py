import pytest
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingest.document_loading import load_pdfs
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
    