import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(directory: str) -> list[Document]:
    """
    Load documents from a directory using a langchain loader .

    Args:
    directory: str - path to directory containing documents.

    Returns:
    list[Document] - list of documents.
    """
    documents = []
    for file in os.listdir(directory):
        if not file.endswith(".pdf"):
            continue
        try:
            loader = PyPDFLoader(os.path.join(directory, file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    return documents