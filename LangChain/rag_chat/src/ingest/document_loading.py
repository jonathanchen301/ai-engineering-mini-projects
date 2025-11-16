import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

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

def split_documents(documents: list[Document], chunk_size: int = 400, chunk_overlap: int = 40) -> list[Document]:
    """
    Split documents into chunks.

    Args:
    documents: list[Document] - list of documents to split.
    chunk_size: int - size of each chunk. Must be greater than 0.
    chunk_overlap: int - overlap between chunks. Must be less than chunk_size and greater than 0.

    Returns:
    list[Document] - list of chunks.
    """
    if chunk_size > 1000:
        print("Warning: chunk_size is greater than 1000. This may cause the model to include irrelevant information.")
    elif chunk_size < 500:
        print("Warning: chunk_size is less than 500. This causes the model to be more precise, but may lose context.")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def create_vectorstore(documents: list[Document], embedding_model: str = "text-embedding-3-small", save_dir: str = "./", save: bool = False):

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=openai_api_key
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    if save:
        vectorstore.save_local(save_dir)

    return vectorstore

def load_vectorstore(save_dir: str = "./", embedding_model: str = "text-embedding-3-small"):

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=openai_api_key
    )

    vectorstore = FAISS.load_local(save_dir, embeddings=embeddings, allow_dangerous_deserialization=True)

    return vectorstore