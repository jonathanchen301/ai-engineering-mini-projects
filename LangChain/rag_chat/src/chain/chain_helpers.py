from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

def retrieve_documents(vectorstore: FAISS, query: str, k: int = 3) -> list[Document]:

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})    
    if retriever is None:
        assert False, "Retriever is not initialized"
    res = retriever.invoke(query)
    if len(res) == 0:
        assert False, "No documents found"
    return res