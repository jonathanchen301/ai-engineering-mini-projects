from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Document Loading
loader = PyPDFLoader("documents/rag_test.pdf")
document = loader.load()

# Text Splitting / Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=2)
chunks = splitter.split_documents(document)

# Embedding
texts = [chunk.page_content for chunk in chunks]
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key = openai_api_key)

# Build a vectore store locally from documents
# vectorstore = FAISS.from_documents(
#     documents=chunks,
#     embedding=embeddings
# )
# Persist to disk (folder)
# vectorstore.save_local("faiss_index")

# Load vectore store from disk
vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)

# Create a retriever
retriever = vectorstore.as_retriever(k=3)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Chain building
template = """
You are a helpful assistant that can answer question about the following context:

CONTEXT:
{context}

QUESTION:
{question}

Using only the information in the context, answer the question. If you don't know the answer, say so.

No commentary, return the JSON in this format:
{{
    "answer": "...",
    "source_text": "Exact text that contains the answer"
}}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

parser = JsonOutputParser()

rag_chain = (
    {                         # Input mapping
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | parser
)

print(rag_chain.invoke("How many hours do cats sleep?"))
print(rag_chain.invoke("How many hours do dogs sleep?"))