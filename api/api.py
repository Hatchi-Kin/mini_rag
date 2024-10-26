import os
import warnings

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain import hub
from langchain_ollama import OllamaLLM
from contextlib import asynccontextmanager


class Query(BaseModel):
    query: str = Field(..., example="What is the url of the swagger documentation of the MegaPi api?")

# Global variables to store documents and Milvus client
documents = []
milvus_client = None
vectorstore = None

def load_documents():
    global documents
    path_pdfs = "api/documents/V2"
    for file in os.listdir(path_pdfs):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(path_pdfs, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global milvus_client, vectorstore

    # Load documents only if the list is empty
    if not documents:
        load_documents()

    # Connect to Milvus and initiate model to generate embeddings
    MILVUS_URL = "api/milvuslite/megapidoc.db"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    _all_splits = text_splitter.split_documents(documents)

    # Vector store creation
    vectorstore = Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        connection_args={
            "uri": MILVUS_URL,
        },
        drop_old=False,
    )

    yield

app = FastAPI(lifespan=lifespan)


@app.post("/query")
async def process_query(query: Query):
    global vectorstore

    llm = OllamaLLM(
        model="llama3.2",
        callbacks=[StreamingStdOutCallbackHandler()],
        stop=["\n"]
    )
    prompt = hub.pull("rlm/rag-prompt")
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": prompt}
    )
    result = qa_chain.invoke({"query": query.query})

    return {"result": result["result"]}


if __name__ == "__main__":
    import uvicorn
    warnings.filterwarnings("ignore", category=UserWarning, message="API key must be provided when using hosted LangSmith API")
    uvicorn.run("api:app", host="0.0.0.0", port=7999, reload=True)
