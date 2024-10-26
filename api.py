import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pymilvus import MilvusClient
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
    query: str = Field(..., example="What is the name of the function that lists all songs from a specific album in the music_library table?")

# Global variable to store documents
documents = []

def load_documents():
    global documents
    path_pdfs = "megapidoc/"
    for file in os.listdir(path_pdfs):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(path_pdfs, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load documents only if the list is empty
    if not documents:
        load_documents()
    yield
    # Clean up resources if needed (e.g., close database connections)


app = FastAPI(lifespan=lifespan)


@app.post("/query")
async def process_query(query: Query):
    # Connect to Milvus
    MILVUS_URL = "./rag101.db"
    client = MilvusClient(uri=MILVUS_URL)
    
    if not client.has_collection("LangChainCollection"):
        client.drop_collection("LangChainCollection")

    # Create embeddings
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

    # Run query
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
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)