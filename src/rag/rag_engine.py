import os
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.tools.retriever import create_retriever_tool
from src.data.urls import urls

PERSIST_DIR = "./chroma_db"

def build_vectorstore():
    """Build or load the persistent vectorstore for RAG."""
    # Deduplicate URLs to reduce fetch/embedding cost
    unique_urls = list(dict.fromkeys(urls))

    emb = OpenAIEmbeddings(model="text-embedding-3-small")  # uses OPENAI_API_KEY from env

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        # Load existing DB if it exists
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
        if vectorstore._collection.count() == 0:
            print(f"{PERSIST_DIR} exists but has no rows. Deleting...")
            shutil.rmtree(PERSIST_DIR)
        else:
            return vectorstore

    # 1) Fetch and preprocess data
    docs = [WebBaseLoader(url).load() for url in unique_urls]

    # 2) Chunk (larger chunks to reduce embedding count)
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=150
    )
    chunks = text_splitter.split_documents(docs_list)

    # 3) Embed & index
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=PERSIST_DIR
    )
    return vectorstore


# Build or load vectorstore on import (lightweight when DB exists)
vectorstore = build_vectorstore()

# Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="retriever_search_results_json",
    description="Use this tool to fetch relevant financial information from the internal knowledge base. Input should be a natural language query. Output will be a JSON string with relevant information.",
    document_separator='\n\n'
    )
