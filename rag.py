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


# Environment variables loaded via load_dotenv()

## RAG Implementation

# Knowledge base URLs
urls = [
    "https://zerodha.com/varsity/module/introduction-to-stock-markets/",
    "https://zerodha.com/varsity/chapter/the-need-to-invest/",
    "https://zerodha.com/varsity/chapter/regulators/",
    "https://zerodha.com/varsity/chapter/financial-intermediaries/",
    "https://zerodha.com/varsity/chapter/the-ipo-markets-part-1/",
    "https://zerodha.com/varsity/chapter/the-ipo-markets-part-2/",
    "https://zerodha.com/varsity/chapter/the-stock-markets/",
    "https://zerodha.com/varsity/chapter/the-stock-markets-index/",
    "https://zerodha.com/varsity/chapter/commonly-used-jargons/",
    "https://zerodha.com/varsity/chapter/the-trading-terminal/",
    "https://zerodha.com/varsity/chapter/clearing-and-settlement-process/",
    "https://zerodha.com/varsity/chapter/five-corporate-actions-and-its-impact-on-stock-prices/",
    "https://zerodha.com/varsity/chapter/key-events-and-their-impact-on-markets/",
    "https://zerodha.com/varsity/chapter/getting-started/",
    "https://zerodha.com/varsity/chapter/supplementary-note-ipo-ofs-fpo/",
    "https://zerodha.com/varsity/chapter/supplementary-note-the-20-market-depth/",
    "https://zerodha.com/varsity/module/technical-analysis/",
    "https://zerodha.com/varsity/chapter/background/",
    "https://zerodha.com/varsity/chapter/introducing-technical-analysis/",
    "https://zerodha.com/varsity/chapter/chart-types/",
    "https://zerodha.com/varsity/chapter/getting-started-candlesticks/",
    "https://zerodha.com/varsity/chapter/single-candlestick-patterns-part-1/",
    "https://zerodha.com/varsity/chapter/single-candlestick-patterns-part-2/",
    "https://zerodha.com/varsity/chapter/single-candlestick-patterns-part-3/",
    "https://zerodha.com/varsity/chapter/multiple-candlestick-patterns-part-1/",
    "https://zerodha.com/varsity/chapter/multiple-candlestick-patterns-part-2/",
    "https://zerodha.com/varsity/chapter/multiple-candlestick-patterns-part-3/",
    "https://zerodha.com/varsity/chapter/support-resistance/",
    "https://zerodha.com/varsity/chapter/volumes/",
    "https://zerodha.com/varsity/chapter/moving-averages/",
    "https://zerodha.com/varsity/chapter/indicators-part-1/",
    "https://zerodha.com/varsity/chapter/indicators-part-2/",
    "https://zerodha.com/varsity/chapter/fibonacci-retracements/",
    "https://zerodha.com/varsity/chapter/dow-theory-part-1/",
    "https://zerodha.com/varsity/chapter/dow-theory-part-2/",
    "https://zerodha.com/varsity/chapter/finale-helping-get-started/",
    "https://zerodha.com/varsity/chapter/supplementary-notes-1/",
    "https://zerodha.com/varsity/chapter/interesting-features-on-tradingview/",
    "https://zerodha.com/varsity/chapter/the-central-pivot-range/",
    "https://zerodha.com/varsity/module/fundamental-analysis/",
    "https://zerodha.com/varsity/chapter/introduction-fundamental-analysis/",
    "https://zerodha.com/varsity/chapter/mindset-investor/",
    "https://zerodha.com/varsity/chapter/read-annual-report-company/",
    "https://zerodha.com/varsity/chapter/understanding-pl-statement-part1/",
    "https://zerodha.com/varsity/chapter/understanding-pl-statement-part2/",
    "https://zerodha.com/varsity/chapter/understanding-balance-sheet-statement-part-1/",
    "https://zerodha.com/varsity/chapter/understanding-balance-sheet-statement-part-2/",
    "https://zerodha.com/varsity/chapter/cash-flow-statement/",
    "https://zerodha.com/varsity/chapter/financial-ratio-analysis/",
    "https://zerodha.com/varsity/chapter/financial-ratios-part-2/",
    "https://zerodha.com/varsity/chapter/financial-ratios-part-3/",
    "https://zerodha.com/varsity/chapter/investment-due-diligence/",
    "https://zerodha.com/varsity/chapter/equity-research-part-1/",
    "https://zerodha.com/varsity/chapter/dcf-primer/",
    "https://zerodha.com/varsity/chapter/equity-research-part-2/",
    "https://zerodha.com/varsity/chapter/finale/",
    "https://zerodha.com/varsity/module/markets-and-taxation/",
    "https://zerodha.com/varsity/chapter/introduction-setting-the-context/",
    "https://zerodha.com/varsity/chapter/basics/",
    "https://zerodha.com/varsity/chapter/classifying-your-market-activity/",
    "https://zerodha.com/varsity/chapter/taxation-for-investors/",
    "https://zerodha.com/varsity/chapter/taxation-for-traders/",
    "https://zerodha.com/varsity/chapter/turnover-balance-sheet-and-pl/",
    "https://zerodha.com/varsity/chapter/itr-forms/",
    "https://zerodha.com/varsity/chapter/foreign-stocks-and-taxation/"
]

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

# Print statement to Verify vectorstore count
# print(vectorstore._collection.count())


retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="retriever_search_results_json",
    description="Use this tool to fetch relevant financial information from the internal knowledge base. Input should be a natural language query. Output will be a JSON string with relevant information.",
    document_separator='\n\n'
    )

# response = retriever_tool.invoke({"query": "What is the stock market"})
# print("response: ", response)
