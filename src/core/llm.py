import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# set LLM 
llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.2,
    max_tokens=800 
)
