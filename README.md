# AI Finance Assistant

This repository contains a modular Multi-Agent Financial Advisor system.

## Project Structure

```
ai_finance_assistant/
├── src/
│   ├── agents/          # Agent definitions (FQAA, MAA, NSA, TEA, GPA)
│   ├── core/            # Core configurations (LLM setup)
│   ├── data/            # Data sources (URLs)
│   ├── rag/             # RAG engine and retriever logic
│   ├── web_app/         # Streamlit UI
│   ├── utils/           # Shared tools and helpers
│   └── workflow/        # LangGraph workflow and state
├── tests/               # Unit tests
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup & Running

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Ensure you have a `.env` file with necessary API keys:
    - `OPENAI_API_KEY`
    - `TAVILY_API_KEY`
    - `NEWSDATA_API_KEY`

3.  **Running the Application**:
    Run the Streamlit app from the root directory:
    ```bash
    streamlit run src/web_app/app.py
    ```

## Agents
- **FQAA (Finance Q&A Agent)**: Handles general financial queries.
- **MAA (Market Analysis Agent)**: Provides real-time market data and analysis.
- **NSA (News Synthesizer Agent)**: Summarizes financial news and sentiment.
- **TEA (Tax Education Agent)**: Explains tax concepts and performs calculations.
- **GPA (Goal Planning Agent)**: Assists with financial goal setting and surplus allocation.

## Workflow
The system uses LangGraph to route queries to the appropriate specialized agent based on the user's input.
