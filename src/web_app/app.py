import streamlit as st
import uuid
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the financial planner graph
# Note: we are importing from src.workflow.graph
from src.workflow.graph import financial_planner

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Financial Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure upload directory exists
UPLOAD_DIR = "./upload"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Premium Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 10px;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    h1 {
        color: #00d4ff;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar
with st.sidebar:
    st.title("Settings & Tools")
    st.info(f"Thread ID: {st.session_state.thread_id}")
    
    st.divider()
    st.subheader("📁 Upload Documents")
    uploaded_file = st.file_uploader("Upload tax PDFs or financial docs", type=["pdf"])
    
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")
        st.info("The Tax Agent can now analyze this document. Mention 'uploaded file' in your query.")

    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Main UI
st.title("💰 Multi-Agent Financial Advisor")
st.caption("AI-Powered Financial Education, Planning, and Analysis")

# Utility functions
def render_chat_history():
    """Renders the common chat history in a scrollable container"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def process_agent_query(user_query):
    """Processes a query and updates the global session state"""
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # LangGraph processing
    try:
        # Prepare initial state - checkpointer handles history
        initial_state = {"messages": [HumanMessage(content=user_query)]}
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Invoke the graph
        result = financial_planner.invoke(initial_state, config)
        
        # Extract the latest response
        agent_response = result["messages"][-1]
        response_content = agent_response if isinstance(agent_response, str) else agent_response.content
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        return response_content
    except Exception as e:
        error_msg = f"Error in agent workflow: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        return error_msg

# Define Tabs
tab_chat, tab_markets, tab_tax, tab_goals, tab_news = st.tabs([
    "💬 Financial general query", 
    "📊 Markets", 
    "📑 Tax Hub", 
    "🎯 Goal Planner", 
    "📰 News Terminal"
])

# TAB 1: FINANCIAL GENERAL QUERY (CHAT)
with tab_chat:
    st.subheader("Conversational Advisor")
    render_chat_history()

# TAB 2: MARKETS
with tab_markets:
    st.subheader("Real-time Market Insights")
    render_chat_history()
    
    st.divider()
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_input = st.text_input("Ticker Symbol", placeholder="e.g. RELIANCE.NS, TSLA", key="market_ticker")
        with col2:
            st.write(" ") # Padding
            if st.button("Analyze", key="market_btn"):
                if ticker_input:
                    with st.spinner(f"Analyzing {ticker_input}..."):
                        process_agent_query(f"What is the current price and analysis for {ticker_input}?")
                        st.rerun()

# TAB 3: TAX HUB
with tab_tax:
    st.subheader("Tax Education & Document Analysis")
    render_chat_history()
    
    st.divider()
    if os.path.exists(UPLOAD_DIR):
        files = os.listdir(UPLOAD_DIR)
        if files:
            st.write("📁 **Files available for analysis:**")
            st.caption(", ".join(files))
            if st.button("Run Comprehensive Tax Analysis", key="tax_btn"):
                with st.spinner("Analyzing documents..."):
                    process_agent_query("Summarize my uploaded tax documents and calculate potential totals/regime impact.")
                    st.rerun()
        else:
            st.info("No documents uploaded. Please use the sidebar to upload PDFs.")

# TAB 4: GOAL PLANNER
with tab_goals:
    st.subheader("Personalized Financial Goals")
    render_chat_history()
    
    st.divider()
    with st.expander("📝 Define New Goal", expanded=False):
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            goal_type = st.selectbox("Goal Type", ["Retirement", "Child Education", "Major Purchase", "Custom"], key="goal_type")
            target_amt = st.number_input("Target Amount", value=1000000, key="goal_amt")
        with g_col2:
            years = st.number_input("Years to Achieve", value=10, min_value=1, key="goal_years")
            expected_ret = st.slider("Expected Return (%)", 5, 15, 10, key="goal_ret")
        
        if st.button("Generate Roadmap", key="goal_btn"):
            with st.spinner("Generating financial roadmap..."):
                query = f"I want to plan for {goal_type} in {years} years. Target is {target_amt}. Expected return is {expected_ret}%. Use calculator tools to provide a month-by-month investment plan."
                process_agent_query(query)
                st.rerun()

# TAB 5: NEWS TERMINAL
with tab_news:
    st.subheader("Financial News & Sentiment")
    render_chat_history()
    
    st.divider()
    col_n1, col_n2 = st.columns([3, 1])
    with col_n1:
        news_query = st.text_input("Topic or Company", placeholder="e.g. Fed Rates, HCLTECH News", key="news_input")
    with col_n2:
        st.write(" ") # Padding
        if st.button("Get News Report", key="news_btn"):
            if news_query:
                with st.spinner(f"Synthesizing news for {news_query}..."):
                    process_agent_query(f"Synthesize today's financial news for {news_query} and explain the market impact/sentiment.")
                    st.rerun()

# Global Chat Input (Available for all tabs at the bottom)
if prompt := st.chat_input("Ask a follow-up or any general question..."):
    with st.spinner("Thinking..."):
        process_agent_query(prompt)
        st.rerun()

# Footer
st.divider()
st.markdown("<p style='text-align: center; color: #666;'>Built with ❤️ using LangGraph and Streamlit</p>", unsafe_allow_html=True)
