import streamlit as st
from main import financial_planner
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import os
from dotenv import load_dotenv

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

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask me about markets, tax, or financial planning..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Processing
    with st.chat_message("assistant"):
        with st.spinner("Processing with financial agents..."):
            try:
                # Prepare initial state for LangGraph
                # We only pass the NEW HumanMessage; LangGraph checkpointer will handle history
                initial_state = {
                    "messages": [HumanMessage(content=prompt)],
                }
                
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # Invoke the graph
                result = financial_planner.invoke(initial_state, config)
                
                # Extract the latest response
                agent_response = result["messages"][-1]
                response_content = agent_response if isinstance(agent_response, str) else agent_response.content
                
                st.markdown(response_content)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                
            except Exception as e:
                st.error(f"Error in agent workflow: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})

# Footer
st.divider()
st.markdown("<p style='text-align: center; color: #666;'>Built with ❤️ using LangGraph and Streamlit</p>", unsafe_allow_html=True)
