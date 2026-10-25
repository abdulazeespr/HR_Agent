import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from hr_assistant import config
from hr_assistant.pipeline import HRAssistantPipeline, build_vector_store_documents
from hr_assistant.vector_store import vector_store_exists, load_vector_store
from hr_assistant.embeddings import get_embedding_model

# Page Configuration
st.set_page_config(
    page_title="Acme Corp | HR Policy AI Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism, Modern Corporate Aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Gradient & Badges */
    .main-title {
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .status-online {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .status-warning {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    
    /* Quick Prompt Cards */
    .prompt-pill {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }
    .prompt-pill:hover {
        background: #EFF6FF;
        border-color: #3B82F6;
        transform: translateY(-1px);
    }
    
    /* Evidence Box */
    .evidence-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.92rem;
        color: #334155;
    }
    
    /* Sidebar styling */
    .sidebar-section-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "quick_prompt_input" not in st.session_state:
    st.session_state.quick_prompt_input = None


# Helper: Initialize or Get Pipeline
@st.cache_resource(show_spinner="Connecting to AI Models & Knowledge Base...")
def get_cached_pipeline(groq_key: str, jina_key: str, model_name: str, temperature: float):
    return HRAssistantPipeline(
        groq_api_key=groq_key,
        jina_api_key=jina_key,
        llm_model=model_name,
        temperature=temperature
    )


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/parse-resumes.png", width=64)
    st.markdown("### **Acme Corp HR Portal**")
    st.caption("AI-Powered Employee Policy Copilot")
    st.divider()

    st.markdown('<div class="sidebar-section-header">API Credentials</div>', unsafe_allow_html=True)
    groq_api_key = st.text_input(
        "Groq API Key",
        value=config.GROQ_API_KEY,
        type="password",
        help="Used for high-speed LLM reasoning and agent generation."
    )
    jina_api_key = st.text_input(
        "Jina API Key",
        value=config.JINA_API_KEY,
        type="password",
        help="Used for dense semantic embeddings."
    )

    st.markdown('<div class="sidebar-section-header">Model Parameters</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "LLM Engine",
        options=config.AVAILABLE_MODELS,
        index=0,
        help="Select model served via Groq Cloud."
    )
    temperature = st.slider(
        "Creativity / Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Lower values guarantee strict factual consistency with HR policy."
    )

    st.markdown('<div class="sidebar-section-header">Knowledge Base Management</div>', unsafe_allow_html=True)
    
    # Vector store status
    db_exists = vector_store_exists()
    if db_exists:
        st.markdown(
            '<div class="status-badge status-online">● FAISS Vector Store: Ready</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-badge status-warning">▲ Vector Store: Not Built</div>',
            unsafe_allow_html=True
        )

    # Rebuild Index Button
    if st.button("🔄 Rebuild Vector Index", use_container_width=True):
        with st.spinner("Re-indexing HR Policy Document into FAISS..."):
            try:
                build_vector_store_documents(
                    force_rebuild=True,
                    jina_api_key=jina_api_key or config.JINA_API_KEY
                )
                st.cache_resource.clear()
                st.success("Knowledge base index updated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Rebuild failed: {e}")

    # Upload Custom Policy File
    with st.expander("📄 Upload Updated Policy File", expanded=False):
        uploaded_file = st.file_uploader("Upload .txt or .md policy document", type=["txt", "md"])
        if uploaded_file is not None:
            if st.button("Apply & Re-Index Uploaded File", use_container_width=True):
                save_dest = os.path.join(config.BASE_DIR, "data", "hr_policy.txt")
                with open(save_dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner("Indexing new document..."):
                    build_vector_store_documents(file_path=save_dest, force_rebuild=True, jina_api_key=jina_key)
                    st.cache_resource.clear()
                    st.success("Custom policy file applied and indexed!")
                    st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.quick_prompt_input = None
        st.rerun()


# Main Application Window
col1, col2 = st.columns([0.75, 0.25])
with col1:
    st.markdown('<div class="main-title">HR Policy Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Ask any question about Acme Corp\'s employee benefits, leave entitlements, remote work, or workplace guidelines.</div>',
        unsafe_allow_html=True
    )

with col2:
    # System Status Indicator Pills
    groq_ok = bool(groq_api_key and not groq_api_key.startswith("jina_"))
    jina_ok = bool(jina_api_key)
    
    if groq_ok and jina_ok and db_exists:
        st.markdown(
            '<div style="text-align: right;"><span class="status-badge status-online">● Agent Active</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="text-align: right;"><span class="status-badge status-warning">▲ Config Required</span></div>',
            unsafe_allow_html=True
        )

# Quick Starter Questions (Pill buttons)
st.markdown("**Quick Topics:**")
q_cols = st.columns(4)
sample_questions = [
    ("🌴 Annual & Sick Leave", "How many days of annual leave and sick leave am I entitled to each year?"),
    ("🏠 Remote Work Policy", "What are the rules and core working hours for working from home?"),
    ("⏳ Probation Terms", "What is the probation period length and can I take leave during probation?"),
    ("🚪 Notice & Resignation", "What is the notice period for resigning during and after probation?")
]

for col, (label, question) in zip(q_cols, sample_questions):
    with col:
        if st.button(label, key=f"btn_{label}", use_container_width=True):
            st.session_state.quick_prompt_input = question

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display source citations if available
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 Retrieved HR Policy Evidence & Excerpts", expanded=False):
                for src in message["sources"]:
                    st.markdown(f'<div class="evidence-box">{src}</div>', unsafe_allow_html=True)
                    
        # Display agent tool calls if available
        if message["role"] == "assistant" and "tool_calls" in message and message["tool_calls"]:
            with st.expander("⚙️ Agent Tool & Reasoning Details", expanded=False):
                for tc in message["tool_calls"]:
                    st.code(f"Tool: {tc.get('name')}\nQuery: {tc.get('args')}", language="yaml")

# User Input Box
chat_input = st.chat_input("Ask a question about HR policies (e.g. 'Can I carry forward unused annual leave?')...")

# Determine active prompt (from quick button or chat input)
active_query = chat_input or st.session_state.quick_prompt_input
if st.session_state.quick_prompt_input and not chat_input:
    # Reset quick prompt trigger after picking it up
    st.session_state.quick_prompt_input = None

if active_query:
    # Add User Message to History
    st.session_state.chat_history.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Consulting HR Policy Handbook..."):
            try:
                # Validate Keys
                if not groq_api_key or groq_api_key.startswith("jina_"):
                    st.error("Please enter a valid Groq API Key in the left sidebar to generate responses.")
                    st.stop()
                if not jina_api_key:
                    st.error("Please enter a valid Jina API Key in the left sidebar for semantic search.")
                    st.stop()

                # Get Pipeline Instance
                pipeline = get_cached_pipeline(
                    groq_key=groq_api_key,
                    jina_key=jina_api_key,
                    model_name=selected_model,
                    temperature=temperature
                )

                # Format conversation history for agent
                history_for_agent = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[:-1]
                ]

                # Run query
                response_data = pipeline.ask(active_query, chat_history=history_for_agent)
                answer_text = response_data.get("answer", "I could not find an answer.")
                sources = response_data.get("sources", [])
                tool_calls = response_data.get("tool_calls", [])

                # Render Answer
                st.markdown(answer_text)

                # Render Evidence & Tool Calls
                if sources:
                    with st.expander("📚 Retrieved HR Policy Evidence & Excerpts", expanded=False):
                        for src in sources:
                            st.markdown(f'<div class="evidence-box">{src}</div>', unsafe_allow_html=True)

                if tool_calls:
                    with st.expander("⚙️ Agent Tool & Reasoning Details", expanded=False):
                        for tc in tool_calls:
                            st.code(f"Tool: {tc.get('name')}\nQuery: {tc.get('args')}", language="yaml")

                # Record in Session State
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                    "tool_calls": tool_calls
                })

            except Exception as e:
                st.error(f"Error processing your query: {e}")
