"""
Streamlit web app for Kisan Call Centre Query Assistant.
Combines semantic search and LLM-based question answering.
"""

import os
import sys
import streamlit as st
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.semantic_search import SemanticSearch
from models.granite_llm import GraniteLLMClient


# Page configuration
st.set_page_config(
    page_title="Kisan Call Centre Query Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2d6a4f;
        margin-bottom: 2rem;
    }
    .subtitle {
        text-align: center;
        color: #52b788;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .answer-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .offline-answer {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    .online-answer {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .reference-box {
        padding: 1rem;
        background-color: #f5f5f5;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ff9800;
    }
    .stats {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0 0.5rem;
    }
    .offline-badge {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    .online-badge {
        background-color: #bbdefb;
        color: #1565c0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_search_engine():
    """
    Load and cache FAISS semantic search engine.
    Uses lazy loading for model and index.
    
    Returns:
        SemanticSearch instance
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        index_path = os.path.join(project_root, 'vector_store', 'faiss.index')
        meta_path = os.path.join(project_root, 'vector_store', 'meta.pkl')
        
        search_engine = SemanticSearch()
        search_engine.load_index(index_path, meta_path)
        
        return search_engine
    except Exception as e:
        st.error(f"❌ Failed to load search engine: {str(e)}")
        st.info("Make sure to run: `python services/semantic_search.py`")
        st.stop()


@st.cache_resource
def load_llm_client():
    """
    Load and cache Granite LLM client.
    
    Returns:
        GraniteLLMClient instance or None if credentials missing
    """
    try:
        client = GraniteLLMClient()
        health = client.health_check()
        
        if health['status'] != 'healthy':
            st.warning(f"⚠️ LLM client not properly configured: {health.get('error', 'Unknown error')}")
            return None
        
        return client
    except ValueError as e:
        st.warning(f"⚠️ LLM credentials not configured: {str(e)}")
        return None
    except Exception as e:
        st.warning(f"⚠️ Failed to load LLM client: {str(e)}")
        return None


def format_reference(result, index):
    """
    Format a single search result as a reference.
    
    Args:
        result: Search result dictionary with 'question', 'answer', 'score' keys
        index: Result index number
        
    Returns:
        Formatted HTML string
    """
    score = result.get('score', 0)
    return f"""
    <div class="reference-box">
        <strong>📌 Reference {index}:</strong><br>
        <strong>Q:</strong> {result.get('question', 'N/A')}<br>
        <strong>A:</strong> {result.get('answer', 'N/A')}<br>
        <small>Relevance Score: {score:.1%}</small>
    </div>
    """


def display_offline_answer(answers_result):
    """
    Display offline answer section with formatted answers.
    
    Args:
        answers_result: Dictionary from get_answers() with offline_answers and search_results
    """
    st.markdown("### 📚 Retrieved Answer (Offline – KCC Data)")
    
    # Handle empty results
    if not answers_result or not answers_result.get('offline_answers'):
        st.info("ℹ️ No matching answers found in the knowledge base.")
        return
    
    # Get search results for detailed view
    search_results = answers_result.get('search_results', [])
    
    if not search_results:
        st.info("ℹ️ No matching answers found in the knowledge base.")
        return
    
    # Main answer (top result)
    top_result = search_results[0]
    
    answer_html = f"""
    <div class="answer-box offline-answer">
        <strong>💡 Best Match Answer:</strong><br><br>
        {top_result['answer']}<br><br>
        <small>Related Question: {top_result['question']}</small><br>
        <small>Relevance Score: {top_result['score']:.1%}</small>
    </div>
    """
    st.markdown(answer_html, unsafe_allow_html=True)
    
    # Show supporting references if multiple results
    if len(search_results) > 1:
        with st.expander(f"📖 View {len(search_results) - 1} Supporting References"):
            for i, result in enumerate(search_results[1:], 2):
                st.markdown(format_reference(result, i), unsafe_allow_html=True)


def display_online_answer(llm_response):
    """
    Display online LLM answer section.
    
    Args:
        llm_response: Response dictionary from Granite LLM (or None)
    """
    st.markdown("### 🤖 LLM Answer (Online – IBM Granite)")
    
    # Handle case where LLM wasn't called
    if llm_response is None:
        st.info("ℹ️ LLM enhancement not enabled. Enable in sidebar settings.")
        return
    
    # Handle successful response
    if llm_response.get('success'):
        answer_html = f"""
        <div class="answer-box online-answer">
            <strong>✨ AI-Enhanced Answer:</strong><br><br>
            {llm_response['answer']}<br><br>
            <small>Model: {llm_response.get('model', 'Unknown')} | Generated: {llm_response.get('timestamp', 'N/A')}</small>
        </div>
        """
        st.markdown(answer_html, unsafe_allow_html=True)
    else:
        # Display error in a non-blocking way
        error_msg = llm_response.get('error', 'Unknown error occurred')
        st.warning(f"⚠️ LLM enhancement failed: {error_msg}")
        st.info("💡 Tip: The offline answer above is still available. Check your API credentials if this persists.")


def main():
    """
    Main Streamlit app function.
    
    Architecture:
    - Backend Logic: Handled by SemanticSearch.get_answers() and GraniteLLMClient
      * get_answers() encapsulates FAISS search and result formatting
      * Returns structured data (offline_answers, llm_context, search_results)
      * LLM client handles token management and API communication
    
    - UI Layer: Streamlit components receive formatted data from backend
      * display_offline_answer() - renders offline results
      * display_online_answer() - renders LLM responses
      * Statistics and metrics - shows query performance
    
    - Data Flow:
      User Input → get_answers() → Formatted Results → Display
                                 → LLM Context → generate_answer() → LLM Response → Display
    """
    
    # Header
    st.markdown(
        '<h1 class="main-header">🌾 Kisan Call Centre Query Assistant</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Get instant answers to agricultural questions</p>',
        unsafe_allow_html=True
    )
    
    # Mode status badges (placeholder - will be updated after loading components)
    st.markdown('<div id="mode-badges"></div>', unsafe_allow_html=True)
    mode_badge_placeholder = st.empty()
    
    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'llm_response' not in st.session_state:
        st.session_state.llm_response = None
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None
    if 'answers_result' not in st.session_state:
        st.session_state.answers_result = None
    
    # Load components
    search_engine = load_search_engine()
    llm_client = load_llm_client()
    
    # Display mode status badges
    badges_html = '<div style="text-align: center; margin: 1.5rem 0;">'
    badges_html += '<span class="mode-badge offline-badge">✓ Offline Mode Active</span>'
    if llm_client is not None:
        badges_html += '<span class="mode-badge online-badge">✓ Online Mode Available</span>'
    badges_html += '</div>'
    mode_badge_placeholder.markdown(badges_html, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        st.markdown("**Search Configuration**")
        num_results = st.slider(
            "Number of reference results:",
            min_value=1,
            max_value=10,
            value=5,
            help="How many similar Q&A pairs to retrieve"
        )
        
        st.markdown("**LLM Configuration**")
        use_llm = st.checkbox(
            "Use AI-powered LLM answer",
            value=True,
            help="Enable IBM Granite LLM for generating enhanced answers"
        )
        
        # Disable LLM if not available
        if use_llm and llm_client is None:
            st.warning("⚠️ LLM credentials not configured. LLM feature disabled.")
            use_llm = False
        
        llm_temperature = 0.7
        llm_max_tokens = 256
        
        if use_llm:
            llm_temperature = st.slider(
                "LLM Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher = more creative, Lower = more deterministic"
            )
            llm_max_tokens = st.slider(
                "Max tokens in LLM response:",
                min_value=50,
                max_value=500,
                value=256,
                step=50,
                help="Maximum length of generated response"
            )
        
        st.markdown("---")
        st.markdown("**About**")
        st.info(
            "This assistant retrieves relevant answers from the Kisan Call Centre "
            "knowledge base and optionally uses AI to generate enhanced responses."
        )
    
    # Main query input
    query = st.text_input(
        "🌱 Enter your agricultural query:",
        placeholder="e.g., How to prevent crop disease? What fertilizer should I use?",
        help="Ask any agricultural question and get instant answers"
    )
    
    # Search button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # ========== INPUT VALIDATION ==========
    if search_button:
        # Validate input
        if not query or not query.strip():
            st.warning("⚠️ Please enter a question to search.")
            return
        
        # Clear previous results when new search starts
        st.session_state.answers_result = None
        st.session_state.llm_response = None
        st.session_state.last_query = query
    
    # ========== PROCESS QUERY ==========
    if st.session_state.last_query is not None:
        query_to_search = st.session_state.last_query
        
        # ===== STEP 1: OFFLINE SEARCH (Always runs) =====
        # Uses get_answers() for clean separation of backend logic
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                st.session_state.answers_result = search_engine.get_answers(
                    query_to_search,
                    top_k=num_results
                )
            except Exception as e:
                st.error(f"❌ Search error: {str(e)}")
                st.session_state.answers_result = {
                    'offline_answers': [],
                    'llm_context': '',
                    'search_results': [],
                    'total_results': 0
                }
    
        # ===== STEP 2: DISPLAY OFFLINE RESULTS (Always shown) =====
        st.markdown("---")
        display_offline_answer(st.session_state.answers_result)
        
        # ===== STEP 3: OPTIONAL LLM ENHANCEMENT =====
        # Uses context prepared by get_answers() for clean LLM integration
        if use_llm and st.session_state.answers_result.get('search_results'):
            with st.spinner("🤖 Generating AI response..."):
                try:
                    # Pass search results as context to LLM
                    context_answers = st.session_state.answers_result.get('search_results', [])
                    
                    st.session_state.llm_response = llm_client.generate_answer(
                        user_query=query_to_search,
                        context_answers=context_answers,
                        max_tokens=llm_max_tokens,
                        temperature=llm_temperature
                    )
                except Exception as e:
                    # LLM failure doesn't crash the app
                    st.session_state.llm_response = {
                        'success': False,
                        'error': f"Connection or API error: {str(e)}"
                    }
        
        # ===== STEP 4: DISPLAY ONLINE RESULTS (If enabled) =====
        if use_llm:
            st.markdown("---")
            # Update badge if LLM response is successful
            if st.session_state.llm_response and st.session_state.llm_response.get('success'):
                badges_html = '<div style="text-align: center; margin: 0.5rem 0;">'
                badges_html += '<span class="mode-badge offline-badge">✓ Offline Mode Active</span>'
                badges_html += '<span class="mode-badge online-badge">✓ Online Mode Active</span>'
                badges_html += '</div>'
                mode_badge_placeholder.markdown(badges_html, unsafe_allow_html=True)
            display_online_answer(st.session_state.llm_response)
        
        # ===== STEP 5: DISPLAY STATISTICS =====
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📊 Offline Results",
                st.session_state.answers_result.get('total_results', 0),
                "from KCC data"
            )
        
        with col2:
            if use_llm and st.session_state.llm_response:
                status = "✓ Generated" if st.session_state.llm_response.get('success') else "✗ Failed"
                st.metric("🤖 LLM Status", status)
            else:
                st.metric("🤖 LLM Status", "Disabled")
        
        with col3:
            st.metric("⏱️ Query Time", datetime.now().strftime('%H:%M:%S'))
    
    # ========== DISCLAIMER ==========
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.85rem; margin-top: 2rem; margin-bottom: 1rem;">
        <strong>⚠️ Disclaimer:</strong><br>
        This assistant provides advisory information only. 
        Farmers should consult local agricultural officers before application.
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
