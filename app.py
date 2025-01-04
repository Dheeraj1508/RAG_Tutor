import streamlit as st
from typing import List
import os
from unstructured.partition.pdf import partition_pdf
from rag_system import EmbedData, QdrantVDB, Retriever, RAG, process_chunks
import base64
# Initialize RAG system
def initialize_rag_system():
    """Initializes the retrieval-augmented generation system."""
    return "RAG System Initialized"

# Process uploaded documents
def process_uploaded_document(file):
    """Processes and indexes the uploaded document."""
    temp_path = f"temp_{file.name}"
    with open(temp_path, "wb") as f:
        f.write(file.read())

    

    batch_size = 32
    with st.spinner("Processing document..."):
        summaries = process_chunks(temp_path)
        embeddata = EmbedData(batch_size=batch_size)

        embeddata.embed(summaries)
        database = QdrantVDB(file.name)
        database.define_client()
        database.create_collection()
        database.ingest_data(embeddata)

    st.session_state["retriever"] = Retriever(database, embeddata)
    st.session_state["indexed_document"] = file.name
    st.session_state["temp_path"] = temp_path

from rag_system import get_rewriting_llm,build_query_rewriting_chain,rewrite_query
# Answer user questions
def answer_question(question: str) -> str:
    """Generates an answer for the given question using the RAG system."""
    retriever = st.session_state.get("retriever")
    if retriever:
        rag = RAG(retriever)
        llm = get_rewriting_llm()
        query_rewriter_chain = build_query_rewriting_chain(llm)
        question = rewrite_query(question, query_rewriter_chain)
        answer = rag.query(question)
        return answer
    else:
        return "No document indexed. Please upload a document first."

# Streamlit UI
st.set_page_config(
    page_title="RAG System",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# st.title("📖 AI TUTOR")
st.markdown(
    """
    <style>
        .main-title {
            margin-top: 6rem; /* Adjust this value to move the title further down */
            text-align: center; /* Optional: keep the title centered */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the title with a custom class
st.markdown('<h1 class="main-title">📖 AI TUTOR</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f9f9f9;
        }
        .block-container {
            padding-top: 2rem;
        }
        .main-header {
            color: #2C3E50;
            text-align: center;
            font-weight: bold;
        }
        .stButton > button {
            background-color: #2C3E50;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.subheader("Upload a Document and Start Asking Questions")
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a document (PDF, TXT, DOCX)", type=["txt", "pdf", "docx"])
 
# from streamlit_pdf_viewer import pdf_viewer

if uploaded_file is not None:
    if "indexed_document" not in st.session_state:
        process_uploaded_document(uploaded_file)
        st.write("You can now start asking questions based on the uploaded document!")



if "indexed_document" in st.session_state:
    # st.markdown(f"### Indexed Document: `{st.session_state['indexed_document']}`")
    

    st.text_input("Ask your question:", key="user_question")
    if st.button("Submit"):
        question = st.session_state.get("user_question", "").strip()
        if question:
            response = answer_question(question)
            st.write(f"**Answer:** {response['response']}")
            st.markdown("### Context Used:")
            # st.markdown(f"```{response['context']}```")
            # st.code(response['context'], language="text")
            st.write(response['context'])
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a document to start.")

st.markdown("---")
st.markdown(
    """
    **Tips:**
    - Upload a document to analyze.
    - Use the question box to ask specific questions about the document.
    """
)
