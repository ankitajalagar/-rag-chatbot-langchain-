"""Streamlit interface for the PDF RAG chatbot."""

import hashlib
import os
import tempfile

import streamlit as st
from pypdf.errors import PdfReadError
from streamlit.errors import StreamlitSecretNotFoundError

from chatbot import build_qa_chain, close_vectorstore
from ingest import build_embeddings, load_pdf, split_documents, store_in_chroma


SECRET_SETTINGS = (
    "LLM_PROVIDER",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
)


def apply_streamlit_secrets():
    try:
        secrets = dict(st.secrets)
    except StreamlitSecretNotFoundError:
        return

    for name in SECRET_SETTINGS:
        if name in secrets and name not in os.environ:
            os.environ[name] = str(secrets[name])


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return build_embeddings()


def clear_processed_document():
    workspace = st.session_state.pop("workspace", None)
    vectorstore = st.session_state.pop("vectorstore", None)
    for key in (
        "qa_chain",
        "document_digest",
        "document_name",
        "vectorstore_path",
        "chat_history",
    ):
        st.session_state.pop(key, None)
    if vectorstore is not None:
        close_vectorstore(vectorstore)
    if workspace is not None:
        workspace.cleanup()


def process_upload(uploaded_file):
    workspace = tempfile.TemporaryDirectory(prefix="rag_pdf_")
    try:
        pdf_path = os.path.join(workspace.name, "uploaded.pdf")
        vectorstore_path = os.path.join(workspace.name, "chroma_db")
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(uploaded_file.getvalue())

        pages = load_pdf(pdf_path)
        chunks = split_documents(pages)
        if not chunks:
            raise ValueError("No text could be extracted from this PDF.")

        vectorstore = store_in_chroma(
            chunks, get_embeddings(), persist_dir=vectorstore_path
        )
    except (OSError, PdfReadError, RuntimeError, ValueError) as error:
        workspace.cleanup()
        st.error(f"Could not process this PDF: {error}")
        return

    old_workspace = st.session_state.get("workspace")
    old_vectorstore = st.session_state.get("vectorstore")
    st.session_state["workspace"] = workspace
    st.session_state["vectorstore"] = vectorstore
    st.session_state["document_digest"] = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()
    st.session_state["document_name"] = uploaded_file.name
    st.session_state["vectorstore_path"] = vectorstore_path
    st.session_state["qa_chain"] = None
    st.session_state["chat_history"] = []
    if old_vectorstore is not None:
        close_vectorstore(old_vectorstore)
    if old_workspace is not None:
        old_workspace.cleanup()

    st.success(f"Processed **{uploaded_file.name}** into {len(chunks)} searchable chunks.")


def render_sources(source_documents):
    if not source_documents:
        return

    with st.expander(f"Retrieved document passages ({len(source_documents)})"):
        for index, document in enumerate(source_documents, start=1):
            page = document.metadata.get("page")
            page_label = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
            st.markdown(f"**Passage {index} · {page_label}**")
            st.caption(document.page_content)


def main():
    st.set_page_config(
        page_title="AI-Powered PDF Question Answering System",
        page_icon="📚",
        layout="wide",
    )
    apply_streamlit_secrets()

    st.markdown(
        """
        <style>
        .block-container {max-width: 1050px; padding-top: 2rem; padding-bottom: 3rem;}
        .hero {padding: 1.6rem 1.8rem; border-radius: 16px;
               background: linear-gradient(120deg, #10243a, #1d4b68); color: white;
               margin-bottom: 1.5rem;}
        .hero p {color: #d7e8f3; margin-bottom: 0;}
        </style>
        <div class="hero">
          <h1>AI-Powered PDF Question Answering System</h1>
          <p>Upload a PDF, retrieve relevant passages, and ask questions grounded in its content.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Your document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file is not None:
            st.caption(f"Selected: {uploaded_file.name}")
            if st.button("Process PDF", type="primary", use_container_width=True):
                with st.spinner("Reading, chunking, and indexing your PDF..."):
                    process_upload(uploaded_file)
        if st.button("Clear document", use_container_width=True):
            clear_processed_document()
            st.rerun()

    if uploaded_file is None:
        st.info("Upload a PDF in the sidebar to get started.")
        return

    current_digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    processed = st.session_state.get("document_digest") == current_digest
    if not processed:
        st.info("Process the selected PDF before asking a question.")
        return

    st.success(f"Ready to answer questions about **{st.session_state['document_name']}**.")
    for entry in st.session_state.get("chat_history", []):
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            render_sources(entry["sources"])

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question",
            placeholder="What are the main points in this document?",
        )
        submitted = st.form_submit_button("Get answer", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question before submitting.")
            return

        try:
            with st.spinner("Retrieving relevant passages and generating an answer..."):
                qa_chain = st.session_state.get("qa_chain")
                if qa_chain is None:
                    qa_chain = build_qa_chain(
                        st.session_state["vectorstore_path"], get_embeddings()
                    )
                    st.session_state["qa_chain"] = qa_chain
                result = qa_chain.invoke({"query": question.strip()})
        except ValueError as error:
            st.error(f"LLM configuration error: {error}")
            return
        except FileNotFoundError as error:
            st.error(f"Document index error: {error}")
            return

        st.session_state["chat_history"].append(
            {
                "question": question.strip(),
                "answer": result["result"],
                "sources": result.get("source_documents", []),
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()
