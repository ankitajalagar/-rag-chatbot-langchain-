"""
chatbot.py
----------
Step 5-8 of the assignment:
  5. Accept questions from the user.
  6. Retrieve the most relevant document chunks (from ChromaDB).
  7. Use an LLM to generate answers based on the retrieved context.
  8. Return accurate and relevant responses.

Run:
    python chatbot.py

Requires ingest.py to have been run first (creates the chroma_db/ folder).
"""

import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from llm_factory import get_llm

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the
context below, which was retrieved from a PDF document. If the answer is not
contained in the context, say you don't know instead of making something up.

Context:
{context}

Question: {question}

Answer clearly and concisely, citing the relevant part of the context where useful.
"""


def load_vectorstore():
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"No ChromaDB found at '{CHROMA_DIR}/'. Run `python ingest.py --pdf <path>` first."
        )
    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{EMBED_MODEL_NAME}")
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="rag_chatbot_collection",
    )
    return vectordb


def build_qa_chain():
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})  # Step 6: top-4 relevant chunks

    llm = get_llm()  # Step 7: chosen LLM (Groq/OpenAI/Gemini/Ollama)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def main():
    print("Building RAG chatbot (loading vector store + LLM)...")
    qa_chain = build_qa_chain()
    print("Ready! Ask questions about your PDF. Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()  # Step 5: accept a question from the user
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue

        result = qa_chain.invoke({"query": question})  # Steps 6-7: retrieve + generate
        answer = result["result"]
        sources = result.get("source_documents", [])

        print(f"\nBot: {answer}\n")  # Step 8: return the response
        if sources:
            pages = sorted({doc.metadata.get("page", "?") for doc in sources})
            print(f"(Sources: page(s) {pages} of the PDF)\n")


if __name__ == "__main__":
    main()
