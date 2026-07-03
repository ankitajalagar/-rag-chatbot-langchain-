"""
llm_factory.py
--------------
Returns a LangChain-compatible chat model based on the LLM_PROVIDER
environment variable. Supports: groq, openai, gemini, ollama.

This lets the chatbot satisfy the assignment's requirement of
"Any Large Language Model (LLM) such as Groq, OpenAI, Gemini, Ollama, etc."
without hardcoding a single provider.
"""

import os


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in your .env file.")
        return ChatGroq(
            api_key=api_key,
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.2,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in your .env file.")
        return ChatOpenAI(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in your .env file.")
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=0.2,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.2,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Choose from: groq, openai, gemini, ollama."
        )
