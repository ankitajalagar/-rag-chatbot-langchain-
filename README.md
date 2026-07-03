# Simple RAG Chatbot (LangChain + ChromaDB)

## Project Overview

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that answers
user questions based on the contents of a PDF document.

Pipeline:
1. Load a PDF document (`PyPDFLoader`).
2. Split it into meaningful, overlapping chunks (`RecursiveCharacterTextSplitter`).
3. Generate embeddings for each chunk using **Sentence Transformers**
   (`all-MiniLM-L6-v2`).
4. Store the embeddings in a persistent **ChromaDB** vector database.
5. Accept a question from the user.
6. Retrieve the most relevant chunks from ChromaDB via similarity search.
7. Pass the retrieved context + question to an **LLM** (Groq, OpenAI, Gemini,
   or Ollama — configurable) to generate an answer.
8. Return an accurate, context-grounded response, along with the source
   page numbers it was drawn from.

## Tech Stack

| Component        | Tool                                  |
|-------------------|----------------------------------------|
| Language          | Python 3.10+                          |
| Orchestration     | LangChain                             |
| Vector database   | ChromaDB (persisted locally)          |
| Embeddings        | Sentence Transformers `all-MiniLM-L6-v2` |
| LLM               | Groq / OpenAI / Gemini / Ollama (pick one) |
| PDF loading       | pypdf                                 |

## Project Structure

```
rag_chatbot/
├── data/                # Put your source PDF here
├── ingest.py            # Steps 1-4: load PDF -> chunk -> embed -> store in ChromaDB
├── chatbot.py           # Steps 5-8: ask -> retrieve -> LLM answer -> return
├── llm_factory.py        # Picks the LLM provider based on .env
├── requirements.txt
├── .env.example
└── README.md
```

(After you run `ingest.py`, a `chroma_db/` folder will also be created — this
is your persisted vector store.)

## Installation Steps

1. **Clone the repository** (or unzip the project folder) and move into it:
   ```bash
   git clone <your-repo-url>
   cd rag_chatbot
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > Note: `requirements.txt` includes packages for all four LLM providers.
   > If you only plan to use one (e.g. Groq), you can safely remove the
   > other providers' lines to speed up installation.

## Required Dependencies

- `langchain`, `langchain-community`, `langchain-text-splitters`
- `chromadb`
- `sentence-transformers`
- `pypdf`
- `python-dotenv`
- One of: `langchain-groq`, `langchain-openai`, `langchain-google-genai`, `langchain-ollama`

## Environment Variables

Copy the example file and fill in your own values:

```bash
cp .env.example .env
```

| Variable          | Description                                              |
|--------------------|------------------------------------------------------------|
| `LLM_PROVIDER`     | One of `groq`, `openai`, `gemini`, `ollama`                |
| `GROQ_API_KEY`     | Required if `LLM_PROVIDER=groq` — get one at https://console.groq.com |
| `OPENAI_API_KEY`   | Required if `LLM_PROVIDER=openai`                          |
| `GOOGLE_API_KEY`   | Required if `LLM_PROVIDER=gemini`                          |
| `OLLAMA_MODEL` / `OLLAMA_BASE_URL` | Used if `LLM_PROVIDER=ollama` (no API key needed, but Ollama must be running locally) |

## Instructions to Run the Application

1. **Add your PDF** to the `data/` folder (or point to any path), e.g.
   `data/source.pdf`.

2. **Ingest the PDF** — this loads, chunks, embeds, and stores it in ChromaDB:
   ```bash
   python ingest.py --pdf data/source.pdf
   ```
   You should see output confirming each of the 4 steps (load → split →
   embed → store).

3. **Run the chatbot:**
   ```bash
   python chatbot.py
   ```

4. **Ask questions** about the PDF directly in the terminal:
   ```
   You: What is this document about?
   Bot: ...
   (Sources: page(s) [1, 2] of the PDF)
   ```

5. Type `exit` or `quit` to stop the chatbot.

## How It Satisfies the Assignment Requirements

| Requirement                          | Where it's implemented                     |
|---------------------------------------|----------------------------------------------|
| Load a PDF document                   | `ingest.py` → `load_pdf()`                   |
| Split into meaningful chunks          | `ingest.py` → `split_documents()`            |
| Generate embeddings                   | `ingest.py` → `build_embeddings()` (all-MiniLM-L6-v2) |
| Store embeddings in ChromaDB          | `ingest.py` → `store_in_chroma()`            |
| Accept questions from the user        | `chatbot.py` → `main()` input loop           |
| Retrieve relevant chunks              | `chatbot.py` → `retriever` (top-4 similarity search) |
| Use an LLM to generate answers        | `llm_factory.py` + `RetrievalQA` chain in `chatbot.py` |
| Return accurate, relevant responses   | Prompt template restricts the LLM to the retrieved context |

## Notes / Troubleshooting

- If `ingest.py` fails with a "PDF not found" error, double-check the `--pdf`
  path.
- If `chatbot.py` fails with a missing API key error, confirm your `.env`
  file is filled in and matches your chosen `LLM_PROVIDER`.
- The first run will download the `all-MiniLM-L6-v2` embedding model
  (~90MB) from Hugging Face — this requires an internet connection once.
- For Ollama, make sure the Ollama app/service is running locally
  (`ollama serve`) and the model is pulled (`ollama pull llama3`) before
  running `chatbot.py`.
