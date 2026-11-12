# AI-Powered PDF Question Answering System

A Streamlit application that lets you upload a PDF and ask questions grounded
in its contents. It uses the existing LangChain retrieval-augmented generation
(RAG) pipeline and can also be run as the original terminal chatbot.

## Architecture and workflow

1. `app.py` accepts a PDF upload and saves it to a temporary workspace.
2. `ingest.py` loads pages with `PyPDFLoader`, splits them into overlapping
   1,000-character chunks, embeds them with
   `sentence-transformers/all-MiniLM-L6-v2`, and indexes them in ChromaDB.
3. `chatbot.py` retrieves the four most relevant chunks and uses a LangChain
   `RetrievalQA` chain to generate an answer from that context. The prompt tells
   the model to say when the answer is not in the retrieved content.
4. The UI displays the answer and, when returned by the chain, the retrieved
   passages with their PDF page numbers.

Each browser session gets its own temporary PDF and vector-store directory so
uploads do not overwrite the CLI's `chroma_db/` or another session's index.
Temporary files are removed when the user clears or replaces the processed
document. Streamlit Community Cloud has ephemeral storage; uploaded PDFs and
indexes are not retained across app restarts. Chroma anonymized telemetry is
disabled by default for the ingestion and retrieval clients.

## Technologies

- Python 3.11 (recommended for local use and selected for deployment)
- Streamlit
- LangChain and LangChain Community
- ChromaDB
- Sentence Transformers (`all-MiniLM-L6-v2`)
- PyPDF
- Groq, OpenAI, Google Gemini, or Ollama via `llm_factory.py`

The default provider is Groq with `openai/gpt-oss-20b`. Provider and model
can be selected using environment variables.

## Local setup

1. Create and activate a Python 3.11 virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Configure the chosen LLM provider (see below). For Groq, for example:

   ```powershell
   # Windows PowerShell; use your own key, and do not commit it.
   $env:GROQ_API_KEY = "your-groq-api-key"
   ```

   ```bash
   # macOS/Linux; use your own key, and do not commit it.
   export GROQ_API_KEY="your-groq-api-key"
   ```

4. Start the web app:

   ```bash
   streamlit run app.py
   ```

The first ingestion downloads the embedding model from Hugging Face and
requires network access. The existing command-line workflow is still
available: place a PDF at `data/source.pdf`, run
`python ingest.py --pdf data/source.pdf`, then `python chatbot.py`.

## LLM configuration

`LLM_PROVIDER` selects the backend. It defaults to `groq`. Add the key matching
your provider and optionally override its model:

| Provider | Required key | Optional model setting | Default model |
|---|---|---|---|
| `groq` | `GROQ_API_KEY` | `GROQ_MODEL` | `openai/gpt-oss-20b` |
| `openai` | `OPENAI_API_KEY` | `OPENAI_MODEL` | `gpt-4o-mini` |
| `gemini` | `GOOGLE_API_KEY` | `GEMINI_MODEL` | `gemini-1.5-flash` |
| `ollama` | None | `OLLAMA_MODEL`, `OLLAMA_BASE_URL` | `llama3`, `http://localhost:11434` |

For Ollama, the Ollama service and selected model must be reachable from the
machine running the app. A local Ollama installation is generally not available
to a hosted Streamlit Community Cloud app.

### Streamlit secrets

For a local secrets file, create `.streamlit/secrets.toml` (keep it out of
version control):

```toml
LLM_PROVIDER = "groq"
GROQ_API_KEY = "your-groq-api-key"
# GROQ_MODEL = "openai/gpt-oss-20b"
```

In Streamlit Community Cloud, add the same settings in the app's **Settings →
Secrets**. Environment variables are also supported; existing environment
values take precedence over Streamlit secrets. Never place real keys in source
code, README files, or a public repository.

## Deploy to Streamlit Community Cloud

1. Push the project to a GitHub repository you control.
2. In Streamlit Community Cloud, create an app from that repository and select
   `app.py` as the entry point.
3. Select Python 3.11 in the app's advanced settings.
4. Add `LLM_PROVIDER` and the matching provider API key under the app's
   **Settings → Secrets**.
5. Deploy. The app installs dependencies from `requirements.txt`; the
   embedding model is downloaded when the first PDF is processed.

The app does not require a repository-local PDF or a prebuilt Chroma directory.
Cloud storage is ephemeral, so users need to upload and process a PDF again
after a restart.

## Notes and limitations

- Answers depend on the uploaded document, retrieved chunks, and configured
  model; the app does not guarantee factual accuracy.
- Scanned/image-only PDFs may not provide extractable text because the current
  loader workflow does not perform OCR.
- The CLI's `ingest.py` still rebuilds the default local `chroma_db/` when run.
- Local model downloads and indexing require enough available memory and disk
  space for the PDF, embedding model, and ChromaDB.
