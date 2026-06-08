# MSME Advisory Assistant

Production-ready FastAPI + LangChain implementation of the original Langflow MSME, Startup, and Entrepreneurship advisory flow.

## 🚀 Try Now

<p align="center">
  <a href="https://msmerag.duckdns.org/">
    <img src="https://img.shields.io/badge/Try%20Now-MSME%20Advisory%20Assistant-blue?style=for-the-badge" alt="Try Now">
  </a>
</p>

## Architecture

- Regex-first guardrails block PII, credentials, jailbreaks, prompt extraction, role-changing attacks, and safety bypass attempts before any downstream component runs.
- Smart routing returns only `MSME` or `General`. Keyword fast-paths reduce LLM calls; ambiguous queries use Groq `llama-3.1-8b-instant`.
- ChromaDB stores the local MSME knowledge base with Ollama `nomic-embed-text` embeddings.
- Tavily is conditional and only runs for latest, recent, budget, announcement, notification, or policy-update style questions.
- Final answer generation uses the configured final model and preserves the Langflow prompt rules.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` with your API keys.

Install and run Ollama, then pull the embedding model:

```bash
ollama pull nomic-embed-text:latest
```

Add PDFs to:

```text
data/pdfs/
```

Start the API:

```bash
uvicorn main:app --reload
```

The app automatically scans `data/pdfs/` on startup and ingests new or changed PDFs into `./ragdb`.

## API

`POST /chat`

Request:

```json
{
  "query": "What is PMEGP?"
}
```

Response:

```json
{
  "route": "MSME",
  "answer": "...",
  "sources": [],
  "metrics": {}
}
```

## Ingestion

The ingestion pipeline:

- Scans all PDFs under `data/pdfs/`.
- Computes file hashes to skip unchanged PDFs.
- Loads pages with `PyPDFLoader`.
- Splits documents into overlapping chunks.
- Computes chunk hashes to avoid duplicate chunk ingestion.
- Stores metadata: `source_file`, `source_path`, `document_type`, `upload_date`, `chunk_id`, `chunk_hash`, and `page`.
- Persists ChromaDB locally in `./ragdb`.

## Metrics

Each request logs and returns timings for:

- `guardrail_ms`
- `router_ms`
- `embedding_generation_ms`
- `chroma_retrieval_ms`
- `tavily_query_rewrite_ms`
- `tavily_search_ms`
- `llm_response_generation_ms`
- `total_request_ms`
- `retrieved_chunks`
- `tavily_used`

## Performance Recommendations

- Keep regex guardrails before all LLM calls.
- Keep routing keyword fast-paths for common MSME terms.
- Use Tavily only for explicitly current or recent questions.
- Preload Chroma and Ollama at startup for lower first-request latency.
- Consider a small local reranker if retrieval quality is weak.
- Cache frequent answers for stable scheme definitions such as PMEGP, Udyam, Mudra, and CGTMSE.
- Increase `RETRIEVAL_TOP_K` only if answer quality requires it.
