# WTF Intelligence Layer

Self-hosted financial intelligence prototype using FastAPI, local RAG, ChromaDB, and Ollama.

The production demo path is local: a client profile and question go into the API, the RAG layer retrieves research chunks from local ChromaDB, the prompt builder combines the profile/question/research context, and Ollama generates a structured JSON response. The benchmark optionally compares the local model with Groq's hosted `openai/gpt-oss-120b` model.

```text
Client Profile + Question
          |
          v
      FastAPI API
          |
          v
   RAG Retrieval Layer
          |
          v
      ChromaDB
          |
       top-k chunks
          |
          v
    Prompt Builder
          |
          v
       Ollama
          |
          v
 Structured JSON Response
```

## Demo

Run these commands from the project root.

### 1. Create and Activate a Python Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

This project declares Python `>=3.13` in `pyproject.toml`.

### 2. Start Ollama and Pull Models

Install Ollama first if it is not already available, then run:

```bash
ollama list
ollama pull llama3.2
ollama pull nomic-embed-text
```

If the Ollama service is not running:

```bash
ollama serve
```

Keep Ollama reachable at:

```text
http://localhost:11434
```

### 3. Configure Environment Variables

Create `.env` from `.env.example`, then use this known-good local demo configuration:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
OLLAMA_EMBED_MODEL=nomic-embed-text

CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=wtf_research

TOP_K=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50

API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Optional, only for the hosted benchmark comparison:
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Use the exact `OLLAMA_MODEL` tag shown by `ollama list` if your local model is named differently, for example `llama3.2` or `llama3.2:8b`.

Do not commit `.env`. If a real key has ever been committed or shared, rotate it.

### 4. Ingest Research Documents

The RAG corpus is loaded from:

```text
data/research/*.txt
```

Start the API:

```bash
uvicorn api.main:app --reload
```

In another terminal, run ingestion through the API:

Windows PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:8000/ingest
```

macOS/Linux:

```bash
curl -X POST http://127.0.0.1:8000/ingest
```

Expected response shape:

```json
{
  "status": "success",
  "documents_loaded": 5,
  "chunks_created": 10,
  "chunks_stored": 10,
  "collection": "wtf_research",
  "embedding_model": "nomic-embed-text",
  "duration_ms": 1234
}
```

Counts and timings depend on the documents and machine.

### 5. Smoke-Test the API

Root:

```bash
curl http://127.0.0.1:8000/
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

Models:

```bash
curl http://127.0.0.1:8000/models
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Ask endpoint, Windows PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d "{\"profile\":{\"primary_goal\":\"Long-term growth\",\"horizon_years\":10,\"risk_score\":7,\"investor_profile\":\"Growth investor\",\"current_holdings\":[\"VTI\",\"VXUS\",\"BTC\"]},\"question\":\"What should I be aware of going into my first advisor meeting?\"}"
```

Ask endpoint, macOS/Linux:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "profile": {
      "primary_goal": "Long-term growth",
      "horizon_years": 10,
      "risk_score": 7,
      "investor_profile": "Growth investor",
      "current_holdings": ["VTI", "VXUS", "BTC"]
    },
    "question": "What should I be aware of going into my first advisor meeting?"
  }'
```

Expected `/ask` response fields:

```json
{
  "response": "Generated answer text...",
  "sources_used": ["data/research/market_dynamics.txt"],
  "model": "llama3.2:latest",
  "inference_ms": 1234,
  "tokens_generated": 120,
  "retrieved_chunks": ["..."],
  "retrieval_ms": 50,
  "total_latency_ms": 1300
}
```

## Project Layout

```text
api/
  main.py        FastAPI app setup
  routes.py      API route handlers
  schemas.py     Pydantic request/response contracts
  config.py      Environment-backed settings
  utils.py       Ollama client/model helpers

rag/
  ingest.py      Loads data/research/*.txt, chunks, embeds, stores in ChromaDB
  retrieve.py    Embeds questions and returns top-k ranked chunks

prompts/
  builder.py     Builds the final model prompt from profile, question, and chunks

benchmark/
  questions.json             Fixed benchmark dataset
  retrieval_snapshot.py      Saves fixed retrieved context for all questions
  run_benchmark.py           Runs local-vs-Groq generation benchmark
  quality_evaluator.py       Interactive human scoring
  evaluate.py                Aggregates results and writes report
  results/                   Benchmark JSON and markdown outputs

data/research/               Local research corpus
chroma_db/                   Local persistent ChromaDB store
```

## API Reference

### `GET /`

Basic service check.

Response:

```json
{
  "service": "wtf-intelligence",
  "status": "ok"
}
```

### `GET /health`

Checks Ollama connectivity and ChromaDB collection status.

Response fields:

```json
{
  "status": "healthy",
  "ollama": {
    "connected": true,
    "model": "llama3.2:latest",
    "error": null
  },
  "chroma": {
    "connected": true,
    "collection": "wtf_research",
    "documents": 10,
    "error": null
  }
}
```

The status is `degraded` when Ollama or ChromaDB is unavailable.

### `GET /models`

Lists models reported by the local Ollama server.

Response:

```json
{
  "models": ["llama3.2:latest", "nomic-embed-text:latest"]
}
```

### `POST /ingest`

Loads text documents from `./data/research`, splits them into chunks, embeds them with `OLLAMA_EMBED_MODEL`, and stores them in the configured ChromaDB collection.

Response fields:

```json
{
  "status": "success",
  "documents_loaded": 5,
  "chunks_created": 10,
  "chunks_stored": 10,
  "collection": "wtf_research",
  "embedding_model": "nomic-embed-text",
  "duration_ms": 1234
}
```

### `POST /ask`

Core intelligence endpoint.

Request schema:

```json
{
  "profile": {
    "primary_goal": "Long-term growth",
    "horizon_years": 10,
    "risk_score": 7,
    "investor_profile": "Growth investor",
    "current_holdings": ["VTI", "VXUS", "BTC"]
  },
  "question": "What should I be aware of going into my first advisor meeting?"
}
```

Validation rules:

- `question` must be non-empty.
- `primary_goal` must be non-empty.
- `horizon_years` must be greater than `0`.
- `risk_score` must be from `1` to `10`.
- `investor_profile` must be non-empty.
- `current_holdings` must contain at least one item.

Response schema:

```json
{
  "response": "Generated answer text...",
  "sources_used": ["data/research/market_dynamics.txt"],
  "model": "llama3.2:latest",
  "inference_ms": 1234,
  "tokens_generated": 120,
  "retrieved_chunks": ["Retrieved context..."],
  "retrieval_ms": 50,
  "total_latency_ms": 1300
}
```

## Benchmark

The benchmark compares:

| System            | Provider | Model                                             |
| ----------------- | -------- | ------------------------------------------------- |
| Local             | Ollama   | configured local model, usually `llama3.2:latest` |
| Hosted comparison | Groq     | `openai/gpt-oss-120b`                             |

The benchmark uses the fixed dataset in `benchmark/questions.json`. It generates a retrieval snapshot first so both models receive the same retrieved research context.

### Configure Groq

Groq is only required for the hosted comparison. Set:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

If `GROQ_API_KEY` is missing, local API usage still works, but the full local-vs-hosted benchmark cannot complete.

### Run the Benchmark

Recommended sequence:

```bash
python -m benchmark.retrieval_snapshot
python -m benchmark.run_benchmark
python -m benchmark.quality_evaluator
python -m benchmark.evaluate
```

`benchmark.run_benchmark` performs:

- system validation
- a single-question sanity test
- retrieval snapshot generation
- Ollama and Groq warm-up
- full benchmark over 10 questions x 3 measured runs
- raw result export

The interactive quality evaluator scores local and Groq responses on relevance, reasoning, personalization, grounding, completeness, factuality, overall quality, hallucination count, and hallucination severity.

### Benchmark Outputs

Generated files:

```text
benchmark/results/retrieval_snapshot.json
benchmark/results/raw_results.json
benchmark/results/quality_scores.json
benchmark/results/analysis.json
benchmark/results/benchmark_report.md
```

Do not edit raw benchmark outputs by hand. Re-run the scripts when data needs to change.

## Docker

Docker is not currently supported by this repository. `Dockerfile` and `docker-compose.yaml` exist but are empty, so the reliable demo path is local Python plus host-installed Ollama.

If Docker support is added later, keep Ollama and the API as separate concerns. Do not assume `localhost:11434` inside a container points to the host Ollama service unless networking is configured explicitly.

## Troubleshooting

### Ollama Is Not Running

Check:

```bash
ollama list
```

Start the service if needed:

```bash
ollama serve
```

Then verify:

```bash
curl http://127.0.0.1:11434/api/tags
```

### Missing Local Model

If `/models` or `/ask` cannot find the configured model:

```bash
ollama pull llama3.2
ollama list
```

Set `OLLAMA_MODEL` to the exact model tag shown by `ollama list`.

### Missing Embedding Model

If ingestion or retrieval fails while embedding text:

```bash
ollama pull nomic-embed-text
ollama list
```

Set `OLLAMA_EMBED_MODEL=nomic-embed-text` or the exact installed tag.

### Empty Chroma Collection

If `/health` shows `documents: 0`, run:

```bash
curl -X POST http://127.0.0.1:8000/ingest
```

Also confirm that `data/research` contains `.txt` files.

### Missing Groq API Key

The API does not require Groq. The benchmark comparison does.

Set:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Then restart the shell or reload the environment before running:

```bash
python -m benchmark.run_benchmark
```

### `POST /ask` Returns Validation Errors

Ensure the request body includes all required profile fields and at least one `current_holdings` item. The schema currently rejects an empty holdings list.

## Known Limitations

- This is a prototype, not a production financial advisory system.
- The local answer quality depends on the installed Ollama model.
- Research is limited to local `.txt` files in `data/research`.
- The API has no authentication, authorization, rate limiting, audit logging, or compliance controls.
- Ingestion currently adds chunks with simple chunk IDs, so repeated ingestion can produce duplicate-ID errors or skip duplicate chunks depending on ChromaDB state.
- Benchmark quality scoring is human-interactive and subjective.
- Docker files are empty and should not be used as the demo path.
- Hosted benchmark results depend on Groq availability, network conditions, and current provider behavior.

## Production Readiness

This repository is suitable as a local demo of private RAG-backed financial intelligence:

- local inference through Ollama
- local embedding through `nomic-embed-text`
- local persistence through ChromaDB
- standard HTTP API through FastAPI
- measurable latency and benchmark artifacts

It is not production-ready without additional work around security, reliability, data governance, observability, deployment, duplicate-safe ingestion, test coverage, and formal financial compliance review.
