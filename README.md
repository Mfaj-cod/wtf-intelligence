# In-House Intelligence Layer

## Project Goal

The target architecture is:

```bash
Client Profile + Question
          |
          v
       FastAPI
          |
          v
   RAG Retrieval Layer
          |
          v
      ChromaDB
          |
      Top-5 Chunks
          |
          v
    Prompt Builder
          |
          v
        Ollama
    Llama 3.2 model
          |
          v
 Structured JSON Response
```

---

# Real-time Progress

- Ollama is reachable.
- Required LLM is available locally.
- `nomic-embed-text` is available locally.
- Research documents can be ingested.
- ChromaDB persists locally.
- Retrieval returns relevant top-5 chunks.
- Prompt builder combines research + profile + question.
- Ollama generates a response.
- `POST /ask` returns structured JSON.
- `/health` works.
- `/models` works.
- `/ingest` works.
- Swagger `/docs` works.
- Source attribution works.
- Latency is measured.
- [ ] Unit tests pass.
- [ ] Docker image builds.
- [ ] docker-compose configuration works.
- [ ] `setup.sh` works or provides actionable setup instructions.
- [ ] Ten benchmark questions are present.
- [ ] Local-vs-commercial benchmark can run.
- [ ] Benchmark results are saved as JSON.
- [ ] No benchmark numbers are fabricated.
- [ ] README allows another developer to run the project quickly.
- [ ] Known limitations are documented.
- [ ] A reasoned production-readiness recommendation exists.

---

# Implementation Order

Implementing in this order:

```text
Phase 1  → Environment + project skeleton
Phase 2  → Ollama connection
Phase 3  → ChromaDB + embeddings
Phase 4  → Document ingestion
Phase 5  → Retrieval
Phase 6  → Prompt engineering
Phase 7  → /ask endpoint
Phase 8  → Remaining API endpoints
Phase 9  → Testing
Phase 10 → Benchmark
Phase 11 → Docker
Phase 12 → setup.sh
Phase 13 → README + final demo
```

---
