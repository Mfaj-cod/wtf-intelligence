import time
from typing import Any

import chromadb
from fastapi import APIRouter, HTTPException

from api.config import settings
from api.schemas import AskRequest, AskResponse, HealthResponse, IngestResponse, ModelsResponse
from api.utils import get_ollama_client, resolve_model_name
from rag.ingest import Ingestor
from rag.retrieve import SemanticRetriever
from prompts.builder import build_prompt

router = APIRouter(tags=["routes"])


@router.post("/ask", response_model=AskResponse)
def generate_response(payload: AskRequest):
    request_started = time.perf_counter()
    question = payload.question.strip()
    profile = payload.profile.model_dump()

    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    active_model = resolve_model_name()

    retriever = SemanticRetriever(
        db_path=settings.CHROMA_PERSIST_DIRECTORY,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_model=settings.OLLAMA_EMBED_MODEL,
        top_k=settings.TOP_K,
    )
    retrieval_started = time.perf_counter()

    try:
        retrieved = retriever.search(question, top_k=settings.TOP_K)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging path
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
    context_chunks = retrieved if retrieved else []
    prompt = build_prompt(profile, question, context_chunks)

    client = get_ollama_client()
    try:
        inference_started = time.perf_counter()
        raw_response = client.generate(
            model=active_model,
            prompt=prompt,
            options={"temperature": 0.2, "num_predict": 500},
            stream=False,
        )
        inference_ms = int((time.perf_counter() - inference_started) * 1000)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        raise HTTPException(status_code=503, detail=f"Ollama inference unavailable: {exc}") from exc

    # Extract text response from Ollama response (dict or object)
    response_text = ""
    if isinstance(raw_response, dict):
        response_text = raw_response.get("response", "").strip()
    elif hasattr(raw_response, "response"):
        response_text = str(getattr(raw_response, "response", "")).strip()
    
    if not response_text:
        response_text = "I couldn’t generate a grounded answer from the available research and profile context."

    sources_used = [
        item.get("source", "unknown")
        for item in context_chunks
        if item.get("source")
    ]
    
    total_latency_ms = int((time.perf_counter() - request_started) * 1000)
    tokens_generated = max(1, len(response_text.split()))

    return AskResponse(
        response=response_text,
        sources_used=sources_used,
        model=active_model,
        inference_ms=inference_ms,
        tokens_generated=tokens_generated,
        retrieved_chunks=[item.get("content", "") for item in context_chunks],
        retrieval_ms=retrieval_ms,
        total_latency_ms=total_latency_ms,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents():
    ingestor = Ingestor(
        data_path="./data/research",
        collection_name=settings.CHROMA_COLLECTION_NAME,
        db_path=settings.CHROMA_PERSIST_DIRECTORY,
        embedding_model=settings.OLLAMA_EMBED_MODEL,
    )

    start = time.perf_counter()
    docs = ingestor.load_documents()
    if not docs:
        raise HTTPException(status_code=404, detail="No research documents were found in ./data/research.")

    chunks = ingestor.split_documents(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=500, detail="Research documents were found but no chunks could be created.")

    emb = ingestor.init_embedding_model()
    if emb is None:
        raise HTTPException(status_code=503, detail="Embedding model could not be initialized.")

    collection = ingestor.init_collection()
    if collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB collection could not be initialized.")

    stored = ingestor.index_chunks()
    duration_ms = int((time.perf_counter() - start) * 1000)

    return IngestResponse(
        status="success",
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        chunks_stored=stored,
        collection=settings.CHROMA_COLLECTION_NAME,
        embedding_model=settings.OLLAMA_EMBED_MODEL,
        duration_ms=duration_ms,
    )


@router.get("/models", response_model=ModelsResponse)
def get_models():
    try:
        client = get_ollama_client()
        response = client.list()
        models = []

        def add_name(name: Any) -> None:
            if isinstance(name, str) and name:
                models.append(name)

        if isinstance(response, dict):
            for model in response.get("models", []):
                if isinstance(model, dict):
                    add_name(model.get("name"))
                else:
                    add_name(model)
        else:
            for item in getattr(response, "models", []) or []:
                if hasattr(item, "model"):
                    add_name(getattr(item, "model"))
                elif isinstance(item, dict):
                    add_name(item.get("name"))
                else:
                    add_name(item)

        return ModelsResponse(models=models)
    except Exception as exc:  # pragma: no cover - runtime environment path
        raise HTTPException(status_code=503, detail=f"Unable to list Ollama models: {exc}") from exc


@router.get("/health", response_model=HealthResponse)
def health_check():
    client = get_ollama_client()
    active_model = resolve_model_name()
    health = {"connected": False, "model": active_model, "error": None}
    chroma_status = {"connected": False, "collection": settings.CHROMA_COLLECTION_NAME, "documents": 0, "error": None}

    try:
        client.list()
        health["connected"] = True
    except Exception as exc:  # pragma: no cover
        health["error"] = str(exc)

    try:
        persistent = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        collection = persistent.get_or_create_collection(settings.CHROMA_COLLECTION_NAME)
        chroma_status["connected"] = True
        chroma_status["documents"] = int(collection.count())
    except Exception as exc:  # pragma: no cover
        chroma_status["error"] = str(exc)

    status = "healthy" if health["connected"] and chroma_status["connected"] else "degraded"
    return HealthResponse(status=status, ollama=health, chroma=chroma_status)

