import json
import time
from pathlib import Path
from typing import Any, Dict, List

import chromadb

from api.config import settings
from rag.retrieve import SemanticRetriever


def generate_retrieval_snapshot(
    questions: List[Dict[str, Any]], output_file: str = "benchmark/results/retrieval_snapshot.json"
) -> Dict[str, Any]:
    """Generate a retrieval snapshot for all benchmark questions.
    
    This ensures both models get identical retrieved context, preventing
    retrieval differences from contaminating model comparison.
    
    Args:
        questions: List of benchmark questions with profiles
        output_file: Path to save retrieval snapshot
        
    Returns:
        Dictionary containing retrieval results for all questions
    """
    retriever = SemanticRetriever(
        db_path=settings.CHROMA_PERSIST_DIRECTORY,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_model=settings.OLLAMA_EMBED_MODEL,
        top_k=settings.TOP_K,
    )

    snapshot = {
        "timestamp": time.time(),
        "configuration": {
            "top_k": settings.TOP_K,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_model": settings.OLLAMA_EMBED_MODEL,
        },
        "retrievals": [],
    }

    for q in questions:
        question_id = q.get("id")
        question_text = q.get("question")

        try:
            start = time.perf_counter()
            retrieved = retriever.search(question_text, top_k=settings.TOP_K)
            end = time.perf_counter()
            retrieval_ms = (end - start) * 1000

            sources = []
            if retrieved:
                for idx, chunk in enumerate(retrieved, 1):
                    sources.append(
                        {
                            "rank": idx,
                            "source": chunk.get("source", "unknown"),
                            "content": chunk.get("content", ""),
                            "metadata": chunk.get("metadata"),
                        }
                    )

            snapshot["retrievals"].append(
                {
                    "question_id": question_id,
                    "question": question_text,
                    "retrieval_ms": retrieval_ms,
                    "top_k": settings.TOP_K,
                    "sources": sources,
                }
            )
        except Exception as e:
            snapshot["retrievals"].append(
                {
                    "question_id": question_id,
                    "question": question_text,
                    "error": str(e),
                }
            )

    # Save snapshot
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"✓ Retrieval snapshot saved to {output_file}")
    return snapshot
