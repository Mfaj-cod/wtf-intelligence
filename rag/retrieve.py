from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()


@dataclass
class RetrievalHit:
    """Structured retrieval output for a single chunk."""

    content: str
    source: str
    score: float
    metadata: dict[str, Any]
    rank: int


class SemanticRetriever:
    """Retrieval service that embeds queries and reranks ChromaDB results."""

    def __init__(
        self,
        db_path: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        top_k: int = 5,
        rerank_limit: int = 12,
    ) -> None:
        self.db_path = db_path or os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "wtf_research")
        self.embedding_model = embedding_model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.top_k = int(top_k or os.getenv("TOP_K", "5"))
        self.rerank_limit = max(rerank_limit, self.top_k)
        self._client: chromadb.PersistentClient | None = None
        self._collection = None
        self._embedder: OllamaEmbeddings | None = None

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.db_path)
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(self.collection_name)
        return self._collection

    def _get_embedder(self) -> OllamaEmbeddings:
        if self._embedder is None:
            self._embedder = OllamaEmbeddings(model=self.embedding_model)
        return self._embedder

    @staticmethod
    def _normalize_distance(distance: float | None) -> float:
        if distance is None:
            return 0.0
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            return 0.0
        # Chroma distance metrics are typically lower is better, so convert to a 0..1 similarity score.
        return 1.0 / (1.0 + max(distance, 0.0))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token]

    @staticmethod
    def _lexical_overlap_score(query: str, text: str) -> float:
        if not query or not text:
            return 0.0
        query_tokens = SemanticRetriever._tokenize(query)
        doc_tokens = SemanticRetriever._tokenize(text)
        if not query_tokens or not doc_tokens:
            return 0.0
        doc_counts = Counter(doc_tokens)
        overlap = sum(min(doc_counts[token], 1) for token in set(query_tokens))
        return overlap / max(len(set(query_tokens)), 1)

    def _rerank_candidates(self, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reranked: list[dict[str, Any]] = []
        for hit in hits:
            content = str(hit.get("content") or "")
            semantic_score = float(hit.get("score") or 0.0)
            lexical_score = self._lexical_overlap_score(query, content)
            composite = (semantic_score * 0.8) + (lexical_score * 0.2)
            reranked.append({**hit, "score": max(0.0, min(1.0, composite))})
        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[: self.top_k]

    def search(self, query: str, top_k: int | None = None, include_metadata: bool = True) -> list[dict[str, Any]]:
        """Search the embedded knowledge base for semantically relevant chunks.

        The Chroma query returns distance scores where lower values are better. We convert them into a
        0..1 similarity score and rerank candidates using a lightweight lexical overlap boost to better
        prioritize relevant chunks.
        """
        if not query or not query.strip():
            raise ValueError("A non-empty query is required for retrieval.")

        collection = self._get_collection()
        if collection.count() == 0:
            return []

        k = self.top_k if top_k is None else int(top_k)
        if k <= 0:
            raise ValueError("top_k must be greater than zero.")

        embedder = self._get_embedder()
        embedding = embedder.embed_query(query)
        candidate_count = max(k * 3, self.rerank_limit)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=candidate_count,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        candidates: list[dict[str, Any]] = []
        for idx, content in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else None
            candidates.append(
                {
                    "content": content,
                    "source": metadata.get("source") or metadata.get("document") or "unknown",
                    "score": self._normalize_distance(distance),
                    "metadata": metadata,
                }
            )

        if not candidates:
            return []

        reranked = self._rerank_candidates(query, candidates)
        final_results: list[dict[str, Any]] = []
        for rank, item in enumerate(reranked[:k], start=1):
            result = {
                "content": item["content"],
                "source": item["source"],
                "score": round(float(item["score"]), 4),
                "metadata": item["metadata"],
                "rank": rank,
            }
            if include_metadata:
                final_results.append(result)
            else:
                final_results.append({
                    "content": result["content"],
                    "source": result["source"],
                    "score": result["score"],
                    "metadata": result["metadata"],
                })

        return final_results

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Compatibility wrapper for retrieval callers."""
        return self.search(query, top_k=top_k)


def retrieve_documents(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    retriever = SemanticRetriever(top_k=top_k)
    return retriever.retrieve(query, top_k=top_k)


# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 2:
#         print("Usage: python -m rag.retrieve \"your research question\"", flush=True)
#         raise SystemExit(1)

#     query = " ".join(sys.argv[1:])
#     results = retrieve_documents(query)

#     if not results:
#         print("No relevant documents found. The collection may be empty or the query may not match the indexed content.", flush=True)
#         raise SystemExit(0)

#     for item in results:
#         print(f"[{item['rank']}] {item['source']} | score={item['score']:.4f}")
#         print(item["content"][:400])
#         print("-" * 80)
