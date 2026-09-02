import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompts.builder import build_prompt
from rag.retrieve import SemanticRetriever

from api.config import settings
from benchmark.models import OllamaModelAdapter, GroqModelAdapter
from benchmark.retrieval_snapshot import generate_retrieval_snapshot


def load_questions(questions_file: str = "benchmark/questions.json") -> List[Dict[str, Any]]:
    """Load benchmark questions from JSON file."""
    with open(questions_file) as f:
        return json.load(f)


def validate_systems(ollama_adapter, groq_adapter) -> bool:
    """Validate that both systems are accessible."""
    print("\n=== Phase 1: Validation ===\n")

    print("Checking Ollama...")
    if ollama_adapter.validate():
        actual_model = ollama_adapter.get_actual_model()
        print(f"  ✓ Ollama online, model: {actual_model}")
    else:
        print("  ✗ Ollama unavailable")
        return False

    print("Checking Groq...")
    if groq_adapter.validate():
        print(f"  ✓ Groq online, model: {groq_adapter.model_name}")
    else:
        print("  ✗ Groq unavailable (optional if not benchmarking Groq)")

    print("Checking ChromaDB...")
    try:
        persistent = __import__("chromadb").PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        collection = persistent.get_or_create_collection(settings.CHROMA_COLLECTION_NAME)
        doc_count = collection.count()
        print(f"  ✓ ChromaDB online, {doc_count} documents")
    except Exception as e:
        print(f"  ✗ ChromaDB error: {e}")
        return False

    return True


def single_question_test(
    ollama_adapter, groq_adapter, question: Dict[str, Any], retrieved_context: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run a single question through both models.
    
    Args:
        ollama_adapter: Ollama model adapter
        groq_adapter: Groq model adapter
        question: Question dict with profile
        retrieved_context: Pre-retrieved context chunks
        
    Returns:
        Result dict with responses from both models
    """
    profile = question.get("profile", {})
    question_text = question.get("question", "")

    # Build prompt using existing prompt builder
    system_prompt = build_prompt(profile, question_text, retrieved_context)

    result = {
        "question_id": question.get("id"),
        "question": question_text,
        "profile": profile,
        "retrieval": {
            "num_chunks": len(retrieved_context),
            "sources": [chunk.get("source", "unknown") for chunk in retrieved_context],
        },
        "local": None,
        "groq": None,
    }

    # Generate from local model
    local_result = ollama_adapter.generate(system_prompt, "")
    result["local"] = {
        "response": local_result.response,
        "latency_ms": local_result.latency_ms,
        "prompt_tokens": local_result.prompt_tokens,
        "completion_tokens": local_result.completion_tokens,
        "total_tokens": local_result.total_tokens,
        "tokens_per_second": local_result.tokens_per_second,
        "model": local_result.model_name,
        "error": local_result.error,
    }

    # Generate from Groq (if available)
    if groq_adapter.client:
        groq_result = groq_adapter.generate(system_prompt, "")
        result["groq"] = {
            "response": groq_result.response,
            "latency_ms": groq_result.latency_ms,
            "prompt_tokens": groq_result.prompt_tokens,
            "completion_tokens": groq_result.completion_tokens,
            "total_tokens": groq_result.total_tokens,
            "tokens_per_second": groq_result.tokens_per_second,
            "model": groq_result.model_name,
            "error": groq_result.error,
        }

    return result


def run_full_benchmark(
    ollama_adapter,
    groq_adapter,
    questions: List[Dict[str, Any]],
    retrieval_snapshot: Dict[str, Any],
    runs_per_question: int = 3,
) -> List[Dict[str, Any]]:
    """Run full benchmark with multiple runs per question.
    
    Args:
        ollama_adapter: Ollama adapter
        groq_adapter: Groq adapter
        questions: List of questions
        retrieval_snapshot: Pre-computed retrieval results
        runs_per_question: Number of runs per question
        
    Returns:
        List of result dicts
    """
    print(f"\n=== Phase 3: Full Benchmark ({len(questions)} questions × {runs_per_question} runs) ===\n")

    all_results = []
    retrieval_map = {r["question_id"]: r["sources"] for r in retrieval_snapshot["retrievals"]}

    for run_num in range(1, runs_per_question + 1):
        print(f"Run {run_num}/{runs_per_question}:")
        for question in questions:
            question_id = question.get("id")
            retrieved = retrieval_map.get(question_id, [])
            context_chunks = [{"source": s["source"], "content": s["content"]} for s in retrieved]

            result = single_question_test(ollama_adapter, groq_adapter, question, context_chunks)
            result["run"] = run_num
            all_results.append(result)

            status = "✓"
            if result["local"].get("error"):
                status = "✗"
            print(f"  {status} Q{question_id} - Local: {result['local']['latency_ms']:.0f}ms", end="")
            if result["groq"]:
                print(f", Groq: {result['groq']['latency_ms']:.0f}ms")
            else:
                print()

    return all_results


def save_raw_results(
    results: List[Dict[str, Any]], output_file: str = "benchmark/results/raw_results.json"
) -> None:
    """Save raw benchmark results to JSON."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    raw = {
        "benchmark": {
            "name": "WTF Intelligence Layer Benchmark",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "questions": len(set(r["question_id"] for r in results)),
            "runs_per_question": max(r.get("run", 1) for r in results),
        },
        "configuration": {
            "top_k": settings.TOP_K,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "temperature": 0.2,
            "max_output_tokens": 500,
        },
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(raw, f, indent=2)

    print(f"✓ Raw results saved to {output_file}")


def main():
    """Main benchmarking entry point."""
    print("=" * 70)
    print("WTF Intelligence Layer — Benchmark Suite")
    print("=" * 70)

    # Initialize adapters
    ollama_adapter = OllamaModelAdapter("llama3.2:latest")
    groq_adapter = GroqModelAdapter()

    # Phase 1: Validation
    if not validate_systems(ollama_adapter, groq_adapter):
        print("\n✗ Validation failed. Exiting.")
        sys.exit(1)

    # Load questions
    questions = load_questions()
    print(f"\n✓ Loaded {len(questions)} benchmark questions")

    # Phase 2: Single question test
    print("\n=== Phase 2: Single-Question Test ===\n")
    print("Running Q1 through both models...")

    retriever = SemanticRetriever(
        db_path=settings.CHROMA_PERSIST_DIRECTORY,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_model=settings.OLLAMA_EMBED_MODEL,
        top_k=settings.TOP_K,
    )
    retrieved = retriever.search(questions[0]["question"], top_k=settings.TOP_K)
    context_chunks = retrieved if retrieved else []

    test_result = single_question_test(ollama_adapter, groq_adapter, questions[0], context_chunks)
    print(f"  Local response length: {len(test_result['local']['response'])} chars")
    if test_result["groq"]:
        print(f"  Groq response length: {len(test_result['groq']['response'])} chars")

    # Generate retrieval snapshot
    print("\n=== Generating Retrieval Snapshot ===\n")
    retrieval_snapshot = generate_retrieval_snapshot(questions)

    # Phase 3: Warm-up
    print("\n=== Phase 3: Warm-up ===\n")
    print("Warming up Ollama...")
    ollama_adapter.warmup()
    print("Warming up Groq...")
    groq_adapter.warmup()

    # Phase 4: Full benchmark
    results = run_full_benchmark(ollama_adapter, groq_adapter, questions, retrieval_snapshot, runs_per_question=3)

    # Save results
    save_raw_results(results)

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
    print(f"Results saved to benchmark/results/")
    print(f"Next step: python -m benchmark.evaluate")


if __name__ == "__main__":
    main()
