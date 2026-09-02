"""
Interactive quality evaluator for the WTF Intelligence Layer benchmark.

Usage:
    python -m benchmark.quality_evaluator

It reads:
    benchmark/results/raw_results.json

It writes incrementally to:
    benchmark/results/quality_scores.json

The evaluator presents each benchmark question with the Local and Groq
responses and asks for 1-5 quality scores.

Scores:
    relevance       - Does the answer directly address the question?
    reasoning       - Is the financial reasoning sound and useful?
    personalization - Does it appropriately use the supplied client profile?
    grounding       - Does it correctly use the supplied research context?
    completeness    - Does it cover the important aspects without unnecessary filler?
    factuality      - Are claims accurate and appropriately qualified?
    overall         - Overall response quality.

Hallucination:
    hallucination_count    - Number of unsupported/fabricated claims you identify.
    hallucination_severity - 0 none, 1 minor, 2 moderate, 3 severe.

The file is saved after every question, so an interruption does not lose
previous evaluations. Re-running the script resumes from unanswered questions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


RAW_RESULTS_FILE = Path("benchmark/results/raw_results.json")
QUALITY_SCORES_FILE = Path("benchmark/results/quality_scores.json")

QUALITY_DIMENSIONS = [
    ("relevance", "Relevance"),
    ("reasoning", "Financial Reasoning"),
    ("personalization", "Profile Personalization"),
    ("grounding", "Research Grounding"),
    ("completeness", "Completeness"),
    ("factuality", "Factuality"),
    ("overall", "Overall Quality"),
]


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Atomically-ish save JSON after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    temp_path.replace(path)


def question_sort_key(question_id: Any) -> tuple:
    """Sort Q1, Q2, ... Q10 numerically where possible."""
    try:
        text = str(question_id)
        if text.upper().startswith("Q"):
            return (0, int(text[1:]))
        return (0, int(text))
    except (ValueError, TypeError):
        return (1, str(question_id))


def normalize_question_id(question_id: Any) -> str:
    """Normalize numeric IDs to Q1, Q2, ... while preserving non-numeric IDs."""
    text = str(question_id)
    if text.upper().startswith("Q"):
        return "Q" + text[1:]
    try:
        return f"Q{int(text)}"
    except ValueError:
        return text


def get_successful_runs(results: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """Return successful results for a model."""
    successful = []

    for result in results:
        model_result = result.get(model)

        if not model_result:
            continue

        if model_result.get("error"):
            continue

        response = model_result.get("response", "")
        if not response or not str(response).strip():
            continue

        successful.append(result)

    return successful


def build_question_groups(
    raw: Dict[str, Any],
    run_number: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Group raw results by question ID.

    If multiple runs exist, use --run N to select a specific run.
    Without --run, the first successful Local/Groq response is selected.
    """
    results = raw.get("results", [])
    groups: Dict[str, Dict[str, Any]] = {}

    for result in results:
        qid = normalize_question_id(result.get("question_id"))

        if run_number is not None:
            result_run = result.get("run", 1)
            if result_run != run_number:
                continue

        if qid not in groups:
            groups[qid] = {
                "question_id": result.get("question_id"),
                "question": result.get("question", ""),
                "profile": result.get("profile", {}),
                "retrieval": result.get("retrieval", {}),
                "local_candidates": [],
                "groq_candidates": [],
            }

        local = result.get("local")
        groq = result.get("groq")

        if local and not local.get("error") and local.get("response"):
            groups[qid]["local_candidates"].append(result)

        if groq and not groq.get("error") and groq.get("response"):
            groups[qid]["groq_candidates"].append(result)

    # Select one response per model/question.
    # By default this is the first successful response, which makes the
    # quality evaluation independent of repeated latency runs.
    for group in groups.values():
        group["local"] = (
            group["local_candidates"][0]
            if group["local_candidates"]
            else None
        )
        group["groq"] = (
            group["groq_candidates"][0]
            if group["groq_candidates"]
            else None
        )

    return dict(sorted(groups.items(), key=lambda item: question_sort_key(item[0])))


def is_question_scored(score_entry: Any) -> bool:
    """Return True only when both Local and Groq have complete scores.

    A pre-created quality_scores.json template may already contain Q1..Q10
    with null values. Those entries must NOT be treated as evaluated.
    """
    if not isinstance(score_entry, dict):
        return False

    required = [
        "relevance",
        "reasoning",
        "personalization",
        "grounding",
        "completeness",
        "factuality",
        "overall",
        "hallucination_count",
        "hallucination_severity",
    ]

    for model in ("local", "groq"):
        model_scores = score_entry.get(model)
        if not isinstance(model_scores, dict):
            return False

        if any(model_scores.get(key) is None for key in required):
            return False

    return True


def load_existing_scores() -> Dict[str, Any]:
    """Load previous scores so evaluation can be resumed."""
    if not QUALITY_SCORES_FILE.exists():
        return {}

    try:
        data = load_json(QUALITY_SCORES_FILE)
        print(f"✓ Loaded existing quality scores: {QUALITY_SCORES_FILE}")
        return data
    except json.JSONDecodeError as exc:
        print(f"✗ Invalid quality_scores.json: {exc}")
        sys.exit(1)


def clear_screen() -> None:
    """Clear terminal for easier response-by-response evaluation."""
    # ANSI works in modern Windows Terminal/VS Code terminals.
    print("\033[2J\033[H", end="")


def print_separator(char: str = "=", width: int = 100) -> None:
    print(char * width)


def print_wrapped_text(title: str, text: Any) -> None:
    print(f"\n--- {title} ---")
    print(str(text).strip() if text else "[No content]")


def print_profile(profile: Dict[str, Any]) -> None:
    print("\n--- Client Profile ---")

    if not profile:
        print("[No profile supplied]")
        return

    for key, value in profile.items():
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        print(f"{key}: {value}")


def print_research_context(retrieval: Dict[str, Any]) -> None:
    """Show retrieved context from raw results.

    The benchmark may store retrieval sources either as structured dictionaries
    or as plain strings. Support both formats so the quality evaluator does not
    crash merely because the retrieval serializer used a different shape.
    """
    print("\n--- Retrieved Research Context ---")

    if not retrieval:
        print("[No retrieval context stored in raw results]")
        return

    # Normal format: {"sources": [...]}
    if isinstance(retrieval, dict):
        sources = retrieval.get("sources", [])

        # Be defensive if the whole retrieval object itself is the source list.
        if not sources and retrieval.get("content"):
            sources = [retrieval]
    elif isinstance(retrieval, list):
        sources = retrieval
    else:
        sources = [retrieval]

    if not sources:
        print("[No retrieval context stored in raw results]")
        return

    for index, source in enumerate(sources, start=1):
        # Structured source object.
        if isinstance(source, dict):
            rank = source.get("rank", index)
            name = source.get(
                "source",
                source.get("filename", source.get("metadata", "unknown"))
            )
            score = source.get("score")

            score_text = (
                f" | score={score:.4f}"
                if isinstance(score, (int, float))
                else ""
            )

            content = source.get(
                "content",
                source.get("text", source.get("page_content", ""))
            )

            print(f"\n[{rank}] {name}{score_text}")
            print(str(content).strip() if content else "[No content]")

        # Plain-string source.
        else:
            print(f"\n[{index}] Retrieved context")
            print(str(source).strip())


def print_response(result: Optional[Dict[str, Any]], label: str) -> None:
    if not result:
        print_wrapped_text(label, "[No successful response recorded]")
        return

    model_result = result.get(label.lower(), {})
    print_wrapped_text(label, model_result.get("response", ""))

    latency = model_result.get("latency_ms")
    tokens = model_result.get("completion_tokens")
    tps = model_result.get("tokens_per_second")

    metrics = []

    if isinstance(latency, (int, float)):
        metrics.append(f"latency={latency:.1f} ms")
    if isinstance(tokens, (int, float)):
        metrics.append(f"completion_tokens={int(tokens)}")
    if isinstance(tps, (int, float)):
        metrics.append(f"tokens/sec={tps:.2f}")

    if metrics:
        print(f"\n[Benchmark metrics: {' | '.join(metrics)}]")


def prompt_score(label: str, allow_blank: bool = False) -> Optional[int]:
    """Prompt for a 1-5 score with validation."""
    while True:
        suffix = " [Enter = skip]" if allow_blank else ""
        raw = input(f"{label} (1-5){suffix}: ").strip()

        if allow_blank and raw == "":
            return None

        if raw in {"1", "2", "3", "4", "5"}:
            return int(raw)

        print("Please enter a whole number from 1 to 5.")


def prompt_nonnegative_int(label: str, allow_blank: bool = False) -> Optional[int]:
    """Prompt for a non-negative integer."""
    while True:
        suffix = " [Enter = skip]" if allow_blank else ""
        raw = input(f"{label}{suffix}: ").strip()

        if allow_blank and raw == "":
            return None

        try:
            value = int(raw)
            if value >= 0:
                return value
        except ValueError:
            pass

        print("Please enter a non-negative integer.")


def prompt_hallucination_severity(allow_blank: bool = False) -> Optional[int]:
    """Prompt for hallucination severity from 0-3."""
    while True:
        suffix = " [Enter = skip]" if allow_blank else ""
        raw = input(
            f"Hallucination severity "
            f"(0=None, 1=Minor, 2=Moderate, 3=Severe){suffix}: "
        ).strip()

        if allow_blank and raw == "":
            return None

        if raw in {"0", "1", "2", "3"}:
            return int(raw)

        print("Please enter 0, 1, 2, or 3.")


def evaluate_model(model_result: Optional[Dict[str, Any]], model_label: str) -> Optional[Dict[str, Any]]:
    """Interactively score one model's response."""
    if not model_result:
        print(f"\nNo successful {model_label} response is available.")
        return None

    print_separator("-")
    print(f"SCORING: {model_label}")
    print_separator("-")

    scores: Dict[str, Any] = {}

    for key, label in QUALITY_DIMENSIONS:
        scores[key] = prompt_score(label)

    print()
    print("Hallucination should mean an unsupported/fabricated claim,")
    print("not merely an answer you personally disagree with.")
    scores["hallucination_count"] = prompt_nonnegative_int(
        "Hallucination count"
    )
    scores["hallucination_severity"] = prompt_hallucination_severity()

    return scores


def evaluate_question(
    qid: str,
    group: Dict[str, Any],
    existing: Dict[str, Any],
) -> bool:
    """Display one question and collect Local/Groq scores."""
    clear_screen()

    print_separator()
    print(f"QUALITY EVALUATION — {qid}")
    print_separator()

    print_wrapped_text("Question", group.get("question", ""))
    print_profile(group.get("profile", {}))

    # The retrieval context is useful for judging grounding.
    # It can be hidden with --hide-context.
    print_research_context(group.get("retrieval", {}))

    print_separator("-")
    print_response(group.get("local"), "Local")
    print_separator("-")
    print_response(group.get("groq"), "Groq")
    print_separator()

    print("\nSCORING RUBRIC")
    print("1 = very poor, 2 = weak, 3 = adequate, 4 = strong, 5 = excellent")
    print("Grounding: judge whether claims are supported by the supplied research.")
    print("Personalization: judge whether the supplied client profile is used appropriately.")
    print()

    current = existing.get(qid, {})

    # Allow the user to resume and overwrite an existing question.
    if current:
        print(f"Existing scores found for {qid}. They will be overwritten if you continue.")
        choice = input("Continue with re-scoring? [y/N]: ").strip().lower()
        if choice != "y":
            return False

    local_result = evaluate_model(group.get("local"), "LOCAL")
    groq_result = evaluate_model(group.get("groq"), "GROQ")

    existing[qid] = {
        "local": local_result or {},
        "groq": groq_result or {},
    }

    save_json(existing, QUALITY_SCORES_FILE)

    print("\n✓ Saved scores for", qid)
    input("Press Enter to continue to the next question...")
    return True


def print_final_summary(scores: Dict[str, Any]) -> None:
    """Print a compact quality summary after evaluation."""
    print()
    print_separator()
    print("QUALITY EVALUATION SUMMARY")
    print_separator()

    for model in ("local", "groq"):
        print(f"\n{model.upper()}")

        dimensions = [
            ("Relevance", "relevance"),
            ("Financial Reasoning", "reasoning"),
            ("Personalization", "personalization"),
            ("Grounding", "grounding"),
            ("Completeness", "completeness"),
            ("Factuality", "factuality"),
            ("Overall", "overall"),
        ]

        values = []

        for label, key in dimensions:
            collected = [
                q[model][key]
                for q in scores.values()
                if q.get(model, {}).get(key) is not None
            ]

            if collected:
                avg = sum(collected) / len(collected)
                values.append(avg)
                print(f"  {label:<22}: {avg:.2f}/5")

        hallucination_counts = [
            q[model].get("hallucination_count")
            for q in scores.values()
            if isinstance(q.get(model, {}).get("hallucination_count"), int)
        ]

        if hallucination_counts:
            questions_with_hallucination = sum(
                1 for value in hallucination_counts if value > 0
            )
            rate = questions_with_hallucination / len(hallucination_counts)
            print(f"  {'Hallucination rate':<22}: {rate:.1%}")
            print(f"  {'Total hallucinations':<22}: {sum(hallucination_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Local-vs-Groq quality evaluator."
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Evaluate a specific benchmark run number when raw_results has repeated runs.",
    )
    parser.add_argument(
        "--hide-context",
        action="store_true",
        help="Do not display retrieved research context.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-evaluate every question even if scores already exist.",
    )
    args = parser.parse_args()

    if not RAW_RESULTS_FILE.exists():
        print(f"✗ Raw results not found: {RAW_RESULTS_FILE}")
        print("Run the benchmark first.")
        sys.exit(1)

    try:
        raw = load_json(RAW_RESULTS_FILE)
    except json.JSONDecodeError as exc:
        print(f"✗ Invalid raw_results.json: {exc}")
        sys.exit(1)

    groups = build_question_groups(raw, args.run)

    if not groups:
        print("✗ No benchmark questions found.")
        sys.exit(1)

    existing_scores = load_existing_scores()

    # Optionally hide context without changing the stored data.
    if args.hide_context:
        for group in groups.values():
            group["retrieval"] = {}

    print_separator()
    print("WTF INTELLIGENCE LAYER — INTERACTIVE QUALITY EVALUATOR")
    print_separator()
    print(f"Raw results:     {RAW_RESULTS_FILE}")
    print(f"Quality output:  {QUALITY_SCORES_FILE}")
    print(f"Questions found: {len(groups)}")

    if args.run is not None:
        print(f"Selected run:    {args.run}")
    else:
        print("Selected run:    first successful response per question/model")

    print()
    print("You will score Local and Groq separately for each question.")
    print("Questions with COMPLETE scores are skipped automatically.")
    print("Blank/template entries (including null scores) will still be evaluated.")
    print("Scores are saved after every completed question.")
    print()

    input("Press Enter to start...")

    completed = 0
    skipped = 0

    for qid, group in groups.items():
        # Skip questions already evaluated unless --all is specified.
        # Only skip a question when it has COMPLETE scores for both models.
        # This is important because quality_scores.json may be a blank template
        # containing Q1..Q10 with null values.
        if not args.all and is_question_scored(existing_scores.get(qid)):
            skipped += 1
            continue

        # Both model responses are required for a fair comparison.
        if not group.get("local") or not group.get("groq"):
            print(f"⚠ Skipping {qid}: Local or Groq response is missing.")
            skipped += 1
            continue

        if evaluate_question(qid, group, existing_scores):
            completed += 1

    clear_screen()
    print_final_summary(existing_scores)

    print()
    print(f"✓ Evaluation session complete.")
    print(f"  Newly evaluated: {completed}")
    print(f"  Skipped:         {skipped}")
    print(f"  Saved to:        {QUALITY_SCORES_FILE}")
    print()
    print("Next step:")
    print("    python -m benchmark.evaluate")


if __name__ == "__main__":
    main()
