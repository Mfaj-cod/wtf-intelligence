import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from statistics import mean, median, stdev


def load_raw_results(results_file: str = "benchmark/results/raw_results.json") -> Dict[str, Any]:
    """Load raw benchmark results."""
    with open(results_file) as f:
        return json.load(f)


def load_quality_scores(
    quality_file: str = "benchmark/results/quality_scores.json",
) -> Dict[str, Any]:
    """Load quality scores keyed by question ID."""
    with open(quality_file, encoding="utf-8") as f:
        return json.load(f)


def calculate_quality_metrics(
    quality_scores: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate 1-5 quality scores and hallucination metrics."""
    dimensions = [
        "relevance",
        "reasoning",
        "personalization",
        "grounding",
        "completeness",
        "factuality",
        "overall",
    ]

    quality = {"local": {}, "groq": {}}

    for model in ("local", "groq"):
        for dimension in dimensions:
            values = [
                float(q[model][dimension])
                for q in quality_scores.values()
                if isinstance(q.get(model, {}).get(dimension), (int, float))
            ]
            if values:
                quality[model][f"mean_{dimension}"] = mean(values)

        hallucination_counts = [
            float(q[model]["hallucination_count"])
            for q in quality_scores.values()
            if isinstance(q.get(model, {}).get("hallucination_count"), (int, float))
        ]

        if hallucination_counts:
            quality[model]["total_hallucinations"] = sum(hallucination_counts)
            quality[model]["hallucination_rate"] = (
                sum(1 for x in hallucination_counts if x > 0)
                / len(hallucination_counts)
            )

    return quality


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile value."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int((percentile / 100.0) * len(sorted_vals))
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def analyze_results(
    raw: Dict[str, Any],
    quality_scores: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze benchmark results and calculate statistics."""
    results = raw.get("results", [])

    analysis = {
        "summary": {
            "total_questions": raw["benchmark"]["questions"],
            "runs_per_question": raw["benchmark"]["runs_per_question"],
            "total_results": len(results),
        },
        "performance": {
            "local": {},
            "groq": {},
        },
        "quality": {
            "local": {},
            "groq": {},
        },
        "question_comparison": [],
    }

    # Collect latencies for each model
    local_latencies = []
    groq_latencies = []
    local_tokens = []
    groq_tokens = []

    for result in results:
        if result["local"]:
            if not result["local"].get("error"):
                local_latencies.append(result["local"]["latency_ms"])
                if result["local"].get("completion_tokens"):
                    local_tokens.append(result["local"]["completion_tokens"])

        if result["groq"]:
            if not result["groq"].get("error"):
                groq_latencies.append(result["groq"]["latency_ms"])
                if result["groq"].get("completion_tokens"):
                    groq_tokens.append(result["groq"]["completion_tokens"])

    # Performance metrics for local
    if local_latencies:
        analysis["performance"]["local"] = {
            "mean_latency_ms": mean(local_latencies),
            "median_latency_ms": median(local_latencies),
            "min_latency_ms": min(local_latencies),
            "max_latency_ms": max(local_latencies),
            "p95_latency_ms": calculate_percentile(local_latencies, 95),
            "std_dev_latency_ms": stdev(local_latencies) if len(local_latencies) > 1 else 0,
        }
        if local_tokens:
            analysis["performance"]["local"]["mean_tokens"] = mean(local_tokens)
            analysis["performance"]["local"]["mean_tokens_per_second"] = (
                mean(local_tokens) / (mean(local_latencies) / 1000)
            )

    # Performance metrics for Groq
    if groq_latencies:
        analysis["performance"]["groq"] = {
            "mean_latency_ms": mean(groq_latencies),
            "median_latency_ms": median(groq_latencies),
            "min_latency_ms": min(groq_latencies),
            "max_latency_ms": max(groq_latencies),
            "p95_latency_ms": calculate_percentile(groq_latencies, 95),
            "std_dev_latency_ms": stdev(groq_latencies) if len(groq_latencies) > 1 else 0,
        }
        if groq_tokens:
            analysis["performance"]["groq"]["mean_tokens"] = mean(groq_tokens)
            analysis["performance"]["groq"]["mean_tokens_per_second"] = (
                mean(groq_tokens) / (mean(groq_latencies) / 1000)
            )

    # Quality metrics
    if quality_scores:
        analysis["quality"] = calculate_quality_metrics(quality_scores)

    # Question-level comparison
    questions_seen = set()
    for result in results:
        qid = result["question_id"]
        if qid not in questions_seen:
            questions_seen.add(qid)
            comparison = {
                "question_id": qid,
                "question": result["question"],
                "local_latency_ms": result["local"]["latency_ms"] if result["local"] else None,
                "groq_latency_ms": result["groq"]["latency_ms"] if result["groq"] else None,
                "local_error": result["local"].get("error") if result["local"] else None,
                "groq_error": result["groq"].get("error") if result["groq"] else None,
            }
            analysis["question_comparison"].append(comparison)

    return analysis


def generate_summary_table(analysis: Dict[str, Any]) -> str:
    """Generate summary comparison table."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("BENCHMARK SUMMARY — Local LLM vs Groq GPT-OSS 120B")
    lines.append("=" * 100)

    perf_local = analysis["performance"]["local"]
    perf_groq = analysis["performance"]["groq"]

    # Metrics to display
    metrics = [
        ("Mean Latency (ms)", "mean_latency_ms"),
        ("Median Latency (ms)", "median_latency_ms"),
        ("P95 Latency (ms)", "p95_latency_ms"),
        ("Min Latency (ms)", "min_latency_ms"),
        ("Max Latency (ms)", "max_latency_ms"),
        ("Mean Output Tokens", "mean_tokens"),
        ("Tokens/Second", "mean_tokens_per_second"),
    ]

    lines.append(f"\n{'Metric':<30} {'Local LLM':>30} {'Groq GPT-OSS 120B':>30}")
    lines.append("-" * 100)

    for label, key in metrics:
        local_val = perf_local.get(key)
        groq_val = perf_groq.get(key)

        local_str = f"{local_val:.1f}" if local_val is not None else "—"
        groq_str = f"{groq_val:.1f}" if groq_val is not None else "—"

        lines.append(f"{label:<30} {local_str:>30} {groq_str:>30}")

    lines.append("=" * 100)

    return "\n".join(lines)


def generate_quality_table(analysis: Dict[str, Any]) -> str:
    """Generate quality comparison table."""
    lines = [
        "\n" + "=" * 100,
        "QUALITY SUMMARY — Local LLM vs Groq GPT-OSS 120B",
        "=" * 100,
    ]

    local = analysis["quality"]["local"]
    groq = analysis["quality"]["groq"]

    metrics = [
        ("Relevance", "mean_relevance"),
        ("Financial Reasoning", "mean_reasoning"),
        ("Personalization", "mean_personalization"),
        ("Research Grounding", "mean_grounding"),
        ("Completeness", "mean_completeness"),
        ("Factuality", "mean_factuality"),
        ("Overall Quality", "mean_overall"),
    ]

    lines.append(f"\n{'Metric':<30} {'Local LLM':>20} {'Groq GPT-OSS 120B':>25}")
    lines.append("-" * 100)

    for label, key in metrics:
        lv = local.get(key)
        gv = groq.get(key)
        ls = f"{lv:.2f}" if lv is not None else "—"
        gs = f"{gv:.2f}" if gv is not None else "—"
        lines.append(f"{label:<30} {ls:>20} {gs:>25}")

    lh = local.get("hallucination_rate")
    gh = groq.get("hallucination_rate")
    ls = f"{lh:.1%}" if lh is not None else "—"
    gs = f"{gh:.1%}" if gh is not None else "—"
    lines.append(f"{'Hallucination Rate':<30} {ls:>20} {gs:>25}")

    lines.append("=" * 100)
    return "\n".join(lines)


def generate_question_comparison(analysis: Dict[str, Any]) -> str:
    """Generate question-by-question comparison."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("QUESTION-BY-QUESTION LATENCY COMPARISON")
    lines.append("=" * 100)

    lines.append(f"\n{'QID':<5} {'Local (ms)':>20} {'Groq (ms)':>20} {'Question':<50}")
    lines.append("-" * 100)

    for comp in analysis["question_comparison"]:
        qid = comp["question_id"]
        local_ms = comp["local_latency_ms"]
        groq_ms = comp["groq_latency_ms"]
        question = comp["question"][:47] + "..." if len(comp["question"]) > 50 else comp["question"]

        local_str = f"{local_ms:.0f}" if local_ms else "error"
        groq_str = f"{groq_ms:.0f}" if groq_ms else "error"

        lines.append(f"{qid:<5} {local_str:>20} {groq_str:>20} {question:<50}")

    lines.append("=" * 100)
    return "\n".join(lines)


def save_analysis(analysis: Dict[str, Any], output_file: str = "benchmark/results/analysis.json") -> None:
    """Save analysis to JSON."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Analysis saved to {output_file}")


def generate_report(analysis: Dict[str, Any], report_file: str = "benchmark/results/benchmark_report.md") -> None:
    """Generate markdown report."""
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Benchmark Report — WTF Intelligence Layer\n")
    lines.append("## Executive Summary\n")
    lines.append(
        f"This report compares the **local self-hosted LLM** (Ollama) against **Groq's GPT-OSS 120B** "
        f"across {analysis['summary']['total_questions']} investment advisory questions.\n"
    )

    lines.append("## Latency Results\n")
    perf_local = analysis["performance"]["local"]
    perf_groq = analysis["performance"]["groq"]

    lines.append("### Local LLM (Ollama)\n")
    if perf_local:
        lines.append(f"- Mean latency: {perf_local.get('mean_latency_ms', '—'):.1f} ms\n")
        lines.append(f"- Median latency: {perf_local.get('median_latency_ms', '—'):.1f} ms\n")
        lines.append(f"- P95 latency: {perf_local.get('p95_latency_ms', '—'):.1f} ms\n")

    lines.append("### Groq GPT-OSS 120B\n")
    if perf_groq:
        lines.append(f"- Mean latency: {perf_groq.get('mean_latency_ms', '—'):.1f} ms\n")
        lines.append(f"- Median latency: {perf_groq.get('median_latency_ms', '—'):.1f} ms\n")
        lines.append(f"- P95 latency: {perf_groq.get('p95_latency_ms', '—'):.1f} ms\n")

    lines.append("## Token Throughput\n")
    if perf_local.get("mean_tokens_per_second"):
        lines.append(f"- Local: {perf_local['mean_tokens_per_second']:.1f} tokens/second\n")
    if perf_groq.get("mean_tokens_per_second"):
        lines.append(f"- Groq: {perf_groq['mean_tokens_per_second']:.1f} tokens/second\n")

    lines.append("## Quality Comparison\n")
    lines.append("| Quality Dimension | Local LLM | Groq GPT-OSS 120B |\n")
    lines.append("|---|---:|---:|\n")

    quality_rows = [
        ("Relevance", "mean_relevance"),
        ("Financial Reasoning", "mean_reasoning"),
        ("Personalization", "mean_personalization"),
        ("Research Grounding", "mean_grounding"),
        ("Completeness", "mean_completeness"),
        ("Factuality", "mean_factuality"),
        ("Overall Quality", "mean_overall"),
    ]

    for label, key in quality_rows:
        lv = analysis["quality"]["local"].get(key)
        gv = analysis["quality"]["groq"].get(key)
        ls = f"{lv:.2f}" if lv is not None else "—"
        gs = f"{gv:.2f}" if gv is not None else "—"
        lines.append(f"| {label} | {ls} | {gs} |\n")

    lh = analysis["quality"]["local"].get("hallucination_rate")
    gh = analysis["quality"]["groq"].get("hallucination_rate")
    ls = f"{lh:.1%}" if lh is not None else "—"
    gs = f"{gh:.1%}" if gh is not None else "—"
    lines.append(f"| Hallucination Rate | {ls} | {gs} |\n")
    lines.append("\nQuality scores are based on the predefined 1–5 evaluation rubric.\n")

    lines.append("## Question Results\n")
    for comp in analysis["question_comparison"]:
        lines.append(f"### Q{comp['question_id']}: {comp['question']}\n")
        lines.append(f"- Local: {comp['local_latency_ms']:.0f}ms\n" if comp["local_latency_ms"] else "- Local: error\n")
        lines.append(f"- Groq: {comp['groq_latency_ms']:.0f}ms\n" if comp["groq_latency_ms"] else "- Groq: error\n")
        lines.append("\n")

    lines.append("## Next Steps\n")
    lines.append("- Review human evaluation scores (when completed)\n")
    lines.append("- Analyze grounding and factuality\n")
    lines.append("- Calculate cost trade-offs\n")

    with open(report_file, "w") as f:
        f.write("".join(lines))

    print(f"✓ Report saved to {report_file}")


def main():
    """Main evaluation entry point."""
    print("=" * 100)
    print("WTF Intelligence Layer — Benchmark Analysis")
    print("=" * 100)

    try:
        raw = load_raw_results()
        print(f"✓ Loaded {len(raw['results'])} raw results")
    except FileNotFoundError:
        print("✗ raw_results.json not found. Run benchmark first: python -m benchmark.run_benchmark")
        sys.exit(1)

    quality_file = "benchmark/results/quality_scores.json"

    try:
        quality_scores = load_quality_scores(quality_file)
        print(f"✓ Loaded quality scores for {len(quality_scores)} questions")
    except FileNotFoundError:
        quality_scores = {}
        print(
            f"⚠ {quality_file} not found. "
            "Performance analysis will run, but quality scores will remain empty."
        )
    except json.JSONDecodeError as e:
        print(f"✗ Invalid quality_scores.json: {e}")
        sys.exit(1)

    analysis = analyze_results(raw, quality_scores)
    save_analysis(analysis)
    generate_report(analysis)

    print(generate_summary_table(analysis))
    print(generate_quality_table(analysis))
    print(generate_question_comparison(analysis))

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()