# Benchmark Phase — Local LLM vs Groq

## 0. All Commands — Run These First

> Run these commands from the **project root**.

### Install / verify dependencies

```bash
pip install -r requirements.txt
```

Verify Ollama:

```bash
ollama list
```

Expected models:

```text
llama3.2:latest
nomic-embed-text:latest
```

If Ollama is not running:

```bash
ollama serve
```

Check the local API:

```bash
curl http://127.0.0.1:11434/api/tags
```

### Set Groq API key

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

Windows CMD:

```cmd
set GROQ_API_KEY=your_groq_api_key
```

Linux/macOS:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

Or put it in `.env` if the project loads environment variables from there:

```env
GROQ_API_KEY=your_groq_api_key
```

### Run the benchmark

Run the retrieval snapshot:

```bash
python -m benchmark.retrieval_snapshot
```

Run the complete model benchmark:

```bash
python -m benchmark.run_benchmark
```

### Evaluate response quality

Run the interactive human evaluator:

```bash
python -m benchmark.quality_evaluator
```

Evaluate all questions again:

```bash
python -m benchmark.quality_evaluator --all
```

Evaluate a specific repeated run:

```bash
python -m benchmark.quality_evaluator --run 2
```

Hide the retrieved research context during evaluation:

```bash
python -m benchmark.quality_evaluator --hide-context
```

### Generate the final analysis and report

```bash
python -m benchmark.evaluate
```

### Recommended complete sequence

```bash
python -m benchmark.retrieval_snapshot
python -m benchmark.run_benchmark
python -m benchmark.quality_evaluator
python -m benchmark.evaluate
```

### Expected final outputs

```text
benchmark/results/
├── retrieval_snapshot.json
├── raw_results.json
├── quality_scores.json
├── analysis.json
└── benchmark_report.md
```

---

# 1. Benchmark Objective

This benchmark evaluates the project's **self-hosted financial AI system** against a commercial hosted model.

The primary comparison is:

| System | Model | Inference |
|---|---|---|
| Local | `llama3.2:latest` | Ollama |
| Commercial | `openai/gpt-oss-120b` | Groq API |

The objective is not simply to determine which model produces the better answer.

The benchmark evaluates:

1. **Response quality**
2. **Financial reasoning**
3. **Research grounding**
4. **Personalization**
5. **Completeness**
6. **Factuality**
7. **Hallucination**
8. **Inference latency**
9. **Token throughput**
10. **Operational cost**
11. **Privacy / deployment implications**

The final question is:

> Is the local model sufficiently capable, fast, private, and economical to justify self-hosted deployment for the intended financial-AI workload?

---

# 2. Benchmark Architecture

The benchmark uses a controlled A/B comparison.

```text
                         Benchmark Questions
                                │
                                ▼
                     Client Profile + Question
                                │
                                ▼
                       Research Retrieval
                                │
                                ▼
                     Retrieval Snapshot
                         ┌──────┴──────┐
                         │             │
                         ▼             ▼
                    Local Model    Groq Model
                    Ollama         API
                         │             │
                         ▼             ▼
                    Local Answer   Groq Answer
                         │             │
                         └──────┬──────┘
                                ▼
                     Performance Metrics
                                │
                                ▼
                       Human Evaluation
                                │
                                ▼
                         Final Analysis
                                │
                                ▼
                    benchmark_report.md
```

The same:

- question
- client profile
- system prompt
- retrieved research
- generation configuration

should be supplied to both models.

The model should be the primary experimental variable.

---

# 3. Benchmark Directory

```text
benchmark/
│
├── README.md
│
├── questions.json
│
├── retrieval_snapshot.py
│
├── run_benchmark.py
├── quality_evaluator.py
├── evaluate.py
│
├── models/
│   ├── base.py
│   ├── ollama_model.py
│   └── groq_model.py
│
└── results/
    ├── retrieval_snapshot.json
    ├── raw_results.json
    ├── quality_scores.json
    ├── analysis.json
    └── benchmark_report.md
```

---

# 4. File Responsibilities

## `questions.json`

Contains the fixed benchmark questions and associated client profiles.

The benchmark questions should not change between model evaluations.

Changing questions after collecting results invalidates direct comparison.

## `retrieval_snapshot.py`

Runs the RAG retrieval pipeline once and saves the retrieved research context.

The snapshot should contain the same retrieved chunks for both models.

This prevents the retrieval system from becoming an uncontrolled variable in the model comparison.

## `models/base.py`

Defines the common model interface.

Both model adapters should expose a consistent interface so that the benchmark runner does not need model-specific logic.

## `models/ollama_model.py`

Adapter for the locally hosted Ollama model.

Current target:

```text
llama3.2:latest
```

## `models/groq_model.py`

Adapter for the Groq-hosted model.

Current target:

```text
openai/gpt-oss-120b
```

## `run_benchmark.py`

Runs the actual generation benchmark.

It should:

1. Load benchmark questions.
2. Load the retrieval snapshot.
3. Build the final prompt.
4. Warm up the models.
5. Run each question through both models.
6. Measure latency.
7. Record token usage.
8. Save raw responses.

## `quality_evaluator.py`

Interactive human evaluation tool.

It displays:

- benchmark question
- client profile
- retrieved research
- Local response
- Groq response

It then asks the evaluator to score each response.

Scores are saved incrementally so an interrupted evaluation can be resumed.

## `evaluate.py`

Aggregates benchmark results.

It generates:

```text
analysis.json
benchmark_report.md
```

The report combines:

- performance measurements
- token statistics
- quality scores
- hallucination statistics
- model comparison
- conclusions

---

# 5. Models

## Local Model

```text
llama3.2:latest
```

Runtime:

```text
Ollama
```

The model runs locally without sending the benchmark prompt or retrieved financial research to an external inference provider.

## Comparison Model

```text
openai/gpt-oss-120b
```

Runtime:

```text
Groq API
```

The Groq model is used as the commercial/hosted comparison system.

The benchmark should use the same application prompt and retrieved context supplied to the local model.

---

# 6. Dataset / Benchmark Questions

The benchmark uses the predefined questions in:

```text
benchmark/questions.json
```

The questions represent the intended financial-advisor workload.

Each question should include the relevant client profile where applicable.

A profile can contain information such as:

```text
primary_goal
horizon_years
risk_score
investor_profile
current_holdings
```

Example:

```text
Primary goal: Build wealth
Horizon: 10 years
Risk score: 7
Investor profile: Growth
Current holdings: Stocks/ETFs + Crypto
```

The benchmark should use the exact stored questions rather than manually entering questions during execution.

---

# 7. RAG Methodology

The RAG pipeline retrieves research documents before generation.

The intended configuration is:

```text
Embedding model: nomic-embed-text
Vector database: ChromaDB
Top-k retrieval: 5
Chunk size: 512
Chunk overlap: 50
```

The retrieval pipeline is executed once for the benchmark dataset.

The resulting retrieval context is stored in:

```text
benchmark/results/retrieval_snapshot.json
```

---

# 8. Why Use a Retrieval Snapshot?

This is important for experimental fairness.

Without a retrieval snapshot:

```text
Question
   │
   ├── Local → Retrieval A → Local model
   │
   └── Groq  → Retrieval B → Groq model
```

A difference in the final answer could therefore be caused by:

- different retrieved documents
- different chunk ordering
- different retrieval latency
- different context

rather than model capability.

With a snapshot:

```text
Question
   │
   ▼
Retrieval Snapshot
   │
   ├──────────────► Local
   │
   └──────────────► Groq
```

Both models receive the same research context.

This isolates the model comparison more effectively.

---

# 9. Prompt Methodology

The benchmark should use the application's existing prompt construction pipeline.

The same:

- system instructions
- client profile
- question
- retrieved context
- output instructions

should be passed to both models.

Do not create a special prompt designed to favor one model.

The benchmark should evaluate the models under the same application conditions.

---

# 10. Benchmark Execution

## Step 1 — Generate Retrieval Snapshot

```bash
python -m benchmark.retrieval_snapshot
```

Verify:

```text
benchmark/results/retrieval_snapshot.json
```

contains the expected questions and retrieved chunks.

## Step 2 — Run Model Benchmark

```bash
python -m benchmark.run_benchmark
```

The benchmark should run the same questions against:

```text
Local → Ollama → llama3.2:latest

Groq → openai/gpt-oss-120b
```

---

# 11. Number of Runs

The intended benchmark configuration is:

```text
10 questions
×
3 measured runs
×
2 models
=
60 measured generations
```

Warm-up calls may be performed before measurement but should not be included in the final latency statistics.

If the runner is configured with:

```python
runs_per_question=1
```

then the current execution is only:

```text
10 × 1 × 2 = 20 measured generations
```

For the final benchmark, use:

```python
runs_per_question=3
```

unless the project configuration explicitly specifies another value.

---

# 12. Warm-Up

The first model request may include:

- model loading
- memory allocation
- runtime initialization
- connection initialization

Therefore the first request should not be treated as representative steady-state latency.

Recommended procedure:

```text
Warm-up
    ↓
Measured run 1
    ↓
Measured run 2
    ↓
Measured run 3
```

Warm-up results should not be included in aggregate latency statistics.

---

# 13. Performance Metrics

The benchmark should collect the following where available.

## Latency

Measure client-side wall-clock latency from:

```text
request start
        ↓
complete response received
```

Use a monotonic timer.

The same measurement approach should be used for both models.

Report:

```text
mean latency
median latency
minimum latency
maximum latency
```

Median latency is particularly useful because a single slow request can distort the mean.

## Token Metrics

Record:

```text
prompt_tokens
completion_tokens
total_tokens
tokens_per_second
```

Groq exposes usage statistics through the API response.

The Ollama adapter should extract equivalent usage information when available.

If a metric is genuinely unavailable, record:

```text
null
```

rather than inventing a value.

---

# 14. Retrieval Latency

Retrieval latency can be measured separately as:

```text
retrieval_ms
```

This measures the RAG retrieval stage.

It should not be confused with model inference latency.

Ideally report:

```text
Retrieval latency
Prompt construction latency
Model inference latency
End-to-end latency
```

If a metric is not currently captured by the implementation, do not fabricate it.

---

# 15. Raw Benchmark Results

The generation benchmark writes:

```text
benchmark/results/raw_results.json
```

This is the raw experimental record.

It should contain, where available:

```text
question ID
model
run number
response
latency
token counts
tokens/sec
retrieved sources
number of retrieved chunks
errors
```

Raw results should be preserved.

Do not manually modify benchmark outputs after collection.

---

# 16. Human Quality Evaluation

Performance metrics alone cannot determine which model is better.

The responses therefore require qualitative evaluation.

Run:

```bash
python -m benchmark.quality_evaluator
```

The evaluator presents each question and the corresponding Local/Groq responses.

Score each response from **1 to 5**.

---

# 17. Quality Rubric

## 17.1 Relevance

Does the answer directly address the user's question?

```text
1 = irrelevant
2 = weakly relevant
3 = partially relevant
4 = mostly relevant
5 = directly relevant
```

## 17.2 Financial Reasoning

Does the response demonstrate sound reasoning about the financial problem?

```text
1 = poor reasoning
2 = major reasoning gaps
3 = acceptable reasoning
4 = strong reasoning
5 = excellent reasoning
```

## 17.3 Personalization

Does the response correctly use the client's profile?

Consider:

- goal
- investment horizon
- risk tolerance
- holdings
- investor profile

```text
1 = ignores profile
2 = minimal personalization
3 = some personalization
4 = good personalization
5 = highly personalized
```

## 17.4 Research Grounding

Does the answer correctly use the retrieved research?

```text
1 = unsupported / contradicts context
2 = weak grounding
3 = partially grounded
4 = well grounded
5 = strongly grounded
```

## 17.5 Completeness

Does the response sufficiently cover the question without important omissions?

```text
1 = severely incomplete
2 = substantially incomplete
3 = adequate
4 = comprehensive
5 = highly comprehensive
```

## 17.6 Factuality

Are the claims accurate and consistent with the supplied context?

```text
1 = major factual errors
2 = several errors
3 = mostly correct
4 = very accurate
5 = highly accurate
```

## 17.7 Overall

Give an independent overall score:

```text
1 = unacceptable
2 = weak
3 = acceptable
4 = strong
5 = excellent
```

The overall score should be an evaluator judgment rather than simply the arithmetic average of the other dimensions.

---

# 18. Hallucination Evaluation

Also record:

```text
hallucination_count
hallucination_severity
```

Recommended severity scale:

```text
0 = none
1 = minor
2 = moderate
3 = severe
```

A hallucination is a claim presented as factual that is unsupported, fabricated, or materially inconsistent with the available research/context.

Do not count reasonable inference as hallucination merely because the exact sentence is not present in the retrieved documents.

---

# 19. Quality Scores File

The human evaluator writes:

```text
benchmark/results/quality_scores.json
```

Expected structure:

```json
{
  "Q1": {
    "local": {
      "relevance": 4,
      "reasoning": 4,
      "personalization": 4,
      "grounding": 4,
      "completeness": 4,
      "factuality": 4,
      "overall": 4,
      "hallucination_count": 0,
      "hallucination_severity": 0
    },
    "groq": {
      "relevance": 5,
      "reasoning": 5,
      "personalization": 5,
      "grounding": 5,
      "completeness": 5,
      "factuality": 5,
      "overall": 5,
      "hallucination_count": 0,
      "hallucination_severity": 0
    }
  }
}
```

The values above are only an example of the format.

**Do not use these example scores as actual benchmark results.**

The scores must come from the evaluation process.

---

# 20. Blind Evaluation Recommendation

For a stronger experiment, hide the model identity from the evaluator.

Instead of:

```text
LOCAL RESPONSE
GROQ RESPONSE
```

prefer:

```text
RESPONSE A
RESPONSE B
```

Then reveal the mapping after scoring.

This reduces evaluator bias.

If the current evaluator does not implement blind evaluation, the benchmark can still be performed, but this should be acknowledged as a limitation.

---

# 21. Quality Aggregation

After completing human evaluation:

```bash
python -m benchmark.evaluate
```

The evaluator aggregates the scores.

The main quality dimensions are:

```text
relevance
reasoning
personalization
grounding
completeness
factuality
```

The `overall` score should remain an independent evaluator metric.

Hallucination statistics should be reported separately.

---

# 22. Final Analysis

The evaluator produces:

```text
benchmark/results/analysis.json
```

This should contain aggregated statistics for both models.

Typical comparisons include:

```text
Average latency
Median latency
Average tokens/sec
Average quality
Dimension-wise quality
Average hallucination count
Hallucination rate
```

---

# 23. Final Benchmark Report

The evaluator also generates:

```text
benchmark/results/benchmark_report.md
```

Generate it with:

```bash
python -m benchmark.evaluate
```

The report should answer:

1. Which model produced better responses?
2. Which model was faster?
3. Which model had better token throughput?
4. Which model was better grounded in research?
5. Which model personalized responses better?
6. Which model hallucinated less?
7. What is the operational cost difference?
8. What are the privacy implications?
9. Is the local model sufficient for the intended use case?

---

# 24. Cost Analysis

The benchmark should distinguish between hosted and local inference costs.

## Hosted model cost

Groq has usage-based inference costs.

Calculate approximate benchmark cost using:

```text
input tokens
+
output tokens
+
provider pricing
```

Use the actual pricing applicable at the time of the benchmark.

## Local model cost

The local model has no per-request API charge.

However, "free" does not mean zero cost.

Relevant costs include:

- electricity
- hardware
- storage
- maintenance
- model hosting
- engineering time

Describe local inference as having **zero external inference API cost**, rather than literally zero total cost.

---

# 25. Privacy Comparison

One major reason for the local architecture is data control.

## Local inference

```text
User
 ↓
FastAPI
 ↓
Local RAG
 ↓
Ollama
 ↓
Local model
```

Sensitive prompts and retrieved research can remain within the deployment environment.

## Hosted inference

```text
User
 ↓
FastAPI
 ↓
RAG
 ↓
Groq API
 ↓
Hosted model
```

Prompts must be sent to the external inference provider.

The benchmark report should discuss this difference independently from raw model quality.

---

# 26. Reproducibility

For reproducible results, keep fixed:

```text
benchmark/questions.json
retrieval snapshot
system prompt
generation configuration
model versions
benchmark code
quality rubric
```

Do not change benchmark questions between model runs.

Do not change retrieved context for one model only.

Do not manually rewrite model responses before evaluation.

Do not mix warm-up requests with measured requests.

---

# 27. Fair Comparison Rules

The following rules should be treated as mandatory.

### Rule 1

Same question.

### Rule 2

Same client profile.

### Rule 3

Same retrieved research.

### Rule 4

Same application prompt.

### Rule 5

Same benchmark dataset.

### Rule 6

Equivalent generation configuration wherever the APIs permit it.

### Rule 7

Do not use tools for one model and not the other.

### Rule 8

Do not manually improve one model's response.

### Rule 9

Keep raw responses unchanged.

### Rule 10

Do not fabricate missing measurements.

---

# 28. Recommended Final Run

Before collecting final results:

```bash
python -m benchmark.retrieval_snapshot
```

Then configure the intended number of measured runs.

For the target benchmark:

```text
runs_per_question = 3
```

Then:

```bash
python -m benchmark.run_benchmark
```

After generation completes:

```bash
python -m benchmark.quality_evaluator
```

Complete the human scoring.

Finally:

```bash
python -m benchmark.evaluate
```

---

# 29. Sanity Checks

Before accepting the benchmark results, verify:

```text
[ ] All 10 questions exist
[ ] Retrieval snapshot exists
[ ] Same retrieval context is used by both models
[ ] Both models completed the same questions
[ ] No benchmark errors were silently ignored
[ ] Warm-up calls are excluded from measured statistics
[ ] Expected number of runs was executed
[ ] Ollama token measurements are populated where available
[ ] Groq token measurements are populated where available
[ ] Quality scores are completed
[ ] Hallucinations were evaluated
[ ] Raw results were preserved
[ ] Final analysis was generated
[ ] Final report was generated
```

---

# 30. Common Problems

## Groq API key missing

Set `GROQ_API_KEY`, then rerun the benchmark.

## Ollama connection error

Check:

```bash
ollama list
```

If necessary:

```bash
ollama serve
```

If Ollama is already running, do not start another server instance.

## Model missing

Check:

```bash
ollama list
```

The required local models should be present.

## Quality evaluation interrupted

Simply rerun:

```bash
python -m benchmark.quality_evaluator
```

The evaluator is designed to resume from saved scores.

## Re-evaluate every question

```bash
python -m benchmark.quality_evaluator --all
```

## Evaluate a specific run

```bash
python -m benchmark.quality_evaluator --run 2
```

## Hide research context

```bash
python -m benchmark.quality_evaluator --hide-context
```

---

# 31. Benchmark Limitations

The final report should acknowledge limitations rather than presenting the benchmark as universally conclusive.

Potential limitations include:

- only 10 benchmark questions
- one primary local model
- one hosted comparison model
- limited hardware
- small human evaluation sample
- subjective quality scoring
- possible evaluator bias
- model-version changes over time
- provider-side API/network variability
- differences in model serving infrastructure
- limited measurement of full production workload behavior

The benchmark is therefore an engineering evaluation of the proposed architecture, not a universal ranking of the two models.

---

# 32. Production Decision Framework

The final decision should not be based on a single metric.

## Prefer the local model when

- quality is acceptable
- latency is acceptable
- hallucination rate is manageable
- privacy is important
- external API dependency is undesirable
- infrastructure cost is acceptable

## Prefer the hosted model when

- significantly higher reasoning quality is required
- latency is acceptable
- external data processing is acceptable
- API cost is justified
- operational simplicity is more important than self-hosting

## Consider a hybrid architecture when

- local inference handles sensitive/default workloads
- hosted inference handles complex requests
- routing is based on confidence or task complexity
- privacy requirements allow selective external inference

---

# 33. Definition of Done

The benchmark phase is complete when:

```text
[✓] Benchmark questions finalized
[✓] Retrieval snapshot generated
[✓] Local model benchmark executed
[✓] Groq benchmark executed
[✓] Required repeated runs completed
[✓] Latency recorded
[✓] Token usage recorded where available
[✓] Raw responses saved
[✓] Human quality evaluation completed
[✓] Hallucination evaluation completed
[✓] analysis.json generated
[✓] benchmark_report.md generated
[✓] Cost analysis completed
[✓] Privacy comparison completed
[✓] Limitations documented
[✓] Production recommendation written
```

---

# 34. Final Deliverables

The benchmark phase should produce:

```text
benchmark/
└── results/
    ├── retrieval_snapshot.json
    ├── raw_results.json
    ├── quality_scores.json
    ├── analysis.json
    └── benchmark_report.md
```

The two most important artifacts are:

```text
analysis.json
```

for machine-readable results, and:

```text
benchmark_report.md
```

for the final human-readable engineering report.

---

# 35. One-Command Summary

After the environment and API key are configured, the intended workflow is:

```bash
python -m benchmark.retrieval_snapshot
python -m benchmark.run_benchmark
python -m benchmark.quality_evaluator
python -m benchmark.evaluate
```

Then inspect:

```text
benchmark/results/analysis.json
benchmark/results/benchmark_report.md
```

These files constitute the final benchmark analysis.
