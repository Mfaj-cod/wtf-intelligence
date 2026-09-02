# Benchmark Report — WTF Intelligence Layer
## Executive Summary
This report compares the **local self-hosted LLM** (Ollama) against **Groq's GPT-OSS 120B** across 10 investment advisory questions.
## Latency Results
### Local LLM (Ollama)
- Mean latency: 75238.3 ms
- Median latency: 78540.5 ms
- P95 latency: 99494.5 ms
### Groq GPT-OSS 120B
- Mean latency: 1884.1 ms
- Median latency: 1814.0 ms
- P95 latency: 2313.9 ms
## Token Throughput
- Local: 5.5 tokens/second
- Groq: 265.4 tokens/second
## Quality Comparison
| Quality Dimension | Local LLM | Groq GPT-OSS 120B |
|---|---:|---:|
| Relevance | 4.60 | 4.80 |
| Financial Reasoning | 3.60 | 4.80 |
| Personalization | 4.20 | 4.80 |
| Research Grounding | 3.90 | 4.30 |
| Completeness | 4.30 | 2.30 |
| Factuality | 3.50 | 4.30 |
| Overall Quality | 3.80 | 3.80 |
| Hallucination Rate | 80.0% | 50.0% |

Quality scores are based on the predefined 1–5 evaluation rubric.
## Question Results
### Q1: What should a growth investor with a 10-year horizon be aware of in current market conditions?
- Local: 59541ms
- Groq: 1769ms

### Q2: Summarise the key risks for a conservative investor holding mostly cash right now.
- Local: 77748ms
- Groq: 2051ms

### Q3: What asset allocation would suit someone who wants to retire in 5 years with a balanced risk profile?
- Local: 83544ms
- Groq: 1738ms

### Q4: How should someone with crypto holdings think about portfolio diversification?
- Local: 98861ms
- Groq: 1926ms

### Q5: What does the current interest rate environment mean for a first-time investor?
- Local: 83470ms
- Groq: 1961ms

### Q6: What are the most important questions an advisor should ask in a first client meeting?
- Local: 87450ms
- Groq: 1831ms

### Q7: Explain the difference between a growth and income investment strategy in plain English.
- Local: 90134ms
- Groq: 1970ms

### Q8: What red flags should an advisor look for in a new client's financial situation?
- Local: 88777ms
- Groq: 1999ms

### Q9: How does inflation affect a long-term wealth-building strategy?
- Local: 81197ms
- Groq: 1683ms

### Q10: What does a Balanced investor profile mean in terms of expected returns and drawdowns?
- Local: 84140ms
- Groq: 1797ms

## Next Steps
- Review human evaluation scores (when completed)
- Analyze grounding and factuality
- Calculate cost trade-offs
