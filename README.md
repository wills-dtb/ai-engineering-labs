# AI Engineering Labs — Retrieval, Agents, Inference

Three runnable Python engineering starter projects exploring reliable AI systems rather than simple API demos.

| Lab | Covers |
| --- | --- |
| [01 — Citation-aware RAG](01-rag) | PDF/text ingestion, word chunking, local embeddings, cosine retrieval, citation IDs and retrieval recall |
| [02 — Guarded agent orchestration](02-agents) | Simulated 40-service migration, JSON plan validation, tool allowlist, 3-worker execution, failure injection and idempotency |
| [03 — Model serving benchmarks](03-serving) | vLLM/OpenAI-compatible serving, streaming TTFT, inter-chunk latency, concurrent throughput and error reporting |

## Tests

Each lab contains Python code, a README and unit tests. In any lab directory, run:

    python -m pip install -r requirements-test.txt
    python -m pytest -q

GitHub Actions runs the suites on pushes and pull requests.

## What this does not prove yet

These are educational implementation baselines, **not** proof of production deployment. RAG citation IDs do not guarantee grounded answers. Agent migrations are simulations with no real service mutations. LLM throughput values must come from real hardware benchmarks, not sample screenshots.

## Milestones for CV-ready evidence

1. Add a labelled held-out evaluation corpus; report retrieval recall@k, unsupported answers and generation faithfulness.
2. Add durable agent state, audit trails, recovery and failure-injection scenarios.
3. Benchmark a real model on documented GPU hardware at multiple concurrency levels and report TTFT p50/p95 and output tokens/sec.

See each lab README for commands and limitations.
