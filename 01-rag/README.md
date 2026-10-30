# 01: Citation-aware RAG

A small but genuine retrieval pipeline: read Markdown, text or PDFs; chunk by words; embed locally with Sentence Transformers; cosine-retrieve evidence; optionally generate an answer and require valid numbered citations.

## Setup

Run from this directory:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m rag_lab.cli build --docs docs
    python -m rag_lab.cli ask --question "How long do refunds take?"

Optional answer generation:

    export OPENAI_API_KEY="..."
    export RAG_MODEL="your-model-id"
    python -m rag_lab.cli ask --question "How long do refunds take?" --generate

Evaluate source retrieval against labelled questions:

    python -m rag_lab.cli evaluate --dataset eval.json --k 3

Tests:

    pip install -r requirements-test.txt
    python -m pytest -q

## Limitations

Word-based chunks are not model tokens. Citation validation confirms reference IDs only, not semantic truth. Source recall does not measure generation faithfulness. Add human-labelled answers, unsupported-question evaluations, prompt-injection defences, and proper experiment logs before calling this production-ready.
