# 02: Guarded multi-agent orchestration

A simulated 40-service migration driven by validated JSON actions, three worker threads, dry-run support, duplicate suppression, and injectable failures.

## Setup

Run from this directory:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

A dry-run makes no state changes:

    python -m agent_lab.cli --services 40 --workers 3

Execute an in-memory simulation:

    python -m agent_lab.cli --services 40 --workers 3 --execute-simulation --out report.json

Inject failure:

    python -m agent_lab.cli --services 40 --execute-simulation --fail-service service-002

Optionally request a JSON plan from an LLM:

    export OPENAI_API_KEY="..."
    export AGENT_MODEL="your-model-id"
    python -m agent_lab.cli --llm-plan

## Safety and limitations

No shell execution, real infrastructure access, or live migration. Model output is only an untrusted proposed plan, checked against an allowlist before any simulation. This is a single-process, in-memory demo; a real workflow needs durable state, authentication, approvals, retries, cancellation, and audit logs. The explicit concurrency setting shows orchestration, not independent autonomous agents.

Run tests:

    python -m pytest -q
