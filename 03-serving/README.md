# 03: Open-model inference

Benchmark an OpenAI-compatible model server for time to first text (TTFT), stream inter-chunk latency, concurrency, and completion-token throughput when usage data is available.

## Setup

Start vLLM with a supported model on a GPU with sufficient VRAM. Example:

    pip install vllm
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000

In another environment:

    pip install -r requirements.txt
    export INFERENCE_BASE_URL="http://localhost:8000"
    export INFERENCE_API_KEY="EMPTY"
    python -m serving_lab.bench --model Qwen/Qwen2.5-0.5B-Instruct --requests 20 --concurrency 4 --out result.json

Run tests without a GPU:

    python -m pytest -q

## Interpretation

A streamed chunk is not necessarily a token. The reported interval is therefore *inter-chunk* latency, not inter-token latency. The benchmark does not establish real-world model performance until the model, GPU, quantization, dataset, and measured result are recorded. No fake metrics are included.
