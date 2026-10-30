from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
import httpx


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    pos = (len(data) - 1) * pct / 100
    lo, hi = int(pos), min(len(data) - 1, int(pos) + 1)
    return round(data[lo] + (data[hi] - data[lo]) * (pos - lo), 4)


def parse_sse(events: list[str]) -> tuple[int | None, list[str]]:
    total = None
    contents = []
    for item in events:
        if item.strip() == "[DONE]":
            continue
        payload = json.loads(item)
        if isinstance(payload.get("usage"), dict):
            count = payload["usage"].get("completion_tokens")
            if isinstance(count, int) and count >= 0:
                total = count
        for choice in payload.get("choices", []):
            content = (choice.get("delta") or {}).get("content")
            if isinstance(content, str) and content:
                contents.append(content)
    return total, contents


async def measure_one(client: httpx.AsyncClient, model: str, question: str, clock=time.perf_counter) -> dict:
    started = clock()
    first_text_at = None
    chunk_timestamps = []
    completion_tokens = None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 96,
        "temperature": 0
    }
    try:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = line[5:].strip()
                if event == "[DONE]":
                    break
                parsed = json.loads(event)
                usage = parsed.get("usage")
                if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                    completion_tokens = usage["completion_tokens"]
                for choice in parsed.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content")
                    if isinstance(delta, str) and delta:
                        moment = clock()
                        if first_text_at is None:
                            first_text_at = moment
                        chunk_timestamps.append(moment)
        ended = clock()
        gaps = [b - a for a, b in zip(chunk_timestamps, chunk_timestamps[1:])]
        duration = ended - started
        return {
            "ok": first_text_at is not None,
            "ttft_ms": round((first_text_at - started) * 1000, 3) if first_text_at else None,
            "mean_inter_chunk_ms": round(statistics.mean(gaps) * 1000, 3) if gaps else None,
            "duration_s": round(duration, 4),
            "completion_tokens": completion_tokens,
            "output_tokens_per_s": round(completion_tokens / duration, 2) if completion_tokens is not None and duration > 0 else None,
            "content_chunks": len(chunk_timestamps)
        }
    except (httpx.HTTPError, ValueError, KeyError) as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


async def run_benchmark(base_url: str, api_key: str, model: str, requests: int, concurrency: int) -> dict:
    if not 1 <= requests <= 1000 or not 1 <= concurrency <= 100:
        raise ValueError("requests and concurrency out of budget")
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(120.0)
    ) as client:
        async def task(i):
            async with semaphore:
                return await measure_one(client, model, f"Explain idempotency in two sentences. Request {i}.")
        started = time.perf_counter()
        outcomes = await asyncio.gather(*(task(i) for i in range(requests)))
        wall = time.perf_counter() - started
    successes = [x for x in outcomes if x["ok"]]
    tokens = [x["completion_tokens"] for x in successes]
    return {
        "model": model,
        "base_url": base_url,
        "requests": requests,
        "concurrency": concurrency,
        "successes": len(successes),
        "failures": len(outcomes) - len(successes),
        "wall_time_s": round(wall, 4),
        "ttft_p50_ms": percentile([x["ttft_ms"] for x in successes], 50),
        "ttft_p95_ms": percentile([x["ttft_ms"] for x in successes], 95),
        "mean_inter_chunk_p50_ms": percentile([x["mean_inter_chunk_ms"] for x in successes if x["mean_inter_chunk_ms"] is not None], 50),
        "output_tokens_per_s_over_wall_time": round(sum(tokens) / wall, 2) if successes and all(t is not None for t in tokens) and wall > 0 else None,
        "results": outcomes
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results = asyncio.run(run_benchmark(
        os.getenv("INFERENCE_BASE_URL", "http://localhost:8000"),
        os.getenv("INFERENCE_API_KEY", "EMPTY"),
        args.model, args.requests, args.concurrency
    ))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
