import asyncio
import json
import httpx
from serving_lab.bench import measure_one, parse_sse, percentile


def test_percentiles():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == 4.8


def test_parse_sse_handles_usage_and_content():
    data = ['{"choices":[{"delta":{"content":"Hello"}}]}', '{"choices":[],"usage":{"completion_tokens":2}}', '[DONE]']
    assert parse_sse(data) == (2, ["Hello"])


def test_streaming_measurement_with_mock_transport():
    body = '\n\n'.join([
        'data: ' + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
        'data: ' + json.dumps({"choices": [{"delta": {"content": " there"}}]}),
        'data: ' + json.dumps({"choices": [], "usage": {"completion_tokens": 2}}),
        'data: [DONE]',
    ]) + '\n\n'

    async def invoke():
        async def handler(request):
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(200, content=body.encode(), headers={"content-type":"text/event-stream"})
        async with httpx.AsyncClient(base_url="http://example.org", transport=httpx.MockTransport(handler)) as client:
            return await measure_one(client, "fake-model", "Hello")
    result = asyncio.run(invoke())
    assert result["ok"] is True
    assert result["completion_tokens"] == 2
    assert result["content_chunks"] == 2
    assert result["ttft_ms"] is not None
