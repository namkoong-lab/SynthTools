"""Tests for LLM(server_url=...) — the OpenAI-compatible HTTP branch.

We mock `openai.OpenAI` so no real server is needed. The mock returns a
`ChatCompletion`-shaped object with `.choices[0].message.content` and
`.usage.{prompt_tokens, completion_tokens}` set, mirroring what `vllm serve
--reasoning-parser openai_gptoss` actually returns over HTTP.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm import LLM, Usage


def _make_mock_response(content: str, prompt_tokens: int, completion_tokens: int):
    """Build a SimpleNamespace mirroring openai.types.chat.ChatCompletion."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _make_mock_client(responses):
    """Mock OpenAI client. `responses` is a list of (content, p_tok, c_tok) tuples."""
    client = MagicMock()
    if isinstance(responses, list):
        client.chat.completions.create.side_effect = [
            _make_mock_response(c, p, ct) for (c, p, ct) in responses
        ]
    else:
        c, p, ct = responses
        client.chat.completions.create.return_value = _make_mock_response(c, p, ct)
    return client


def test_server_url_single_message_returns_string():
    mock_client = _make_mock_client(("hello world", 12, 5))
    with patch("openai.OpenAI", return_value=mock_client):
        llm = LLM("Qwen3-32B", server_url="http://fake/v1")
        out = llm([{"role": "user", "content": "hi"}])
    assert out == "hello world"
    assert mock_client.chat.completions.create.call_count == 1


def test_server_url_string_input_auto_wrapped():
    mock_client = _make_mock_client(("response", 4, 2))
    with patch("openai.OpenAI", return_value=mock_client):
        llm = LLM("Qwen3-32B", server_url="http://fake/v1")
        out = llm("hi")
    assert out == "response"
    sent = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert sent == [{"role": "user", "content": "hi"}]


def test_server_url_sets_last_usage_from_response():
    mock_client = _make_mock_client(("ok", 12, 5))
    with patch("openai.OpenAI", return_value=mock_client):
        llm = LLM("Qwen3-32B", server_url="http://fake/v1")
        llm([{"role": "user", "content": "x"}])
    assert llm.last_usage == Usage(prompt_tokens=12, completion_tokens=5)
    assert llm.last_usage_per_request is None


def test_server_url_batched_returns_list():
    mock_client = _make_mock_client([
        ("a-resp", 10, 3),
        ("b-resp", 20, 4),
        ("c-resp", 30, 5),
    ])
    with patch("openai.OpenAI", return_value=mock_client):
        llm = LLM("Qwen3-32B", server_url="http://fake/v1")
        out = llm([
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
            [{"role": "user", "content": "c"}],
        ])
    assert sorted(out) == sorted(["a-resp", "b-resp", "c-resp"])
    assert mock_client.chat.completions.create.call_count == 3


def test_server_url_sets_last_usage_per_request_for_batch():
    mock_client = _make_mock_client([
        ("a", 10, 3),
        ("b", 20, 4),
    ])
    with patch("openai.OpenAI", return_value=mock_client):
        llm = LLM("Qwen3-32B", server_url="http://fake/v1")
        llm([
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ])
    assert llm.last_usage_per_request is not None
    assert len(llm.last_usage_per_request) == 2
    total_p = sum(u.prompt_tokens for u in llm.last_usage_per_request)
    total_c = sum(u.completion_tokens for u in llm.last_usage_per_request)
    assert llm.last_usage == Usage(prompt_tokens=total_p, completion_tokens=total_c)
    assert total_p == 30
    assert total_c == 7


def test_server_url_does_not_load_vllm():
    """Constructing + calling LLM with server_url should never import vllm."""
    # Drop any cached vllm import so we can detect re-import cleanly.
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("vllm")}
    try:
        mock_client = _make_mock_client(("ok", 1, 1))
        with patch("openai.OpenAI", return_value=mock_client):
            llm = LLM("GPT-OSS-120B", server_url="http://fake/v1")
            llm([{"role": "user", "content": "hi"}])
        assert not any(k.startswith("vllm") for k in sys.modules), \
            f"vllm got imported: {[k for k in sys.modules if k.startswith('vllm')]}"
    finally:
        # Restore so other tests aren't affected.
        sys.modules.update(saved)


def test_server_url_uses_model_id_from_registry():
    """The HTTP request must send cfg.model_id, not the registry key."""
    mock_client = _make_mock_client(("ok", 1, 1))
    with patch("openai.OpenAI", return_value=mock_client):
        llm = LLM("GPT-OSS-120B", server_url="http://fake/v1")
        llm([{"role": "user", "content": "hi"}])
    sent_model = mock_client.chat.completions.create.call_args.kwargs["model"]
    assert sent_model == "openai/gpt-oss-120b"


def test_server_url_passes_sampling_params():
    mock_client = _make_mock_client(("ok", 1, 1))
    with patch("openai.OpenAI", return_value=mock_client):
        llm = LLM("Qwen3-32B", server_url="http://fake/v1", max_tokens=999)
        llm([{"role": "user", "content": "hi"}])
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 999
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.95


def test_server_url_client_is_lazy_and_cached():
    """The OpenAI client is constructed once and reused."""
    mock_client = _make_mock_client([("a", 1, 1), ("b", 1, 1)])
    with patch("openai.OpenAI", return_value=mock_client) as mock_ctor:
        llm = LLM("Qwen3-32B", server_url="http://fake/v1")
        # Not constructed at __init__
        assert mock_ctor.call_count == 0
        llm([{"role": "user", "content": "hi"}])
        llm([{"role": "user", "content": "hi"}])
        assert mock_ctor.call_count == 1


def test_in_process_branch_unchanged_when_server_url_none():
    """Sanity: with server_url=None, no openai client is created and __repr__ matches old shape."""
    with patch("openai.OpenAI") as mock_ctor:
        llm = LLM("Qwen3-32B")  # no server_url
        assert llm.server_url is None
        assert "not loaded" in repr(llm)
        assert mock_ctor.call_count == 0
