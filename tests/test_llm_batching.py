"""Unit tests for LLM batched vs single usage tracking.

We don't load a real vLLM engine. Instead we monkey-patch `_ensure_engine`
to return a fake engine whose `.chat()` returns deterministic per-request results.
"""

from types import SimpleNamespace

import pytest

from llm import LLM, Usage


def _make_fake_result(prompt_ids, completion_ids, text="resp"):
    return SimpleNamespace(
        prompt_token_ids=prompt_ids,
        outputs=[SimpleNamespace(token_ids=completion_ids, text=text)],
    )


class _FakeEngine:
    def __init__(self, results):
        self._results = results

    def chat(self, messages, sampling_params=None, use_tqdm=True):
        return self._results


def _build_llm_with_fake_engine(results):
    llm = LLM("GPT-OSS-20B")
    llm._engine = _FakeEngine(results)
    llm._ensure_engine = lambda: llm._engine  # no-op wrapper
    return llm


def test_single_call_sets_last_usage_and_clears_per_request():
    fake = _make_fake_result([1, 2, 3], [9, 9])
    llm = _build_llm_with_fake_engine([fake])
    out = llm([{"role": "user", "content": "hi"}])
    assert out == "resp"
    assert llm.last_usage == Usage(prompt_tokens=3, completion_tokens=2)
    assert llm.last_usage_per_request is None


def test_batched_call_populates_per_request():
    results = [
        _make_fake_result([1, 2, 3, 4], [1, 2], text="a"),
        _make_fake_result([5, 6], [1, 2, 3, 4, 5], text="b"),
        _make_fake_result([7], [9], text="c"),
    ]
    llm = _build_llm_with_fake_engine(results)
    out = llm([
        [{"role": "user", "content": "q1"}],
        [{"role": "user", "content": "q2"}],
        [{"role": "user", "content": "q3"}],
    ])
    assert out == ["a", "b", "c"]
    assert llm.last_usage_per_request == [
        Usage(4, 2),
        Usage(2, 5),
        Usage(1, 1),
    ]
    assert llm.last_usage == Usage(prompt_tokens=7, completion_tokens=8)


def test_batched_after_single_resets_per_request_on_next_single():
    """Order: batched then single. `last_usage_per_request` must reset to None on the single call."""
    batched = [_make_fake_result([1], [1], "a"), _make_fake_result([2], [2], "b")]
    single = _make_fake_result([9, 9, 9], [3, 3], "s")

    llm = _build_llm_with_fake_engine(batched)
    llm([[{"role": "user", "content": "q1"}], [{"role": "user", "content": "q2"}]])
    assert llm.last_usage_per_request is not None

    llm._engine = _FakeEngine([single])
    llm([{"role": "user", "content": "q"}])
    assert llm.last_usage_per_request is None
    assert llm.last_usage == Usage(3, 2)
