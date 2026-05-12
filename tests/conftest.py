"""Shared test fixtures: a FakeLLM that doesn't need vLLM loaded."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20


class FakeLLM:
    """In-memory LLM for unit tests. Queue responses; they pop in call order.

    Queue a plain string for a single-message call:
        llm.queue("the response")
    Queue a list for a batched call (one string per request):
        llm.queue_batch(["resp1", "resp2", "resp3"])

    Each recorded entry in `self.calls` has: {"messages": ..., "batched": bool}.
    """

    def __init__(self, model: str = "fake-model"):
        self.model = model
        self._queue: List[Union[str, List[str]]] = []
        self.calls: List[dict] = []
        self.last_usage: Optional[FakeUsage] = None
        self.last_usage_per_request: Optional[List[FakeUsage]] = None

    # Queue helpers -----------------------------------------------------
    def queue(self, response: str) -> None:
        self._queue.append(response)

    def queue_batch(self, responses: List[str]) -> None:
        self._queue.append(list(responses))

    # vLLM-compat no-op
    def _ensure_engine(self):
        pass

    # Main call ---------------------------------------------------------
    def __call__(self, messages):
        if not self._queue:
            raise AssertionError("FakeLLM queue empty — did you forget to queue a response?")
        next_response = self._queue.pop(0)

        batched = isinstance(messages, list) and messages and isinstance(messages[0], list)
        self.calls.append({"messages": messages, "batched": batched})

        if batched:
            if not isinstance(next_response, list):
                raise AssertionError(
                    f"Batched call expected a list of responses but got str: {next_response!r}"
                )
            if len(next_response) != len(messages):
                raise AssertionError(
                    f"Batched call: got {len(messages)} messages but {len(next_response)} queued responses"
                )
            self.last_usage_per_request = [FakeUsage(10 + i, 20 + i) for i in range(len(next_response))]
            p = sum(u.prompt_tokens for u in self.last_usage_per_request)
            c = sum(u.completion_tokens for u in self.last_usage_per_request)
            self.last_usage = FakeUsage(p, c)
            return next_response

        if isinstance(next_response, list):
            raise AssertionError(
                "Single call expected a string response but got a list (did you use queue_batch?)"
            )
        self.last_usage = FakeUsage(10, 20)
        self.last_usage_per_request = None
        return next_response


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    d = tmp_path / "out"
    d.mkdir()
    return d
