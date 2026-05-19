"""Tests for utils.run_parallel — the process-pool helper shared by
task_generation and trajectory_generation.

Real ProcessPoolExecutor is awkward in tests (worker fns must be top-level
picklable, FakeLLM has state). Instead we test:
  - serial path (concurrency <= 1) runs in-process, in input order
  - on_result callback fires for every item with running (done, total) counts
  - per-item exceptions are swallowed into {"error": "..."} dicts
  - empty input returns []
"""

import pytest

from utils import run_parallel


def _double(x):
    return x * 2


def _raises_on_three(x):
    if x == 3:
        raise ValueError("three is bad")
    return x * 10


def test_run_parallel_empty():
    assert run_parallel([], _double, concurrency=4) == []


def test_run_parallel_serial_preserves_order():
    """concurrency=1 runs in-process; results in input order."""
    out = run_parallel([1, 2, 3, 4, 5], _double, concurrency=1)
    assert out == [2, 4, 6, 8, 10]


def test_run_parallel_serial_invokes_callback():
    """on_result called once per item with (done_count, total, result)."""
    log = []
    run_parallel(
        [1, 2, 3],
        _double,
        concurrency=1,
        on_result=lambda done, total, r: log.append((done, total, r)),
    )
    assert log == [(1, 3, 2), (2, 3, 4), (3, 3, 6)]


def test_run_parallel_serial_swallows_exceptions():
    """A failing item yields a {'error': ...} dict; the rest still run."""
    out = run_parallel([1, 2, 3, 4], _raises_on_three, concurrency=1)
    assert out[0] == 10
    assert out[1] == 20
    assert isinstance(out[2], dict) and "ValueError" in out[2]["error"]
    assert out[2]["item"] == "3"
    assert out[3] == 40
