"""Unit tests for traj_generation.build_sequences.

Covers:
- Skip logic: specs with non-empty `sequences` are left alone (no LLM call).
- Reliability filter: `eval_only=True` (default) excludes tools without an
  audit entry; `min_reliability` excludes audited-but-low tools; `no-eval-only`
  keeps unaudited tools.
- Post-processing: unknown tool names are dropped, duplicates dropped,
  separator-style mismatches tolerated.
- Write-back: sequences land in the env_spec JSON on disk.
- Field mode iterates matching specs; single-spec mode targets one file.
- Too-few tools: empty sequences returned, no crash.
- The `n_sequences` / `seq_length` values flow into the LLM prompt.
"""

import json
from pathlib import Path

import pytest

from traj_generation.build_sequences import build_sequences


def _sequences_response(seqs: dict) -> str:
    return "```json\n" + json.dumps(seqs) + "\n```"


def _make_tool(name: str) -> dict:
    return {
        "tool_name": name,
        "tool_description": f"desc for {name}",
        "parameters": {"x": {"type": "string", "required": True}},
        "error_messages": [],
        "usage": "usage",
        "output_details": {"y": {"type": "string"}},
    }


def _make_spec(
    spec_id: str,
    field: str,
    tool_names_with_rel: list,
    sequences: dict = None,
    subfield: str = "S",
    task: str = "T",
) -> dict:
    """Build a minimal env_spec with tools and an audit block.

    tool_names_with_rel: list of (tool_name, reliability_or_None_or_Missing).
        Use the sentinel "MISSING" for "no audit entry for this tool".
    """
    tools = []
    audit_tools = []
    for name, rel in tool_names_with_rel:
        tools.append(_make_tool(name))
        if rel == "MISSING":
            continue
        audit_tools.append({
            "id": f"{spec_id}.{name}",
            "tool_name": name,
            "reliability": rel,
            "reliability_per_mode": {},
            "n_test_calls": 9,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "evaluated_at": "2026-04-20T00:00:00Z",
        })
    return {
        "schema_version": "env_spec.v2",
        "spec_id": spec_id,
        "field": field,
        "subfield": subfield,
        "task": task,
        "generated_at": "2026-04-20T00:00:00Z",
        "model": "fake-model",
        "model_config": {},
        "tools": tools,
        "sequences": sequences if sequences is not None else {},
        "generation_log": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "field_elapsed_s": 0.0,
        "audit": {
            "schema_version": "audit.v1",
            "last_evaluated_at": "2026-04-20T00:00:00Z",
            "model": "fake-model",
            "model_config": {},
            "tools": audit_tools,
            "aggregate": {
                "n_tools_total": len(tools),
                "n_tools_evaluated": len(audit_tools),
                "n_tools_pending": 0,
                "mean_reliability": None,
                "usage_total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        },
    }


def _write_spec(dir_: Path, spec: dict) -> Path:
    p = dir_ / f"{spec['spec_id']}.json"
    p.write_text(json.dumps(spec, indent=2))
    return p


def _read_spec(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------

def test_build_sequences_skips_if_present(fake_llm, tmp_output):
    """If sequences dict is non-empty, the LLM is never called and the file is untouched."""
    spec = _make_spec(
        "myfield_spec_000", "MyField",
        [("Alpha", 1.0), ("Beta", 1.0)],
        sequences={"seq1": ["Alpha", "Beta"]},
    )
    path = _write_spec(tmp_output, spec)
    mtime_before = path.stat().st_mtime

    build_sequences(
        env_specs_dir=tmp_output,
        llm=fake_llm,
        spec_path=path,
        n_sequences=3,
        seq_length=2,
    )

    # No LLM call, file content preserved
    assert fake_llm.calls == []
    assert _read_spec(path)["sequences"] == {"seq1": ["Alpha", "Beta"]}


# ---------------------------------------------------------------------------
# Reliability filter
# ---------------------------------------------------------------------------

def test_build_sequences_filters_unevaluated_when_eval_only(fake_llm, tmp_output):
    """Default eval_only=True drops tools missing an audit entry OR with reliability=None."""
    spec = _make_spec(
        "f_spec_000", "F",
        [("Alpha", 1.0), ("Beta", None), ("Gamma", "MISSING"), ("Delta", 1.0)],
    )
    path = _write_spec(tmp_output, spec)

    # LLM returns a sequence using only tools that should have passed the filter
    fake_llm.queue(_sequences_response({"seq1": ["Alpha", "Delta"]}))

    build_sequences(
        env_specs_dir=tmp_output,
        llm=fake_llm,
        spec_path=path,
        n_sequences=1,
        seq_length=2,
    )

    # Inspect the prompt actually sent — only Alpha + Delta appear in tool list
    assert len(fake_llm.calls) == 1
    prompt = fake_llm.calls[0]["messages"]
    # The role passes a string directly to the runner; normalize for assertion
    prompt_text = prompt if isinstance(prompt, str) else prompt[0]["content"]
    assert "Alpha" in prompt_text
    assert "Delta" in prompt_text
    assert '"tool_name": "Beta"' not in prompt_text
    assert '"tool_name": "Gamma"' not in prompt_text

    saved = _read_spec(path)
    assert saved["sequences"] == {"seq1": ["Alpha", "Delta"]}


def test_build_sequences_filters_by_min_reliability(fake_llm, tmp_output):
    """Tools with reliability below min_reliability are dropped."""
    spec = _make_spec(
        "f_spec_001", "F",
        [("Alpha", 0.9), ("Beta", 0.3), ("Gamma", 0.7)],
    )
    path = _write_spec(tmp_output, spec)

    fake_llm.queue(_sequences_response({"seq1": ["Alpha", "Gamma"]}))

    build_sequences(
        env_specs_dir=tmp_output,
        llm=fake_llm,
        spec_path=path,
        n_sequences=1,
        seq_length=2,
        min_reliability=0.5,
    )

    prompt_text = fake_llm.calls[0]["messages"]
    prompt_text = prompt_text if isinstance(prompt_text, str) else prompt_text[0]["content"]
    assert "Alpha" in prompt_text
    assert "Gamma" in prompt_text
    assert '"tool_name": "Beta"' not in prompt_text


def test_build_sequences_no_eval_only(fake_llm, tmp_output):
    """With eval_only=False, unaudited tools pass the filter."""
    spec = _make_spec(
        "f_spec_002", "F",
        [("Alpha", 1.0), ("Beta", "MISSING"), ("Gamma", None)],
    )
    path = _write_spec(tmp_output, spec)

    fake_llm.queue(_sequences_response({"seq1": ["Alpha", "Beta", "Gamma"]}))

    build_sequences(
        env_specs_dir=tmp_output,
        llm=fake_llm,
        spec_path=path,
        n_sequences=1,
        seq_length=3,
        eval_only=False,
    )

    prompt_text = fake_llm.calls[0]["messages"]
    prompt_text = prompt_text if isinstance(prompt_text, str) else prompt_text[0]["content"]
    assert "Alpha" in prompt_text
    assert "Beta" in prompt_text
    assert "Gamma" in prompt_text

    assert _read_spec(path)["sequences"] == {"seq1": ["Alpha", "Beta", "Gamma"]}


# ---------------------------------------------------------------------------
# Too few tools: safe no-op
# ---------------------------------------------------------------------------

def test_build_sequences_empty_when_filter_too_strict(fake_llm, tmp_output):
    """Filter leaves fewer tools than seq_length → write empty sequences, no LLM call."""
    spec = _make_spec(
        "f_spec_003", "F",
        [("Alpha", 0.1), ("Beta", 0.2)],
    )
    path = _write_spec(tmp_output, spec)

    build_sequences(
        env_specs_dir=tmp_output,
        llm=fake_llm,
        spec_path=path,
        n_sequences=5,
        seq_length=6,   # > filtered tools count
        min_reliability=0.5,
    )

    assert fake_llm.calls == []
    assert _read_spec(path)["sequences"] == {}


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def test_build_sequences_drops_unknown_tool_names(fake_llm, tmp_output):
    """Sequences referencing a tool not in the filtered pool are dropped."""
    spec = _make_spec(
        "f_spec_004", "F",
        [("Alpha", 1.0), ("Beta", 1.0)],
    )
    path = _write_spec(tmp_output, spec)

    seqs = {
        "seq1": ["Alpha", "Beta"],           # keeper
        "seq2": ["Alpha", "Gamma"],          # Gamma not in pool → dropped
        "seq3": ["Alpha", "Beta"],           # exact duplicate of seq1 → dropped
        "seq4": ["Alpha", "Alpha", "Beta"],  # consecutive dup → dropped
    }
    fake_llm.queue(_sequences_response(seqs))

    build_sequences(
        env_specs_dir=tmp_output,
        llm=fake_llm,
        spec_path=path,
        n_sequences=4,
        seq_length=2,
    )

    assert _read_spec(path)["sequences"] == {"seq1": ["Alpha", "Beta"]}


def test_build_sequences_tolerates_separator_style_mismatch(fake_llm, tmp_output):
    """Tools declared with hyphens/spaces match bare PascalCase in sequences."""
    spec = _make_spec(
        "f_spec_005", "F",
        [("Trade-off Analyzer", 1.0), ("Cost Estimator", 1.0)],
    )
    path = _write_spec(tmp_output, spec)

    seqs = {
        "seq1": ["TradeoffAnalyzer", "CostEstimator"],  # normalized match
        "seq2": ["TotallyFake", "CostEstimator"],        # genuinely unknown
    }
    fake_llm.queue(_sequences_response(seqs))

    build_sequences(
        env_specs_dir=tmp_output,
        llm=fake_llm,
        spec_path=path,
        n_sequences=2,
        seq_length=2,
    )

    assert set(_read_spec(path)["sequences"].keys()) == {"seq1"}


def test_build_sequences_threads_n_sequences_and_seq_length_to_prompt(fake_llm, tmp_output):
    """CLI knobs reach the LLM prompt (regression for seqs_per_spec / seq_length)."""
    tools = [(f"Tool{i}", 1.0) for i in range(15)]
    spec = _make_spec("f_spec_006", "F", tools)
    path = _write_spec(tmp_output, spec)

    fake_llm.queue(_sequences_response({"seq1": ["Tool0"] * 11}))  # will be dropped (consec dup), fine

    build_sequences(
        env_specs_dir=tmp_output,
        llm=fake_llm,
        spec_path=path,
        n_sequences=7,
        seq_length=11,
    )

    prompt_text = fake_llm.calls[0]["messages"]
    prompt_text = prompt_text if isinstance(prompt_text, str) else prompt_text[0]["content"]
    assert "Produce **7 sequences**" in prompt_text
    assert "length 11" in prompt_text
    assert "seq7" in prompt_text


# ---------------------------------------------------------------------------
# Field mode
# ---------------------------------------------------------------------------

def test_build_sequences_field_mode_iterates_matching_specs(fake_llm, tmp_output):
    """Field mode processes every scenario whose `field` matches in one batched call."""
    spec_a = _make_spec("field_a_spec_000", "FieldA", [("Alpha", 1.0), ("Beta", 1.0)])
    spec_b = _make_spec("field_a_spec_001", "FieldA", [("Alpha", 1.0), ("Beta", 1.0)])
    spec_c = _make_spec("field_b_spec_000", "FieldB", [("Gamma", 1.0), ("Delta", 1.0)])
    for s in [spec_a, spec_b, spec_c]:
        _write_spec(tmp_output, s)

    # Two FieldA specs are batched into one LLM call
    fake_llm.queue_batch([
        _sequences_response({"seq1": ["Alpha", "Beta"]}),
        _sequences_response({"seq1": ["Beta", "Alpha"]}),
    ])

    build_sequences(
        env_specs_dir=tmp_output,
        llm=fake_llm,
        field="FieldA",
        n_sequences=1,
        seq_length=2,
    )

    # Exactly ONE LLM call (batched), not N per spec
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["batched"] is True
    assert len(fake_llm.calls[0]["messages"]) == 2

    a = _read_spec(tmp_output / "field_a_spec_000.json")
    b = _read_spec(tmp_output / "field_a_spec_001.json")
    c = _read_spec(tmp_output / "field_b_spec_000.json")

    assert a["sequences"] == {"seq1": ["Alpha", "Beta"]}
    assert b["sequences"] == {"seq1": ["Beta", "Alpha"]}
    assert c["sequences"] == {}  # untouched


def test_build_sequences_records_usage_in_generation_log(fake_llm, tmp_output):
    """Each spec gets a `sequences_build` entry in generation_log with prompt/response/usage."""
    spec = _make_spec("f_spec_007", "F", [("Alpha", 1.0), ("Beta", 1.0)])
    path = _write_spec(tmp_output, spec)

    fake_llm.queue(_sequences_response({"seq1": ["Alpha", "Beta"]}))

    build_sequences(
        env_specs_dir=tmp_output,
        llm=fake_llm,
        spec_path=path,
        n_sequences=3,
        seq_length=2,
        min_reliability=0.5,
        eval_only=True,
    )

    saved = _read_spec(path)
    entries = [e for e in saved["generation_log"] if e.get("phase") == "sequences_build"]
    assert len(entries) == 1
    entry = entries[0]

    # Recorded filter + request settings
    assert entry["n_sequences_requested"] == 3
    assert entry["seq_length"] == 2
    assert entry["min_reliability"] == 0.5
    assert entry["eval_only"] is True
    assert entry["n_tools_filtered"] == 2
    assert entry["n_tools_total"] == 2

    # Recorded provenance
    assert entry["model"] == "fake-model"
    assert "model_config" in entry
    assert entry["generated_at"].endswith("Z")
    assert "Alpha" in entry["prompt"]
    assert "Beta" in entry["prompt"]
    assert "seq1" in entry["response"]

    # Token usage present (FakeLLM: 10 prompt + 20 completion for a single call)
    assert entry["usage"]["prompt_tokens"] == 10
    assert entry["usage"]["completion_tokens"] == 20


def test_build_sequences_batched_across_field_records_per_request_usage(fake_llm, tmp_output):
    """In a batched call, each spec's log entry gets its own per-request usage (not a split of the aggregate)."""
    spec_a = _make_spec("field_a_spec_000", "FieldA", [("Alpha", 1.0), ("Beta", 1.0)])
    spec_b = _make_spec("field_a_spec_001", "FieldA", [("Gamma", 1.0), ("Delta", 1.0)])
    for s in [spec_a, spec_b]:
        _write_spec(tmp_output, s)

    fake_llm.queue_batch([
        _sequences_response({"seq1": ["Alpha", "Beta"]}),
        _sequences_response({"seq1": ["Gamma", "Delta"]}),
    ])

    build_sequences(
        env_specs_dir=tmp_output,
        llm=fake_llm,
        field="FieldA",
        n_sequences=1,
        seq_length=2,
    )

    # FakeLLM assigns per-request usage of (10+i, 20+i) for i in 0..N-1
    a = _read_spec(tmp_output / "field_a_spec_000.json")
    b = _read_spec(tmp_output / "field_a_spec_001.json")
    ua = [e for e in a["generation_log"] if e.get("phase") == "sequences_build"][0]["usage"]
    ub = [e for e in b["generation_log"] if e.get("phase") == "sequences_build"][0]["usage"]
    assert (ua["prompt_tokens"], ua["completion_tokens"]) == (10, 20)
    assert (ub["prompt_tokens"], ub["completion_tokens"]) == (11, 21)


def test_build_sequences_field_mode_skips_already_populated(fake_llm, tmp_output):
    """Field mode skips specs that already have sequences."""
    done = _make_spec(
        "field_a_spec_000", "FieldA", [("Alpha", 1.0), ("Beta", 1.0)],
        sequences={"seq1": ["Alpha", "Beta"]},
    )
    todo = _make_spec("field_a_spec_001", "FieldA", [("Alpha", 1.0), ("Beta", 1.0)])
    for s in [done, todo]:
        _write_spec(tmp_output, s)

    fake_llm.queue(_sequences_response({"seq1": ["Beta", "Alpha"]}))

    build_sequences(
        env_specs_dir=tmp_output,
        llm=fake_llm,
        field="FieldA",
        n_sequences=1,
        seq_length=2,
    )

    # Exactly one call — the already-done spec was skipped
    assert len(fake_llm.calls) == 1
    assert _read_spec(tmp_output / "field_a_spec_000.json")["sequences"] == {"seq1": ["Alpha", "Beta"]}
    assert _read_spec(tmp_output / "field_a_spec_001.json")["sequences"] == {"seq1": ["Beta", "Alpha"]}


def test_build_sequences_no_filter_means_all_specs(fake_llm, tmp_output):
    """Calling with neither spec_path nor field iterates every spec in the dir."""
    s1 = _make_spec("a_spec_000", "A", [("Alpha", 1.0), ("Beta", 1.0)])
    s2 = _make_spec("b_spec_000", "B", [("Gamma", 1.0), ("Delta", 1.0)])
    _write_spec(tmp_output, s1)
    _write_spec(tmp_output, s2)
    fake_llm.queue_batch([
        json.dumps({"seq1": ["Alpha", "Beta"]}),
        json.dumps({"seq1": ["Gamma", "Delta"]}),
    ])
    out = build_sequences(env_specs_dir=tmp_output, llm=fake_llm, seq_length=2)
    assert len(out) == 2


def test_build_sequences_rejects_both_spec_and_field(fake_llm, tmp_output):
    spec = _make_spec("f_spec_000", "F", [("Alpha", 1.0), ("Beta", 1.0)])
    path = _write_spec(tmp_output, spec)
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_sequences(
            env_specs_dir=tmp_output,
            llm=fake_llm,
            spec_path=path,
            field="F",
        )
