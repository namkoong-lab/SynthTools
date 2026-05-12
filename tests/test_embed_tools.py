"""Tests for env_audit/embed_tools.py — one NPZ per tool, keyed by id."""

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pytest

from env_audit.embed_tools import build_tool_text, embed_tools


class StubEncoder:
    """Returns deterministic vectors. Records every batch it sees in self.calls."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: List[Sequence[str]] = []

    def encode_batch(self, texts):
        self.calls.append(list(texts))
        out = []
        for i, _ in enumerate(texts):
            v = [0.0] * self.dim
            v[i % self.dim] = 1.0
            out.append(v)
        return out


def _write_dataset(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_dataset(path: Path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _load_npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return data["embedding"]


# ---------------------------------------------------------------------------
# build_tool_text — prose renderer
# ---------------------------------------------------------------------------

def test_build_tool_text_includes_all_fields():
    tool = {
        "tool_name": "ReturnRequestValidator",
        "tool_description": "Reports the status of an existing return.",
        "parameters": {
            "return_request_id": {
                "type": "string", "required": True,
                "description": "The unique return request ID.",
            },
        },
        "usage": "Pass a known RET ID to confirm approval before downstream.",
        "output_details": {
            "status": {"type": "string", "description": "approved / rejected / pending"},
        },
        "error_messages": [
            "404 if the return request does not exist.",
            "400 for malformed IDs.",
        ],
    }
    text = build_tool_text(tool)
    assert "Tool: ReturnRequestValidator" in text
    assert "Description: Reports the status" in text
    assert "return_request_id (string, required)" in text
    assert "Usage: Pass a known RET ID" in text
    assert "Output:" in text
    assert "status (string)" in text
    assert "Possible errors:" in text
    assert "404 if the return request does not exist." in text


def test_build_tool_text_renders_optional_param_flag():
    tool = {"tool_name": "X",
            "parameters": {"limit": {"type": "integer", "required": False, "description": "Max"}}}
    assert "limit (integer, optional)" in build_tool_text(tool)


def test_build_tool_text_handles_empty_tool():
    assert build_tool_text({}) == ""


def test_build_tool_text_does_not_dump_json():
    tool = {"tool_name": "X", "tool_description": "y",
            "parameters": {"a": {"type": "string", "required": True}}}
    text = build_tool_text(tool)
    assert "{" not in text
    assert '"' not in text


# ---------------------------------------------------------------------------
# embed_tools — one NPZ per tool
# ---------------------------------------------------------------------------

def test_embed_tools_writes_one_npz_per_tool(tmp_path: Path):
    path = tmp_path / "tools.jsonl"
    _write_dataset(path, [
        {"id": "a", "tool": {"tool_name": "ToolA"}},
        {"id": "b", "tool": {"tool_name": "ToolB"}},
        {"id": "c", "tool": {"tool_name": "ToolC"}},
    ])
    enc = StubEncoder(dim=4)
    counts = embed_tools(path, model_id="stub", target_dim=4, encoder=enc)
    assert counts["embedded"] == 3
    assert counts["skipped"] == 0

    out_dir = tmp_path / "embeddings"
    files = sorted(p.name for p in out_dir.iterdir())
    assert files == ["a.npz", "b.npz", "c.npz", "meta.json"]

    for tid in ("a", "b", "c"):
        emb = _load_npz(out_dir / f"{tid}.npz")
        assert emb.shape == (4,)
        assert emb.dtype == np.float32


def test_embed_tools_npz_filename_matches_tool_id(tmp_path: Path):
    """Tool ids contain dots (spec_id.tool_name) — must round-trip in filename."""
    path = tmp_path / "tools.jsonl"
    tool_id = "aerospace_and_defense_spec_000.ParseMissionRequirements"
    _write_dataset(path, [{"id": tool_id, "tool": {"tool_name": "ParseMissionRequirements"}}])
    embed_tools(path, model_id="stub", target_dim=4, encoder=StubEncoder(4))
    assert (tmp_path / "embeddings" / f"{tool_id}.npz").exists()


def test_embed_tools_does_not_pollute_jsonl(tmp_path: Path):
    path = tmp_path / "tools.jsonl"
    _write_dataset(path, [
        {"id": "a", "field": "X", "tool": {"tool_name": "A"}, "reliability": 0.8},
    ])
    embed_tools(path, model_id="stub", target_dim=4, encoder=StubEncoder(4))
    rows = _load_dataset(path)
    r = rows[0]
    assert "embedding" not in r
    assert "embedding_model" not in r
    assert "embedding_dim" not in r
    assert r["id"] == "a"
    assert r["field"] == "X"
    assert r["reliability"] == 0.8


def test_embed_tools_skip_existing_npz(tmp_path: Path):
    path = tmp_path / "tools.jsonl"
    _write_dataset(path, [
        {"id": "a", "tool": {"tool_name": "A"}},
        {"id": "b", "tool": {"tool_name": "B"}},
    ])
    enc = StubEncoder(dim=4)
    embed_tools(path, model_id="stub", target_dim=4, encoder=enc)
    # add c, re-run
    _write_dataset(path, [
        {"id": "a", "tool": {"tool_name": "A"}},
        {"id": "b", "tool": {"tool_name": "B"}},
        {"id": "c", "tool": {"tool_name": "C"}},
    ])
    enc2 = StubEncoder(dim=4)
    counts = embed_tools(path, model_id="stub", target_dim=4, encoder=enc2)
    assert counts == {**counts, "total": 3, "skipped": 2, "embedded": 1}
    assert len(enc2.calls) == 1 and len(enc2.calls[0]) == 1
    assert (tmp_path / "embeddings" / "c.npz").exists()


def test_embed_tools_resume_preserves_existing_vectors(tmp_path: Path):
    path = tmp_path / "tools.jsonl"
    _write_dataset(path, [{"id": "a", "tool": {"tool_name": "A"}}])
    embed_tools(path, model_id="stub", target_dim=4, encoder=StubEncoder(4))
    a_before = _load_npz(tmp_path / "embeddings" / "a.npz")

    _write_dataset(path, [
        {"id": "a", "tool": {"tool_name": "A"}},
        {"id": "b", "tool": {"tool_name": "B"}},
    ])
    embed_tools(path, model_id="stub", target_dim=4, encoder=StubEncoder(4))

    a_after = _load_npz(tmp_path / "embeddings" / "a.npz")
    np.testing.assert_array_equal(a_after, a_before)


def test_embed_tools_no_pending_still_refreshes_meta(tmp_path: Path):
    path = tmp_path / "tools.jsonl"
    _write_dataset(path, [{"id": "a", "tool": {"tool_name": "A"}}])
    embed_tools(path, model_id="stub", target_dim=4, encoder=StubEncoder(4))
    enc2 = StubEncoder(dim=4)
    counts = embed_tools(path, model_id="stub", target_dim=4, encoder=enc2)
    assert counts == {**counts, "total": 1, "skipped": 1, "embedded": 0}
    # encoder not invoked when there's nothing to do
    assert len(enc2.calls) == 0
    # meta still updated (n_tools equals total npz files)
    meta = json.loads((tmp_path / "embeddings" / "meta.json").read_text())
    assert meta["n_tools"] == 1


def test_embed_tools_meta_uses_relative_paths(tmp_path: Path):
    path = tmp_path / "tools.jsonl"
    _write_dataset(path, [{"id": "a", "tool": {"tool_name": "A"}}])
    embed_tools(path, model_id="stub", target_dim=4, encoder=StubEncoder(4))
    meta = json.loads((tmp_path / "embeddings" / "meta.json").read_text())
    assert meta["model_id"] == "stub"
    assert meta["embedding_dim"] == 4
    assert meta["source_dataset"] == "tools.jsonl"
    assert meta["embeddings_dir"] == "embeddings"
    assert not meta["source_dataset"].startswith("/")
    assert not meta["embeddings_dir"].startswith("/")
    assert "generated_at" in meta


def test_embed_tools_rejects_model_dim_mismatch(tmp_path: Path):
    """Re-running with a different model in the same dir must error out."""
    path = tmp_path / "tools.jsonl"
    _write_dataset(path, [{"id": "a", "tool": {"tool_name": "A"}}])
    embed_tools(path, model_id="stub-A", target_dim=4, encoder=StubEncoder(4))
    with pytest.raises(RuntimeError, match="stub-B"):
        embed_tools(path, model_id="stub-B", target_dim=4, encoder=StubEncoder(4))


def test_embed_tools_rejects_dim_mismatch(tmp_path: Path):
    path = tmp_path / "tools.jsonl"
    _write_dataset(path, [{"id": "a", "tool": {"tool_name": "A"}}])
    embed_tools(path, model_id="stub", target_dim=4, encoder=StubEncoder(4))
    with pytest.raises(RuntimeError, match="dim="):
        embed_tools(path, model_id="stub", target_dim=8, encoder=StubEncoder(8))


def test_embed_tools_atomic_no_tmp_left(tmp_path: Path):
    path = tmp_path / "tools.jsonl"
    _write_dataset(path, [{"id": "a", "tool": {"tool_name": "A"}}])
    embed_tools(path, model_id="stub", target_dim=4, encoder=StubEncoder(4))
    leftovers = [p.name for p in (tmp_path / "embeddings").iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_embed_tools_empty_dataset(tmp_path: Path):
    path = tmp_path / "tools.jsonl"
    _write_dataset(path, [])
    enc = StubEncoder(dim=4)
    counts = embed_tools(path, model_id="stub", target_dim=4, encoder=enc)
    assert counts["total"] == 0
    assert counts["embedded"] == 0
    assert len(enc.calls) == 0
    # No files created when the dataset is empty
    assert not (tmp_path / "embeddings" / "meta.json").exists()


def test_embed_tools_custom_output_dir(tmp_path: Path):
    path = tmp_path / "tools.jsonl"
    _write_dataset(path, [{"id": "a", "tool": {"tool_name": "A"}}])
    custom = tmp_path / "elsewhere"
    embed_tools(path, output_dir=custom, model_id="stub", target_dim=4, encoder=StubEncoder(4))
    assert (custom / "a.npz").exists()
    assert (custom / "meta.json").exists()
