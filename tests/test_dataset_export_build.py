"""End-to-end: build + verify on a tmp dir with a fake tokenizer. Needs datasets."""
import json

import pytest

datasets = pytest.importorskip("datasets")

from dataset_export import build_hf_dataset as b, tokens, verify_hf_dataset as v  # noqa: E402
from dataset_export.row_schema import COLUMNS, features  # noqa: E402
from tests.test_dataset_export_assemble import HAPPY, TOOLS, asst, tool, traj  # noqa: E402
from tests.test_dataset_export_tokens import FakeTok  # noqa: E402


def write_inputs(tmp_path):
    labels, trajs = tmp_path / "labels", tmp_path / "trajectories"
    labels.mkdir(); trajs.mkdir()
    judge, tasks = [], []
    chats = {"demo_spec_000_seq1": HAPPY, "demo_spec_000_seq2": HAPPY[:4], "demo_spec_001_seq1": HAPPY}
    for k, (task_id, chat) in enumerate(chats.items()):
        t = traj(chat); t["task_id"] = task_id
        (trajs / f"{task_id}.s100{k}.json").write_text(json.dumps(t))
        judge.append({"task_id": task_id, "seed": 1000 + k, "decision": "SUCCESSFUL", "path": f"/old/{task_id}.s100{k}.json"})
        tasks.append({"id": task_id, "field": "Demo", "summary": "Do it.", "gt_tool_calls": ["Search(q='x')"]})
    judge.append({"task_id": "demo_spec_001_seq1", "seed": 1009, "decision": "SUCCESSFUL", "path": "/old/missing.json"})
    (labels / b.LABELS_FILE).write_text("\n".join(json.dumps(r) for r in judge) + "\n")
    (labels / b.TASK_CONTENT_FILE).write_text("\n".join(json.dumps(r) for r in tasks) + "\n")
    return labels, trajs


def test_build_and_verify(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "load_tokenizer", lambda name: FakeTok())
    labels, trajs = write_inputs(tmp_path)
    out = tmp_path / "out"
    manifest = b.build(labels, trajs, out, tokenizer="fake", workers=1, rows_per_shard=2)
    assert manifest["counts"] == {"records": 4, "written": 2, "dropped": {"no_final_stop": 1, "unreadable_file": 1}}
    assert manifest["n_shards"] == 2 and manifest["tokenizer"] == "fake" and manifest["rule"]
    assert sorted(p.name for p in (out / "data").iterdir()) == ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
    ds = v.load_rows(out)
    assert ds.features == features()
    rows = list(ds)
    assert [r["id"] for r in rows] == ["demo_spec_000_seq1.s1000", "demo_spec_001_seq1.s1002"]
    msgs = rows[0]["messages"]
    assert msgs[1]["tool_calls"][0]["function"] == {"name": "Search", "arguments": '{"q": "x"}'}   # OpenAI convention
    assert msgs[0]["reasoning_content"] is None and msgs[0]["tool_calls"] is None                   # null where absent
    tools = json.loads(rows[0]["tools"])
    assert msgs[-1]["content"] == "Done: saved a." and tools[0]["function"]["name"] == TOOLS[0]["tool_name"]
    assert isinstance(tools[0]["function"]["parameters"], dict)
    assert rows[0]["field"] == "Demo" and rows[0]["gt_tool_calls"] == ["Search(q='x')"]
    assert 0 < rows[0]["n_target_tokens_nothink"] < rows[0]["n_target_tokens"] < rows[0]["n_total_tokens"]
    # what was counted at build time is what the stored row renders to
    assert tokens.count_tokens(FakeTok(), msgs, tools) == {k: rows[0][k] for k in ("n_target_tokens", "n_target_tokens_nothink", "n_total_tokens")}
    report = v.verify(out, sample=5)
    assert report["problems"] == [] and report["rows"] == 2 and report["sample_mismatches"] == 0


def test_verify_catches_a_broken_row(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "load_tokenizer", lambda name: FakeTok())
    labels, trajs = write_inputs(tmp_path)
    out = tmp_path / "out"
    b.build(labels, trajs, out, tokenizer="fake", workers=1, rows_per_shard=10)
    path = out / "data" / "train-00000-of-00001.parquet"
    rows = list(v.load_rows(out))
    rows[0]["messages"] = rows[0]["messages"][:-1]  # ends with a tool reply now
    datasets.Dataset.from_dict({c: [r[c] for r in rows] for c in COLUMNS}, features=features()).to_parquet(path)
    report = v.verify(out, sample=0)
    assert any("1 rows violate" in p for p in report["problems"])


def test_row_problems_on_good_row():
    from dataset_export.assemble import assemble
    conv, _ = assemble(traj(HAPPY))
    row = {"id": "a.s1", "field": "f", "gt_tool_calls": ["x"], "n_target_tokens": 2, "n_target_tokens_nothink": 1,
           "n_total_tokens": 3, "tools": json.dumps(conv["tools"]), "messages": conv["messages"]}
    assert v.row_problems(row) == []
    bad = dict(row); bad["n_total_tokens"] = 1
    assert v.row_problems(bad)
