"""End-to-end: build + verify on a tmp dir with a fake tokenizer. Needs pyarrow."""
import json

import pytest

pytest.importorskip("pyarrow")

from dataset_export import build_hf_dataset as b, tokens, verify_hf_dataset as v  # noqa: E402
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
    import pyarrow.parquet as pq
    rows = pq.read_table(out / "data").to_pylist()
    assert [r["id"] for r in rows] == ["demo_spec_000_seq1.s1000", "demo_spec_001_seq1.s1002"]
    msgs = json.loads(rows[0]["messages"])
    assert msgs[1]["tool_calls"][0]["function"] == {"name": "Search", "arguments": {"q": "x"}}
    assert msgs[-1]["content"] == "Done: saved a." and json.loads(rows[0]["tools"])[0]["function"]["name"] == TOOLS[0]["tool_name"]
    assert rows[0]["field"] == "Demo" and json.loads(rows[0]["gt_tool_calls"]) == ["Search(q='x')"]
    assert 0 < rows[0]["n_target_tokens_nothink"] < rows[0]["n_target_tokens"] < rows[0]["n_total_tokens"]
    report = v.verify(out, sample=5)
    assert report["problems"] == [] and report["rows"] == 2 and report["sample_mismatches"] == 0


def test_verify_catches_a_broken_row(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "load_tokenizer", lambda name: FakeTok())
    labels, trajs = write_inputs(tmp_path)
    out = tmp_path / "out"
    b.build(labels, trajs, out, tokenizer="fake", workers=1, rows_per_shard=10)
    import pyarrow as pa, pyarrow.parquet as pq
    path = out / "data" / "train-00000-of-00001.parquet"
    rows = pq.read_table(path).to_pylist()
    msgs = json.loads(rows[0]["messages"]); msgs.pop()  # ends with a tool reply now
    rows[0]["messages"] = json.dumps(msgs)
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
    report = v.verify(out, sample=0)
    assert any("1 rows violate" in p for p in report["problems"])


def test_row_problems_on_good_row():
    from dataset_export.assemble import assemble
    conv, _ = assemble(traj(HAPPY))
    row = {"id": "a.s1", "field": "f", "gt_tool_calls": json.dumps(["x"]), "n_target_tokens": 2, "n_target_tokens_nothink": 1,
           "n_total_tokens": 3, "tools": json.dumps(conv["tools"]), "messages": json.dumps(conv["messages"])}
    assert v.row_problems(row) == []
    bad = dict(row); bad["n_total_tokens"] = 1
    assert v.row_problems(bad)
