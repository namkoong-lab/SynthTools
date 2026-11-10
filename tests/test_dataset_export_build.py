"""End-to-end: build + verify on a tmp dir with a fake tokenizer. Needs datasets."""
import json

import pytest

datasets = pytest.importorskip("datasets")

from dataset_export import build_hf_dataset as b, tokens, verify_hf_dataset as v  # noqa: E402
from dataset_export.row_schema import COLUMNS, features  # noqa: E402
from tests.test_dataset_export_assemble import HAPPY, TOOLS, asst, tool, traj  # noqa: E402
from tests.test_dataset_export_tokens import FakeTok  # noqa: E402

FIELDS = {"demo_spec_000_seq1": "Demo", "demo_spec_000_seq2": "Demo", "demo_spec_001_seq1": "Legal"}


def write_inputs(tmp_path):
    labels, trajs = tmp_path / "labels", tmp_path / "trajectories"
    labels.mkdir(); trajs.mkdir()
    judge, tasks = [], []
    chats = {"demo_spec_000_seq1": HAPPY, "demo_spec_000_seq2": HAPPY[:4], "demo_spec_001_seq1": HAPPY}
    for k, (task_id, chat) in enumerate(chats.items()):
        t = traj(chat); t["task_id"] = task_id
        (trajs / f"{task_id}.s100{k}.json").write_text(json.dumps(t))
        judge.append({"task_id": task_id, "seed": 1000 + k, "decision": "SUCCESSFUL", "path": f"/old/{task_id}.s100{k}.json"})
        tasks.append({"id": task_id, "field": FIELDS[task_id], "summary": "Do it.", "gt_tool_calls": ["Search(q='x')"]})
    judge.append({"task_id": "demo_spec_001_seq1", "seed": 1009, "decision": "SUCCESSFUL", "path": "/old/missing.json"})
    (labels / b.LABELS_FILE).write_text("\n".join(json.dumps(r) for r in judge) + "\n")
    (labels / b.TASK_CONTENT_FILE).write_text("\n".join(json.dumps(r) for r in tasks) + "\n")
    return labels, trajs


def test_build_and_verify(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "load_tokenizer", lambda name: FakeTok())
    labels, trajs = write_inputs(tmp_path)
    out = tmp_path / "out"
    manifest = b.build(labels, trajs, out, tokenizer="fake", workers=1, rows_per_shard=2,
                       holdout_fields=["Legal"], test_fraction=0.0)
    assert manifest["counts"] == {"records": 4, "written": 2, "written_by_split": {"train": 1, "test": 1},
                                  "dropped": {"no_final_stop": 1, "unreadable_file": 1}}
    assert manifest["split"] == {"unit": "task", "holdout_fields": ["Legal"], "test_fraction": 0.0, "seed": 0}
    assert manifest["n_shards"] == {"train": 1, "test": 1} and manifest["tokenizer"] == "fake" and manifest["rule"]
    assert sorted(p.name for p in (out / "data").iterdir()) == ["test-00000-of-00001.parquet", "train-00000-of-00001.parquet"]
    train, test = v.load_rows(out, "train"), v.load_rows(out, "test")
    assert train.features == features() and [r["id"] for r in train] == ["demo_spec_000_seq1.s1000"]
    assert [r["id"] for r in test] == ["demo_spec_001_seq1.s1002"] and test[0]["field"] == "Legal"
    row = train[0]
    msgs = row["messages"]
    assert msgs[1]["tool_calls"][0]["function"] == {"name": "Search", "arguments": '{"q": "x"}'}   # OpenAI convention
    assert msgs[0]["reasoning_content"] is None and msgs[0]["tool_calls"] is None                   # null where absent
    tools = json.loads(row["tools"])
    assert msgs[-1]["content"] == "Done: saved a." and tools[0]["function"]["name"] == TOOLS[0]["tool_name"]
    assert isinstance(tools[0]["function"]["parameters"], dict)
    assert row["field"] == "Demo" and row["gt_tool_calls"] == ["Search(q='x')"]
    assert 0 < row["n_target_tokens_nothink"] < row["n_target_tokens"] < row["n_total_tokens"]
    # what was counted at build time is what the stored row renders to
    assert tokens.count_tokens(FakeTok(), msgs, tools) == {k: row[k] for k in ("n_target_tokens", "n_target_tokens_nothink", "n_total_tokens")}
    assert manifest["tokens"] == {k: manifest["tokens_by_split"]["train"][k] + manifest["tokens_by_split"]["test"][k] for k in manifest["tokens"]}
    report = v.verify(out, sample=5)
    assert report["problems"] == [] and report["rows"] == 2 and report["sample_mismatches"] == 0
    assert report["tasks_in_both"] == 0 and report["holdout_fields"] == ["Legal"]


def test_test_fraction_is_by_task_and_stable(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "load_tokenizer", lambda name: FakeTok())
    labels, trajs = write_inputs(tmp_path)
    records = b.load_records(labels)
    everything = b.assign_splits(records, [], test_fraction=1.0, split_seed=0)
    assert len(everything["test"]) == 4 and everything["train"] == []
    once = b.assign_splits(records, [], 0.5, 0)
    assert once == b.assign_splits(records, [], 0.5, 0)
    by_task = {}
    for split, recs in once.items():
        for rec, _ in recs:
            by_task.setdefault(rec["task_id"], set()).add(split)
    assert all(len(s) == 1 for s in by_task.values())
    with pytest.raises(ValueError, match="holdout fields"):
        b.assign_splits(records, ["Nope"], 0.0, 0)


def test_verify_catches_a_broken_row_and_a_leak(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "load_tokenizer", lambda name: FakeTok())
    labels, trajs = write_inputs(tmp_path)
    out = tmp_path / "out"
    b.build(labels, trajs, out, tokenizer="fake", workers=1, rows_per_shard=10, holdout_fields=["Legal"], test_fraction=0.0)
    path = out / "data" / "train-00000-of-00001.parquet"
    rows = list(v.load_rows(out, "train"))
    rows[0]["messages"] = rows[0]["messages"][:-1]  # ends with a tool reply now
    datasets.Dataset.from_dict({c: [r[c] for r in rows] for c in COLUMNS}, features=features()).to_parquet(path)
    report = v.verify(out, sample=0)
    assert any("1 rows violate" in p for p in report["problems"])
    # a test row copied into train: same task in both splits and a held-out field in train
    leak = list(v.load_rows(out, "test"))
    datasets.Dataset.from_dict({c: [r[c] for r in rows + leak] for c in COLUMNS}, features=features()).to_parquet(path)
    problems = v.verify(out, sample=0)["problems"]
    assert any("both splits" in p for p in problems) and any("has a row in train" in p for p in problems)


def test_row_problems_on_good_row():
    from dataset_export.assemble import assemble
    conv, _ = assemble(traj(HAPPY))
    row = {"id": "a.s1", "field": "f", "gt_tool_calls": ["x"], "n_target_tokens": 2, "n_target_tokens_nothink": 1,
           "n_total_tokens": 3, "tools": json.dumps(conv["tools"]), "messages": conv["messages"]}
    assert v.row_problems(row) == []
    bad = dict(row); bad["n_total_tokens"] = 1
    assert v.row_problems(bad)
