"""Verify a built dataset: manifest arithmetic, the invariants of every row, the split (no task in both
splits, held-out fields only in test, per-split counts), token counts and render structure recomputed
on a random sample with the real tokenizer.

    python -m dataset_export.verify_hf_dataset --out <dataset dir> [--sample 50]

Exit code 1 on any problem.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset_export import tokens
from dataset_export.row_schema import COLUMNS


SPLITS = ("train", "test")


def load_rows(out: Path, split: str = "train"):
    """One split as `datasets` returns it, read straight from its parquet files."""
    import pyarrow.parquet as pq
    from datasets import Dataset
    files = sorted(str(p) for p in (out / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} shards in {out / 'data'}")
    return Dataset(pq.read_table(files))


def row_problems(row: Dict[str, Any]) -> List[str]:
    """The invariants of one row (as `datasets` returns it); empty list when it is well formed."""
    problems = []
    if any(c not in row for c in COLUMNS):
        return ["missing columns"]
    msgs = row["messages"]
    try:
        tools = json.loads(row["tools"])
    except (TypeError, ValueError):
        return ["tools is not a JSON string"]
    names = {t["function"]["name"] for t in tools}
    roles = [m["role"] for m in msgs]
    if not tools:
        problems.append("no tools")
    if not (roles[0] == "user" and msgs[0]["content"].strip()):
        problems.append("first message is not a non-empty user message")
    if len(msgs) < 4 or len(msgs) % 2:
        problems.append("message count is not user + pairs + answer")
    if not (roles[-1] == "assistant" and not msgs[-1].get("tool_calls") and msgs[-1]["content"].strip()):
        problems.append("last message is not a non-empty final answer")
    for i in range(1, len(msgs) - 1, 2):
        a, t = msgs[i], msgs[i + 1] if i + 1 < len(msgs) - 1 else None
        if a["role"] != "assistant" or not a.get("tool_calls") or a.get("content") != "" or not isinstance(a.get("reasoning_content"), str):
            problems.append(f"message {i} is not a tool-call turn")
            continue
        fn = a["tool_calls"][0]["function"]
        try:
            arguments = json.loads(fn["arguments"])
        except (TypeError, ValueError):
            arguments = None
        if fn["name"] not in names or not isinstance(arguments, dict):
            problems.append(f"message {i}: unknown tool or arguments not a JSON object string")
        if t is None or t["role"] != "tool":
            problems.append(f"message {i + 1} is not a tool reply")
            continue
        try:
            payload = json.loads(t["content"])
        except (TypeError, ValueError):
            problems.append(f"message {i + 1}: reply is not JSON")
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("status_code"), int) or "explanation" in payload:
            problems.append(f"message {i + 1}: reply without int status_code or with explanation")
    if not (isinstance(row["gt_tool_calls"], list) and row["gt_tool_calls"]):
        problems.append("gt_tool_calls is not a non-empty list")
    if ".s" not in row["id"]:
        problems.append("id is not <task_id>.s<seed>")
    if not 0 < row["n_target_tokens_nothink"] < row["n_target_tokens"] < row["n_total_tokens"]:
        problems.append("token counts are not ordered nothink < target < total")
    return problems


def verify(out: Path, sample: int = 50, tokenizer: Optional[str] = None) -> Dict[str, Any]:
    manifest = json.loads((out / "manifest.json").read_text())
    counts = manifest["counts"]
    holdout = set(manifest["split"]["holdout_fields"])
    problems: List[str] = []
    if counts["records"] != counts["written"] + sum(counts["dropped"].values()):
        problems.append("manifest: records != written + dropped")
    if counts["written"] != sum(counts["written_by_split"].values()):
        problems.append("manifest: written != sum over splits")
    datasets = {s: load_rows(out, s) for s in SPLITS}
    ids = {s: list(datasets[s]["id"]) for s in SPLITS}
    all_ids = ids["train"] + ids["test"]
    if not (len(all_ids) == counts["written"] == len(set(all_ids))):
        problems.append("row count or id uniqueness does not match the manifest")
    if any(len(ids[s]) != counts["written_by_split"][s] for s in SPLITS):
        problems.append("per-split row counts do not match the manifest")
    tasks = {s: {i.rsplit(".s", 1)[0] for i in ids[s]} for s in SPLITS}
    if tasks["train"] & tasks["test"]:
        problems.append(f"{len(tasks['train'] & tasks['test'])} tasks appear in both splits")
    invalid = 0
    fields = set()
    sums = {"n_target_tokens": 0, "n_target_tokens_nothink": 0, "n_total_tokens": 0}
    sums_by_split = {s: dict(sums) for s in SPLITS}
    for s in SPLITS:
        for row in datasets[s]:
            fields.add(row["field"])
            for k in sums:
                sums[k] += row[k]
                sums_by_split[s][k] += row[k]
            if row_problems(row):
                invalid += 1
            if s == "train" and row["field"] in holdout:
                problems.append(f"held-out field {row['field']!r} has a row in train")
                break
    if invalid:
        problems.append(f"{invalid} rows violate the row invariants")
    if sums != manifest["tokens"] or sums_by_split != manifest["tokens_by_split"]:
        problems.append("token sums differ from the manifest")
    missing = holdout - set(datasets["test"]["field"])
    if missing:
        problems.append(f"held-out fields absent from test: {sorted(missing)}")
    dataset = datasets["train"]
    ids = ids["train"]
    mismatches = 0
    if sample and ids:
        tok = tokens.load_tokenizer(tokenizer or manifest["tokenizer"])
        chosen = sorted(random.Random(0).sample(range(len(ids)), min(sample, len(ids))))
        for row in dataset.select(chosen):
            msgs, tools = row["messages"], json.loads(row["tools"])
            n_asst = sum(1 for m in msgs if m["role"] == "assistant")
            text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
            same_counts = tokens.count_tokens(tok, msgs, tools) == {k: row[k] for k in sums}
            same_structure = (text.count(tokens.ASSISTANT_HEADER) == n_asst
                              and text.count('<tool_call>\n{"name": "') == n_asst - 1
                              and text.count("<tool_response>") == n_asst - 1)
            if not (same_counts and same_structure):
                mismatches += 1
        if mismatches:
            problems.append(f"{mismatches} sampled rows: token counts or render structure differ")
    report = {"rows": len(all_ids), "rows_by_split": {s: len(datasets[s]) for s in SPLITS},
              "tasks_by_split": {s: len(tasks[s]) for s in SPLITS}, "tasks_in_both": len(tasks["train"] & tasks["test"]),
              "holdout_fields": sorted(holdout), "invalid_rows": invalid, "fields": len(fields), "token_sums": sums,
              "sampled": min(sample, len(ids)), "sample_mismatches": mismatches, "problems": problems}
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=int, default=50)
    p.add_argument("--tokenizer", default=None, help="defaults to the manifest's tokenizer")
    a = p.parse_args(argv)
    report = verify(a.out, a.sample, a.tokenizer)
    print(json.dumps(report, indent=1))
    print("VERIFY END", "OK" if not report["problems"] else "PROBLEMS")
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
