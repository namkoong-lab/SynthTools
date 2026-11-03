"""Verify a built dataset: manifest arithmetic, the invariants of every row, token counts and render
structure recomputed on a random sample with the real tokenizer.

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


def row_problems(row: Dict[str, Any]) -> List[str]:
    """The invariants of one row; empty list when it is well formed."""
    problems = []
    if any(c not in row for c in COLUMNS):
        return ["missing columns"]
    msgs = json.loads(row["messages"])
    tools = json.loads(row["tools"])
    names = {t["function"]["name"] for t in tools}
    roles = [m["role"] for m in msgs]
    if not tools:
        problems.append("no tools")
    if not (roles[0] == "user" and msgs[0]["content"].strip()):
        problems.append("first message is not a non-empty user message")
    if len(msgs) < 4 or len(msgs) % 2:
        problems.append("message count is not user + pairs + answer")
    if not (roles[-1] == "assistant" and "tool_calls" not in msgs[-1] and msgs[-1]["content"].strip()):
        problems.append("last message is not a non-empty final answer")
    for i in range(1, len(msgs) - 1, 2):
        a, t = msgs[i], msgs[i + 1] if i + 1 < len(msgs) - 1 else None
        if a["role"] != "assistant" or not a.get("tool_calls") or a.get("content") != "" or not isinstance(a.get("reasoning_content"), str):
            problems.append(f"message {i} is not a tool-call turn")
            continue
        fn = a["tool_calls"][0]["function"]
        if fn["name"] not in names or not isinstance(fn["arguments"], dict):
            problems.append(f"message {i}: unknown tool or non-dict arguments")
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
    if not (isinstance(json.loads(row["gt_tool_calls"]), list) and json.loads(row["gt_tool_calls"])):
        problems.append("gt_tool_calls is not a non-empty list")
    if ".s" not in row["id"]:
        problems.append("id is not <task_id>.s<seed>")
    if not 0 < row["n_target_tokens_nothink"] < row["n_target_tokens"] < row["n_total_tokens"]:
        problems.append("token counts are not ordered nothink < target < total")
    return problems


def verify(out: Path, sample: int = 50, tokenizer: Optional[str] = None) -> Dict[str, Any]:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    manifest = json.loads((out / "manifest.json").read_text())
    counts = manifest["counts"]
    problems: List[str] = []
    if counts["records"] != counts["written"] + sum(counts["dropped"].values()):
        problems.append("manifest: records != written + dropped")
    dataset = ds.dataset(str(out / "data"), format="parquet")
    ids = dataset.to_table(columns=["id"])["id"].to_pylist()
    if not (len(ids) == counts["written"] == len(set(ids))):
        problems.append("row count or id uniqueness does not match the manifest")
    invalid = 0
    fields = set()
    sums = {"n_target_tokens": 0, "n_target_tokens_nothink": 0, "n_total_tokens": 0}
    for batch in dataset.to_batches(batch_size=2000):
        for row in batch.to_pylist():
            fields.add(row["field"])
            for k in sums:
                sums[k] += row[k]
            if row_problems(row):
                invalid += 1
    if invalid:
        problems.append(f"{invalid} rows violate the row invariants")
    if sums != manifest["tokens"]:
        problems.append("token sums differ from the manifest")
    mismatches = 0
    if sample and ids:
        tok = tokens.load_tokenizer(tokenizer or manifest["tokenizer"])
        chosen = sorted(random.Random(0).sample(ids, min(sample, len(ids))))
        for row in dataset.to_table(filter=pc.field("id").isin(chosen)).to_pylist():
            msgs, tools = json.loads(row["messages"]), json.loads(row["tools"])
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
    report = {"rows": len(ids), "invalid_rows": invalid, "fields": len(fields), "token_sums": sums,
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
