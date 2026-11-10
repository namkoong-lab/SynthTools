"""Build the HF-style trajectory dataset: one parquet row per kept SUCCESSFUL trajectory, split train/test.

    python -m dataset_export.build_hf_dataset \\
        --labels-dir       /shared/share_mala/tc3527/synthtools/labels_20261029 \\
        --trajectories-dir /shared/share_mala/tc3527/synthtools/tool_content_full/trajectories \\
        --out              /shared/share_mala/tc3527/synthtools/hf_dataset/synthtools_trajectories \\
        --holdout-fields "Legal Services" "Healthcare Medical" ... [--test-fraction 0.05] [--split-seed 0] \\
        [--tokenizer Qwen/Qwen3-1.7B] [--workers 24] [--rows-per-shard 5000] [--limit N]

Inputs: `<labels-dir>/trajectories.successful.jsonl` (task_id, seed, path) and
`<labels-dir>/task_content.valid.jsonl` (field, gt_tool_calls). Rollouts are opened by name,
`<trajectories-dir>/<basename of path>`; the directory is never listed.

Split: the unit is the task, so every rollout of a task lands in one split. The `test` split holds every
task of the `--holdout-fields` (out-of-distribution fields, never trained on) plus a `--test-fraction`
of the other tasks chosen by a stable hash of (split_seed, task_id); everything else is `train`.
Records are sorted by (task_id, seed) within each split and cut into shards; each shard is one worker
job (`assemble` + `count_tokens`).

Output: `<out>/data/{train,test}-XXXXX-of-NNNNN.parquet` (written with `datasets`, so
`load_dataset("parquet", data_dir=...)` returns both splits) and `<out>/manifest.json` (inputs with
sha256, git commit, tokenizer, the split parameters, counts and token totals per split, drop reasons).
Run on a CPU node with the SynthTools venv and `HF_HOME` pointing at the cache that holds the tokenizer
(`HF_HUB_OFFLINE=1`). A full build of 210,925 records takes about four minutes on 24 workers; verify it
with `verify_hf_dataset`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dataset_export import tokens
from dataset_export.assemble import assemble
from dataset_export.row_schema import COLUMNS, features, make_row

LABELS_FILE = "trajectories.successful.jsonl"
TASK_CONTENT_FILE = "task_content.valid.jsonl"
SPLITS = ("train", "test")
RULE = ("SUCCESSFUL trajectories of VALID tasks whose every turn converts: readable envelope, "
        "plain-literal call, known tool, JSON tool reply with int status_code, ending with a non-empty "
        "<STOP> answer; calls the environment rejected for semantic reasons are kept.")

_TOK: Any = None


def _init_worker(tokenizer: str) -> None:
    global _TOK
    _TOK = tokens.load_tokenizer(tokenizer)


def load_records(labels_dir: Path) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """(judge record, task row) pairs sorted by (task_id, seed)."""
    tasks: Dict[str, Dict[str, Any]] = {}
    with open(labels_dir / TASK_CONTENT_FILE) as f:
        for line in f:
            r = json.loads(line)
            tasks[r["id"]] = {"field": r.get("field") or "", "gt_tool_calls": r.get("gt_tool_calls") or []}
    records = []
    with open(labels_dir / LABELS_FILE) as f:
        for line in f:
            j = json.loads(line)
            records.append(({"task_id": j["task_id"], "seed": int(j["seed"]), "path": j["path"]}, tasks[j["task_id"]]))
    records.sort(key=lambda rt: (rt[0]["task_id"], rt[0]["seed"]))
    return records


def split_of(task_id: str, seed: int, test_fraction: float) -> str:
    """Stable train/test assignment of a task: sha256 of (seed, task_id) mapped to [0, 1)."""
    u = int.from_bytes(hashlib.sha256(f"{seed}:{task_id}".encode()).digest()[:8], "big") / 2 ** 64
    return "test" if u < test_fraction else "train"


def assign_splits(records: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], holdout_fields: Sequence[str],
                  test_fraction: float, split_seed: int) -> Dict[str, list]:
    """Records per split; a held-out field goes entirely to test, other tasks by `split_of`."""
    known = {task["field"] for _, task in records}
    unknown = set(holdout_fields) - known
    if unknown:
        raise ValueError(f"holdout fields not in task_content: {sorted(unknown)}")
    holdout = set(holdout_fields)
    out: Dict[str, list] = {s: [] for s in SPLITS}
    for rec, task in records:
        split = "test" if task["field"] in holdout else split_of(rec["task_id"], split_seed, test_fraction)
        out[split].append((rec, task))
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def shard_name(split: str, index: int, total: int) -> str:
    return f"{split}-{index:05d}-of-{total:05d}.parquet"


def build_shard(job: Tuple[str, int, int, list, str, str]) -> Dict[str, Any]:
    """One shard: convert its records, write the parquet file atomically, return the stats."""
    from datasets import Dataset

    split, index, n_shards, records, out, traj_dir = job
    rows: List[Dict[str, Any]] = []
    dropped: Counter = Counter()
    for rec, task in records:
        try:
            with open(Path(traj_dir) / os.path.basename(rec["path"])) as f:
                traj = json.load(f)
        except (OSError, ValueError):
            dropped["unreadable_file"] += 1
            continue
        conv, reason = assemble(traj)
        if conv is None:
            dropped[reason] += 1
            continue
        try:
            counts = tokens.count_tokens(_TOK, conv["messages"], conv["tools"])
        except Exception:  # a template failure is a data defect to count, not a crash
            dropped["render_error"] += 1
            continue
        rows.append(make_row(f"{rec['task_id']}.s{rec['seed']}", task["field"], task["gt_tool_calls"],
                             conv["tools"], conv["messages"], counts))
    data_dir = Path(out) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    name = shard_name(split, index, n_shards)
    tmp = data_dir / (name + ".tmp")
    columns = {c: [r[c] for r in rows] for c in COLUMNS}
    Dataset.from_dict(columns, features=features()).to_parquet(tmp)   # datasets' own writer (snappy)
    os.replace(tmp, data_dir / name)
    return {
        "split": split,
        "shard": name,
        "records": len(records),
        "written": len(rows),
        "dropped": dict(dropped),
        "tokens": {k: sum(r[k] for r in rows) for k in ("n_target_tokens", "n_target_tokens_nothink", "n_total_tokens")},
    }


def build(labels_dir: Path, trajectories_dir: Path, out: Path, tokenizer: str, workers: int = 24,
          rows_per_shard: int = 5000, limit: Optional[int] = None, holdout_fields: Sequence[str] = (),
          test_fraction: float = 0.05, split_seed: int = 0) -> Dict[str, Any]:
    started = time.time()
    records = load_records(labels_dir)
    if limit:
        records = records[:limit]
    by_split = assign_splits(records, holdout_fields, test_fraction, split_seed)
    jobs = []
    n_shards = {}
    for split in SPLITS:
        recs = by_split[split]
        n_shards[split] = max(1, (len(recs) + rows_per_shard - 1) // rows_per_shard)
        jobs += [(split, i, n_shards[split], recs[i * rows_per_shard:(i + 1) * rows_per_shard], str(out), str(trajectories_dir))
                 for i in range(n_shards[split])]
    out.mkdir(parents=True, exist_ok=True)
    stats: List[Dict[str, Any]] = []
    if workers > 1:
        with Pool(workers, initializer=_init_worker, initargs=(tokenizer,)) as pool:
            for k, s in enumerate(pool.imap_unordered(build_shard, jobs), 1):
                stats.append(s)
                print(f"{k}/{len(jobs)} {s['shard']} rows={s['written']}", flush=True)
    else:
        _init_worker(tokenizer)
        for k, job in enumerate(jobs, 1):
            stats.append(build_shard(job))
            print(f"{k}/{len(jobs)} {stats[-1]['shard']} rows={stats[-1]['written']}", flush=True)
    stats.sort(key=lambda s: s["shard"])
    by = {split: [s for s in stats if s["split"] == split] for split in SPLITS}
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": round(time.time() - started, 1),
        "command": sys.argv,
        "git_commit": git_commit(),
        "inputs": {
            "labels": {"path": str(labels_dir / LABELS_FILE), "sha256": sha256_of(labels_dir / LABELS_FILE)},
            "task_content": {"path": str(labels_dir / TASK_CONTENT_FILE), "sha256": sha256_of(labels_dir / TASK_CONTENT_FILE)},
            "trajectories_dir": str(trajectories_dir),
        },
        "tokenizer": tokenizer,
        "rows_per_shard": rows_per_shard,
        "split": {"unit": "task", "holdout_fields": list(holdout_fields), "test_fraction": test_fraction, "seed": split_seed},
        "n_shards": n_shards,
        "counts": {
            "records": len(records),
            "written": sum(s["written"] for s in stats),
            "written_by_split": {split: sum(s["written"] for s in by[split]) for split in SPLITS},
            "dropped": dict(sum((Counter(s["dropped"]) for s in stats), Counter())),
        },
        "tokens": dict(sum((Counter(s["tokens"]) for s in stats), Counter())),
        "tokens_by_split": {split: dict(sum((Counter(s["tokens"]) for s in by[split]), Counter())) for split in SPLITS},
        "rule": RULE,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("BUILD END", json.dumps(manifest["counts"]), manifest["tokens"], flush=True)
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--trajectories-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--holdout-fields", nargs="*", default=[], help="fields whose tasks all go to test")
    p.add_argument("--test-fraction", type=float, default=0.05, help="fraction of the other tasks that go to test")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--rows-per-shard", type=int, default=5000)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    manifest = build(a.labels_dir, a.trajectories_dir, a.out, a.tokenizer, a.workers, a.rows_per_shard, a.limit,
                     a.holdout_fields, a.test_fraction, a.split_seed)
    return 0 if manifest["counts"]["written"] else 1


if __name__ == "__main__":
    sys.exit(main())
