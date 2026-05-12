"""Split a judge_trajectories.py audit JSONL into per-category task_id files.

Read-only — no trajectory files are touched. Just classifies the judge's
explanation into buckets so downstream tools can act on each bucket
separately.

Categories (in match priority order):
  - truncation              regen LLM ran out of tokens / Harmony eat-up
  - ast_quoting             unquoted strings in tool_call (4xx-recovery candidates)
  - schema_type_violation   float for int param etc.
  - misread_prior           regen LLM cited wrong prior fact
  - ignored_task_part       only addressed half the task
  - uncorrected_4xx         trajectory ends on a failed call without recovery
  - wrong_tool              picked the wrong tool altogether
  - confabulated_value      value exists nowhere in chat-visible context
  - unmatched               none of the above

Usage:
    python -m scripts.audit_split \\
        --audit-jsonl /tmp/traj_judge_full.jsonl \\
        --out-dir /tmp/audit_buckets/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import get_logger  # noqa: E402

logger = get_logger("synthtools")


# Order matters: first match wins. More specific patterns first.
_CATEGORY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("truncation",            re.compile(r"truncat|cut[- ]off|incomplete sentence", re.IGNORECASE)),
    ("ast_quoting",           re.compile(r"omit.*quot|missing quot|unquot|invalid syntax|non[- ‐­]?quoted", re.IGNORECASE)),
    ("schema_type_violation", re.compile(r"non[‐‑‒–—\-]?integer|float[^.]*?(?:integer|int)|requires.*integer|type[- ]?violat|wrong type|integer schema", re.IGNORECASE)),
    ("misread_prior",         re.compile(r"contradict|misstat|misread|misrepres|incorrectly refers|incorrectly references", re.IGNORECASE)),
    ("ignored_task_part",     re.compile(r"ignored|missed|skipped|did not address|never (?:addressed|tackled)", re.IGNORECASE)),
    ("uncorrected_4xx",       re.compile(r"(?:400|404|4\d\d|5\d\d).*(?:never (?:corrected|retried)|not corrected|unaddressed|unresolved)|(?:never (?:corrected|retried)|unaddressed).*(?:400|4\d\d|error|failure)", re.IGNORECASE)),
    ("wrong_tool",            re.compile(r"misused.*tool|wrong tool|used.*tool instead", re.IGNORECASE)),
    ("confabulated_value",    re.compile(r"invent|fabricat|made up|never (?:created|provided|returned|defined|recorded|stored)|not (?:in|present|named) the task|not in any prior", re.IGNORECASE)),
]


def classify(explanation: str) -> str:
    if not explanation:
        return "unmatched"
    for label, pat in _CATEGORY_PATTERNS:
        if pat.search(explanation):
            return label
    return "unmatched"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/audit_buckets"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[str]] = {}
    samples: dict[str, list[str]] = {}
    counts = Counter()

    with open(args.audit_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = r.get("quality")
            if q != 0 and q is not None:
                continue
            tid = r.get("task_id")
            exp = r.get("explanation") or ""
            cat = classify(exp)
            buckets.setdefault(cat, []).append(tid)
            counts[cat] += 1
            samples.setdefault(cat, [])
            if len(samples[cat]) < 3:
                samples[cat].append(exp[:160])

    total = sum(counts.values())
    print(f"\n=== audit split (total bad/unparseable: {total:,}) ===")
    for cat in sorted(buckets, key=lambda c: -counts[c]):
        n = counts[cat]
        out = args.out_dir / f"{cat}.txt"
        out.write_text("\n".join(sorted(buckets[cat])) + ("\n" if buckets[cat] else ""))
        pct = 100 * n / max(total, 1)
        print(f"  {cat:<26}: {n:>5} ({pct:5.1f}%)  → {out}")

    (args.out_dir / "summary.json").write_text(json.dumps({
        "total_bad": total, "counts": dict(counts),
    }, indent=2))

    print("\n=== sample explanations per category (first 3) ===")
    for cat in sorted(buckets, key=lambda c: -counts[c]):
        print(f"\n  [{cat}]")
        for s in samples.get(cat, []):
            print(f"    - {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
