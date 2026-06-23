#!/usr/bin/env python3
"""Normalize JSON-style boolean/null literals to Python literals inside the
`gt_tool_calls` strings of task_content.jsonl.

A call like  Foo(open_access=true, note=null)  is parsed by Python's ast with
`true`/`false`/`null` as bareword Name nodes (not booleans). We rewrite ONLY
those Name nodes (true->True, false->False, null->None) and re-emit the call.
String values are ast.Constant nodes and are never touched, so 'true' inside a
quoted string is preserved.

A call is rewritten ONLY if it contains at least one such Name; otherwise the
original string is kept byte-identical. Unparseable calls are left as-is.

Usage:
  python dev/normalize_gt_literals.py --in <jsonl> [--apply] [--field gt_tool_calls]

Without --apply it is a dry run (counts only, no write).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils import normalize_call_literals  # noqa: E402


def normalize_call(call: str):
    """Return (new_call, changed) using the canonical pipeline normalizer."""
    new_call = normalize_call_literals(call)
    return new_call, (1 if new_call != call else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--field", default="gt_tool_calls")
    ap.add_argument("--apply", action="store_true",
                    help="write the normalized file (default: dry run)")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        sys.exit(f"no such file: {inp}")

    rows = 0
    rows_changed = 0
    calls_total = 0
    calls_changed = 0
    repl_total = 0
    samples = []

    tmp = inp.with_suffix(inp.suffix + ".normtmp")
    out = open(tmp, "w") if args.apply else None
    try:
        with open(inp) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    if out:
                        out.write(line + "\n")
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    if out:
                        out.write(line + "\n")
                    continue
                rows += 1
                calls = d.get(args.field)
                changed_here = False
                if isinstance(calls, list):
                    new_calls = []
                    for c in calls:
                        calls_total += 1
                        nc, n = normalize_call(c)
                        if n:
                            calls_changed += 1
                            repl_total += n
                            changed_here = True
                            if len(samples) < 8:
                                samples.append((c[:90], nc[:90]))
                        new_calls.append(nc)
                    d[args.field] = new_calls
                if changed_here:
                    rows_changed += 1
                if out:
                    out.write(json.dumps(d, ensure_ascii=False) + "\n")
    finally:
        if out:
            out.close()

    print(f"rows scanned        : {rows}")
    print(f"rows changed        : {rows_changed}")
    print(f"call strings total  : {calls_total}")
    print(f"call strings changed: {calls_changed}")
    print(f"literal replacements: {repl_total}")
    print("samples (before -> after):")
    for a, b in samples:
        print(f"  {a}")
        print(f"    -> {b}")

    if args.apply:
        # Atomic-ish replace, leaving a backup the caller created beforehand.
        os.replace(tmp, inp)
        print(f"\nAPPLIED: wrote {inp}")
    else:
        if tmp.exists():
            tmp.unlink()
        print("\nDRY RUN: no file written (pass --apply to write)")


if __name__ == "__main__":
    main()
