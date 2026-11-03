"""Corpus-wide checks of the conversion rules against what the environment recorded. Run after any
change to tool_calls.py, tool_schemas.py or assemble.py; each subcommand prints a report. They read
the raw corpus, so run them on a CPU node with the SynthTools venv.

    python -m dataset_export.corpus_checks calls    --labels-dir L --trajectories-dir T [--unsuccessful] [--report-dir R]
    python -m dataset_export.corpus_checks schemas  --labels-dir L --trajectories-dir T --tools-dataset <tools_dataset.jsonl>
    python -m dataset_export.corpus_checks assemble --labels-dir L --trajectories-dir T

calls     every call of every trajectory: converter status x environment reply; the parse status must
          agree with the environment's syntax verdict; the call string and tool name must equal the
          recorded `turns[k]`; every reply must exist; ground-truth calls of the VALID tasks must be
          plain literals. Disagreements go to <report-dir>/anomalies.jsonl.
schemas   every released tool converts to a valid JSON Schema (2020-12 meta-schema, no `float`, no
          simulator-only key, `required` names exist); the arguments of every environment-accepted
          plain-literal call validate against the converted schema (residue = simulator leniency).
assemble  drop reasons over all SUCCESSFUL trajectories, invariants of the kept rows, raw-chat facts.

Reference numbers of 2026-11-03 are in README.md; a rerun must reproduce them.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import extract_json_objects
from dataset_export.assemble import assemble
from dataset_export.row_schema import STOP
from dataset_export.tool_calls import OK, convert_call, is_plain_literal_call
from dataset_export.tool_schemas import tool_to_function
from dataset_export.verify_hf_dataset import row_problems

_CFG: Dict[str, Any] = {}


def _init(cfg: Dict[str, Any]) -> None:
    _CFG.update(cfg)


def load_records(labels_dir: Path, unsuccessful: bool = False) -> List[Dict[str, Any]]:
    names = ["trajectories.successful.jsonl"] + (["trajectories.unsuccessful.jsonl"] if unsuccessful else [])
    records = []
    for name in names:
        with open(labels_dir / name) as f:
            for line in f:
                j = json.loads(line)
                records.append({"task_id": j["task_id"], "seed": int(j["seed"]), "path": j["path"]})
    return records


def open_traj(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        with open(Path(_CFG["traj_dir"]) / os.path.basename(rec["path"])) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def iter_calls(chat: List[Dict[str, Any]]):
    """(index, call string) for every assistant message whose envelope carries a non-STOP call."""
    for i, m in enumerate(chat):
        if m.get("role") != "assistant":
            continue
        objs = extract_json_objects(m.get("content")) if isinstance(m.get("content"), str) else []
        tc = objs[0].get("tool_call") if objs and isinstance(objs[0], dict) else None
        if isinstance(tc, str) and tc.strip() and STOP not in tc:
            yield i, tc


def reply_of(chat: List[Dict[str, Any]], i: int) -> Tuple[Optional[int], str, str]:
    """(status_code, error_message, kind) of the reply to the call at index i."""
    nxt = chat[i + 1] if i + 1 < len(chat) else None
    if not nxt or nxt.get("role") != "tool":
        return None, "", "no_reply"
    try:
        j = json.loads(nxt["content"])
    except (TypeError, ValueError):
        return None, "", "no_status"
    if not isinstance(j, dict) or not isinstance(j.get("status_code"), int):
        return None, "", "no_status"
    resp = j.get("response")
    if isinstance(resp, dict) and str(resp.get("error", "")).startswith("Unknown tool name"):
        return j["status_code"], "", "unknown_tool"
    sc = j["status_code"]
    return sc, j.get("error_message") or "", ("2xx" if 200 <= sc < 300 else f"fail_{sc}")


# The simulator echoes the Python parser's message after one of these prefixes and a colon. A message
# like "Syntax error in rule_expression: ..." is a semantic check of an argument's content, not a parse error.
_SYNTAX_PREFIX = re.compile(r"^(Syntax error|syntax_error|Parse error|Tool call syntax error|Offending issue)\s*:", re.I)


def is_syntax_reply(err: str, parser_message: Optional[str]) -> bool:
    """Did the environment reject this call for a Python syntax error?"""
    return bool(parser_message and parser_message in err) or bool(_SYNTAX_PREFIX.match(err or ""))


def _merge(results, n_examples=2):
    tot: collections.Counter = collections.Counter()
    anomalies: List[Dict[str, Any]] = []
    examples: Dict[Any, list] = collections.defaultdict(list)
    for c, a, e in results:
        tot.update(c)
        anomalies.extend(a)
        for k, v in e.items():
            examples[k].extend(v[: max(0, n_examples - len(examples[k]))])
    return tot, anomalies, examples


# ---------------------------------------------------------------------------
# calls
# ---------------------------------------------------------------------------

def _calls_worker(rec):
    c, anomalies, ex = collections.Counter(), [], collections.defaultdict(list)
    traj = open_traj(rec)
    rid = f"{rec['task_id']}.s{rec['seed']}"
    if traj is None:
        c[("unreadable_file",)] += 1
        return c, anomalies, ex
    chat, turns = traj["solver_chat"], traj.get("turns") or []
    k = 0
    for i, tc in iter_calls(chat):
        conv = convert_call(tc)
        sc, err, kind = reply_of(chat, i)
        mine = conv.status + ("+stringified" if conv.stringified else "") + ("+positional" if conv.positional_ignored else "")
        c[("xtab", mine, kind)] += 1
        if len(ex[(mine, kind)]) < 2:
            ex[(mine, kind)].append((rid, i, tc[:140], err[:80]))
        if kind == "no_reply":
            anomalies.append({"id": rid, "i": i, "kind": "no_reply", "raw": tc[:200]})
            c[("ANOMALY", "no_reply")] += 1
        elif kind not in ("unknown_tool", "no_status") and (conv.status != OK) != is_syntax_reply(err, conv.error):
            anomalies.append({"id": rid, "i": i, "kind": "syntax_disagreement", "status": conv.status,
                              "env": kind, "env_err": err[:160], "my_err": conv.error, "raw": tc[:300]})
            c[("ANOMALY", "syntax_disagreement")] += 1
        elif kind in ("unknown_tool", "no_status"):
            c[("indeterminate", kind, conv.status)] += 1
        turn = turns[k] if k < len(turns) else None
        k += 1
        if turn is None:
            c[("ANOMALY", "no_turn_record")] += 1
            anomalies.append({"id": rid, "i": i, "kind": "no_turn_record", "raw": tc[:200]})
            continue
        if (turn.get("tool_call") or "").strip() != tc.strip():
            c[("ANOMALY", "turn_call_mismatch")] += 1
            anomalies.append({"id": rid, "i": i, "kind": "turn_call_mismatch", "raw": tc[:200], "turn": str(turn.get("tool_call"))[:200]})
        if conv.status == OK and turn.get("tool_name") not in (None, conv.name):
            c[("ANOMALY", "name_mismatch")] += 1
            anomalies.append({"id": rid, "i": i, "kind": "name_mismatch", "mine": conv.name, "turn": turn.get("tool_name")})
        if conv.status == OK:
            try:
                json.dumps(conv.tool_call(), ensure_ascii=False, allow_nan=False)
            except ValueError:
                c[("ANOMALY", "not_strict_json")] += 1
    if k != len(turns):
        c[("ANOMALY", "turn_count_mismatch")] += 1
        anomalies.append({"id": rid, "kind": "turn_count_mismatch", "tool_msgs": k, "turns": len(turns)})
    return c, anomalies, ex


def check_calls(a: argparse.Namespace) -> None:
    records = load_records(a.labels_dir, a.unsuccessful)
    print(f"trajectories {len(records)}", flush=True)
    with Pool(a.workers, initializer=_init, initargs=({"traj_dir": str(a.trajectories_dir)},)) as pool:
        results = pool.map(_calls_worker, records, chunksize=400)
    tot, anomalies, examples = _merge(results)
    print("\nconverter status x environment reply:")
    for mine, env, v in sorted(((k[1], k[2], v) for k, v in tot.items() if k[0] == "xtab"), key=lambda x: -x[2]):
        print(f"  {mine:28s} {env:14s} {v:9d}")
    print("\nindeterminate replies (unknown tool / null payload) by converter status:",
          {(k[1], k[2]): v for k, v in tot.items() if k[0] == "indeterminate"} or "none")
    print("anomalies:", {k[1]: v for k, v in tot.items() if k[0] == "ANOMALY"} or "none")
    print("\nexamples per cell:")
    for key, v in sorted(examples.items(), key=lambda kv: str(kv[0])):
        for e in v:
            print("  ", key, "|", e)
    a.report_dir.mkdir(parents=True, exist_ok=True)
    with open(a.report_dir / "anomalies.jsonl", "w") as f:
        for x in anomalies:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"\nanomalies written: {len(anomalies)} -> {a.report_dir / 'anomalies.jsonl'}")
    gt: collections.Counter = collections.Counter()
    with open(a.labels_dir / "task_content.valid.jsonl") as f:
        for line in f:
            for g in json.loads(line).get("gt_tool_calls") or []:
                c = convert_call(g)
                gt["plain_literal" if is_plain_literal_call(c) else c.status + ("+stringified" if c.stringified else "")] += 1
    print("ground-truth calls of the VALID tasks:", dict(gt))
    print("CALLS END")


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

def check_tools_file(tools_dataset: Path) -> None:
    import jsonschema

    def walk(s, acc):
        if isinstance(s, dict):
            t = s.get("type")
            if t == "float" or (isinstance(t, list) and "float" in t):
                acc.append("float_type_left")
            for k, v in s.items():
                if k == "required" and isinstance(v, bool):
                    acc.append("boolean_required_left")
                walk(v, acc)
        elif isinstance(s, list):
            for x in s:
                walk(x, acc)

    n, bad, ex = 0, collections.Counter(), []
    with open(tools_dataset) as f:
        for line in f:
            row = json.loads(line)
            fn = tool_to_function(row["tool"])
            n += 1
            if set(fn) != {"type", "function"} or set(fn["function"]) != {"name", "description", "parameters"}:
                bad["extra_keys"] += 1
            acc: List[str] = []
            walk(fn["function"]["parameters"], acc)
            for x in acc:
                bad[x] += 1
            try:
                jsonschema.Draft202012Validator.check_schema(fn["function"]["parameters"])
            except jsonschema.SchemaError as e:
                bad["invalid_schema"] += 1
                if len(ex) < 5:
                    ex.append((row["id"], str(e.message)[:120]))
            p = fn["function"]["parameters"]
            req = p.get("required", [])
            if len(req) != len(set(req)) or any(r not in p["properties"] for r in req):
                bad["required_not_in_properties"] += 1
    print(f"released tools {n}; structural problems: {dict(bad) or 'none'}")
    for e in ex:
        print("  ", e)


def _schemas_worker(rec):
    import jsonschema

    c, ex = collections.Counter(), collections.defaultdict(list)
    traj = open_traj(rec)
    if traj is None:
        return c, [], ex
    schemas = {t["tool_name"]: tool_to_function(t)["function"]["parameters"] for t in traj["tools"]}
    list_required = {t["tool_name"]: {n for n, p in (t.get("parameters") or {}).items() if isinstance(p, dict) and isinstance(p.get("required"), list)}
                     for t in traj["tools"]}
    validators: Dict[str, Any] = {}
    chat = traj["solver_chat"]
    for i, tc in iter_calls(chat):
        conv = convert_call(tc)
        if not is_plain_literal_call(conv) or conv.name not in schemas:
            continue
        sc, err, kind = reply_of(chat, i)
        if sc is None or kind == "unknown_tool":
            continue
        v = validators.get(conv.name) or validators.setdefault(conv.name, jsonschema.Draft202012Validator(schemas[conv.name]))
        errors = list(v.iter_errors(conv.arguments))
        kinds = ",".join(sorted({e.validator for e in errors}))
        outcome = ("accepted" if kind == "2xx" else
                   "rejected_type_or_required" if re.search(r"Invalid type|Missing required|expected (object|array|string|integer|number|boolean)", err) else
                   "rejected_other")
        c[("xtab", outcome, "valid" if not errors else "invalid:" + kinds)] += 1
        if errors and len(ex[(outcome, kinds)]) < 2:
            e = errors[0]
            ex[(outcome, kinds)].append((f"{rec['task_id']}.s{rec['seed']}", conv.name, "/".join(map(str, e.absolute_path))[:50], str(e.message)[:90], err[:70]))
        for name in list_required.get(conv.name, ()):
            c[("list_required", "accepted" if kind == "2xx" else "rejected", "present" if name in conv.arguments else "absent")] += 1
    return c, [], ex


def check_schemas(a: argparse.Namespace) -> None:
    check_tools_file(a.tools_dataset)
    records = load_records(a.labels_dir)
    with Pool(a.workers, initializer=_init, initargs=({"traj_dir": str(a.trajectories_dir)},)) as pool:
        results = pool.map(_schemas_worker, records, chunksize=400)
    tot, _, examples = _merge(results)
    print("\nplain-literal calls of SUCCESSFUL trajectories, environment outcome x validation against the converted schema:")
    for outcome, validity, v in sorted(((k[1], k[2], v) for k, v in tot.items() if k[0] == "xtab"), key=lambda x: (x[0], -x[2])):
        print(f"  {outcome:28s} {validity:60s} {v:9d}")
    acc = sum(v for k, v in tot.items() if k[0] == "xtab" and k[1] == "accepted")
    acc_valid = tot[("xtab", "accepted", "valid")]
    rej = sum(v for k, v in tot.items() if k[0] == "xtab" and k[1] == "rejected_type_or_required")
    rej_valid = tot[("xtab", "rejected_type_or_required", "valid")]
    print(f"\naccepted calls valid against the converted schema: {acc_valid}/{acc} = {acc_valid / max(acc, 1):.2%}")
    print(f"type/required-rejected calls invalid against it:   {rej - rej_valid}/{rej} = {(rej - rej_valid) / max(rej, 1):.2%}")
    print("object parameters with a required LIST and no flag (accepted/rejected x present/absent):",
          {(k[1], k[2]): v for k, v in tot.items() if k[0] == "list_required"})
    print("\nexamples:")
    for key, v in examples.items():
        for e in v:
            print("  ", key, "|", e)
    print("SCHEMAS END")


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------

def _assemble_worker(rec):
    c, ex = collections.Counter(), {}
    traj = open_traj(rec)
    rid = f"{rec['task_id']}.s{rec['seed']}"
    if traj is None:
        c[("drop", "unreadable_file")] += 1
        return c, [], ex
    for m in traj["solver_chat"][1:]:
        if m["role"] == "assistant":
            objs = extract_json_objects(m["content"]) if isinstance(m.get("content"), str) else []
            if len(objs) > 1:
                c[("fact", "assistant_reply_with_several_json_objects")] += 1
            if objs and isinstance(objs[0], dict):
                tc, r = objs[0].get("tool_call"), objs[0].get("reason")
                if isinstance(tc, str) and STOP in tc and tc.strip() != STOP:
                    c[("fact", "stop_with_extra_text")] += 1
                if isinstance(tc, str) and STOP not in tc and tc.strip() and not (isinstance(r, str) and r.strip()):
                    c[("fact", "call_with_empty_reason")] += 1
                if set(objs[0]) - {"reason", "tool_call"}:
                    c[("fact", "envelope_with_extra_keys")] += 1
        elif m["role"] == "tool" and '"explanation"' in m["content"]:
            c[("fact", "tool_reply_with_explanation")] += 1
    if (traj.get("ground_truth") or {}).get("summary") != _CFG["summaries"].get(rec["task_id"]):
        c[("fact", "summary_differs_from_task_content")] += 1
    conv, reason = assemble(traj)
    if conv is None:
        c[("drop", reason)] += 1
        return c, [], ex
    c[("kept",)] += 1
    c[("kept_calls",)] += conv["n_calls"]
    c[("kept_failed_calls",)] += conv["n_failed"]
    if conv["n_failed"]:
        c[("kept_with_recovery",)] += 1
    row = {"id": rid, "field": "", "gt_tool_calls": json.dumps(["x"]), "n_target_tokens": 2, "n_target_tokens_nothink": 1,
           "n_total_tokens": 3, "tools": json.dumps(conv["tools"]), "messages": json.dumps(conv["messages"])}
    problems = row_problems(row)
    if problems:
        c[("INVARIANT_VIOLATION",)] += 1
        ex[rid] = problems[:3]
    return c, [], ex


def check_assemble(a: argparse.Namespace) -> None:
    summaries = {}
    with open(a.labels_dir / "task_content.valid.jsonl") as f:
        for line in f:
            r = json.loads(line)
            summaries[r["id"]] = r["summary"]
    records = load_records(a.labels_dir)
    with Pool(a.workers, initializer=_init, initargs=({"traj_dir": str(a.trajectories_dir), "summaries": summaries},)) as pool:
        results = pool.map(_assemble_worker, records, chunksize=400)
    tot, _, examples = _merge(results)
    n, kept = len(records), tot[("kept",)]
    print(f"SUCCESSFUL trajectories {n}; kept {kept} ({kept / n:.1%}); dropped {n - kept}")
    print("drop reasons:")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        if k[0] == "drop":
            print(f"  {k[1]:28s} {v:7d}")
    print(f"kept rows: calls {tot[('kept_calls',)]}, recovered (non-2xx) calls {tot[('kept_failed_calls',)]}, rows with >=1 recovery {tot[('kept_with_recovery',)]}")
    print("invariant violations:", tot[("INVARIANT_VIOLATION",)], list(examples.items())[:3])
    print("facts over all raw chats:")
    for k, v in sorted(tot.items()):
        if k[0] == "fact":
            print(f"  {k[1]:40s} {v:7d}")
    print("ASSEMBLE END")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="check", required=True)
    for name in ("calls", "schemas", "assemble"):
        s = sub.add_parser(name)
        s.add_argument("--labels-dir", type=Path, required=True)
        s.add_argument("--trajectories-dir", type=Path, required=True)
        s.add_argument("--workers", type=int, default=24)
        if name == "calls":
            s.add_argument("--unsuccessful", action="store_true", help="also the UNSUCCESSFUL trajectories")
            s.add_argument("--report-dir", type=Path, default=Path("."))
        if name == "schemas":
            s.add_argument("--tools-dataset", type=Path, required=True)
    a = p.parse_args(argv)
    {"calls": check_calls, "schemas": check_schemas, "assemble": check_assemble}[a.check](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
