"""Environment audit orchestrator.

Build a flat tools dataset from env_generation scenarios, then score each tool's
reliability via a targeted simulator-evaluation loop.

Pipeline (all phases batched where they involve an LLM):
    1. Build   — flatten env_specs/*.json into tools_dataset.jsonl rows
                 (new rows get reliability=null, fresh per-tool eval log file)
    2. Metadata — per tool (N)          — generate fake world state
    3. Tests    — per tool (N)          — generate K test calls across fm1/fm2/fm3
    4. Check    — per (tool, test) N*K  — parameter_check each test call
    5. Simulate — per passing (tool, test)  — simulate tool response
    6. Judge    — per (tool, test) N*K  — judge correctness with failure_mode
    7. Aggregate + write reliability + per-mode breakdown back to jsonl

Per-tool checkpoint file `tool_eval_logs/{tool_id}.json` carries a
`phase_completed` field; each phase processes only tools/pairs where
`phase_completed < current_phase`, so crash-recovery is natural.

Usage:
    from llm import LLM
    from env_audit.generate import audit_tools

    llm = LLM("GPT-OSS-120B")
    audit_tools(
        env_specs_dir=Path("tool_content/env_specs"),
        dataset_path=Path("tool_content/tools_dataset.jsonl"),
        eval_logs_dir=Path("tool_content/tool_eval_logs"),
        llm=llm,
    )
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from env_audit.utils import (
    eval_log_path as _eval_log_path,
    load_eval_log as _load_eval_log,
    load_jsonl as _load_jsonl,
    model_config_for as _model_config_for,
    now_iso as _now_iso,
    refresh_env_spec_audit,
    save_eval_log as _save_eval_log,
    write_jsonl as _write_jsonl,
)
from roles.env_generator import EnvironmentGenerator
from roles.tool_simulator import ToolSimulator
from roles.judge_simulator import JudgeSimulator
from utils import (
    batch_call,
    extract_json_objects,
    get_logger,
    UsageTracker,
    usage_to_dict,
)

logger = get_logger("synthtools")

EVAL_LOG_SCHEMA_VERSION = "env_audit_log.v1"
DATASET_SCHEMA_VERSION = "tools_dataset.v1"
EVAL_VERSION = "env_audit.v1"


# ---------------------------------------------------------------------------
# Phase 1: Build
# ---------------------------------------------------------------------------

def _collect_tools_from_env_specs(env_specs_dir: Path) -> List[Dict]:
    """Walk env_specs_dir and flatten to {id, field, subfield, task, tool_name, tool, spec_id} rows."""
    rows = []
    for p in sorted(env_specs_dir.glob("*_spec_*.json")):
        try:
            spec = json.loads(p.read_text())
        except Exception as e:
            logger.warning(f"Skipping unreadable spec {p.name}: {e}")
            continue
        spec_id = spec.get("spec_id") or p.stem
        field = spec.get("field", "")
        subfield = spec.get("subfield", "")
        task = spec.get("task", "")
        for tool in spec.get("tools", []):
            if not isinstance(tool, dict):
                continue
            name = tool.get("tool_name")
            if not name:
                continue
            rows.append({
                "id": f"{spec_id}.{name}",
                "spec_id": spec_id,
                "field": field,
                "subfield": subfield,
                "task": task,
                "tool_name": name,
                "tool": tool,
            })
    return rows


def build_dataset(
    env_specs_dir: Path,
    dataset_path: Path,
    eval_logs_dir: Path,
    llm,
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Flatten env_specs_dir → append new tools to dataset_path, seed eval logs.

    Pure bookkeeping, no LLM calls. Idempotent: tools already in dataset are kept
    as-is (reliability values preserved). New tools get reliability=null and a
    fresh eval_log at phase_completed=1.

    `fields`: optional list of field names to keep (exact match). Specs whose
    `field` isn't in the list are skipped.

    Returns `{"n_discovered": N, "n_already_in_dataset": M, "n_new": K, "n_skipped_by_field": S}`.
    """
    env_specs_dir = Path(env_specs_dir)
    dataset_path = Path(dataset_path)
    eval_logs_dir = Path(eval_logs_dir)
    eval_logs_dir.mkdir(parents=True, exist_ok=True)

    field_set = set(fields) if fields else None

    existing_rows = _load_jsonl(dataset_path)
    existing_ids = {r["id"] for r in existing_rows if "id" in r}

    discovered_all = _collect_tools_from_env_specs(env_specs_dir)
    if field_set is not None:
        discovered = [r for r in discovered_all if r["field"] in field_set]
        n_skipped_by_field = len(discovered_all) - len(discovered)
    else:
        discovered = discovered_all
        n_skipped_by_field = 0

    new_rows = []
    n_new = 0
    model_config = _model_config_for(llm)
    now = _now_iso()

    for row in discovered:
        if row["id"] in existing_ids:
            continue
        n_new += 1
        new_rows.append({
            "schema_version": DATASET_SCHEMA_VERSION,
            "id": row["id"],
            "field": row["field"],
            "subfield": row["subfield"],
            "task": row["task"],
            "tool_name": row["tool_name"],
            "tool": row["tool"],
            "reliability": None,
            "reliability_per_mode": None,
            "eval_version": None,
            "evaluated_at": None,
            "model": None,
        })
        log_path = _eval_log_path(eval_logs_dir, row["id"])
        if not log_path.exists():
            _save_eval_log(log_path, {
                "schema_version": EVAL_LOG_SCHEMA_VERSION,
                "tool_id": row["id"],
                "spec_id": row["spec_id"],
                "field": row["field"],
                "subfield": row["subfield"],
                "task": row["task"],
                "tool": row["tool"],
                "generated_at": now,
                "model": llm.model,
                "model_config": model_config,
                "phase_completed": 1,
                "metadata": None,
                "test_calls": [],
                "results": [],
                "reliability": None,
                "reliability_per_mode": None,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

    # Skip the dataset rewrite when nothing new to add — critical for
    # concurrent --evaluate-only jobs that each call build_dataset up front.
    if new_rows:
        full_rows = existing_rows + new_rows
        _write_jsonl(dataset_path, full_rows)
    n_already = len(discovered) - n_new
    logger.info(
        f"[build] discovered={len(discovered_all)} kept_by_field={len(discovered)} "
        f"already_in_dataset={n_already} new={n_new}"
    )
    return {
        "n_discovered": len(discovered_all),
        "n_already_in_dataset": n_already,
        "n_new": n_new,
        "n_skipped_by_field": n_skipped_by_field,
    }


# ---------------------------------------------------------------------------
# Phase 2: Metadata
# ---------------------------------------------------------------------------

def _phase2_metadata(
    tools_pending: List[Dict],
    role: EnvironmentGenerator,
    llm,
    eval_logs_dir: Path,
    tracker: UsageTracker,
) -> None:
    """For every tool at phase_completed<2, generate metadata via one batched LLM call."""
    targets = [t for t in tools_pending if t["phase_completed"] < 2]
    if not targets:
        return
    prompts = [
        role.get_prompt("metadata", Data=json.dumps(_metadata_input(t), indent=2))
        for t in targets
    ]
    results = batch_call(llm, prompts)
    for t, r in zip(targets, results):
        tracker.track(r["usage"])
        objs = extract_json_objects(r["response"])
        t["metadata"] = objs[0] if objs else {}
        t["phase_completed"] = 2
        _bump_usage(t, r["usage"])
        _save_eval_log(_eval_log_path(eval_logs_dir, t["tool_id"]), t)
    logger.info(f"[phase 2] metadata generated for {len(targets)} tools")


def _metadata_input(t: Dict) -> Dict[str, Any]:
    return {
        "field_name": t.get("field"),
        "subfield": t.get("subfield"),
        "task": t.get("task"),
        "tools": [t["tool"]],
    }


# ---------------------------------------------------------------------------
# Phase 3: Test-call generation
# ---------------------------------------------------------------------------

def _phase3_test_calls(
    tools_pending: List[Dict],
    tool_tester_template: str,
    llm,
    eval_logs_dir: Path,
    tracker: UsageTracker,
) -> None:
    """For every tool at phase_completed<3, generate K test calls across 3 failure modes."""
    targets = [t for t in tools_pending if t["phase_completed"] < 3]
    if not targets:
        return
    prompts = [tool_tester_template.format(tool_details=json.dumps(t["tool"], ensure_ascii=False)) for t in targets]
    results = batch_call(llm, prompts)
    for t, r in zip(targets, results):
        tracker.track(r["usage"])
        t["test_calls"] = _parse_test_calls(r["response"])
        t["phase_completed"] = 3
        _bump_usage(t, r["usage"])
        _save_eval_log(_eval_log_path(eval_logs_dir, t["tool_id"]), t)
    logger.info(f"[phase 3] test-calls generated for {len(targets)} tools "
                f"(avg {sum(len(t['test_calls']) for t in targets) / max(1, len(targets)):.1f} per tool)")


def _parse_test_calls(response: str) -> List[Dict]:
    """Extract the test-call dict from the LLM response and normalize to a flat list.

    Output shape:
        [{"idx": 0, "failure_mode_group": "fm1|fm2|fm3", "failure_mode": "...",
          "parameters": {...}, "tool_call_message": "ToolName(...)"}]
    """
    objs = extract_json_objects(response)
    if not objs or not isinstance(objs[0], dict):
        return []
    raw = objs[0]
    out = []
    for idx, (key, entry) in enumerate(raw.items()):
        if not isinstance(entry, dict):
            continue
        fm = entry.get("Failure mode") or entry.get("failure_mode") or ""
        params = entry.get("Tool parameters") or entry.get("tool_parameters") or {}
        message = entry.get("Tool call message") or entry.get("tool_call_message") or ""
        # Clean surrounding brackets like "[TOOL_NAME(...)]" → "TOOL_NAME(...)"
        if isinstance(message, str):
            message = message.strip()
            if message.startswith("[") and message.endswith("]"):
                message = message[1:-1]
        out.append({
            "idx": idx,
            "failure_mode_group": _classify_failure_mode(fm),
            "failure_mode": fm,
            "parameters": params,
            "tool_call_message": message,
        })
    return out


def _classify_failure_mode(fm: str) -> str:
    """Heuristic: map a free-text failure_mode to fm1/fm2/fm3."""
    if not fm:
        return "fm3"
    low = fm.lower()
    fm1_markers = ["missing", "misspell", "wrong type", "unknown param", "range", "out of bounds",
                   "schema", "invalid type", "size", "enum invalid"]
    fm2_markers = ["cross-field", "mutually exclusive", "quota", "limit", "rate", "temporal",
                   "date", "dependency", "contradict"]
    for m in fm1_markers:
        if m in low:
            return "fm1"
    for m in fm2_markers:
        if m in low:
            return "fm2"
    return "fm3"


# ---------------------------------------------------------------------------
# Phase 4: Parameter check
# ---------------------------------------------------------------------------

def _phase4_param_check(
    tools_pending: List[Dict],
    sim_role: ToolSimulator,
    llm,
    eval_logs_dir: Path,
    tracker: UsageTracker,
) -> None:
    """For every (tool, test_call) pair where tool's phase_completed<4, run a batched param-check."""
    targets = [t for t in tools_pending if t["phase_completed"] < 4]
    if not targets:
        return
    # Flatten into one big batch
    pairs: List[Tuple[Dict, Dict]] = []
    for t in targets:
        for tc in t.get("test_calls", []):
            pairs.append((t, tc))
    if not pairs:
        # Tools with no parsed test calls — still bump phase so we don't retry
        for t in targets:
            t["phase_completed"] = 4
            _save_eval_log(_eval_log_path(eval_logs_dir, t["tool_id"]), t)
        logger.info("[phase 4] no test calls to check — bumped phase")
        return

    prompts = [sim_role.get_prompt(
        "parameter_check",
        tool_name=t["tool"].get("tool_name", ""),
        tool_description=_fmt(t["tool"].get("tool_description", "")),
        parameters=_fmt(t["tool"].get("parameters", {})),
        error_messages=_fmt(t["tool"].get("error_messages", [])),
        usage=_fmt(t["tool"].get("usage", "")),
    ) + "\n" + tc["tool_call_message"] for t, tc in pairs]

    results = batch_call(llm, prompts)
    # Zip results back onto tools → results list on each tool
    by_tool: Dict[str, Dict] = {t["tool_id"]: t for t in targets}
    for t in targets:
        t.setdefault("results", [])
        t["results"] = []  # reset — this phase (re)writes them
    for (t, tc), r in zip(pairs, results):
        tracker.track(r["usage"])
        objs = extract_json_objects(r["response"])
        parsed = objs[0] if objs else None
        passed = False
        if isinstance(parsed, dict):
            passed = (parsed.get("status") == "PASS" or parsed.get("status_code") == 200)
        if not passed:
            passed = ("Status: PASS" in r["response"] or "Status Code: 200" in r["response"])
        by_tool[t["tool_id"]]["results"].append({
            "idx": tc["idx"],
            "param_check": parsed if parsed is not None else {"status": "FAIL", "error_message": "unparseable"},
            "param_check_passed": passed,
            "simulator_response": None,
            "judgment": None,
            "correct": None,
        })
        _bump_usage(by_tool[t["tool_id"]], r["usage"])
    for t in targets:
        t["phase_completed"] = 4
        _save_eval_log(_eval_log_path(eval_logs_dir, t["tool_id"]), t)
    logger.info(f"[phase 4] parameter-checked {len(pairs)} (tool, test_call) pairs across {len(targets)} tools")


def _fmt(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


# ---------------------------------------------------------------------------
# Phase 5: Simulate
# ---------------------------------------------------------------------------

def _phase5_simulate(
    tools_pending: List[Dict],
    sim_role: ToolSimulator,
    llm,
    eval_logs_dir: Path,
    tracker: UsageTracker,
) -> None:
    """Simulate only test calls that passed the parameter check."""
    targets = [t for t in tools_pending if t["phase_completed"] < 5]
    if not targets:
        return
    # Gather passing (tool, result_entry, test_call) triples
    triples: List[Tuple[Dict, Dict, Dict]] = []
    for t in targets:
        test_by_idx = {tc["idx"]: tc for tc in t.get("test_calls", [])}
        for entry in t.get("results", []):
            if entry.get("param_check_passed"):
                tc = test_by_idx.get(entry["idx"])
                if tc:
                    triples.append((t, entry, tc))

    if not triples:
        for t in targets:
            t["phase_completed"] = 5
            _save_eval_log(_eval_log_path(eval_logs_dir, t["tool_id"]), t)
        logger.info("[phase 5] no passing param_check — bumped phase")
        return

    prompts = []
    for t, _, tc in triples:
        prompt = sim_role.get_prompt(
            "simulate",
            tool_name=t["tool"].get("tool_name", ""),
            tool_description=_fmt(t["tool"].get("tool_description", "")),
            parameters=_fmt(t["tool"].get("parameters", {})),
            error_messages=_fmt(t["tool"].get("error_messages", [])),
            usage=_fmt(t["tool"].get("usage", "")),
            initial_config=_fmt(t["tool"].get("initial_config", {})),
            tool_call=_fmt(tc["tool_call_message"]),
            output_details=_fmt(t["tool"].get("output_details", {})),
            metadata=_fmt(t.get("metadata") or {}),
        )
        prompts.append(prompt)

    results = batch_call(llm, prompts)
    for (t, entry, _), r in zip(triples, results):
        tracker.track(r["usage"])
        objs = extract_json_objects(r["response"])
        entry["simulator_response"] = objs[0] if objs else {"raw": r["response"]}
        _bump_usage(t, r["usage"])
    for t in targets:
        t["phase_completed"] = 5
        _save_eval_log(_eval_log_path(eval_logs_dir, t["tool_id"]), t)
    logger.info(f"[phase 5] simulated {len(triples)} tool calls across {len(targets)} tools")


# ---------------------------------------------------------------------------
# Phase 6: Judge
# ---------------------------------------------------------------------------

def _phase6_judge(
    tools_pending: List[Dict],
    judge_role: JudgeSimulator,
    llm,
    eval_logs_dir: Path,
    tracker: UsageTracker,
) -> None:
    """Judge every (tool, test_call) pair, passing failure_mode to the judge prompt."""
    targets = [t for t in tools_pending if t["phase_completed"] < 6]
    if not targets:
        return
    pairs: List[Tuple[Dict, Dict, Dict]] = []
    for t in targets:
        test_by_idx = {tc["idx"]: tc for tc in t.get("test_calls", [])}
        for entry in t.get("results", []):
            tc = test_by_idx.get(entry["idx"])
            if tc:
                pairs.append((t, entry, tc))

    if not pairs:
        for t in targets:
            t["phase_completed"] = 6
            _save_eval_log(_eval_log_path(eval_logs_dir, t["tool_id"]), t)
        logger.info("[phase 6] no results to judge — bumped phase")
        return

    prompts = []
    for t, entry, tc in pairs:
        # If simulator didn't run, pass the param_check result as the "response"
        response_for_judge = entry["simulator_response"] if entry["simulator_response"] else entry["param_check"]
        prompts.append(judge_role.get_prompt(
            "judge",
            tool_details=_fmt(t["tool"]),
            message=_fmt(tc["tool_call_message"]),
            response=_fmt(response_for_judge),
            meta_data=_fmt(t.get("metadata") or {}),
            failure_mode=_fmt(tc.get("failure_mode") or "none"),
        ))

    results = batch_call(llm, prompts)
    for (t, entry, _), r in zip(pairs, results):
        tracker.track(r["usage"])
        objs = extract_json_objects(r["response"])
        judgment = objs[0] if objs else {"error": "could_not_parse", "raw": r["response"]}
        entry["judgment"] = judgment
        entry["correct"] = (isinstance(judgment, dict) and judgment.get("judgment") == "correct")
        _bump_usage(t, r["usage"])
    for t in targets:
        t["phase_completed"] = 6
        _save_eval_log(_eval_log_path(eval_logs_dir, t["tool_id"]), t)
    logger.info(f"[phase 6] judged {len(pairs)} (tool, test_call) pairs across {len(targets)} tools")


# ---------------------------------------------------------------------------
# Phase 7: Aggregate + write
# ---------------------------------------------------------------------------

def _compute_reliability_from_results(t: Dict) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    """Given an eval log dict (with `results` + `test_calls`), return overall
    reliability + per-failure-mode reliability. Kept standalone so sweep_all
    can recompute from finished eval logs without re-running phase 7."""
    results = t.get("results", [])
    n_total = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    reliability = n_correct / n_total if n_total > 0 else None

    test_by_idx = {tc["idx"]: tc for tc in t.get("test_calls", [])}
    per_mode: Dict[str, Tuple[int, int]] = {"fm1": (0, 0), "fm2": (0, 0), "fm3": (0, 0)}
    for r in results:
        tc = test_by_idx.get(r["idx"])
        if not tc:
            continue
        grp = tc.get("failure_mode_group") or "fm3"
        correct = bool(r.get("correct"))
        total, ok = per_mode.get(grp, (0, 0))
        per_mode[grp] = (total + 1, ok + (1 if correct else 0))
    reliability_per_mode = {
        grp: (ok / total if total > 0 else None)
        for grp, (total, ok) in per_mode.items()
    }
    return reliability, reliability_per_mode


def _phase7_aggregate(
    tools_pending: List[Dict],
    dataset_path: Path,
    env_specs_dir: Path,
    eval_logs_dir: Path,
    llm,
    skip_dataset_write: bool = False,
) -> None:
    targets = [t for t in tools_pending if t["phase_completed"] < 7]
    if not targets:
        return

    for t in targets:
        reliability, reliability_per_mode = _compute_reliability_from_results(t)
        t["reliability"] = reliability
        t["reliability_per_mode"] = reliability_per_mode
        t["phase_completed"] = 7
        _save_eval_log(_eval_log_path(eval_logs_dir, t["tool_id"]), t)

    if not skip_dataset_write:
        # Rewrite dataset with updated per-tool fields
        dataset_rows = _load_jsonl(dataset_path)
        reliability_by_id = {t["tool_id"]: t["reliability"] for t in targets}
        per_mode_by_id = {t["tool_id"]: t["reliability_per_mode"] for t in targets}
        now = _now_iso()
        model = llm.model
        for row in dataset_rows:
            if row["id"] in reliability_by_id:
                row["reliability"] = reliability_by_id[row["id"]]
                row["reliability_per_mode"] = per_mode_by_id[row["id"]]
                row["eval_version"] = EVAL_VERSION
                row["evaluated_at"] = now
                row["model"] = model
        _write_jsonl(dataset_path, dataset_rows)

    # Refresh the audit block of every env_spec touched this phase.
    # Scoped to `targets`' specs only — disjoint across parallel --fields jobs.
    affected_specs = {t["spec_id"] for t in targets if t.get("spec_id")}
    llm_model_config = _model_config_for(llm)
    for spec_id in affected_specs:
        try:
            refresh_env_spec_audit(env_specs_dir, spec_id, eval_logs_dir, llm.model, llm_model_config)
        except FileNotFoundError as e:
            logger.warning(f"env_spec for {spec_id} not found; skipping audit refresh: {e}")
    dataset_note = "" if not skip_dataset_write else " (dataset rewrite skipped)"
    logger.info(f"[phase 7] aggregated + wrote reliability for {len(targets)} tools "
                f"(refreshed audit on {len(affected_specs)} env_specs){dataset_note}")


# ---------------------------------------------------------------------------
# Tracking utility
# ---------------------------------------------------------------------------

def _bump_usage(t: Dict, usage: Any) -> None:
    d = usage_to_dict(usage) or {}
    u = t.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    u["prompt_tokens"] += d.get("prompt_tokens", 0) or 0
    u["completion_tokens"] += d.get("completion_tokens", 0) or 0
    u["total_tokens"] = u["prompt_tokens"] + u["completion_tokens"]


def _sweep_refresh_all_audit_blocks(env_specs_dir: Path, eval_logs_dir: Path, llm) -> None:
    """Walk every eval_log; refresh the audit block of every spec that has any.

    Cheap (no LLM). Idempotent: re-running with unchanged eval_logs produces the
    same env_spec content. Backfills env_specs that were audited before the
    audit-block writer existed.
    """
    spec_ids = set()
    for p in eval_logs_dir.glob("*.json"):
        log = _load_eval_log(p)
        if log and log.get("spec_id"):
            spec_ids.add(log["spec_id"])
    if not spec_ids:
        return
    model_config = _model_config_for(llm)
    refreshed = 0
    for spec_id in sorted(spec_ids):
        try:
            refresh_env_spec_audit(env_specs_dir, spec_id, eval_logs_dir, llm.model, model_config)
            refreshed += 1
        except FileNotFoundError:
            logger.warning(f"sweep: env_spec for {spec_id} not found, skipping")
    logger.info(f"sweep: refreshed audit block on {refreshed}/{len(spec_ids)} env_specs")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def evaluate_tools(
    dataset_path: Path,
    env_specs_dir: Path,
    eval_logs_dir: Path,
    llm,
    ids: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    skip_dataset_write: bool = False,
    skip_sweep: bool = False,
) -> Dict[str, Any]:
    """Phases 2-7 — evaluate un-evaluated tools; filters applied as intersection.

    Pending set = tools with phase_completed < 7 AND matching filters.
    Per-tool eval logs are updated as each phase completes; env_specs get their
    `audit` block refreshed whenever a tool reaches phase 7.

    `skip_dataset_write` + `skip_sweep` = True in parallel `--evaluate-only` mode:
    phase 7 skips the global dataset rewrite, and the end-of-run sweep is
    omitted. Run `sweep_all()` serially afterwards to fold in every job's work.

    Returns `{"n_pending": ..., "n_skipped_by_filter": ..., "total_tokens": ..., "elapsed_s": ...}`.
    """
    dataset_path = Path(dataset_path)
    env_specs_dir = Path(env_specs_dir)
    eval_logs_dir = Path(eval_logs_dir)
    eval_logs_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(llm, "_ensure_engine"):
        llm._ensure_engine()
    logger.info(f"Model: {llm.model}")
    start = time.time()
    tracker = UsageTracker()

    id_set = set(ids) if ids else None
    field_set = set(fields) if fields else None

    all_unfinished: List[Dict] = []
    for p in sorted(eval_logs_dir.glob("*.json")):
        eval_log = _load_eval_log(p)
        if eval_log is None or eval_log.get("phase_completed", 0) >= 7:
            continue
        all_unfinished.append(eval_log)

    pending: List[Dict] = []
    for log in all_unfinished:
        if id_set is not None and log.get("tool_id") not in id_set:
            continue
        if field_set is not None and log.get("field") not in field_set:
            continue
        pending.append(log)
    n_skipped_by_filter = len(all_unfinished) - len(pending)

    logger.info(f"Pending: {len(pending)}  (filtered out: {n_skipped_by_filter})")
    if not pending:
        # Even with nothing to evaluate, sweep audit blocks so env_specs reflect
        # the current state of eval_logs (handles backfill / manual edits).
        if not skip_sweep:
            _sweep_refresh_all_audit_blocks(env_specs_dir, eval_logs_dir, llm)
        return {
            "n_pending": 0,
            "n_skipped_by_filter": n_skipped_by_filter,
            "total_tokens": tracker.total["total_tokens"],
            "elapsed_s": round(time.time() - start, 2),
        }

    eg_role = EnvironmentGenerator(_env_generator_template_files(), llm)
    tool_tester_template = _load_tool_tester_template()
    sim_role = ToolSimulator(llm)
    judge_role = JudgeSimulator(llm)

    _phase2_metadata(pending, eg_role, llm, eval_logs_dir, tracker)
    _phase3_test_calls(pending, tool_tester_template, llm, eval_logs_dir, tracker)
    _phase4_param_check(pending, sim_role, llm, eval_logs_dir, tracker)
    _phase5_simulate(pending, sim_role, llm, eval_logs_dir, tracker)
    _phase6_judge(pending, judge_role, llm, eval_logs_dir, tracker)
    _phase7_aggregate(pending, dataset_path, env_specs_dir, eval_logs_dir, llm,
                      skip_dataset_write=skip_dataset_write)

    if not skip_sweep:
        _sweep_refresh_all_audit_blocks(env_specs_dir, eval_logs_dir, llm)

    elapsed = round(time.time() - start, 2)
    totals = tracker.total
    logger.info(f"Total usage: prompt={totals['prompt_tokens']}, completion={totals['completion_tokens']}, total={totals['total_tokens']}")
    logger.info(f"Run time: {elapsed}s")
    return {
        "n_pending": len(pending),
        "n_skipped_by_filter": n_skipped_by_filter,
        "total_tokens": totals["total_tokens"],
        "elapsed_s": elapsed,
    }


def sweep_all(
    dataset_path: Path,
    env_specs_dir: Path,
    eval_logs_dir: Path,
    llm=None,
) -> Dict[str, Any]:
    """Rebuild `tools_dataset.jsonl` + refresh every env_spec `audit` block
    from every eval log at `phase_completed == 7`. No LLM calls.

    Intended to run once after N parallel `--evaluate-only` jobs complete,
    folding each job's per-tool eval_logs into the two shared artifacts.

    Idempotent: running twice with unchanged eval_logs produces the same files.
    """
    dataset_path = Path(dataset_path)
    env_specs_dir = Path(env_specs_dir)
    eval_logs_dir = Path(eval_logs_dir)

    existing_rows = _load_jsonl(dataset_path)
    existing_by_id = {r["id"]: r for r in existing_rows if "id" in r}

    now = _now_iso()
    # Pick model info from any completed eval log (all should agree on the model).
    model_name = llm.model if llm is not None else None
    model_config = _model_config_for(llm) if llm is not None else None

    updated = 0
    for p in sorted(eval_logs_dir.glob("*.json")):
        log = _load_eval_log(p)
        if log is None or log.get("phase_completed", 0) < 7:
            continue
        tool_id = log.get("tool_id")
        if tool_id not in existing_by_id:
            continue
        if model_name is None:
            model_name = log.get("model")
            model_config = log.get("model_config")
        row = existing_by_id[tool_id]
        row["reliability"] = log.get("reliability")
        row["reliability_per_mode"] = log.get("reliability_per_mode")
        row["eval_version"] = EVAL_VERSION
        row["evaluated_at"] = now
        row["model"] = log.get("model") or model_name
        updated += 1
    _write_jsonl(dataset_path, list(existing_by_id.values()))

    # Now refresh audit blocks for every spec that has an eval log.
    # Use the model/config picked above (or a sentinel if nothing was found).
    class _Shim:
        def __init__(self, model, cfg):
            self.model = model
            self.cfg = type("Cfg", (), cfg or {})() if cfg else None
            self.max_tokens = (cfg or {}).get("max_tokens")
    shim_llm = llm if llm is not None else _Shim(model_name or "unknown", model_config)
    _sweep_refresh_all_audit_blocks(env_specs_dir, eval_logs_dir, shim_llm)

    logger.info(f"[sweep] rebuilt {updated} dataset rows from eval_logs")
    return {"n_dataset_rows_updated": updated, "n_dataset_rows_total": len(existing_rows)}


def audit_tools(
    env_specs_dir: Path,
    dataset_path: Path,
    eval_logs_dir: Path,
    llm,
    fields: Optional[List[str]] = None,
    ids: Optional[List[str]] = None,
    mode: str = "full",
) -> Dict[str, Any]:
    """Build + evaluate tools. Idempotent.

    `mode`:
      - "full" (default): build → evaluate → sweep (monolithic, unchanged).
      - "build": phase 1 only. No LLM. Seeds dataset + eval_log stubs.
      - "evaluate": phases 2–6 + per-spec refresh in phase 7. Skips global dataset rewrite AND end-of-run sweep. Safe for N concurrent jobs with disjoint `--fields`.
      - "sweep": no LLM. Rebuild dataset + refresh every env_spec audit block from eval_logs.
    """
    if mode not in ("full", "build", "evaluate", "sweep"):
        raise ValueError(f"mode must be one of full|build|evaluate|sweep, got {mode!r}")

    if mode == "sweep":
        return sweep_all(
            dataset_path=dataset_path,
            env_specs_dir=env_specs_dir,
            eval_logs_dir=eval_logs_dir,
            llm=llm,
        )

    build_summary = build_dataset(
        env_specs_dir=env_specs_dir,
        dataset_path=dataset_path,
        eval_logs_dir=eval_logs_dir,
        llm=llm,
        fields=fields,
    )

    if mode == "build":
        return {
            "mode": "build",
            "n_new_tools": build_summary["n_new"],
            "n_discovered": build_summary["n_discovered"],
            "n_skipped_by_field": build_summary["n_skipped_by_field"],
        }

    skip_globals = (mode == "evaluate")
    eval_summary = evaluate_tools(
        dataset_path=dataset_path,
        env_specs_dir=env_specs_dir,
        eval_logs_dir=eval_logs_dir,
        llm=llm,
        ids=ids,
        fields=fields,
        skip_dataset_write=skip_globals,
        skip_sweep=skip_globals,
    )
    return {
        "mode": mode,
        "n_new_tools": build_summary["n_new"],
        "n_discovered": build_summary["n_discovered"],
        "n_skipped_by_field": build_summary["n_skipped_by_field"],
        "n_evaluated": eval_summary["n_pending"],
        "n_skipped_by_filter": eval_summary["n_skipped_by_filter"],
        "total_tokens": eval_summary["total_tokens"],
        "elapsed_s": eval_summary["elapsed_s"],
    }


def _env_generator_template_files() -> Dict[str, str]:
    base = Path(__file__).resolve().parent.parent / "prompt_templates" / "env_generator"
    return {
        # Only need metadata for env_audit; placeholders for the other keys so
        # EnvironmentGenerator can still instantiate cleanly if callers dispatch.
        "subfield":  str(base / "subfield.yml"),
        "task":      str(base / "task.yml"),
        "tool":      str(base / "tool.yml"),
        "sequences": str(base / "sequences.yml"),
        "metadata":  str(base / "metadata.yml"),
    }


def _load_tool_tester_template() -> str:
    path = Path(__file__).resolve().parent.parent / "prompt_templates" / "tool_simulator" / "test_calls.yml"
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["template"]
