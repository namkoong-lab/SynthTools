"""Environment generation orchestrator.

Usage:
    from llm import LLM
    from env_generation.generate import generate_environments

    llm = LLM("GPT-OSS-120B")
    generate_environments(
        fields=["Aerospace and Defense"],
        output_dir=Path("tool_content/env_specs"),
        llm=llm,
        max_subfields=2,
        max_tasks_per_subfield=3,
    )

Pipeline (per field):
    1. field -> subfields               (1 call, serial)
    2. subfield -> tasks                (N calls, batched)
    3. (subfield, task) -> tools        (N*M calls, batched)

Sequences (formerly phase 4) are NOT generated here anymore — they are produced
post-audit by `task_generation.build_sequences`, which can filter the tool pool
by reliability before asking the LLM to chain tools. The scenario JSON we save
includes `sequences: {}` as a placeholder; build_sequences fills it in later.

One JSON file per scenario = `(field, subfield, task)`:
    `{field_slug}_spec_{NNN:03d}.json`
Plus one sibling `{field_slug}_field_gen.json` per field carrying the shared
phase-1 and phase-2 LLM calls, so scenarios don't duplicate those strings.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from roles.env_generator import EnvironmentGenerator
from utils import (
    batch_call,
    extract_json_objects,
    get_logger,
    next_artifact_index,
    parse_list,
    UsageTracker,
    usage_to_dict,
    write_json_atomic,
)

logger = get_logger("synthtools")

SPEC_SCHEMA_VERSION = "env_spec.v1"
FIELD_GEN_SCHEMA_VERSION = "env_field_gen.v1"
MANIFEST_SCHEMA_VERSION = "env_manifest.v1"
MANIFEST_FILENAME = "manifest.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(field: str) -> str:
    return field.replace(" ", "_").replace("&", "and").lower()


def _unique(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items or []:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _phase_entry(phase: str, prompt: str, response: str, usage: Any, parsed: Any = None) -> Dict[str, Any]:
    return {
        "phase": phase,
        "prompt": prompt,
        "response": response,
        "parsed": parsed,
        "usage": usage_to_dict(usage),
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _model_config_for(llm) -> Dict[str, Any]:
    """Extract reproducibility-relevant sampling config from an LLM instance."""
    cfg = getattr(llm, "cfg", None)
    return {
        "temperature": getattr(cfg, "temperature", None),
        "top_p": getattr(cfg, "top_p", None),
        "max_tokens": getattr(llm, "max_tokens", None),
    }


def _append_manifest(output_dir: Path, entry: Dict[str, Any]) -> None:
    """Append one scenario summary to manifest.jsonl (creates it if missing)."""
    path = output_dir / MANIFEST_FILENAME
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _existing_spec_count(output_dir: Path, field_slug: str) -> int:
    """Count `{field_slug}_spec_NNN.json` files already on disk for top-up re-runs."""
    if not output_dir.exists():
        return 0
    pattern = re.compile(rf"^{re.escape(field_slug)}_spec_(\d+)\.json$")
    return sum(1 for p in output_dir.iterdir() if pattern.match(p.name))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def generate_environments(
    fields: List[str],
    output_dir: Path,
    llm,
    max_subfields: int = 1,
    max_tasks_per_subfield: int = 1,
) -> List[Dict]:
    """Generate scenario specs for one or more fields.

    Args:
        fields: List of field names.
        output_dir: Directory for scenario JSON files.
        llm: LLM instance (single + batched callable).
        max_subfields: How many subfields to expand per field.
        max_tasks_per_subfield: How many tasks per subfield.

    Returns:
        List of saved scenario dicts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    role = EnvironmentGenerator(_default_template_files(), llm)

    # Warm up the engine once (matches task_generation behaviour).
    if hasattr(llm, "_ensure_engine"):
        llm._ensure_engine()
    logger.info(f"Model: {llm.model}")

    saved: List[Dict] = []
    for field in fields:
        saved.extend(_generate_for_field(
            field=field,
            output_dir=output_dir,
            llm=llm,
            role=role,
            max_subfields=max_subfields,
            max_tasks_per_subfield=max_tasks_per_subfield,
        ))
    return saved


def _generate_for_field(
    field: str,
    output_dir: Path,
    llm,
    role: EnvironmentGenerator,
    max_subfields: int,
    max_tasks_per_subfield: int,
) -> List[Dict]:
    field_slug = _slug(field)
    logger.info(f"Field: {field}")

    existing_count = _existing_spec_count(output_dir, field_slug)
    target_count = max_subfields * max_tasks_per_subfield
    if existing_count >= target_count:
        logger.info(f"  {existing_count} existing specs >= target {target_count}; skipping field")
        return []
    to_generate = target_count - existing_count
    if existing_count > 0:
        logger.info(f"  found {existing_count} existing specs; topping up to {target_count} ({to_generate} more)")

    usage_tracker = UsageTracker()
    field_log: List[Dict[str, Any]] = []
    start = time.time()

    # Phase 1: field -> subfields
    p1_prompt = role.get_prompt("subfield", field_name=field)
    p1_response = llm([{"role": "user", "content": p1_prompt}])
    p1_usage = llm.last_usage
    usage_tracker.track(p1_usage)
    subfields_all = _unique(parse_list(p1_response))
    subfields = subfields_all[:max_subfields]
    field_log.append(_phase_entry("subfields", p1_prompt, p1_response, p1_usage, parsed=subfields))
    logger.info(f"  subfields: {len(subfields_all)} parsed, using {len(subfields)}")

    if not subfields:
        logger.warning(f"No subfields parsed for field '{field}' — skipping")
        return []

    # Phase 2: subfield -> tasks (batched across subfields)
    p2_prompts = [role.get_prompt("task", field_name=field, subfield_name=s) for s in subfields]
    p2_results = batch_call(llm, p2_prompts)
    tasks_per_subfield: List[List[str]] = []
    for s_idx, (s, r) in enumerate(zip(subfields, p2_results)):
        usage_tracker.track(r["usage"])
        parsed = _unique(parse_list(r["response"]))[:max_tasks_per_subfield]
        tasks_per_subfield.append(parsed)
        field_log.append(_phase_entry(
            "tasks", p2_prompts[s_idx], r["response"], r["usage"], parsed=parsed
        ))
    logger.info(f"  tasks: {[len(t) for t in tasks_per_subfield]}")

    # Flatten scenarios = [(subfield_idx, subfield, task), ...]
    scenarios: List[Dict[str, Any]] = []
    for s_idx, (subfield, tasks) in enumerate(zip(subfields, tasks_per_subfield)):
        for task in tasks:
            scenarios.append({"subfield_idx": s_idx, "subfield": subfield, "task": task})
    if len(scenarios) > to_generate:
        scenarios = scenarios[:to_generate]
        logger.info(f"  trimmed scenarios to {to_generate} (top-up cap)")
    logger.info(f"  scenarios: {len(scenarios)}")

    if not scenarios:
        logger.warning(f"No scenarios for field '{field}'")
        _save_field_log(output_dir, field_slug, field, llm, field_log, usage_tracker.total, time.time() - start)
        return []

    # Phase 3: tools (batched across scenarios)
    p3_prompts = [
        role.get_prompt("tool", field_name=field, subfield_name=sc["subfield"], task_name=sc["task"])
        for sc in scenarios
    ]
    p3_results = batch_call(llm, p3_prompts)
    for sc, prompt, r in zip(scenarios, p3_prompts, p3_results):
        usage_tracker.track(r["usage"])
        tools = _parse_tools(r["response"])
        sc["tools"] = tools
        sc["p3_prompt"] = prompt
        sc["p3_response"] = r["response"]
        sc["p3_usage"] = r["usage"]
    logger.info(f"  tools per scenario: {[len(sc['tools']) for sc in scenarios]}")

    # Save per-field log once
    elapsed = time.time() - start
    field_log_path = _save_field_log(
        output_dir, field_slug, field, llm, field_log, usage_tracker.total, elapsed
    )

    # Save one JSON per scenario. Parse-failure policy: skip writing a scenario
    # whose tools came back empty. Sequences are populated later by
    # `task_generation.build_sequences`; env_generation just emits `sequences: {}`.
    saved: List[Dict] = []
    for sc in scenarios:
        if not sc["tools"]:
            logger.warning(
                f"Skipping scenario (empty tools): subfield={sc['subfield']!r} task={sc['task']!r}"
            )
            continue

        scenario_tracker = UsageTracker()
        # Phases 1-2 are shared across scenarios; we record only their token usage in the
        # scenario file and link to the full prompt+response via field_log_path.
        scenario_tracker.track(p1_usage)
        scenario_tracker.track(p2_results[sc["subfield_idx"]]["usage"])
        scenario_tracker.track(sc["p3_usage"])

        spec_id = f"{field_slug}_spec_{next_artifact_index(f'{field_slug}_spec', output_dir):03d}"
        generated_at = _now_iso()
        model_config = _model_config_for(llm)
        scenario_payload = {
            "schema_version": SPEC_SCHEMA_VERSION,
            "spec_id": spec_id,
            "field": field,
            "subfield": sc["subfield"],
            "task": sc["task"],
            "generated_at": generated_at,
            "model": llm.model,
            "model_config": model_config,
            "tools": sc["tools"],
            "sequences": {},
            "generation_log": [
                {
                    "phase": "subfields",
                    "shared_with_field_log": field_log_path.name,
                    "usage": usage_to_dict(p1_usage),
                },
                {
                    "phase": "tasks",
                    "shared_with_field_log": field_log_path.name,
                    "subfield_idx": sc["subfield_idx"],
                    "usage": usage_to_dict(p2_results[sc["subfield_idx"]]["usage"]),
                },
                {
                    "phase": "tools",
                    "prompt": sc["p3_prompt"],
                    "response": sc["p3_response"],
                    "usage": usage_to_dict(sc["p3_usage"]),
                },
            ],
            "usage": scenario_tracker.total,
            "field_elapsed_s": round(elapsed, 2),
        }
        path = output_dir / f"{spec_id}.json"
        write_json_atomic(scenario_payload, path)
        logger.info(f"Saved scenario: {path}")
        _append_manifest(output_dir, {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "spec_id": spec_id,
            "field": field,
            "subfield": sc["subfield"],
            "task": sc["task"],
            "n_tools": len(sc["tools"]),
            "n_sequences": 0,
            "generated_at": generated_at,
            "model": llm.model,
            "usage": scenario_tracker.total,
        })
        saved.append(scenario_payload)

    return saved


def _save_field_log(
    output_dir: Path,
    field_slug: str,
    field: str,
    llm,
    entries: List[Dict],
    totals: Dict[str, int],
    elapsed: float,
) -> Path:
    """Merge new phase-1/phase-2 entries into the field log, accumulating tokens + elapsed.

    A re-run of env_generation for the same field would otherwise overwrite the
    prior prompts/responses. Scenario files from the first run reference the
    field log via `shared_with_field_log`, so losing those entries breaks
    reproducibility. Append-semantics keeps every phase-1/phase-2 LLM call
    ever made against this field on disk.
    """
    path = output_dir / f"{field_slug}_field_gen.json"
    prior_entries: List[Dict] = []
    prior_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prior_elapsed = 0.0
    if path.exists():
        try:
            prior = json.loads(path.read_text())
            prior_entries = prior.get("generation_log") or []
            pu = prior.get("usage") or {}
            prior_usage = {
                "prompt_tokens": pu.get("prompt_tokens", 0) or 0,
                "completion_tokens": pu.get("completion_tokens", 0) or 0,
                "total_tokens": pu.get("total_tokens", 0) or 0,
            }
            prior_elapsed = float(prior.get("elapsed_s", 0) or 0)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"field log at {path} unreadable — starting fresh ({exc})")

    merged_usage = {
        "prompt_tokens": prior_usage["prompt_tokens"] + (totals.get("prompt_tokens", 0) or 0),
        "completion_tokens": prior_usage["completion_tokens"] + (totals.get("completion_tokens", 0) or 0),
        "total_tokens": prior_usage["total_tokens"] + (totals.get("total_tokens", 0) or 0),
    }
    payload = {
        "schema_version": FIELD_GEN_SCHEMA_VERSION,
        "field": field,
        "generated_at": _now_iso(),
        "model": llm.model,
        "model_config": _model_config_for(llm),
        "generation_log": prior_entries + entries,
        "usage": merged_usage,
        "elapsed_s": round(prior_elapsed + elapsed, 2),
    }
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(path)
    logger.info(f"Saved field log: {path} (entries={len(payload['generation_log'])})")
    return path


def _parse_tools(response: str) -> List[Dict]:
    """Extract tool schemas from an LLM response and dedup by tool_name."""
    objs = extract_json_objects(response)
    seen = set()
    out = []
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        name = obj.get("tool_name")
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(obj)
    return out


def _default_template_files() -> Dict[str, str]:
    base = Path(__file__).resolve().parent.parent / "prompt_templates" / "env_generator"
    return {
        "subfield":  str(base / "env_generator_subfield_template.yml"),
        "task":      str(base / "env_generator_task_template.yml"),
        "tool":      str(base / "env_generator_tool_template.yml"),
        "sequences": str(base / "env_generator_sequences_template.yml"),
    }
