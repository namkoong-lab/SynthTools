"""env_audit-local helpers.

Cross-stage helpers live in the top-level ``utils`` module. Anything specific to
env_audit (env_spec audit-block I/O, per-tool eval log I/O, timestamp/model
config helpers used only here) belongs in this file.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _pid_tmp(path: Path) -> Path:
    """Return a PID-suffixed `.tmp` path so concurrent writers to the same
    target don't collide on the staging filename."""
    return path.with_suffix(path.suffix + f".tmp.{os.getpid()}")


# ---------------------------------------------------------------------------
# Timestamp + model config
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def model_config_for(llm) -> Dict[str, Any]:
    cfg = getattr(llm, "cfg", None)
    return {
        "temperature": getattr(cfg, "temperature", None),
        "top_p": getattr(cfg, "top_p", None),
        "max_tokens": getattr(llm, "max_tokens", None),
    }


# ---------------------------------------------------------------------------
# JSONL + per-tool eval log I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """Atomic jsonl write — PID-suffixed `.tmp` + rename. Safe under concurrent
    writers to the same target file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _pid_tmp(path)
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def eval_log_path(eval_logs_dir: Path, tool_id: str) -> Path:
    return eval_logs_dir / f"{tool_id}.json"


def load_eval_log(path: Path) -> Optional[Dict]:
    """Load an eval log, returning None on missing file or malformed JSON
    (e.g. from a prior crash mid-save). Caller treats None as "needs work"."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None


def save_eval_log(path: Path, payload: Dict) -> None:
    """Atomic per-tool log write — PID-suffixed `.tmp` + rename. Safe under
    concurrent writers (disjoint tool_ids) and clean on SIGKILL mid-save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _pid_tmp(path)
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# env_spec read/write + audit-block refresh
# ---------------------------------------------------------------------------

ENV_SPEC_V2 = "env_spec.v2"
AUDIT_SCHEMA_VERSION = "audit.v1"


def find_env_spec_path(env_specs_dir: Path, spec_id: str) -> Path:
    """Return the JSON path for a given spec_id under env_specs_dir.

    Relies on the env_generation naming: ``{spec_id}.json`` directly.
    """
    return env_specs_dir / f"{spec_id}.json"


def read_env_spec(env_specs_dir: Path, spec_id: str) -> Dict[str, Any]:
    path = find_env_spec_path(env_specs_dir, spec_id)
    if not path.exists():
        raise FileNotFoundError(f"env_spec not found: {path}")
    with open(path) as f:
        return json.load(f)


def write_env_spec(env_specs_dir: Path, spec: Dict[str, Any]) -> Path:
    """Write a spec JSON atomically (temp file + rename)."""
    spec_id = spec["spec_id"]
    final = find_env_spec_path(env_specs_dir, spec_id)
    tmp = final.with_suffix(".json.tmp")
    final.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(final)
    return final


def refresh_env_spec_audit(
    env_specs_dir: Path,
    spec_id: str,
    eval_logs_dir: Path,
    llm_model: str,
    llm_model_config: Dict[str, Any],
) -> None:
    """Recompute and write the entire `audit` block of env_specs/{spec_id}.json.

    Reads every `tool_eval_logs/{spec_id}.*.json` file, summarizes each, and
    writes back a v2 env_spec. Idempotent: running twice with the same state
    produces the same file content.

    Tools whose eval log reached ``phase_completed == 7`` are included in
    ``audit.tools`` with their full summary. Pending tools (phase < 7) are
    counted in ``audit.aggregate.n_tools_pending`` but not listed.
    """
    spec = read_env_spec(env_specs_dir, spec_id)
    declared_tool_ids = {f"{spec_id}.{t.get('tool_name')}" for t in spec.get("tools", [])
                         if isinstance(t, dict) and t.get("tool_name")}

    evaluated_entries: List[Dict[str, Any]] = []
    n_pending = 0
    total_prompt = 0
    total_completion = 0
    reliabilities: List[float] = []

    for p in sorted(eval_logs_dir.glob(f"{spec_id}.*.json")):
        log = load_eval_log(p)
        if log is None:
            continue
        if log.get("tool_id") not in declared_tool_ids:
            # Keep the check loose so renamed/removed tools don't leak in.
            continue
        if log.get("phase_completed", 0) < 7:
            n_pending += 1
            continue
        usage = log.get("usage") or {}
        total_prompt += usage.get("prompt_tokens", 0) or 0
        total_completion += usage.get("completion_tokens", 0) or 0
        rel = log.get("reliability")
        if isinstance(rel, (int, float)):
            reliabilities.append(float(rel))
        evaluated_entries.append({
            "id": log.get("tool_id"),
            "tool_name": log.get("tool_id", "").rsplit(".", 1)[-1],
            "reliability": rel,
            "reliability_per_mode": log.get("reliability_per_mode"),
            "n_test_calls": len(log.get("test_calls") or []),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                "completion_tokens": usage.get("completion_tokens", 0) or 0,
                "total_tokens": (usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0),
            },
            "evaluated_at": log.get("generated_at"),
        })

    n_total = len(declared_tool_ids)
    mean_rel = (sum(reliabilities) / len(reliabilities)) if reliabilities else None
    audit_block = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "last_evaluated_at": now_iso(),
        "model": llm_model,
        "model_config": llm_model_config,
        "tools": evaluated_entries,
        "aggregate": {
            "n_tools_total": n_total,
            "n_tools_evaluated": len(evaluated_entries),
            "n_tools_pending": n_pending,
            "mean_reliability": mean_rel,
            "usage_total": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            },
        },
    }

    spec["schema_version"] = ENV_SPEC_V2
    spec["audit"] = audit_block
    write_env_spec(env_specs_dir, spec)
