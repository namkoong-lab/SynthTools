"""Build candidate tool sequences for existing env_specs, post-audit.

Why this stage exists (and why it's not in env_generation anymore):
    env_generation emits scenario specs with an empty `sequences: {}` block.
    After env_audit scores each tool's reliability, this module generates
    sequences using *only* the subset of tools that pass the reliability
    filter. Result: sequences are guaranteed to reference tools that are
    known to work, rather than being baked from every tool including the
    unreliable ones.

Targets (mutually exclusive):
    - `spec_path`: one specific env_spec JSON
    - `field`:     every `{slug}_spec_*.json` whose `field` matches

Behaviour:
    - If the target spec already has a non-empty `sequences` dict, we skip it
      (no LLM call, no rewrite). To force regeneration, clear `sequences` in
      the file first.
    - Tool pool is filtered from `spec["tools"]` using `spec["audit"]["tools"]`:
        * `eval_only=True` (default): drop tools without an audit entry or
          with `reliability == None`.
        * `reliability < min_reliability`: drop.
    - If the filtered pool has fewer than `seq_length` tools, we log a warning
      and leave `sequences = {}` (not an error).
    - Sequences are passed through `_clean_sequences` post-processing: drop
      exact dupes, drop consecutive-dup sequences, drop sequences referencing
      an unknown tool name (after normalization).

Usage:
    from llm import LLM
    from task_generation.build_sequences import build_sequences

    llm = LLM("GPT-OSS-120B")
    build_sequences(
        env_specs_dir=Path("tool_content/env_specs"),
        llm=llm,
        field="Aerospace and Defense",
        n_sequences=10,
        seq_length=8,
        min_reliability=0.5,
    )
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_MIN_RELIABILITY, DEFAULT_MODEL, DEFAULT_N_SEQUENCES, DEFAULT_SEQ_LENGTH
from env_audit.utils import model_config_for, now_iso, write_env_spec
from llm import LLM, MODEL_REGISTRY
from roles.env_generator import EnvironmentGenerator
from utils import batch_call, extract_json_objects, get_logger, usage_to_dict

logger = get_logger("synthtools")


# ---------------------------------------------------------------------------
# Sequence post-processing (moved here from env_generation)
# ---------------------------------------------------------------------------

def _normalize_tool_name(name: str) -> str:
    """Compare tool names after stripping whitespace, hyphens, and underscores.

    Defensive: tool generation now enforces PascalCase (no separators), but
    sequences have been observed with separators stripped. Normalizing both
    sides makes the membership check robust to either.
    """
    return (name or "").replace(" ", "").replace("-", "").replace("_", "")


def _clean_sequences(
    sequences: Dict[str, List[str]],
    allowed_tool_names: List[str],
) -> Dict[str, List[str]]:
    """Drop sequences that are invalid.

    Drops in this order:
      1. Sequences containing a consecutive-duplicate tool name.
      2. Exact-duplicate sequences.
      3. Sequences referencing a tool name not in `allowed_tool_names`
         (after normalizing both sides via `_normalize_tool_name`).
    """
    if not sequences:
        return {}
    allowed_norm = {_normalize_tool_name(n) for n in allowed_tool_names}
    seen: List[List[str]] = []
    out: Dict[str, List[str]] = {}
    for key, seq in sequences.items():
        if not isinstance(seq, list):
            continue
        if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
            continue
        if seq in seen:
            continue
        unknown = [step for step in seq if _normalize_tool_name(step) not in allowed_norm]
        if unknown:
            logger.warning(f"  dropping sequence {key}: references unknown tool(s) {unknown}")
            continue
        seen.append(seq)
        out[key] = seq
    return out


# ---------------------------------------------------------------------------
# Reliability filter
# ---------------------------------------------------------------------------

def filter_tools_by_reliability(
    spec: Dict[str, Any],
    min_reliability: float,
    eval_only: bool,
) -> List[Dict[str, Any]]:
    """Return the subset of spec['tools'] whose audit reliability clears the bar.

    Policy:
      - eval_only=True drops tools with no audit entry and tools whose
        reliability is None.
      - Otherwise those tools pass through; only audited tools below threshold
        are dropped.
    """
    audit_block = spec.get("audit") or {}
    audit_entries = audit_block.get("tools") or []
    audit_by_name = {e.get("tool_name"): e for e in audit_entries if isinstance(e, dict)}

    kept: List[Dict[str, Any]] = []
    for tool in spec.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("tool_name")
        entry = audit_by_name.get(name)

        if entry is None:
            if eval_only:
                continue
            kept.append(tool)
            continue

        rel = entry.get("reliability")
        if rel is None:
            if eval_only:
                continue
            kept.append(tool)
            continue

        if rel >= min_reliability:
            kept.append(tool)

    return kept


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def _default_template_files() -> Dict[str, str]:
    base = Path(__file__).resolve().parent.parent / "prompt_templates" / "env_generator"
    return {
        "subfield":  str(base / "subfield.yml"),
        "task":      str(base / "task.yml"),
        "tool":      str(base / "tool.yml"),
        "sequences": str(base / "sequences.yml"),
    }


def _iter_spec_paths(env_specs_dir: Path, field: Optional[str]) -> List[Path]:
    """Return every scenario JSON in env_specs_dir matching `field` (if given)."""
    paths = []
    for p in sorted(env_specs_dir.glob("*_spec_*.json")):
        if p.name.endswith(".tmp"):
            continue
        try:
            with open(p) as f:
                spec = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"  skipping {p.name}: unreadable ({exc})")
            continue
        if field is not None and spec.get("field") != field:
            continue
        paths.append(p)
    return paths


def build_sequences(
    env_specs_dir: Path,
    llm,
    spec_path: Optional[Path] = None,
    field: Optional[str] = None,
    n_sequences: int = DEFAULT_N_SEQUENCES,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    min_reliability: float = DEFAULT_MIN_RELIABILITY,
    eval_only: bool = True,
) -> List[Dict[str, Any]]:
    """Populate the `sequences` block for one or more env_specs.

    Uses a single batched LLM call across every spec that needs generation,
    regardless of whether the caller passed a single `spec_path` or a `field`.
    Records the prompt, response, usage, and filter settings in each spec's
    `generation_log` under a new `sequences_build` phase entry.

    Args:
        env_specs_dir: Directory containing env_spec JSON files.
        llm: LLM instance (callable).
        spec_path: Optional path to a single scenario JSON. Mutually exclusive
            with `field`.
        field: Optional field name. All scenarios whose `field` matches are
            processed. Mutually exclusive with `spec_path`.
        n_sequences: Number of sequences to request from the LLM per spec.
        seq_length: Length of each sequence.
        min_reliability: Drop tools with audit reliability below this threshold.
        eval_only: If True (default), drop tools lacking an audit entry or
            with `reliability == None`.

    Returns:
        List of updated spec dicts.
    """
    if spec_path is not None and field is not None:
        raise ValueError("spec_path and field are mutually exclusive")

    env_specs_dir = Path(env_specs_dir)

    role = EnvironmentGenerator(_default_template_files(), llm)
    if hasattr(llm, "_ensure_engine"):
        llm._ensure_engine()

    if spec_path is not None:
        target_paths = [Path(spec_path)]
    else:
        target_paths = _iter_spec_paths(env_specs_dir, field)
        logger.info(f"Field '{field}': found {len(target_paths)} spec(s)")

    start = time.time()
    updated: List[Dict[str, Any]] = []
    to_generate: List[Dict[str, Any]] = []

    # Phase 1: load + filter. Specs that are skipped or have an empty pool are
    # written (if changed) immediately; the rest get queued for a batched call.
    for p in target_paths:
        with open(p) as f:
            spec = json.load(f)
        spec_id = spec["spec_id"]

        if spec.get("sequences"):
            logger.info(f"  {spec_id}: sequences already present ({len(spec['sequences'])}) — skipping")
            updated.append(spec)
            continue

        filtered = filter_tools_by_reliability(spec, min_reliability, eval_only)
        logger.info(
            f"  {spec_id}: filtered tools {len(filtered)}/{len(spec.get('tools') or [])} "
            f"(min_reliability={min_reliability}, eval_only={eval_only})"
        )
        if len(filtered) < seq_length:
            logger.warning(
                f"  {spec_id}: only {len(filtered)} tools pass the filter, "
                f"need >= {seq_length} — leaving spec file UNTOUCHED"
            )
            updated.append(spec)
            continue

        tool_data = {
            "field_name": spec.get("field"),
            "subfield": spec.get("subfield"),
            "task": spec.get("task"),
            "tools": filtered,
        }
        prompt = role.get_prompt(
            "sequences",
            Data=json.dumps(tool_data, indent=2),
            seqs_per_spec=n_sequences,
            seq_length=seq_length,
        )
        to_generate.append({"spec": spec, "filtered": filtered, "prompt": prompt})

    # Phase 2: one batched LLM call across every spec that needs generation.
    if to_generate:
        prompts = [item["prompt"] for item in to_generate]
        logger.info(f"build_sequences: batched LLM call for {len(prompts)} spec(s)")
        results = batch_call(llm, prompts)

        for item, r in zip(to_generate, results):
            spec = item["spec"]
            spec_id = spec["spec_id"]
            filtered = item["filtered"]
            response = r["response"]
            usage = r["usage"]

            objs = extract_json_objects(response)
            raw_seqs = objs[0] if objs and isinstance(objs[0], dict) else {}
            allowed_names = [t.get("tool_name", "") for t in filtered]
            sequences = _clean_sequences(raw_seqs, allowed_names)

            logger.info(
                f"  {spec_id}: generated {len(raw_seqs)} raw, kept {len(sequences)} after cleaning "
                f"(usage: {usage_to_dict(usage)})"
            )

            spec["sequences"] = sequences
            gen_log = spec.setdefault("generation_log", [])
            gen_log.append({
                "phase": "sequences_build",
                "generated_at": now_iso(),
                "model": getattr(llm, "model", None),
                "model_config": model_config_for(llm),
                "n_sequences_requested": n_sequences,
                "seq_length": seq_length,
                "min_reliability": min_reliability,
                "eval_only": eval_only,
                "n_tools_filtered": len(filtered),
                "n_tools_total": len(spec.get("tools") or []),
                "prompt": item["prompt"],
                "response": response,
                "usage": usage_to_dict(usage),
            })
            write_env_spec(env_specs_dir, spec)
            updated.append(spec)

    logger.info(f"build_sequences: processed {len(updated)} spec(s) in {time.time() - start:.1f}s")
    return updated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate tool sequences for env_specs after audit.")
    parser.add_argument("--env-specs-dir", type=Path, required=True,
                        help="Directory containing env_spec JSON files")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_REGISTRY))
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--env-spec", type=Path, help="Path to a single env_spec JSON")
    target.add_argument("--field", type=str, help="Field name: process every matching scenario")
    parser.add_argument("--n-sequences", type=int, default=DEFAULT_N_SEQUENCES)
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--min-reliability", type=float, default=DEFAULT_MIN_RELIABILITY,
                        help="Drop tools below this reliability threshold")
    parser.add_argument("--no-eval-only", dest="eval_only", action="store_false",
                        help="Include tools without audit data (default: exclude them)")
    parser.set_defaults(eval_only=True)
    args = parser.parse_args()

    llm = LLM(args.model)
    build_sequences(
        env_specs_dir=args.env_specs_dir,
        llm=llm,
        spec_path=args.env_spec,
        field=args.field,
        n_sequences=args.n_sequences,
        seq_length=args.seq_length,
        min_reliability=args.min_reliability,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
