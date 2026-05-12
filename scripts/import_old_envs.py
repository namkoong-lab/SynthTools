"""
One-off migration: import old-pipeline env_yaml files (v0 schema) into the new
env_spec.v2 JSON format.

Source: /pscratch/sd/t/tcaste/synthtools_dataset/tool_content_3/env_yaml/*.yml
Target: /pscratch/sd/t/tcaste/tool_content/env_specs/{field_slug}_spec_NNN.json
        + manifest.jsonl

Schema transformations applied per imported tool:
  - tool_name: PascalCase-normalized (strips spaces/hyphens/punctuation)
  - parameters[*].type "number"  → "float" (or "integer" if default is int-like)
  - parameters[*].type "object" without `properties` → "string" + note in description
  - parameters[*].type "array" without `items` → items: {type: string}
  - output_details[*]: same type rewrites as parameters
  - environment_metadata (seq1, seq2, ...) → sequences (dict of seq_key → [new_names])

Non-copyable fields on the spec envelope are marked:
  model: "imported"
  model_config: null
  generation_log: []
  usage: zeros
  field_elapsed_s: null
  audit: null

Per-spec `imported_from` block records source provenance (path + source_schema "v0").

Run:
  python -m scripts.import_old_envs \
      --source-dir /pscratch/sd/t/tcaste/synthtools_dataset/tool_content_3/env_yaml \
      --output-dir /pscratch/sd/t/tcaste/tool_content/env_specs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

SCHEMA_VERSION_SPEC = "env_spec.v2"
SCHEMA_VERSION_MANIFEST = "env_manifest.v1"
SOURCE_SCHEMA = "v0"

IMPORT_MODEL = "GPT-OSS-120B"
IMPORT_MODEL_CONFIG = {"temperature": 0.2, "top_p": 0.95, "max_tokens": 16384}

PASCAL_RE = re.compile(r"[A-Za-z0-9]+")
SLUG_RE = re.compile(r"[^a-z0-9]+")

# Keywords that hint a numeric parameter is really a float.
_FLOAT_DESC_HINTS = (
    "percent", "percentage", "ratio", "rate", "probability",
    "score", "factor", "coefficient", "threshold", "temperature",
    "fraction", "decimal", "multiplier",
)
# Keywords that hint a numeric parameter is really an integer.
_INT_DESC_HINTS = (
    "count", "number of", "how many", "integer only", "minitems",
    "maxitems", "maxcount",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify_field(field_name: str) -> str:
    """Convert "Aerospace And Defense" → "aerospace_and_defense"."""
    s = unicodedata.normalize("NFKC", field_name or "").lower()
    s = SLUG_RE.sub("_", s).strip("_")
    return s or "unknown_field"


def normalize_tool_name(raw: str) -> str:
    """Convert "Aircraft Data Retriever" → "AircraftDataRetriever"."""
    s = unicodedata.normalize("NFKC", raw or "")
    parts = PASCAL_RE.findall(s)
    if not parts:
        return "UnnamedTool"
    out = "".join(w[0].upper() + w[1:] for w in parts)
    if out[0].isdigit():
        out = "T" + out
    return out


def build_name_map(tools: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map each old tool_name → unique new PascalCase name within this spec."""
    name_map: Dict[str, str] = {}
    used: Dict[str, int] = defaultdict(int)
    for t in tools:
        old = t.get("tool_name", "")
        new = normalize_tool_name(old)
        if new in used:
            used[new] += 1
            new = f"{new}{used[new]}"
        else:
            used[new] = 1
        name_map[old] = new
    return name_map


def resolve_numeric_type(schema: Dict[str, Any]) -> str:
    """Decide whether an old `number` param is really integer or float."""
    dflt = schema.get("default")
    if isinstance(dflt, bool):
        return "boolean"
    if isinstance(dflt, float):
        return "float"
    if isinstance(dflt, int):
        return "integer"
    if isinstance(dflt, str):
        try:
            if "." in dflt or "e" in dflt.lower():
                float(dflt)
                return "float"
            int(dflt)
            return "integer"
        except ValueError:
            pass
    for bound_key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        v = schema.get(bound_key)
        if isinstance(v, float) and not v.is_integer():
            return "float"
    desc = (schema.get("description") or "").lower()
    if any(h in desc for h in _FLOAT_DESC_HINTS):
        return "float"
    if any(h in desc for h in _INT_DESC_HINTS):
        return "integer"
    return "float"


def rewrite_type_schema(
    schema: Any,
    *,
    path: str,
    warnings: List[Dict[str, str]],
) -> Any:
    """Apply the 4 type-level rewrites recursively to a param or output schema."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    t = out.get("type")
    if t == "number":
        out["type"] = resolve_numeric_type(out)
    elif t == "object" and "properties" not in out:
        out["type"] = "string"
        out["description"] = (
            (out.get("description") or "").rstrip()
            + " (JSON-encoded object; shape described in the tool's `usage` field.)"
        ).strip()
        warnings.append({"path": path, "issue": "object_without_properties"})
    elif t == "array" and "items" not in out:
        out["items"] = {"type": "string"}
        warnings.append({"path": path, "issue": "array_without_items"})
    if isinstance(out.get("items"), dict):
        out["items"] = rewrite_type_schema(
            out["items"], path=f"{path}.items", warnings=warnings
        )
    if isinstance(out.get("properties"), dict):
        out["properties"] = {
            k: rewrite_type_schema(v, path=f"{path}.properties.{k}", warnings=warnings)
            for k, v in out["properties"].items()
        }
    return out


def rewrite_tool(
    tool: Dict[str, Any],
    name_map: Dict[str, str],
    *,
    spec_id: str,
    warnings: List[Dict[str, str]],
) -> Dict[str, Any]:
    old_name = tool.get("tool_name", "")
    new_name = name_map.get(old_name, normalize_tool_name(old_name))

    new_params: Dict[str, Any] = {}
    for pname, pschema in (tool.get("parameters") or {}).items():
        new_params[pname] = rewrite_type_schema(
            pschema, path=f"{spec_id}:{new_name}.params.{pname}", warnings=warnings
        )

    new_outputs: Dict[str, Any] = {}
    for oname, oschema in (tool.get("output_details") or {}).items():
        if isinstance(oschema, str):
            new_outputs[oname] = {"type": oschema, "description": ""}
            continue
        new_outputs[oname] = rewrite_type_schema(
            oschema, path=f"{spec_id}:{new_name}.output.{oname}", warnings=warnings
        )

    return {
        "tool_name": new_name,
        "tool_description": tool.get("tool_description", ""),
        "parameters": new_params,
        "error_messages": list(tool.get("error_messages") or []),
        "usage": tool.get("usage", ""),
        "output_details": new_outputs,
    }


def rewrite_sequences(
    env_metadata: Dict[str, Any],
    name_map: Dict[str, str],
    *,
    spec_id: str,
    warnings: List[Dict[str, str]],
) -> Dict[str, List[str]]:
    """Map each seq's tool-name list through name_map; drop unresolved entries.

    Old tc_3 YAMLs sometimes have sequence entries already PascalCased even
    though the tools list still uses spaced names, so accept either form.
    """
    if not isinstance(env_metadata, dict):
        return {}
    valid_new = set(name_map.values())
    out: Dict[str, List[str]] = {}
    for seq_key, tool_list in env_metadata.items():
        if not isinstance(tool_list, list):
            continue
        resolved: List[str] = []
        for nm in tool_list:
            if nm in name_map:
                resolved.append(name_map[nm])
            elif nm in valid_new:
                resolved.append(nm)
            elif normalize_tool_name(nm) in valid_new:
                resolved.append(normalize_tool_name(nm))
            else:
                warnings.append({
                    "path": f"{spec_id}:sequences.{seq_key}",
                    "issue": "unknown_tool_in_sequence",
                    "value": str(nm),
                })
        if resolved:
            out[seq_key] = resolved
    return out


def convert_one(
    src_path: Path,
    *,
    spec_idx: int,
    field_slug: str,
    import_ts: str,
    warnings: List[Dict[str, str]],
    secondary_src: Path = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (spec_json, manifest_row) for a single old env_yaml file.

    If `secondary_src` (tc_2 sibling) is provided and readable, union its
    sequences into the output under `tc2_{seq_key}` keys.
    """
    raw = yaml.safe_load(src_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{src_path}: top-level YAML is not a dict")

    field_display = raw.get("field_name") or ""
    subfield = raw.get("subfield") or ""
    task = raw.get("task") or ""
    tools_raw = raw.get("tools") or []
    env_meta = raw.get("environment_metadata") or {}

    spec_id = f"{field_slug}_spec_{spec_idx:03d}"
    name_map = build_name_map(tools_raw)
    tools_new = [
        rewrite_tool(t, name_map, spec_id=spec_id, warnings=warnings)
        for t in tools_raw
    ]
    sequences = rewrite_sequences(env_meta, name_map, spec_id=spec_id, warnings=warnings)

    secondary_info = None
    if secondary_src is not None and secondary_src.is_file():
        try:
            raw2 = yaml.safe_load(secondary_src.read_text())
        except Exception as e:
            warnings.append({
                "path": f"{spec_id}:secondary",
                "issue": "secondary_parse_error",
                "value": f"{secondary_src}: {e}",
            })
            raw2 = None
        if isinstance(raw2, dict):
            env_meta2 = raw2.get("environment_metadata") or {}
            prefixed = {f"tc2_{k}": v for k, v in env_meta2.items() if isinstance(v, list)}
            seqs2 = rewrite_sequences(
                prefixed, name_map, spec_id=spec_id, warnings=warnings
            )
            sequences.update(seqs2)
            secondary_info = {
                "source": str(secondary_src),
                "n_sequences_merged": len(seqs2),
            }

    spec = {
        "schema_version": SCHEMA_VERSION_SPEC,
        "spec_id": spec_id,
        "field": field_display,
        "subfield": subfield,
        "task": task,
        "generated_at": import_ts,
        "model": IMPORT_MODEL,
        "model_config": dict(IMPORT_MODEL_CONFIG),
        "tools": tools_new,
        "sequences": sequences,
        "generation_log": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "field_elapsed_s": None,
        "audit": None,
        "imported_from": {
            "source": str(src_path),
            "secondary_source": secondary_info,
            "source_schema": SOURCE_SCHEMA,
            "old_tool_name_map": name_map,
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION_MANIFEST,
        "spec_id": spec_id,
        "field": field_display,
        "subfield": subfield,
        "task": task,
        "n_tools": len(tools_new),
        "n_sequences": len(sequences),
        "generated_at": import_ts,
        "model": IMPORT_MODEL,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return spec, manifest


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-dir",
        default="/pscratch/sd/t/tcaste/synthtools_dataset/tool_content_3/env_yaml",
    )
    ap.add_argument(
        "--secondary-source-dir",
        default=None,
        help="Optional sibling env_yaml dir (e.g. tc_2) whose sequences are "
             "unioned into each matching spec under `tc2_{seq_key}` keys.",
    )
    ap.add_argument(
        "--output-dir",
        default="/pscratch/sd/t/tcaste/tool_content/env_specs",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.source_dir)
    out = Path(args.output_dir)
    secondary_dir = Path(args.secondary_source_dir) if args.secondary_source_dir else None
    if not src.is_dir():
        print(f"ERROR: source dir {src} does not exist", file=sys.stderr)
        return 2
    if secondary_dir is not None and not secondary_dir.is_dir():
        print(f"ERROR: secondary source dir {secondary_dir} does not exist", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    all_files = sorted(src.glob("*.yml")) + sorted(src.glob("*.yaml"))
    if args.limit:
        all_files = all_files[: args.limit]
    print(f"Found {len(all_files)} source yamls")

    # Group by field_slug → preserve source ordering
    by_field: Dict[str, List[Path]] = defaultdict(list)
    skipped: List[Tuple[Path, str]] = []
    for p in all_files:
        try:
            raw = yaml.safe_load(p.read_text())
        except Exception as e:
            skipped.append((p, f"yaml-parse-error: {e}"))
            continue
        if not isinstance(raw, dict) or not raw.get("field_name"):
            skipped.append((p, "missing field_name"))
            continue
        by_field[slugify_field(raw["field_name"])].append(p)

    print(f"Grouped into {len(by_field)} fields; {len(skipped)} skipped pre-parse")

    import_ts = now_iso()
    warnings: List[Dict[str, str]] = []
    manifest_rows: List[Dict[str, Any]] = []
    n_written = 0

    for field_slug in sorted(by_field):
        files = by_field[field_slug]
        for new_idx, src_path in enumerate(files):
            sibling = (secondary_dir / src_path.name) if secondary_dir else None
            try:
                spec, manifest = convert_one(
                    src_path,
                    spec_idx=new_idx,
                    field_slug=field_slug,
                    import_ts=import_ts,
                    warnings=warnings,
                    secondary_src=sibling,
                )
            except Exception as e:
                skipped.append((src_path, f"convert-error: {e}"))
                continue
            if args.dry_run:
                n_written += 1
                manifest_rows.append(manifest)
                continue
            out_path = out / f"{spec['spec_id']}.json"
            atomic_write_json(out_path, spec)
            manifest_rows.append(manifest)
            n_written += 1

    if not args.dry_run:
        manifest_path = out / "manifest.jsonl"
        with manifest_path.open("w") as f:
            for row in manifest_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        warn_path = out / "_import_warnings.jsonl"
        with warn_path.open("w") as f:
            for w in warnings:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")

        skip_path = out / "_import_skipped.jsonl"
        with skip_path.open("w") as f:
            for p, reason in skipped:
                f.write(json.dumps({"source": str(p), "reason": reason}) + "\n")

    print()
    print(f"Wrote:    {n_written} env_spec JSONs to {out}")
    print(f"Warnings: {len(warnings)}  →  _import_warnings.jsonl")
    print(f"Skipped:  {len(skipped)}  →  _import_skipped.jsonl")
    print(f"Fields:   {len(by_field)}")
    if args.dry_run:
        print("(dry-run — nothing written to disk)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
