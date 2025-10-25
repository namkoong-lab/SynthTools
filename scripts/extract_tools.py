#!/usr/bin/env python3
"""
Extract tool JSON blocks from YAML specs and write each tool to its own JSON file.

Supports:
- Fenced JSON blocks like ```json { ... } ```
- A top-level or nested "tools" key (list or dict)
- Raw JSON strings inside any string field

Usage (convert a directory of YAML specs):
  python scripts/extract_tools_to_json.py \
    tool_content/full_tool_specs/my_tools \
    -o tool_content/full_tool_specs/my_tools/tool_json

Usage (convert specific YAML files):
  python scripts/extract_tools_to_json.py \
    /path/to/spec1.yml /path/to/spec2.yaml \
    -o /path/to/out_dir

"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


FENCED_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def find_yaml_files(paths: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(p.rglob("*.yml"))
            files.extend(p.rglob("*.yaml"))
        elif p.is_file() and p.suffix.lower() in {".yml", ".yaml"}:
            files.append(p)
        else:
            print(f"Skipping non-YAML path: {p}", file=sys.stderr)
    seen = set()
    unique: List[Path] = []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            unique.append(f)
    return unique


def slugify(name: str, max_len: int = 80) -> str:
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return (s[:max_len] or "tool").strip("_-")


def json_objects_from_string(s: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block in FENCED_JSON_RE.findall(s):
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                out.append(obj)
            elif isinstance(obj, list):
                out.extend([x for x in obj if isinstance(x, dict)])
        except Exception:
            pass
    s_stripped = s.strip()
    if (s_stripped.startswith("{") and s_stripped.endswith("}")) or (
        s_stripped.startswith("[") and s_stripped.endswith("]")
    ):
        try:
            obj = json.loads(s_stripped)
            if isinstance(obj, dict):
                out.append(obj)
            elif isinstance(obj, list):
                out.extend([x for x in obj if isinstance(x, dict)])
        except Exception:
            pass
    return out


def walk_collect_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from walk_collect_strings(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_collect_strings(item)
    elif isinstance(obj, str):
        yield obj


def walk_collect_tools_key(obj: Any) -> List[Dict[str, Any]]:
    """Find values under any key named 'tools' (case-insensitive)."""
    found: List[Dict[str, Any]] = []

    def _walk(x: Any):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k.strip().lower() == "tools":
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                found.append(item)
                    elif isinstance(v, dict):
                        for item in v.values():
                            if isinstance(item, dict):
                                found.append(item)
                _walk(v)
        elif isinstance(x, list):
            for it in x:
                _walk(it)

    _walk(obj)
    return found


def dedupe_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for t in tools:
        key = json.dumps(t, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def extract_tools_from_yaml(yaml_path: Path) -> List[Dict[str, Any]]:
    text = yaml_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except Exception as e:
        print(f"[WARN] Failed to parse YAML: {yaml_path}: {e}", file=sys.stderr)
        data = None

    tools: List[Dict[str, Any]] = []

    for s in walk_collect_strings(data) if data is not None else [text]:
        tools.extend(json_objects_from_string(s))

    if data is not None:
        tools.extend(walk_collect_tools_key(data))

    tools = [t for t in tools if isinstance(t, dict)]
    tools = dedupe_tools(tools)
    tools = [
        t
        for t in tools
        if any(
            isinstance(t.get(k), str) and t.get(k).strip()
            for k in ("tool_name", "name", "id", "title")
        )
    ]
    return tools


def choose_tool_name(tool: Dict[str, Any], fallback_index: int) -> str:
    for k in ("tool_name", "name", "id", "title"):
        v = tool.get(k)
        if isinstance(v, str) and v.strip():
            return slugify(v)
    return f"tool_{fallback_index:02d}"


def write_tool_files(tools: List[Dict[str, Any]], base: Path, out_dir: Path) -> List[Path]:
    written: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, tool in enumerate(tools, start=1):
        tname = choose_tool_name(tool, idx)
        fname = f"{base.stem}__{tname}.json"
        path = out_dir / fname
        with path.open("w", encoding="utf-8") as f:
            json.dump(tool, f, indent=2, ensure_ascii=False)
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description="Extract tool JSONs from YAML specs.")
    ap.add_argument("paths", nargs="+", help="YAML file(s) or directory(ies) to scan")
    ap.add_argument("-o", "--out", default="tools_out", help="Output directory")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.paths]
    out_dir = Path(args.out)

    yaml_files = find_yaml_files(inputs)
    if not yaml_files:
        print("No YAML files found.", file=sys.stderr)
        sys.exit(1)

    total_tools = 0
    for yml in yaml_files:
        tools = extract_tools_from_yaml(yml)
        if not tools:
            print(f"[INFO] No tools found in {yml}")
            continue
        written = write_tool_files(tools, base=yml, out_dir=out_dir)
        total_tools += len(written)
        for p in written:
            print(f"[OK] Wrote: {p}")

    print(f"\nDone. Extracted {total_tools} tool file(s) into: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
