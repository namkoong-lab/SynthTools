#!/usr/bin/env python3
"""
Generate one human-facing YAML per task from environment .environment.yml files.

Description:
- Accepts inputs that are files and/or directories (directories are scanned for *.environment.yml).
- For each task in each environment file, writes a YAML containing meta_data, task_data, and tools_data.
- Tools are resolved from --tools-dir as JSON files named: {spec_prefix}__{Tool_Name_With_Underscores}.json
- Output files are written to --out-dir and named: {spec_prefix}.human_task{N}.yml

Example usage:
  Single file:
    python3 scaling_tools/scripts/generate_task_to_human.py \
      tool_content/tool_final/task_meta/ecommerce_and_retail_tool_spec_43.environment.yml \
      --tools-dir tool_content/tool_final/tool_json \
      --out-dir  tool_content/tool_final/task_meta/output

  Multiple inputs (mix of directories and files):
    python3 scaling_tools/scripts/generate_task_to_human.py \
      tool_content/tool_final/task_meta \
      tool_content/tool_final/task_meta/ecommerce_and_retail_tool_spec_43.environment.yml \
      --tools-dir tool_content/tool_final/tool_json \
      --out-dir  tool_content/tool_final/task_meta/output
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml


def _select_mapping_from_docs(docs: List[Any]) -> Dict[str, Any]:
    def has_keys(d: Any) -> bool:
        if not isinstance(d, dict):
            return False
        lowered = {str(k).lower() for k in d.keys()}
        return ("tasks" in lowered) or ("meta_data" in lowered) or ("metadata" in lowered) or ("meta" in lowered)

    for d in docs:
        if has_keys(d):
            return d
    for d in docs:
        if isinstance(d, dict):
            return d
    raise ValueError("No mapping documents found in provided YAML content")


def _parse_tasks_from_markdown(text: str) -> List[Dict[str, Any]]:
    """Parse tasks from markdown-like sections in environment files.

    Expects blocks like:
      **Task 1**: <task text>
      **Task-difficulty**: easy
      **Number-tools-required**: 3
      **Tools-required**: ["Tool A", "Tool B"]
      **Task-validity**: valid
    """
    tasks: List[Dict[str, Any]] = []
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m_task = re.match(r"^\s*\*?\*?Task\s*(?:\d+)?\*?\*?\s*:\s*(?P<task>.+)$", line, re.IGNORECASE)
        if not m_task:
            i += 1
            continue
        task_text = m_task.group("task").strip()
        difficulty = None
        num_tools = None
        tools_list: List[str] = []
        validity = None
        j = i + 1
        while j < len(lines) and lines[j].strip():
            l = lines[j]
            m_diff = re.match(r"^\s*\*?\*?Task-difficulty\*?\*?\s*:\s*([A-Za-z_-]+)", l, re.IGNORECASE)
            if m_diff:
                difficulty = m_diff.group(1).strip()

            m_num = re.match(r"^\s*\*?\*?Number-tools-required\*?\*?\s*:\s*(\d+)", l, re.IGNORECASE)
            if m_num:
                try:
                    num_tools = int(m_num.group(1))
                except Exception:
                    num_tools = None

            m_tools = re.match(r"^\s*\*?\*?Tools-required\*?\*?\s*:\s*\[(.*?)\]", l, re.IGNORECASE)
            if m_tools:
                inner = m_tools.group(1).strip()
                if inner:
                    parts = [p.strip() for p in re.split(r",\s*", inner)]
                    cleaned_tokens: List[str] = []
                    for token in parts:
                        tok = token
                        if len(tok) >= 2 and ((tok[0] == '"' and tok[-1] == '"') or (tok[0] == "'" and tok[-1] == "'")):
                            tok = tok[1:-1]
                        cleaned_tokens.append(tok)
                    tools_list = [t for t in cleaned_tokens if t]

            m_valid = re.match(r"^\s*\*?\*?Task-validity\*?\*?\s*:\s*([A-Za-z_-]+)", l, re.IGNORECASE)
            if m_valid:
                validity = m_valid.group(1).strip()

            if re.match(r"^\s*\*?\*?Task\s*(?:\d+)?\*?\*?\s*:\s*", l, re.IGNORECASE):
                break
            j += 1

        tasks.append({
            "task": task_text,
            "task_difficulty": (difficulty or "").lower() if difficulty else None,
            "number_tools_required": num_tools,
            "tools_required": tools_list,
            "task_validity": (validity or "").lower() if validity else None,
        })
        i = j

    return [t for t in tasks if t.get("task")]


def load_env_inner_yaml(env_path: Path) -> Dict[str, Any]:
    raw_text = env_path.read_text()
    outer = yaml.safe_load(raw_text)
    inner_text = None
    if isinstance(outer, dict) and "tool_set_yml" in outer:
        inner_text = outer.get("tool_set_yml")
    if inner_text is not None and isinstance(inner_text, str):
        try:
            inner = yaml.safe_load(inner_text)
        except Exception:
            try:
                docs = list(yaml.safe_load_all(inner_text))
                inner = _select_mapping_from_docs(docs)
            except Exception:
                def extract_balanced_json(s: str) -> tuple[str, int]:
                    start = s.find("{")
                    if start == -1:
                        raise ValueError("No JSON object start '{' found")
                    depth = 0
                    i = start
                    in_str = False
                    esc = False
                    while i < len(s):
                        ch = s[i]
                        if in_str:
                            if esc:
                                esc = False
                            elif ch == "\\":
                                esc = True
                            elif ch == '"':
                                in_str = False
                        else:
                            if ch == '"':
                                in_str = True
                            elif ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    return s[start:i+1], i+1
                        i += 1
                    raise ValueError("Unbalanced JSON braces")

                try:
                    json_blob, end_idx = extract_balanced_json(inner_text)
                    meta_json = json.loads(json_blob)
                except Exception as je:
                    raise ValueError(f"Failed to parse leading JSON meta in {env_path}: {je}")
                inner = {}
                meta_data = {}
                for k in ["field_name", "subfield", "task", "tool_budget", "tool_sequences", "metadata", "meta_data"]:
                    if k in meta_json:
                        if k == "meta_data":
                            meta_data.update(meta_json[k])
                        else:
                            meta_data[k] = meta_json[k]
                inner["meta_data"] = meta_data
                remainder = inner_text[end_idx:]
                tasks_list = []
                task_re = re.compile(
                    r"Task:\s*\"(?P<task>.*?)\"\s*,\s*Task-difficulty:\s*\"(?P<difficulty>.*?)\"\s*,\s*Number-tools-required:\s*(?P<num>\d+)\s*,\s*Tools-required:\s*\[(?P<tools>.*?)\]\s*,\s*Task-validity:\s*\"(?P<validity>.*?)\"",
                    re.IGNORECASE | re.DOTALL,
                )
                for m in task_re.finditer(remainder):
                    tools_raw = m.group("tools").strip()
                    tool_names = []
                    for part in re.split(r",\s*", tools_raw):
                        part = part.strip()
                        if part.startswith("\"") and part.endswith("\""):
                            part = part[1:-1]
                        tool_names.append(part)
                    difficulty = m.group("difficulty").strip()
                    validity = m.group("validity").strip()
                    tasks_list.append({
                        "task": m.group("task").strip(),
                        "task_difficulty": difficulty.lower(),
                        "number_tools_required": int(m.group("num")),
                        "tools_required": tool_names,
                        "task_validity": validity.lower(),
                    })
                inner["tasks"] = tasks_list
        if not (isinstance(inner, dict) and inner.get("tasks")):
            md_tasks = _parse_tasks_from_markdown(inner_text)
            if md_tasks:
                if not isinstance(inner, dict):
                    inner = {}
                inner["tasks"] = md_tasks
    else:
        if isinstance(outer, dict):
            inner = outer
        elif isinstance(outer, list):
            inner = _select_mapping_from_docs(outer)
        else:
            docs = list(yaml.safe_load_all(raw_text))
            inner = _select_mapping_from_docs(docs)

    if not isinstance(inner, dict):
        raise ValueError(f"Unexpected env YAML structure in {env_path}")
    return inner


def normalize_tool_name(tool_name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", tool_name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def discover_env_files(inputs: List[Path]) -> List[Path]:
    env_files: List[Path] = []
    for p in inputs:
        if p.is_dir():
            env_files.extend(sorted(p.rglob("*.environment.yml")))
        elif p.is_file():
            if p.suffixes[-2:] == [".environment", ".yml"] or p.name.endswith(".environment.yml"):
                env_files.append(p)
        else:
            raise FileNotFoundError(f"Input path does not exist: {p}")
    seen = set()
    unique_files: List[Path] = []
    for f in env_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    return unique_files


def find_tool_json(tools_dir: Path, spec_prefix: str, tool_name: str) -> Optional[Path]:
    fname = f"{spec_prefix}__{normalize_tool_name(tool_name)}.json"
    fpath = tools_dir / fname
    return fpath if fpath.exists() else None


def _find_case_insensitive_key(d: Dict[str, Any], key: str) -> Optional[str]:
    key_l = key.lower()
    for k in d.keys():
        if str(k).lower() == key_l:
            return k
    return None


def _extract_meta_data(env_inner: Dict[str, Any]) -> Dict[str, Any]:
    metadata_section = None
    for candidate in ("meta_data", "metadata", "meta"):
        k = _find_case_insensitive_key(env_inner, candidate)
        if k and isinstance(env_inner[k], dict):
            metadata_section = env_inner[k]
            break
    
    data_section = None
    for candidate in ("data", "data_section", "content"):
        k = _find_case_insensitive_key(env_inner, candidate)
        if k and isinstance(env_inner[k], dict):
            data_section = env_inner[k]
            break
    
    result = {}
    if metadata_section:
        result.update(metadata_section)
    if data_section:
        result.update(data_section)
    
    if not result:
        result = {k: v for k, v in env_inner.items() if str(k).lower() not in {"tasks"}}
    
    return result


def _normalize_tasks_structure(tasks_val: Any) -> List[tuple]:
    if isinstance(tasks_val, dict):
        flat: List[tuple] = []
        for group_key, group_val in tasks_val.items():
            if isinstance(group_val, list):
                for idx, item in enumerate(group_val, start=1):
                    tkey = f"{group_key}_{idx}"
                    flat.append((tkey, item))
            else:
                flat.append((str(group_key), group_val))
        return flat
    if isinstance(tasks_val, list):
        pairs = []
        for idx, item in enumerate(tasks_val, start=1):
            if isinstance(item, dict):
                task_key = item.get("task_type") or item.get("id") or f"task_{idx}"
                pairs.append((str(task_key), item))
        return pairs
    return []


def _get_value_any(task_obj: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        ck = _find_case_insensitive_key(task_obj, k)
        if ck is not None:
            return task_obj.get(ck)
    return None


def build_task_doc(env_inner: Dict[str, Any], task_key: str, task_obj: Dict[str, Any], tools_dir: Path, spec_prefix: str) -> Dict[str, Any]:
    meta_data = _extract_meta_data(env_inner)

    tools_required = _get_value_any(task_obj, [
        "tools_required",
        "tools-required",
        "tools",
    ]) or []
    tools_data: List[Dict[str, Any]] = []
    for tname in tools_required:
        entry: Dict[str, Any] = {"tool_name": tname}
        json_path = find_tool_json(tools_dir, spec_prefix, tname)
        if json_path:
            try:
                entry["tool_json"] = json.loads(json_path.read_text())
            except Exception:
                entry["tool_json"] = None
        else:
            entry["tool_json"] = None
        tools_data.append(entry)

    task_data = {
        "task_type": task_key,
        "task": _get_value_any(task_obj, [
            "task",
            "ask",
            "description",
        ]),
        "task_diffuculty": _get_value_any(task_obj, [
            "task_diffuculty",
            "task_difficulty",
            "task-difficulty",
            "difficulty",
        ]),
        "number_tools_required": _get_value_any(task_obj, [
            "number_tools_required",
            "number-tools-required",
            "num_tools",
            "tools_count",
        ]),
        "tools_required": tools_required,
        "task_validity": _get_value_any(task_obj, [
            "task_validity",
            "task-validity",
            "validity",
        ]),
    }

    return {
        "meta_data": meta_data,
        "task_data": task_data,
        "tools_data": tools_data,
    }


def write_task_docs(env_file: Path, env_inner: Dict[str, Any], tools_dir: Path, out_dir: Path) -> List[Path]:
    tasks_key = _find_case_insensitive_key(env_inner, "tasks")
    tasks_val = env_inner.get(tasks_key) if tasks_key else None
    task_pairs = _normalize_tasks_structure(tasks_val)
    if not task_pairs:
        return []

    spec_prefix = env_file.name.replace(".environment.yml", "")
    out_paths: List[Path] = []

    for idx, (task_key, task_obj) in enumerate(task_pairs, start=1):
        doc = build_task_doc(env_inner, task_key, task_obj, tools_dir, spec_prefix)
        out_name = f"{spec_prefix}.human_task{idx}.yml"
        out_path = out_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
        out_path.write_text(text)
        out_paths.append(out_path)
    return out_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate human-facing per-task YAMLs from environment files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Single file:\n"
            "    python3 scaling_tools/scripts/generate_task_to_human.py \\\n+\n"
            "      tool_content/tool_final/task_meta/ecommerce_and_retail_tool_spec_43.environment.yml \\\n+\n"
            "      --tools-dir tool_content/tool_final/tool_json \\\n+\n"
            "      --out-dir  tool_content/tool_final/task_meta/output\n\n"
            "  Multiple inputs (mix of directories and files):\n"
            "    python3 scaling_tools/scripts/generate_task_to_human.py \\\n+\n"
            "      tool_content/tool_final/task_meta \\\n+\n"
            "      tool_content/tool_final/task_meta/ecommerce_and_retail_tool_spec_43.environment.yml \\\n+\n"
            "      --tools-dir tool_content/tool_final/tool_json \\\n+\n"
            "      --out-dir  tool_content/tool_final/task_meta/output\n"
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Environment .environment.yml files or directories containing them")
    parser.add_argument("--tools-dir", required=True, type=Path, help="Directory containing tool JSON files")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for generated YAML files")
    args = parser.parse_args()

    env_files = discover_env_files([p.resolve() for p in args.inputs])
    if not env_files:
        print("No environment files found.")
        return

    generated: List[Path] = []
    for env_file in env_files:
        try:
            env_inner = load_env_inner_yaml(env_file)
            generated.extend(write_task_docs(env_file, env_inner, args.tools_dir.resolve(), args.out_dir.resolve()))
        except Exception as e:
            print(f"Error processing {env_file}: {e}")

    for p in generated:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()


