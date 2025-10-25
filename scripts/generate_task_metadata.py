#!/usr/bin/env python3
"""
Generate environment tasks metadata using ONLY the environment-tasks-metadata-alternate template
and a tool YAML as input.

This script loads the fixed template:
  prompt_templates/generate_environment_tasks_meta_data/generate_environment_tasks_metadata_alternate_template.yml
inserts the full tool YAML into the {Data} placeholder (with correct indentation), and calls an LLM
via the clients in utils.client_utils to produce environment tasks metadata. The output is saved
as <input>.environment.yml next to the input tool YAML (or in --out-dir if provided).

Examples (Anthropic):
  python scripts/generate_environment_tasks_metadata.py \
    --input-yaml tool_content/tool_final/tool_yaml/agriculture_environmental_tool_spec_248.yml \
    --model_provider anthropic \
    --model claude-sonnet-4-20250514 \
    --max-tokens 5000 \
    --api-keys configs/api_keys.json

Examples (OpenAI):
  python scripts/generate_environment_tasks_metadata.py \
    --input-yaml tool_content/tool_final/tool_yaml/agriculture_environmental_tool_spec_248.yml \
    --model_provider openai \
    --model gpt-4o \
    --max-tokens 4000
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from utils.client_utils import AnthropicClient, OpenAIClient


PROJ_ROOT = Path(__file__).parent.parent


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def indent_block(text: str, indent: str) -> str:
    lines = text.splitlines()
    return "\n".join(indent + line for line in lines) if lines else indent


def fill_template_with_data(template: str, data_block: str) -> str:
    m = re.search(r'(?m)^(?P<indent>\s*)\{Data\}\s*$', template)
    if not m:
        raise ValueError("Placeholder '{Data}' not found in template.")
    indent = m.group("indent")
    indented = indent_block(data_block.rstrip("\n"), indent)
    return template[:m.start()] + indented + template[m.end():]


def load_api_keys_from_path_copy_if_needed(api_keys_path_arg: Path) -> None:
    expected_path = (PROJ_ROOT / "configs" / "api_keys.json").resolve()
    if not api_keys_path_arg:
        return
    try:
        src = api_keys_path_arg.resolve()
        if src != expected_path and src.exists():
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(src), str(expected_path))
    except Exception as e:
        print(f"Warning: Failed to sync API keys from {api_keys_path_arg} to {expected_path}: {e}")


def call_model_with_template(system_prompt: str, provider: str, model: str, max_tokens: int, temperature: float) -> str:
    provider_lc = (provider or "anthropic").lower()
    if provider_lc == "anthropic":
        client = AnthropicClient(model=model, max_tokens=max_tokens, temperature=temperature, prompt=system_prompt)
    elif provider_lc == "openai":
        client = OpenAIClient(model=model, max_tokens=max_tokens, temperature=temperature, prompt=system_prompt)
    else:
        raise SystemExit(f"Unsupported model provider: {provider}")
    user_msg = "Return ONLY a single valid YAML document. No prose."
    return client.message(user_msg)


def derive_output_path(input_yaml: Path) -> Path:
    if input_yaml.suffix.lower() in {".yml", ".yaml"}:
        return input_yaml.with_name(f"{input_yaml.stem}.environment{input_yaml.suffix}")
    return input_yaml.with_suffix(input_yaml.suffix + ".environment.yml")


YAML_FENCE_RE = re.compile(
    r"```(?:ya?ml)?\s*(?P<body>[\s\S]*?)```",
    flags=re.IGNORECASE,
)


def strip_yaml_fences(text: str) -> str:
    """Extract body from a markdown YAML fenced block if present."""
    m = YAML_FENCE_RE.search(text)
    if m:
        return m.group("body")
    if text.lstrip().startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        return "\n".join(lines)
    return text


def expand_input_to_files(input_arg: str) -> list[Path]:
    """Expand a single CLI input into a list of file paths.

    Supports:
      - single file path
      - comma-separated list of file paths
      - JSON array of file paths
      - directory path (non-recursive), collecting *.yml and *.yaml
    Paths not absolute are resolved relative to project root.
    """
    candidates: list[str]
    s = (input_arg or "").strip()
    candidates = []
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                candidates = [str(x) for x in arr]
        except Exception:
            pass
    if not candidates and ("," in s):
        candidates = [part.strip() for part in s.split(",") if part.strip()]
    if not candidates:
        candidates = [s]

    proj_root = PROJ_ROOT.resolve()
    files: list[Path] = []
    for cand in candidates:
        p = Path(cand)
        if not p.is_absolute():
            p = (proj_root / p).resolve()
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in {".yml", ".yaml"}:
                    files.append(child.resolve())
        elif p.is_file():
            files.append(p.resolve())
        else:
            print(f"Warning: input path not found: {p}")

    seen = set()
    unique_files: list[Path] = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    return unique_files


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate environment tasks metadata from template and tool YAML input.")
    ap.add_argument(
        "--input-yaml",
        required=True,
        help=(
            "Input tool YAML path. Also supports: "
            "comma-separated list, JSON array of paths, or a directory (non-recursive) "
            "from which all *.yml/*.yaml will be processed."
        ),
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Optional directory to write output file. Defaults to input file directory",
    )
    ap.add_argument("--model_provider", default="anthropic", help="Model provider: anthropic or openai")
    ap.add_argument("--model", default="claude-sonnet-4-20250514", help="Model name for the chosen provider")
    ap.add_argument("--max-tokens", type=int, default=5000, help="Max output tokens")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument(
        "--api-keys",
        default=str((PROJ_ROOT / "configs" / "api_keys.json").resolve()),
        help="Path to API keys JSON (will be copied to scaling_tools/configs/api_keys.json for clients)",
    )
    ap.add_argument(
        "--template",
        default=str((PROJ_ROOT / "prompt_templates" / "generate_environment_tasks_meta_data" / "generate_environment_tasks_metadata_alternate_template.yml").resolve()),
        help="Path to the generation template YAML",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    tpl_path = Path(args.template)
    if not tpl_path.is_absolute():
        tpl_path = (PROJ_ROOT / tpl_path).resolve()

    load_api_keys_from_path_copy_if_needed(Path(args.api_keys))

    provider = (args.model_provider or "anthropic").lower()
    model = args.model
    if provider == "openai":
        if not model or "claude" in model.lower():
            model = "gpt-4o"

    input_files = expand_input_to_files(args.input_yaml)
    if not input_files:
        raise SystemExit("No valid input YAML files found.")

    template_text = read_text(tpl_path)

    out_dir: Path | None = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (PROJ_ROOT / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    for input_yaml_path in input_files:
        input_data_text = read_text(input_yaml_path)

        system_prompt = fill_template_with_data(template_text, input_data_text)

        raw_yaml = call_model_with_template(
            system_prompt=system_prompt,
            provider=provider,
            model=model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        raw_yaml = strip_yaml_fences(raw_yaml).strip()

        out_path = derive_output_path(input_yaml_path)
        if out_dir is not None:
            out_path = out_dir / out_path.name

        write_text(out_path, raw_yaml)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
