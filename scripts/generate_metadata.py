#!/usr/bin/env python3
"""
Run a prompt template as the system prompt for many tool JSON files,
and save the model's response as JSON per input.

Supports Anthropic and OpenAI providers (mirrors provider/model handling used by
generate_tool_from_field_alternate_prompt.py).

Usage (Anthropic):
  python generate_metadata.py \
    --template /mnt/data/generate_metadata_alternate_template.yml \
    --json-glob "/mnt/data/*.json" \
    --out-dir /mnt/data/claude_metadata_out \
    --model_provider anthropic \
    --model claude-3-7-sonnet-20250219 \
    --max-tokens 2000 \
    --api-keys configs/api_keys.json

Usage (OpenAI):
  python generate_metadata.py \
    --template /mnt/data/generate_metadata_alternate_template.yml \
    --json-glob "/mnt/data/*.json" \
    --out-dir /mnt/data/openai_metadata_out \
    --model_provider openai \
    --model gpt-4o \
    --max-tokens 2000
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import List

from utils.client_utils import AnthropicClient, OpenAIClient

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def indent_block(text: str, indent: str) -> str:
    lines = text.splitlines()
    return "\n".join(indent + line for line in lines) if lines else indent

def fill_template_with_tool_description(template: str, tool_desc: str) -> str:
    m = re.search(r'(?m)^(?P<indent>\s*)\{tool_description\}\s*$', template)
    if not m:
        raise ValueError("Placeholder '{tool_description}' not found in template.")
    indent = m.group("indent")
    indented_desc = indent_block(tool_desc.rstrip("\n"), indent)
    return template[:m.start()] + indented_desc + template[m.end():]

def extract_tool_description_from_json(json_path: str) -> str:
    raw = read_text(json_path)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(data, indent=2, ensure_ascii=False)

JSON_FENCE_RE = re.compile(
    r"```(?:json|javascript|js)?\s*(?P<body>\{.*?\})\s*```",
    flags=re.DOTALL | re.IGNORECASE,
)

def coerce_to_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass

    m = JSON_FENCE_RE.search(text)
    if m:
        candidate = m.group("body")
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Could not parse model output as JSON.")

def call_model(system_prompt: str, provider: str, model: str, max_tokens: int, temperature: float = 0.0) -> str:
    user_msg = "Return ONLY a single valid JSON object. No prose."
    provider_lc = (provider or "anthropic").lower()

    if provider_lc == "anthropic":
        client = AnthropicClient(model=model, max_tokens=max_tokens, temperature=temperature, prompt=system_prompt)
    elif provider_lc == "openai":
        client = OpenAIClient(model=model, max_tokens=max_tokens, temperature=temperature, prompt=system_prompt)
    else:
        raise SystemExit(f"Unsupported model provider: {provider}")

    return client.message(user_msg)

def main():
    ap = argparse.ArgumentParser(description="Batch Claude run with template prompt and multiple tool JSONs.")
    ap.add_argument("--template", required=True, help="Path to prompt template with {tool_description} placeholder")
    ap.add_argument("--json-glob", required=True, help="Glob for input tool JSON files")
    ap.add_argument("--out-dir", default="claude_metadata_out", help="Where to write JSON outputs")
    ap.add_argument("--model_provider", default="anthropic", help="Model provider: anthropic or openai")
    ap.add_argument("--model", default="claude-sonnet-4-20250514", help="Model name for the chosen provider")
    ap.add_argument("--max-tokens", type=int, default=2000, help="Max output tokens")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument(
        "--api-keys",
        default="configs/api_keys.json",
        help="Path (relative to project root or absolute) to API keys JSON containing an 'anthropic' field",
    )
    args = ap.parse_args()

    if args.api_keys:
        try:
            from pathlib import Path as _Path
            key_path = _Path(args.api_keys)
            if not key_path.is_absolute():
                key_path = (init_path.PROJ_ROOT / key_path).resolve()
            with open(str(key_path), "r", encoding="utf-8") as f:
                key_data = json.load(f)
            if isinstance(key_data, dict) and key_data.get("anthropic"):
                os.environ["ANTHROPIC_API_KEY"] = key_data["anthropic"]
        except Exception as e:
            print(f"Warning: Failed to load API keys from {args.api_keys}: {e}")

    template = read_text(args.template)
    json_paths = sorted(glob.glob(args.json_glob))
    if not json_paths:
        raise SystemExit(f"No JSON files matched: {args.json_glob}")

    os.makedirs(args.out_dir, exist_ok=True)

    for jp in json_paths:
        base = Path(jp).stem
        out_json = Path(args.out_dir) / f"{base}__output.json"
        
        if out_json.exists():
            print(f"⏭️  Skipping {jp} - output already exists: {out_json}")
            continue
            
        print(f"→ Processing {jp}")

        try:
            tool_desc = extract_tool_description_from_json(jp)
            system_prompt = fill_template_with_tool_description(template, tool_desc)

            raw_text = call_model(
                system_prompt=system_prompt,
                provider=args.model_provider,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            try:
                data = coerce_to_json(raw_text)
            except Exception as e:
                data = {"_parse_error": str(e), "_raw": raw_text}

            write_json(str(out_json), data)
            print(f"Wrote {out_json}")

        except Exception as e:
            err_path = Path(args.out_dir) / f"{base}__error.json"
            write_json(str(err_path), {"error": str(e)})
            print(f"Failed {jp} -> {err_path}")

if __name__ == "__main__":
    main()
