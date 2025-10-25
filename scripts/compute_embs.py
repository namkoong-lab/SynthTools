#!/usr/bin/env python3
"""
Compute and store L2-normalized embeddings for tool JSON files.

Summary:
- Reads tool JSON(s) from a directory, a list of files, or a single file
- Builds deterministic text with build_tool_text_default(name, description, parameters, usage)
- Calls OpenAI embeddings in batches and L2-normalizes each vector
- Saves one compressed .npz per tool to the output directory

Defaults:
- Output directory: if -o/--out is omitted and inputs are in .../my_tools/tool_json,
  outputs go to .../my_tools/tool_emb (created if missing)
- Existing files are skipped: if <stem>.npz already exists, it won't be recomputed
- Output filename: full JSON stem is preserved (e.g., financial_trading_tool_spec_2__Economic_Indicator_Fetcher.npz)

Usage (directory):
  python scripts/compute_embs.py \
    tool_content/full_tool_specs/my_tools/tool_json \
    -o tool_content/full_tool_specs/my_tools/tool_emb \
    --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np

from utils.client_utils import OpenAIEmbeddingClient


def sanitize_filename(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch if ch in allowed else "_" for ch in name)


def find_json_files(inputs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.json")))
        elif p.is_file() and p.suffix.lower() == ".json":
            files.append(p)
        else:
            print(f"[WARN] Skipping non-JSON path: {p}")
    seen = set()
    unique: List[Path] = []
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(f)
    return unique


def derive_tool_id_from_path(p: Path) -> str:
    return p.stem


def batched(iterable: Iterable[Any], batch_size: int) -> Generator[List[Any], None, None]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def save_embedding_npz(
    out_dir: Path,
    rec_id: str,
    embedding: List[float],
    model: str,
    text_len: int,
    dtype: str = "float32",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(rec_id)
    out_path = out_dir / f"{safe_name}.npz"
    arr = np.asarray(embedding, dtype=dtype)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    np.savez_compressed(
        out_path,
        embedding=arr,
        id=np.array(rec_id),
        model=np.array(model),
        text_length=np.array(text_len, dtype=np.int64),
        embedding_dim=np.array(arr.shape[0], dtype=np.int64),
        dtype=np.array(str(arr.dtype)),
    )


def load_top_level_tool(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, list):
        if not obj:
            return None
        candidate = obj[0]
    else:
        candidate = obj
    return candidate if isinstance(candidate, dict) else None


def build_tool_text_default(tool: Dict[str, Any]) -> str:
    def norm_space(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    def norm_lower_text(s: str) -> str:
        return norm_space(s).lower()

    name = norm_lower_text(tool.get("tool_name", ""))
    desc = norm_lower_text(tool.get("tool_description", ""))
    usage = norm_lower_text(tool.get("usage", ""))
    params_obj = tool.get("parameters") or {}
    if isinstance(params_obj, dict):
        param_items = sorted(params_obj.items(), key=lambda kv: str(kv[0]).lower())
    else:
        param_items = sorted(
            [(p.get("name", ""), p) for p in (params_obj or [])],
            key=lambda kv: str(kv[0]).lower(),
        )
    entries: List[str] = []
    for p_name, p in param_items:
        pname = norm_space(str(p_name))
        ptype = norm_space(str(p.get("type", "")))
        preq_val = p.get("required")
        preq = "none" if preq_val is None else str(preq_val)
        if "default" in p:
            dval = p.get("default")
            pdef = "none" if dval is None else norm_space(str(dval))
        else:
            pdef = "none"
        pdesc = norm_lower_text(p.get("description", ""))
        entries.append(f"{pname}|{ptype}|{preq}|{pdef}|{pdesc}")
    params_block = "[{}]".format(" ; ".join(entries))
    return "\n".join([
        f"name: {name}",
        f"description: {desc}",
        f"parameters: {params_block}",
        f"usage: {usage}",
    ])


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute OpenAI embeddings for tool JSONs using build_tool_text_default; save one npz per tool."
        )
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Input JSON file(s) and/or directory(ies)",
    )
    p.add_argument(
        "-o",
        "--out",
        dest="output_dir",
        default=None,
        help=(
            "Output directory for per-tool embeddings (.npz). If omitted, will be set to"
            " the parent directory of the tools folder joined with 'tool_emb'."
        ),
    )
    p.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Number of texts per embeddings request",
    )
    p.add_argument(
        "--dtype",
        dest="dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Floating dtype used when saving embeddings",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    inputs = [Path(p).resolve() for p in args.inputs]
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        base = inputs[0] if inputs[0].is_dir() else inputs[0].parent
        out_dir = (base.parent / "tool_emb").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = find_json_files(inputs)
    if not json_files:
        print("No JSON files found.")
        sys.exit(1)

    client = OpenAIEmbeddingClient()
    model = client.get_model_name()

    def stream_items() -> Generator[Tuple[str, str], None, None]:
        for fpath in json_files:
            tool_id = derive_tool_id_from_path(fpath)
            safe = sanitize_filename(tool_id)
            if (out_dir / f"{safe}.npz").exists():
                continue
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception as e:
                sys.stderr.write(f"[WARN] Failed to read {fpath}: {e}\n")
                continue
            tool_obj = load_top_level_tool(data)
            if tool_obj is None:
                sys.stderr.write(f"[WARN] Skipping {fpath}: top-level is not an object\n")
                continue
            text = build_tool_text_default(tool_obj)
            yield tool_id, text

    total_processed = 0
    total_errors = 0

    for batch in batched(stream_items(), args.batch_size):
        ids = [tool_id for tool_id, _ in batch]
        texts = [text.replace("\n", " ") for _, text in batch]

        try:
            embs = client.embed_many(texts)
        except Exception as e:
            embs = []
            for tool_id, text in batch:
                try:
                    embs.append(client.embed(text))
                except Exception as inner:
                    total_errors += 1
                    sys.stderr.write(f"Error embedding id={tool_id}: {inner}\n")
                    embs.append(None)

        for tool_id, emb, text in zip(ids, embs, texts):
            if emb is None:
                continue
            try:
                save_embedding_npz(out_dir, tool_id, emb, model, text_len=len(text), dtype=args.dtype)
                total_processed += 1
            except Exception as write_err:
                total_errors += 1
                sys.stderr.write(f"Error saving id={tool_id}: {write_err}\n")

    print(json.dumps({"processed": total_processed, "errors": total_errors, "output_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


