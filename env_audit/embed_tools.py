"""Compute embeddings for each tool in tools_dataset.jsonl.

One NPZ file per tool, named after the tool id, in a sibling `embeddings/`
directory:

    tool_content/
      ├── tools_dataset.jsonl
      └── embeddings/
          ├── meta.json                                                 (shared)
          ├── aerospace_and_defense_spec_000.ParseMissionRequirements.npz
          ├── aerospace_and_defense_spec_000.SelectAirframeConfiguration.npz
          └── ...

Each per-tool NPZ contains exactly one array, `embedding: float32[dim]`,
L2-normalized. The shared `meta.json` records the model_id, embedding_dim,
last generated_at, and source dataset (relative path).

Resume: tools whose `<tool_id>.npz` already exists are skipped. Re-runs
only embed missing tools. To swap to a different model or dim, point
`--output-dir` at a fresh directory (or delete the existing one) — the
script errors out if the existing meta.json disagrees with the requested
(model, dim).

Usage:
    python -m env_audit.embed_tools --dataset-path tool_content/tools_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import get_logger, write_json_atomic

logger = get_logger("synthtools")

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_DIM = 1024


# ---------------------------------------------------------------------------
# Prose renderer — what the embedder actually sees
# ---------------------------------------------------------------------------

def build_tool_text(tool: Dict[str, Any]) -> str:
    """Render a tool's full definition as natural-language prose.

    Includes name, description, every parameter (name + type + required-flag +
    description), the freeform `usage` instructions, the `output_details`
    schema in prose, and `error_messages`.
    """
    name = tool.get("tool_name", "")
    desc = tool.get("tool_description", "")
    params = tool.get("parameters", {}) or {}
    usage = tool.get("usage", "")
    output_details = tool.get("output_details", {}) or {}
    error_messages = tool.get("error_messages", []) or []

    lines: List[str] = []
    if name:
        lines.append(f"Tool: {name}")
    if desc:
        lines.append(f"Description: {desc}")

    if params:
        lines.append("Parameters:")
        for pname, pinfo in params.items():
            pinfo = pinfo or {}
            ptype = pinfo.get("type", "")
            preq = "required" if pinfo.get("required") else "optional"
            pdesc = pinfo.get("description", "")
            line = f"- {pname} ({ptype}, {preq})"
            if pdesc:
                line += f": {pdesc}"
            lines.append(line)

    if usage:
        lines.append(f"Usage: {usage}")

    if output_details:
        lines.append("Output:")
        for fname, finfo in output_details.items():
            if isinstance(finfo, dict):
                ftype = finfo.get("type", "")
                fdesc = finfo.get("description", "")
                line = f"- {fname} ({ftype})"
                if fdesc:
                    line += f": {fdesc}"
            else:
                line = f"- {fname}: {finfo}"
            lines.append(line)

    if error_messages:
        lines.append("Possible errors:")
        for err in error_messages:
            lines.append(f"- {err}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Encoder — lazy-loaded so tests can mock without importing transformers/torch
# ---------------------------------------------------------------------------

class _QwenEmbedder:
    """Last-token pooling + L2 norm, MRL truncation. Matches the official
    Qwen3-Embedding usage."""

    def __init__(self, model_id: str, target_dim: int, device: str = "auto",
                 max_seq_length: int = 4096):
        import torch
        from transformers import AutoTokenizer, AutoModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.target_dim = target_dim
        self.max_seq_length = max_seq_length

        logger.info(f"Loading {model_id} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self.model_id = model_id

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        import torch
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lens]

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode one batch in a single forward pass. Caller controls batching."""
        import torch
        import torch.nn.functional as F

        batch = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch)
        embeds = self._last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        if self.target_dim and self.target_dim < embeds.shape[1]:
            embeds = embeds[:, :self.target_dim]
        embeds = F.normalize(embeds, p=2, dim=1)
        return embeds.float().cpu().tolist()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_npz_atomic(npz_path: Path, embedding: List[float]) -> None:
    import numpy as np
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(embedding, dtype=np.float32)
    # np.savez auto-appends .npz if missing — keep .npz on the tmp path so the
    # rename target matches.
    tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez(tmp, embedding=arr)
    os.replace(tmp, npz_path)


def _check_meta_consistency(meta_path: Path, model_id: str, target_dim: int) -> None:
    """If a meta.json exists with a different model/dim, raise — protects against
    accidentally mixing embeddings from two models in the same directory."""
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    if meta.get("model_id") != model_id or meta.get("embedding_dim") != target_dim:
        raise RuntimeError(
            f"Existing {meta_path} was written with model={meta.get('model_id')!r} "
            f"dim={meta.get('embedding_dim')}, but you asked for model={model_id!r} dim={target_dim}. "
            f"Use a different --output-dir, or delete the existing directory."
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def embed_tools(
    dataset_path: Path,
    output_dir: Optional[Path] = None,
    model_id: str = DEFAULT_MODEL,
    target_dim: int = DEFAULT_DIM,
    batch_size: int = 16,
    device: str = "auto",
    encoder: Any = None,
) -> Dict[str, Any]:
    """Read the dataset; for each tool whose `<tool_id>.npz` doesn't exist yet
    in `output_dir`, embed and write it. Update `output_dir/meta.json`.

    `encoder` is for tests — pass an object with `.encode(texts) -> list[list[float]]`
    to avoid loading transformers/torch.
    """
    dataset_path = Path(dataset_path)
    if output_dir is None:
        output_dir = dataset_path.parent / "embeddings"
    output_dir = Path(output_dir)
    meta_path = output_dir / "meta.json"
    _check_meta_consistency(meta_path, model_id, target_dim)

    rows = _load_jsonl(dataset_path)
    pending: List[Dict[str, Any]] = []
    for row in rows:
        tid = row.get("id")
        if not tid:
            continue
        if not (output_dir / f"{tid}.npz").exists():
            pending.append(row)

    counts = {
        "total": len(rows),
        "skipped": len(rows) - len(pending),
        "embedded": 0,
        "output_dir": str(output_dir),
    }

    if not pending:
        logger.info(f"All {len(rows)} tools already embedded in {output_dir}; nothing to do.")
        # Still refresh meta so generated_at reflects the latest visit.
        if rows:
            _write_meta(meta_path, model_id, target_dim, dataset_path, output_dir)
        return counts

    logger.info(
        f"Embedding {len(pending)}/{len(rows)} tools with {model_id} "
        f"(dim={target_dim}, batch={batch_size}) → {output_dir}/<tool_id>.npz"
    )
    if encoder is None:
        encoder = _QwenEmbedder(model_id=model_id, target_dim=target_dim, device=device)

    try:
        from tqdm import tqdm
        bar = tqdm(total=len(pending), desc="Embedding", unit="tool")
    except ImportError:
        bar = None

    # Encode batch-by-batch; write each NPZ as it's produced. This gives
    # a live progress bar AND makes the run resume-safe at batch granularity.
    embedded = 0
    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        texts = [build_tool_text(row.get("tool", {}) or {}) for row in chunk]
        embs = encoder.encode_batch(texts)
        if len(embs) != len(chunk):
            raise RuntimeError(f"encoder returned {len(embs)} vectors for {len(chunk)} inputs")
        for row, emb in zip(chunk, embs):
            _save_npz_atomic(output_dir / f"{row['id']}.npz", emb)
        embedded += len(chunk)
        if bar is not None:
            bar.update(len(chunk))
    if bar is not None:
        bar.close()
    counts["embedded"] = embedded

    _write_meta(meta_path, model_id, target_dim, dataset_path, output_dir)
    logger.info(
        f"Wrote {counts['embedded']} new tool embeddings to {output_dir} "
        f"({counts['skipped']} already present)."
    )
    return counts


def _write_meta(meta_path: Path, model_id: str, target_dim: int,
                dataset_path: Path, output_dir: Path) -> None:
    n_tools = len(list(output_dir.glob("*.npz")))
    base = dataset_path.parent
    rel_dataset = dataset_path.name
    try:
        rel_dir = str(output_dir.relative_to(base))
    except ValueError:
        rel_dir = os.path.relpath(output_dir, base)
    meta = {
        "model_id": model_id,
        "embedding_dim": target_dim,
        "n_tools": n_tools,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_dataset": rel_dataset,
        "embeddings_dir": rel_dir,
    }
    write_json_atomic(meta, meta_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute per-tool embeddings for tools_dataset.jsonl")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for per-tool NPZ files (default: <dataset_dir>/embeddings).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM,
                        help=f"Truncated embedding dimension (default: {DEFAULT_DIM}; native is 2560)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    args = parser.parse_args()

    counts = embed_tools(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_id=args.model,
        target_dim=args.dim,
        batch_size=args.batch_size,
        device=args.device,
    )
    logger.info(f"Done: {counts}")


if __name__ == "__main__":
    main()
