#!/usr/bin/env python3
"""
Run degree-drop deduplication on tool embeddings (per field) and write tool_list.json.

Usage:
  python run_deduplication.py \
    --tool-json-dir <path/to/tool_json> \
    --emb-dir <path/to/tool_emb> \
    --yaml-dir <path/to/tool_yaml> \
    --out <path/to/tool_list.json> \
    --out-dir <path/to/deduped_tool_json_out_dir> \
    --taus 0.92 0.85 \
    --random-seed 42 \
    --log-level INFO

What it does:
- Enumerates tools present in BOTH JSON and embedding folders.
- Exact dedup 1: drop within-field exact-name duplicates.
- Exact dedup 2: drop exact-text duplicates.
- Semantic dedup: per field, build a τ-graph (cosine), run degree-drop until no edges remain.
- Writes tool_list.json JSONL with per-τ drop flags and logs a summary.

JSONL record fields:
  id, name, field, subfield, task, spec, dropped_{tau}...
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import random
import logging
import shutil


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_src = script_dir.parent / "tool_json"
    default_emb = script_dir.parent / "tool_emb"
    default_yaml = script_dir.parent / "tool_yaml"

    ap = argparse.ArgumentParser(description="Write tool_list.json (JSONL) for common tools.")
    ap.add_argument(
        "--tool-json-dir",
        dest="tool_json_dir",
        default=str(default_src),
        help="Folder containing tool JSON files (*.json). Default: sibling 'tool_json'",
    )
    ap.add_argument(
        "--emb-dir",
        dest="emb_dir",
        default=str(default_emb),
        help="Folder containing tool embeddings (*.npz). Default: sibling 'tool_emb'",
    )
    ap.add_argument(
        "--yaml-dir",
        dest="yaml_dir",
        default=str(default_yaml),
        help="Folder containing tool YAML files (*.yml). Default: sibling 'tool_yaml'",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help="Optional explicit output path. If omitted, writes 'tool_list.json' next to tool_json_dir",
    )
    ap.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help=(
            "Optional directory to write deduplicated tool JSONs. If multiple --taus are provided, "
            "creates one directory per tau as '<out-dir>_<tau>'. Existing directories are overwritten."
        ),
    )
    ap.add_argument(
        "--taus",
        dest="taus",
        type=float,
        nargs="+",
        required=True,
        help="One or more thresholds to evaluate (space-separated). Example: --taus 0.75 0.85 0.92",
    )
    ap.add_argument(
        "--random-seed",
        dest="random_seed",
        type=int,
        default=42,
        help="Random seed for tie-breaking in components of size 2 (default: 42)",
    )
    ap.add_argument(
        "--log-file",
        dest="log_file",
        default=None,
        help=(
            "Path to write logs. Default: sibling to --out (or tool_list.json) as 'run_deduplication.log'"
        ),
    )
    ap.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    ap.add_argument(
        "--exact-ded",
        dest="exact_ded",
        type=lambda s: str(s).lower() in ("1", "true", "t", "yes", "y"),
        nargs="?",
        const=True,
        default=True,
        help="Run exact dedup before cosine dedup (default: True). Pass False to skip.",
    )
    return ap.parse_args()


def list_stems(path: Path, suffix: str) -> List[str]:
    return sorted(p.stem for p in path.glob(f"*.{suffix}") if p.is_file())


FIELD_SPEC_RE = re.compile(r"^(?P<field>[a-z0-9_]+)_tool_spec_(?P<spec>\d+)(?:__|$)", re.IGNORECASE)


def extract_name_spec_field(tool_id: str) -> Tuple[str, Optional[int], str]:
    """Extract tool name (after "__"), spec number, and field from an id.

    Examples:
      id: financial_trading_tool_spec_2__Economic_Indicator_Fetcher
        -> name: Economic_Indicator_Fetcher, spec: 2, field: financial_trading
    """
    name = tool_id.split("__", 1)[1] if "__" in tool_id else tool_id
    m = FIELD_SPEC_RE.match(tool_id)
    if m:
        field = m.group("field")
        spec_num: Optional[int] = int(m.group("spec"))
    else:
        field = "unknown"
        spec_num = None
    return name, spec_num, field


def _seeded_hash(value: str, seed: int) -> int:
    """Deterministic hash for tie-breaking using the provided seed.

    Larger integer value wins when used with max().
    """
    s = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest(), byteorder="big", signed=False)


@lru_cache(maxsize=None)
def load_yaml_metadata(yaml_dir_str: str, field: str, spec: int) -> Tuple[Optional[str], Optional[str]]:
    yaml_dir = Path(yaml_dir_str)
    yaml_path = yaml_dir / f"{field}_tool_spec_{spec}.yml"
    if not yaml_path.exists() or not yaml_path.is_file():
        return None, None
    try:
        try:
            import yaml
        except Exception:
            return None, None
        with yaml_path.open("r", encoding="utf-8") as f:
            data: Dict = yaml.safe_load(f) or {}
    except Exception:
        return None, None
    subfield = data.get("subfield")
    task = data.get("task")
    return subfield, task


def write_jsonl(
    path: Path,
    ids: List[str],
    yaml_dir: Path,
    dropped_ids: Optional[Set[str]] = None,
    extra_fields_by_id: Optional[Dict[str, Dict[str, object]]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for tool_id in ids:
            name, spec_num, field = extract_name_spec_field(tool_id)
            if spec_num is not None:
                subfield, task = load_yaml_metadata(str(yaml_dir), field, spec_num)
            else:
                subfield, task = None, None
            rec = {
                "id": tool_id,
                "name": name,
                "field": field,
                "subfield": subfield,
                "task": task,
                "spec": spec_num,
            }
            if extra_fields_by_id and tool_id in extra_fields_by_id:
                rec.update(extra_fields_by_id[tool_id])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_embedding(npz_path: Path) -> Optional[np.ndarray]:
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "embedding" not in data:
            return None
        emb = np.asarray(data["embedding"], dtype=np.float32)
        if emb.ndim != 1:
            return None
        return emb
    except Exception:
        return None


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if norms.dtype != matrix.dtype:
        norms = norms.astype(matrix.dtype)
    norms[norms == 0.0] = 1.0
    out = matrix / norms
    if out.dtype != matrix.dtype:
        out = out.astype(matrix.dtype)
    return out


def _count_connected_components(adj_bool: np.ndarray) -> int:
    n = adj_bool.shape[0]
    if n == 0:
        return 0
    visited = np.zeros(n, dtype=bool)
    components = 0
    for i in range(n):
        if not visited[i]:
            components += 1
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                neighbors = np.flatnonzero(adj_bool[u])
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
    return components


def compute_components_by_field(ids: List[str], emb_dir: Path, tau: float) -> Dict[str, int]:
    field_to_ids: Dict[str, List[str]] = {}
    for tool_id in ids:
        _, _, field = extract_name_spec_field(tool_id)
        field_to_ids.setdefault(field, []).append(tool_id)

    result: Dict[str, int] = {}
    for field, field_ids in sorted(field_to_ids.items(), key=lambda kv: kv[0]):
        embeddings: List[np.ndarray] = []
        for tool_id in field_ids:
            npz_path = emb_dir / f"{tool_id}.npz"
            emb = _load_embedding(npz_path)
            if emb is not None:
                embeddings.append(emb)
        if not embeddings:
            result[field] = 0
            continue
        X = np.vstack(embeddings).astype(np.float32)
        Xn = _normalize_rows(X)
        S = Xn @ Xn.T
        adj = (S >= tau)
        np.fill_diagonal(adj, False)
        num_components = _count_connected_components(adj)
        result[field] = int(num_components)
    return result


def _connected_components_nodes(adj_bool: np.ndarray) -> List[List[int]]:
    n = adj_bool.shape[0]
    components: List[List[int]] = []
    if n == 0:
        return components
    visited = np.zeros(n, dtype=bool)
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            comp_nodes: List[int] = []
            while stack:
                u = stack.pop()
                comp_nodes.append(u)
                neighbors = np.flatnonzero(adj_bool[u])
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            components.append(comp_nodes)
    return components


def deduplicate_by_field(ids: List[str], emb_dir: Path, tau: float, seed: int) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    rng = random.Random(seed)
    field_to_ids: Dict[str, List[str]] = {}
    for tool_id in ids:
        _, _, field = extract_name_spec_field(tool_id)
        field_to_ids.setdefault(field, []).append(tool_id)

    kept_by_field: Dict[str, Set[str]] = {}
    dropped_by_field: Dict[str, Set[str]] = {}

    for field, field_ids in sorted(field_to_ids.items(), key=lambda kv: kv[0]):
        embeddings: List[np.ndarray] = []
        valid_ids: List[str] = []
        for tool_id in field_ids:
            npz_path = emb_dir / f"{tool_id}.npz"
            emb = _load_embedding(npz_path)
            if emb is not None:
                embeddings.append(emb)
                valid_ids.append(tool_id)

        kept: Set[str] = set(field_ids)
        dropped: Set[str] = set()

        if embeddings:
            X = np.vstack(embeddings).astype(np.float32)
            Xn = _normalize_rows(X)
            S = Xn @ Xn.T
            adj = (S >= tau)
            np.fill_diagonal(adj, False)
            comps = _connected_components_nodes(adj)

            for comp in comps:
                if len(comp) == 1:
                    continue

                comp_indices = np.array(comp, dtype=int)
                S_sub = S[np.ix_(comp_indices, comp_indices)].astype(np.float32)
                S_sub = np.nan_to_num(S_sub, nan=-np.inf, posinf=1.0, neginf=-1.0)
                np.fill_diagonal(S_sub, -np.inf)

                alive: List[int] = list(range(len(comp_indices)))
                drop_logger = logging.getLogger("dedup.similarity")

                def _has_any_edge() -> bool:
                    if not alive:
                        return False
                    A = (S_sub[np.ix_(alive, alive)] >= tau)
                    return bool(np.any(A))

                if len(alive) == 2:
                    i_drop_local = rng.choice(alive)
                    i_keep_local = alive[0] if i_drop_local == alive[1] else alive[1]
                    drop_global = comp_indices[i_drop_local]
                    keep_global = comp_indices[i_keep_local]
                    drop_id = valid_ids[drop_global]
                    keep_id = valid_ids[keep_global]
                    sim_val = float(S_sub[i_drop_local, i_keep_local])
                    assert sim_val >= tau, f"Invariant failed: sim {sim_val:.6f} < tau {tau:.6f} for {drop_id} vs {keep_id}"
                    drop_logger.info(
                        f"DROP similarity: {drop_id} because cosine similarity with {keep_id} was {sim_val:.4f}"
                    )
                    if drop_id in kept:
                        kept.remove(drop_id)
                        dropped.add(drop_id)
                    continue

                while _has_any_edge():
                    S_alive = S_sub[np.ix_(alive, alive)]
                    A_alive = (S_alive >= tau)
                    deg = A_alive.sum(axis=1).astype(int)
                    sum_incident = np.where(A_alive, S_alive, 0.0).sum(axis=1)

                    max_deg = int(deg.max()) if deg.size else 0
                    cand_pos = np.flatnonzero(deg == max_deg)
                    if cand_pos.size > 1:
                        sums = sum_incident[cand_pos]
                        max_sum = float(np.max(sums))
                        cand_pos = cand_pos[np.flatnonzero(sums == max_sum)]
                    if cand_pos.size > 1:
                        cand_locals = [alive[int(p)] for p in cand_pos]
                        cand_hash_vals = [
                            _seeded_hash(valid_ids[int(comp_indices[idx])], seed) for idx in cand_locals
                        ]
                        best_idx = int(np.argmax(np.array(cand_hash_vals)))
                        i_drop_local = cand_locals[best_idx]
                    elif cand_pos.size == 1:
                        i_drop_local = alive[int(cand_pos[0])]
                    else:
                        cand_locals = list(alive)
                        cand_hash_vals = [
                            _seeded_hash(valid_ids[int(comp_indices[idx])], seed) for idx in cand_locals
                        ]
                        best_idx = int(np.argmax(np.array(cand_hash_vals)))
                        i_drop_local = cand_locals[best_idx]

                    neighbor_mask = (S_sub[i_drop_local, alive] >= tau)
                    if not np.any(neighbor_mask):
                        break
                    neighbor_positions = np.flatnonzero(neighbor_mask)
                    neighbor_locals = [alive[int(p)] for p in neighbor_positions]
                    neighbor_sims = [float(S_sub[i_drop_local, j]) for j in neighbor_locals]
                    best_neighbor_idx = int(np.argmax(np.array(neighbor_sims)))
                    i_keep_local = neighbor_locals[best_neighbor_idx]

                    drop_global = comp_indices[i_drop_local]
                    keep_global = comp_indices[i_keep_local]
                    drop_id = valid_ids[drop_global]
                    keep_id = valid_ids[keep_global]
                    sim_val = float(S_sub[i_drop_local, i_keep_local])
                    assert sim_val >= tau, f"Invariant failed: sim {sim_val:.6f} < tau {tau:.6f} for {drop_id} vs {keep_id}"
                    drop_logger.info(
                        f"DROP similarity: {drop_id} because cosine similarity with {keep_id} was {sim_val:.4f}"
                    )

                    if drop_id in kept:
                        kept.remove(drop_id)
                        dropped.add(drop_id)

                    alive = [idx for idx in alive if idx != i_drop_local]

                if alive:
                    survivors_global = [valid_ids[int(comp_indices[i])] for i in alive]
                    for a_pos in range(len(alive)):
                        for b_pos in range(a_pos + 1, len(alive)):
                            ia = alive[a_pos]
                            ib = alive[b_pos]
                            sim_val = float(S_sub[ia, ib])
                            assert sim_val < tau, (
                                f"Invariant failed: survivors {survivors_global[a_pos]} and {survivors_global[b_pos]} "
                                f"have cosine {sim_val:.6f} >= tau {tau:.6f}"
                            )

        kept_by_field[field] = kept
        dropped_by_field[field] = dropped

    return kept_by_field, dropped_by_field


def main() -> None:
    args = parse_args()
    json_dir = Path(args.tool_json_dir).resolve()
    emb_dir = Path(args.emb_dir).resolve()
    yaml_dir = Path(args.yaml_dir).resolve()

    if not json_dir.exists() or not json_dir.is_dir():
        raise SystemExit(f"tool_json_dir not found or not a directory: {json_dir}")
    if not emb_dir.exists() or not emb_dir.is_dir():
        raise SystemExit(f"emb_dir not found or not a directory: {emb_dir}")
    if not yaml_dir.exists() or not yaml_dir.is_dir():
        raise SystemExit(f"yaml_dir not found or not a directory: {yaml_dir}")

    json_stems: Set[str] = set(list_stems(json_dir, "json"))
    emb_stems: Set[str] = set(list_stems(emb_dir, "npz"))

    common_stems = sorted(json_stems.intersection(emb_stems))

    if args.out_path:
        out_path = Path(args.out_path).resolve()
    else:
        out_path = (json_dir.parent / "tool_list.json").resolve()

    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    if args.log_file:
        log_file_path = Path(args.log_file).resolve()
    else:
        log_file_path = out_path.with_name("run_deduplication.log")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler])
    logging.getLogger("dedup").info(json.dumps({
        "event": "start",
        "algorithm": "DefDepUp",
        "taus": [float(t) for t in args.taus],
        "random_seed": int(args.random_seed)
    }))

    log_name = logging.getLogger("dedup.name")
    log_text = logging.getLogger("dedup.text")
    log_sim = logging.getLogger("dedup.similarity")

    def _safe_read_json(json_path: Path) -> Optional[Dict]:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _normalize_string_basic(text: str) -> str:
        t = text.lower()
        t = re.sub(r"[^a-z0-9]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _normalized_name_for_id(tool_id: str, src_dir: Path) -> str:
        rec = _safe_read_json(src_dir / f"{tool_id}.json")
        if isinstance(rec, dict):
            name_val = rec.get("tool_name") or rec.get("name")
        else:
            name_val = None
        if not name_val:
            name_val, _, _ = extract_name_spec_field(tool_id)
        return _normalize_string_basic(str(name_val))

    def _normalized_text_for_id(tool_id: str, src_dir: Path) -> str:
        rec = _safe_read_json(src_dir / f"{tool_id}.json")
        if not isinstance(rec, dict):
            return _normalized_name_for_id(tool_id, src_dir)
        pieces: List[str] = []
        name_val = rec.get("tool_name") or rec.get("name") or ""
        desc_val = rec.get("tool_description") or rec.get("description") or ""
        pieces.append(str(name_val))
        pieces.append(str(desc_val))
        params = rec.get("parameters")
        if isinstance(params, dict):
            for p_name, p_spec in sorted(params.items(), key=lambda kv: kv[0]):
                pieces.append(str(p_name))
                if isinstance(p_spec, dict):
                    pieces.append(str(p_spec.get("description", "")))
                    pieces.append(str(p_spec.get("type", "")))
                    if "default" in p_spec:
                        pieces.append(str(p_spec.get("default")))
        errs = rec.get("error_messages")
        if isinstance(errs, list):
            for e in errs:
                pieces.append(str(e))
        usage = rec.get("usage")
        if usage:
            pieces.append(str(usage))
        outs = rec.get("output_details")
        if isinstance(outs, dict):
            for o_name, o_spec in sorted(outs.items(), key=lambda kv: kv[0]):
                pieces.append(str(o_name))
                if isinstance(o_spec, dict):
                    pieces.append(str(o_spec.get("description", "")))
                    pieces.append(str(o_spec.get("type", "")))
        joined = "\n".join(pieces)
        return _normalize_string_basic(joined)

    def _fingerprint(text_norm: str) -> str:
        return hashlib.sha1(text_norm.encode("utf-8")).hexdigest()[:12]

    seen_name_by_field: Dict[str, Dict[str, str]] = {}
    dropped_exact_name: Set[str] = set()
    kept_after_name: List[str] = []
    if not getattr(args, "exact_ded", True):
        kept_after_name = list(common_stems)
    else:
        for tool_id in common_stems:
            norm_name = _normalized_name_for_id(tool_id, json_dir)
            _, _, fld = extract_name_spec_field(tool_id)
            fld_map = seen_name_by_field.setdefault(fld, {})
            if norm_name in fld_map:
                kept_id = fld_map[norm_name]
                dropped_exact_name.add(tool_id)
                log_name.info(f"DROP exact-name: {tool_id} because normalized name '{norm_name}' equals {kept_id} in field '{fld}'")
            else:
                fld_map[norm_name] = tool_id
                kept_after_name.append(tool_id)
    log_name.info(json.dumps({"step": "exact_name", "scope": "within_field", "dropped": len(dropped_exact_name), "kept": len(kept_after_name)}))

    seen_text: Dict[str, str] = {}
    dropped_exact_text: Set[str] = set()
    kept_after_text: List[str] = []
    if not getattr(args, "exact_ded", True):
        kept_after_text = list(kept_after_name)
    else:
        for tool_id in kept_after_name:
            norm_text = _normalized_text_for_id(tool_id, json_dir)
            fp = _fingerprint(norm_text)
            if norm_text in seen_text:
                kept_id = seen_text[norm_text]
                dropped_exact_text.add(tool_id)
                log_text.info(f"DROP exact-text: {tool_id} because normalized text fp={fp} matches {kept_id}")
            else:
                seen_text[norm_text] = tool_id
                kept_after_text.append(tool_id)
    log_text.info(json.dumps({"step": "exact_text", "dropped": len(dropped_exact_text), "kept": len(kept_after_text)}))

    def _format_tau_key(t: float) -> str:
        s = f"{t:.6f}"
        s = s.rstrip("0").rstrip(".") if "." in s else s
        return s

    taus_list: List[float] = list(args.taus)
    tau_to_dropped: Dict[str, Set[str]] = {}
    for t in taus_list:
        logging.getLogger("dedup").info(json.dumps({
            "event": "run_tau",
            "tau": float(t)
        }))
        _, dropped_by_field = deduplicate_by_field(kept_after_text, emb_dir, float(t), int(args.random_seed))
        dropped_set: Set[str] = set()
        for s in dropped_by_field.values():
            dropped_set.update(s)
        tau_key = _format_tau_key(float(t))
        tau_to_dropped[tau_key] = dropped_set

        if getattr(args, "out_dir", None):
            survivors_tau: List[str] = [tid for tid in kept_after_text if tid not in dropped_set]
            base_out_dir = Path(str(args.out_dir)).resolve()
            if len(taus_list) == 1:
                out_dir_for_tau = base_out_dir
            else:
                out_dir_for_tau = Path(str(f"{base_out_dir}_{tau_key}"))

            if out_dir_for_tau.exists():
                try:
                    if out_dir_for_tau.is_dir():
                        shutil.rmtree(out_dir_for_tau)
                    else:
                        out_dir_for_tau.unlink()
                except Exception:
                    pass
            out_dir_for_tau.mkdir(parents=True, exist_ok=True)

            for tid in survivors_tau:
                src = json_dir / f"{tid}.json"
                dst = out_dir_for_tau / f"{tid}.json"
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    continue

    dropped_ids_all: Set[str] = set(dropped_exact_name) | set(dropped_exact_text)
    for dset in tau_to_dropped.values():
        dropped_ids_all.update(dset)

    extras: Dict[str, Dict[str, object]] = {}
    for tid in common_stems:
        flags: Dict[str, object] = {}
        is_exact_drop_membership = (tid in dropped_exact_name) or (tid in dropped_exact_text)
        flags["dropped_exact"] = bool(is_exact_drop_membership if getattr(args, "exact_ded", True) else False)
        for tau_key, dset in tau_to_dropped.items():
            flags[f"dropped_{tau_key}"] = bool(is_exact_drop_membership or (tid in dset))
        extras[tid] = flags

    # Write JSONL with only per-tau flags (no union flag)
    write_jsonl(out_path, common_stems, yaml_dir, dropped_ids=None, extra_fields_by_id=extras)
    logging.getLogger("dedup").info(
        json.dumps(
            {
                "tool_json_dir": str(json_dir),
                "emb_dir": str(emb_dir),
                "yaml_dir": str(yaml_dir),
                "written": len(common_stems),
                "output": str(out_path),
            },
            ensure_ascii=False,
        )
    )

    # ---- Summary ----
    final_kept_ids: List[str] = [tid for tid in common_stems if tid not in dropped_ids_all]
    # For summary, show component counts at the smallest tau (most edges) for survivors
    components = compute_components_by_field(final_kept_ids, emb_dir, float(min(taus_list)))

    fields_all: Set[str] = set()
    for tid in common_stems:
        _, _, fld = extract_name_spec_field(tid)
        fields_all.add(fld)
    per_field_counts: Dict[str, Dict[str, int]] = {}
    for fld in sorted(fields_all):
        fld_ids = [tid for tid in common_stems if extract_name_spec_field(tid)[2] == fld]
        dropped_cnt = sum(1 for tid in fld_ids if tid in dropped_ids_all)
        remaining_cnt = len(fld_ids) - dropped_cnt
        per_field_counts[fld] = {"dropped": dropped_cnt, "remaining": remaining_cnt}

    # Per-tau summaries
    for tau_key, dset in tau_to_dropped.items():
        logging.getLogger("dedup").info(json.dumps({
            "tau": tau_key,
            "drops": len(dset)
        }))

    # Global summary
    drops_by_tau: Dict[str, int] = {k: len(v) for k, v in tau_to_dropped.items()}
    logging.getLogger("dedup").info(json.dumps({
        "taus": [float(t) for t in taus_list],
        "components_by_field": components,
        "dedup_counts_by_field": per_field_counts,
        "drops": {"similarity_by_tau": drops_by_tau},
        "total_dropped": len(dropped_ids_all),
        "total_remaining": len(final_kept_ids),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
