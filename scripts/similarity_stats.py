"""
Compute cosine-similarity statistics for tool embeddings and optionally save plots.

Usage:
  python scripts/similarity_stats.py \
    /path/to/tool_list.json \
    /path/to/embeddings_dir \
    --save \
    --output-dir tool_content/analysis

Example:
  python scripts/similarity_stats.py \
    tool_content/tool_list.json \
    tool_content/embeddings \
    --taus 0.75 0.85 0.95 \
    --save
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute similarity statistics from tool list and embeddings, optionally plotting PCA/t-SNE.",
    )
    parser.add_argument("tool_list_path", type=Path)
    parser.add_argument("embedding_dir", type=Path)
    parser.add_argument("--include-dropped", action="store_true")
    parser.add_argument("--taus", type=float, nargs="+", default=None, help="List of tau thresholds to evaluate (space-separated). If omitted, auto-detect from dropped_{tau} fields in tool list.")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


FIELD_RE = re.compile(r"^(?P<field>[a-z0-9_]+)_tool_spec_(?P<num>\d+)__", re.IGNORECASE)


def _read_jsonl_or_json(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    records: List[dict] = []
    has_newlines = "\n" in text
    if has_newlines:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                records = []
                break
        if records:
            return records
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _id_to_field(rec_id: str) -> str:
    m = FIELD_RE.match(rec_id)
    return m.group("field") if m else "unknown"


def _id_to_num(rec_id: str) -> str:
    m = FIELD_RE.match(rec_id)
    return m.group("num") if m else "unknown"


def _list_npz(dir_path: Path) -> List[Path]:
    return sorted(p for p in dir_path.glob("*.npz") if p.is_file())


def _load_npz(path: Path) -> Tuple[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "embedding" not in data:
        raise ValueError(f"missing embedding in {path}")
    emb = np.asarray(data["embedding"], dtype=np.float64)
    if emb.ndim != 1:
        raise ValueError(f"embedding must be one dimensional in {path}")
    rec_id = str(data["id"]) if "id" in data and str(data["id"]) else path.stem
    return rec_id, emb


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def compute_similarity_statistics_from_jsonl_and_embeddings(
    tool_list_path: str | Path,
    embedding_dir: str | Path,
    include_dropped: bool = False,
    save: bool = False,
    output_dir: str | Path | None = None,
    taus: Sequence[float] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tool_list_path = Path(tool_list_path)
    embedding_dir = Path(embedding_dir)

    if not tool_list_path.exists():
        raise FileNotFoundError(f"Tool list not found: {tool_list_path}")
    if not embedding_dir.exists():
        raise FileNotFoundError(f"Embedding dir not found: {embedding_dir}")

    records = _read_jsonl_or_json(tool_list_path)
    id_to_field: Dict[str, str] = {}
    id_to_subfield: Dict[str, str] = {}
    id_to_task: Dict[str, str] = {}
    rec_id_to_tau_flags: Dict[str, Dict[str, bool]] = {}

    for rec in records:
        rec_id = str(rec.get("id", ""))
        if not rec_id:
            continue
        id_to_field[rec_id] = str(rec.get("field", _id_to_field(rec_id)))
        id_to_subfield[rec_id] = str(rec.get("subfield", "unknown"))
        id_to_task[rec_id] = str(rec.get("task", "unknown"))

        tau_flags: Dict[str, bool] = {}
        for k, v in rec.items():
            if isinstance(k, str) and k.startswith("dropped_"):
                tau_key = k[len("dropped_"):]
                tau_flags[tau_key] = bool(v)
        rec_id_to_tau_flags[rec_id] = tau_flags

    def _format_tau_key(val: float) -> str:
        s = f"{val:.6f}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    if taus is not None and len(taus) > 0:
        tau_keys: List[str] = [_format_tau_key(float(t)) for t in taus]
    else:
        tau_values: List[float] = []
        seen: set[float] = set()
        for flags in rec_id_to_tau_flags.values():
            for tau_key in flags.keys():
                try:
                    tau_val = float(tau_key)
                except Exception:
                    continue
                if tau_val not in seen:
                    seen.add(tau_val)
                    tau_values.append(tau_val)
        tau_values.sort()
        tau_keys = [_format_tau_key(v) for v in tau_values]
        if not tau_keys:
            raise ValueError("No dropped_{tau} fields found and no --taus provided. Cannot compute per-tau statistics.")

    files = _list_npz(embedding_dir)
    if not files:
        raise ValueError(f"No .npz files found in {embedding_dir}")
    id_to_vec: Dict[str, np.ndarray] = {}
    for p in files:
        rec_id, vec = _load_npz(p)
        id_to_vec[rec_id] = vec

    def _compute_for_ids(active_ids: List[str], plot_suffix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not active_ids:
            raise ValueError("No embeddings matched the selected IDs for tau=" + plot_suffix)
        X = np.vstack([id_to_vec[i] for i in active_ids]).astype(np.float64)
        Xn = _normalize_rows(X)
        S = Xn @ Xn.T
        n = S.shape[0]

        field_to_num_to_idxs: Dict[str, Dict[str, List[int]]] = {}
        field_to_idxs: Dict[str, List[int]] = {}
        subfield_to_idxs: Dict[Tuple[str, str], List[int]] = {}
        task_to_idxs: Dict[Tuple[str, str, str], List[int]] = {}

        for idx, rec_id in enumerate(active_ids):
            fld = id_to_field.get(rec_id, _id_to_field(rec_id))
            subfld = id_to_subfield.get(rec_id, "unknown")
            task = id_to_task.get(rec_id, "unknown")
            num = _id_to_num(rec_id)

            field_to_num_to_idxs.setdefault(fld, {}).setdefault(num, []).append(idx)
            field_to_idxs.setdefault(fld, []).append(idx)
            subfield_to_idxs.setdefault((fld, subfld), []).append(idx)
            task_to_idxs.setdefault((fld, subfld, task), []).append(idx)

        if n >= 2:
            iu = np.triu_indices(n, k=1)
            all_vals = S[iu]
            global_mean = float(all_vals.mean())
            global_std = float(all_vals.std(ddof=0))
            global_min = float(all_vals.min())
            global_max = float(all_vals.max())
        else:
            global_mean = float("nan")
            global_std = float("nan")
            global_min = float("nan")
            global_max = float("nan")

        rows_field: List[dict] = []
        for fld, idxs in sorted(field_to_idxs.items(), key=lambda kv: kv[0]):
            idxs_arr = np.array(idxs, dtype=int)
            cnt = len(idxs_arr)

            if cnt >= 2:
                Sg = S[np.ix_(idxs_arr, idxs_arr)]
                iu_g = np.triu_indices(cnt, k=1)
                vals_g = Sg[iu_g]
                within_mean = float(vals_g.mean())
                within_std = float(vals_g.std(ddof=0))
                within_min = float(vals_g.min())
                within_max = float(vals_g.max())
            else:
                within_mean = float("nan")
                within_std = float("nan")
                within_min = float("nan")
                within_max = float("nan")

            others = np.setdiff1d(np.arange(n, dtype=int), idxs_arr, assume_unique=True)
            if others.size >= 1 and cnt >= 1:
                S_go = S[np.ix_(idxs_arr, others)]
                vals_go = S_go.reshape(-1)
                vs_mean = float(vals_go.mean())
                vs_std = float(vals_go.std(ddof=0))
                vs_min = float(vals_go.min())
                vs_max = float(vals_go.max())
            else:
                vs_mean = float("nan")
                vs_std = float("nan")
                vs_min = float("nan")
                vs_max = float("nan")

            rows_field.append(
                {
                    "field": fld,
                    "count": int(cnt),
                    "within_mean": within_mean,
                    "within_std": within_std,
                    "within_min": within_min,
                    "within_max": within_max,
                    "vs_others_mean": vs_mean,
                    "vs_others_std": vs_std,
                    "vs_others_min": vs_min,
                    "vs_others_max": vs_max,
                }
            )

        field_df = pd.DataFrame(rows_field).sort_values(by="field").reset_index(drop=True)

        rows_subfield: List[dict] = []
        for (fld, subfld), idxs in sorted(subfield_to_idxs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            idxs_arr = np.array(idxs, dtype=int)
            cnt = len(idxs_arr)

            if cnt >= 2:
                Sg = S[np.ix_(idxs_arr, idxs_arr)]
                iu_g = np.triu_indices(cnt, k=1)
                vals_g = Sg[iu_g]
                within_mean = float(vals_g.mean())
                within_std = float(vals_g.std(ddof=0))
                within_min = float(vals_g.min())
                within_max = float(vals_g.max())
            else:
                within_mean = float("nan")
                within_std = float("nan")
                within_min = float("nan")
                within_max = float("nan")

            others = np.setdiff1d(np.arange(n, dtype=int), idxs_arr, assume_unique=True)
            if others.size >= 1 and cnt >= 1:
                S_go = S[np.ix_(idxs_arr, others)]
                vals_go = S_go.reshape(-1)
                vs_mean = float(vals_go.mean())
                vs_std = float(vals_go.std(ddof=0))
                vs_min = float(vals_go.min())
                vs_max = float(vals_go.max())
            else:
                vs_mean = float("nan")
                vs_std = float("nan")
                vs_min = float("nan")
                vs_max = float("nan")

            rows_subfield.append(
                {
                    "field": fld,
                    "subfield": subfld,
                    "count": int(cnt),
                    "within_mean": within_mean,
                    "within_std": within_std,
                    "within_min": within_min,
                    "within_max": within_max,
                    "vs_others_mean": vs_mean,
                    "vs_others_std": vs_std,
                    "vs_others_min": vs_min,
                    "vs_others_max": vs_max,
                }
            )

        subfield_df = (
            pd.DataFrame(rows_subfield)
            .sort_values(by=["field", "subfield"]).reset_index(drop=True)
        )

        rows_task: List[dict] = []
        for (fld, subfld, task), idxs in sorted(task_to_idxs.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            idxs_arr = np.array(idxs, dtype=int)
            cnt = len(idxs_arr)

            if cnt >= 2:
                Sg = S[np.ix_(idxs_arr, idxs_arr)]
                iu_g = np.triu_indices(cnt, k=1)
                vals_g = Sg[iu_g]
                within_mean = float(vals_g.mean())
                within_std = float(vals_g.std(ddof=0))
                within_min = float(vals_g.min())
                within_max = float(vals_g.max())
            else:
                within_mean = float("nan")
                within_std = float("nan")
                within_min = float("nan")
                within_max = float("nan")

            others = np.setdiff1d(np.arange(n, dtype=int), idxs_arr, assume_unique=True)
            if others.size >= 1 and cnt >= 1:
                S_go = S[np.ix_(idxs_arr, others)]
                vals_go = S_go.reshape(-1)
                vs_mean = float(vals_go.mean())
                vs_std = float(vals_go.std(ddof=0))
                vs_min = float(vals_go.min())
                vs_max = float(vals_go.max())
            else:
                vs_mean = float("nan")
                vs_std = float("nan")
                vs_min = float("nan")
                vs_max = float("nan")

            rows_task.append(
                {
                    "field": fld,
                    "subfield": subfld,
                    "task": task,
                    "count": int(cnt),
                    "within_mean": within_mean,
                    "within_std": within_std,
                    "within_min": within_min,
                    "within_max": within_max,
                    "vs_others_mean": vs_mean,
                    "vs_others_std": vs_std,
                    "vs_others_min": vs_min,
                    "vs_others_max": vs_max,
                }
            )

        task_df = (
            pd.DataFrame(rows_task)
            .sort_values(by=["field", "subfield", "task"]).reset_index(drop=True)
        )

        rows_field_num: List[dict] = []
        for fld, num_map in sorted(field_to_num_to_idxs.items(), key=lambda kv: kv[0]):
            def _num_sort_key(k: str) -> Tuple[int, str]:
                try:
                    return (0, int(k))
                except Exception:
                    return (1, k)

            for num, idxs in sorted(num_map.items(), key=lambda kv: _num_sort_key(kv[0])):
                idxs_arr = np.array(idxs, dtype=int)
                cnt = len(idxs_arr)

                if cnt >= 2:
                    Sg = S[np.ix_(idxs_arr, idxs_arr)]
                    iu_g = np.triu_indices(cnt, k=1)
                    vals_g = Sg[iu_g]
                    within_mean = float(vals_g.mean())
                    within_std = float(vals_g.std(ddof=0))
                    within_min = float(vals_g.min())
                    within_max = float(vals_g.max())
                else:
                    within_mean = float("nan")
                    within_std = float("nan")
                    within_min = float("nan")
                    within_max = float("nan")

                others = np.setdiff1d(np.arange(n, dtype=int), idxs_arr, assume_unique=True)
                if others.size >= 1 and cnt >= 1:
                    S_go = S[np.ix_(idxs_arr, others)]
                    vals_go = S_go.reshape(-1)
                    vs_mean = float(vals_go.mean())
                    vs_std = float(vals_go.std(ddof=0))
                    vs_min = float(vals_go.min())
                    vs_max = float(vals_go.max())
                else:
                    vs_mean = float("nan")
                    vs_std = float("nan")
                    vs_min = float("nan")
                    vs_max = float("nan")

                rows_field_num.append(
                    {
                        "field": fld,
                        "num": str(num),
                        "count": int(cnt),
                        "within_mean": within_mean,
                        "within_std": within_std,
                        "within_min": within_min,
                        "within_max": within_max,
                        "vs_others_mean": vs_mean,
                        "vs_others_std": vs_std,
                        "vs_others_min": vs_min,
                        "vs_others_max": vs_max,
                    }
                )

        field_num_df = pd.DataFrame(rows_field_num)
        if not field_num_df.empty:
            field_num_df["num_sort"] = pd.to_numeric(field_num_df["num"], errors="coerce")
            field_num_df["num_sort"] = field_num_df["num_sort"].fillna(10**9).astype(int)
            field_num_df = field_num_df.sort_values(by=["field", "num_sort"]).reset_index(drop=True)
            field_num_df = field_num_df.drop(columns=["num_sort"]) 

        global_summary = pd.DataFrame(
            [
                {
                    "num_vectors": int(n),
                    "overall_mean": global_mean,
                    "overall_std": global_std,
                    "overall_min": global_min,
                    "overall_max": global_max,
                }
            ]
        )


        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        COLORS = {
            "ink": "#222222",
            "grid": "#E6E6E6",
            "tau": "#C46A3D",
        }

        mpl.rcParams.update({
            "figure.figsize": (12, 8),
            "figure.dpi": 150,
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linestyle": ":",
            "grid.linewidth": 1.0,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "legend.frameon": True,
            "font.family": "serif",
        })

        fields: List[str] = [_id_to_field(rid) for rid in active_ids]
        fields_fmt: List[str] = [f.replace("_", " ").title() for f in fields]
        unique_fields = sorted(pd.unique(np.asarray(fields_fmt, dtype=object)))
        
        palette = sns.color_palette("Set2", n_colors=len(unique_fields))
        palette_map = {name: palette[i] for i, name in enumerate(unique_fields)}

        pca = PCA(n_components=2, random_state=42)
        X2_pca = pca.fit_transform(Xn)
        plot_df_pca = pd.DataFrame({
            "x": X2_pca[:, 0],
            "y": X2_pca[:, 1],
            "field": fields_fmt,
            "id": active_ids,
        })
        
        fig_pca, ax_pca = plt.subplots(figsize=(12, 8))
        
        sns.scatterplot(
            data=plot_df_pca,
            x="x",
            y="y",
            hue="field",
            hue_order=unique_fields,
            palette=palette_map,
            s=40,
            alpha=0.9,
            edgecolor=COLORS["ink"],
            linewidth=0.3,
            ax=ax_pca,
        )
        
        ax_pca.set_title("PCA of Synthetic Tools")
        ax_pca.set_xlabel("PCA 1")
        ax_pca.set_ylabel("PCA 2")
        
        leg = ax_pca.legend(title="Field", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        if leg is not None:
            frame = leg.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("#AAAAAA")
            frame.set_linewidth(0.8)
            frame.set_alpha(0.95)
            leg.set_title("Field", prop={"weight": "bold"})
            for txt in leg.get_texts():
                txt.set_color(COLORS["ink"])
        
        plt.tight_layout()
        out_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_pca.savefig(out_dir / f"tools_pca_tau_{plot_suffix}.png", dpi=150, bbox_inches='tight')
        plt.close(fig_pca)

        perplexity = 30 if n > 31 else max(5, n - 1) if n > 1 else 1
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
            metric="cosine",
        )
        X2_tsne = tsne.fit_transform(Xn)
        plot_df_tsne = pd.DataFrame({
            "x": X2_tsne[:, 0],
            "y": X2_tsne[:, 1],
            "field": fields_fmt,
            "id": active_ids,
        })
        
        fig_tsne, ax_tsne = plt.subplots(figsize=(12, 8))
        
        sns.scatterplot(
            data=plot_df_tsne,
            x="x",
            y="y",
            hue="field",
            hue_order=unique_fields,
            palette=palette_map,
            s=40,
            alpha=0.9,
            edgecolor=COLORS["ink"],
            linewidth=0.3,
            ax=ax_tsne,
        )
        
        ax_tsne.set_title("t-SNE of Synthetic Tools")
        ax_tsne.set_xlabel("t-SNE 1")
        ax_tsne.set_ylabel("t-SNE 2")
        
        leg = ax_tsne.legend(title="Field", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        if leg is not None:
            frame = leg.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("#AAAAAA")
            frame.set_linewidth(0.8)
            frame.set_alpha(0.95)
            leg.set_title("Field")
            for txt in leg.get_texts():
                txt.set_color(COLORS["ink"])
        
        plt.tight_layout()
        fig_tsne.savefig(out_dir / f"tools_tsne_tau_{plot_suffix}.png", dpi=150, bbox_inches='tight')
        plt.close(fig_tsne)

        return field_df, field_num_df, global_summary, subfield_df, task_df

    field_by_tau: Dict[str, pd.DataFrame] = {}
    subfield_by_tau: Dict[str, pd.DataFrame] = {}
    task_by_tau: Dict[str, pd.DataFrame] = {}
    global_by_tau: Dict[str, pd.DataFrame] = {}

    for tau_key in tau_keys:
        allowed_ids = [rid for rid, flags in rec_id_to_tau_flags.items() if (not flags.get(tau_key, False)) or include_dropped]
        active_ids = [rid for rid in allowed_ids if rid in id_to_vec]
        field_df, field_num_df, global_summary, subfield_df, task_df = _compute_for_ids(active_ids, tau_key)
        field_by_tau[tau_key] = field_df
        subfield_by_tau[tau_key] = subfield_df
        task_by_tau[tau_key] = task_df
        global_by_tau[tau_key] = global_summary

    def _suffix_and_prepare(df: pd.DataFrame, key_cols: List[str], tau_key: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=key_cols)
        suffixed = df.copy()
        value_cols = [c for c in suffixed.columns if c not in key_cols]
        rename_map = {c: f"{c}_tau_{tau_key}" for c in value_cols}
        suffixed = suffixed.rename(columns=rename_map)
        return suffixed

    def _combine_frames(frames_by_tau: Dict[str, pd.DataFrame], key_cols: List[str]) -> pd.DataFrame:
        combined: pd.DataFrame | None = None
        for tau_key in tau_keys:
            df_tau = _suffix_and_prepare(frames_by_tau.get(tau_key, pd.DataFrame()), key_cols, tau_key)
            if combined is None:
                combined = df_tau
            else:
                combined = combined.merge(df_tau, on=key_cols, how="outer")
        if combined is None:
            combined = pd.DataFrame(columns=key_cols)
        combined = combined.sort_values(by=key_cols).reset_index(drop=True)
        return combined

    combined_field_df = _combine_frames(field_by_tau, ["field"])
    combined_subfield_df = _combine_frames(subfield_by_tau, ["field", "subfield"])
    combined_task_df = _combine_frames(task_by_tau, ["field", "subfield", "task"])

    combined_global = pd.DataFrame([{}])
    for tau_key in tau_keys:
        g = global_by_tau.get(tau_key)
        if g is None or g.empty:
            continue
        flat = {f"{col}_tau_{tau_key}": g.iloc[0][col] for col in g.columns}
        combined_global = pd.concat([combined_global, pd.DataFrame([flat])], axis=1)
    combined_global = combined_global.loc[:, ~combined_global.columns.duplicated()].copy()
    if combined_global.empty:
        combined_global = pd.DataFrame()

    if save:
        out_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        if not combined_global.empty:
            combined_global.to_csv(out_dir / "aggregate.csv", index=False)
        combined_field_df.to_csv(out_dir / "by_field.csv", index=False)
        combined_subfield_df.to_csv(out_dir / "by_subfield.csv", index=False)
        combined_task_df.to_csv(out_dir / "by_task.csv", index=False)

    return combined_field_df, pd.DataFrame(), combined_global


__all__ = [
    "compute_similarity_statistics_from_jsonl_and_embeddings",
]


if __name__ == "__main__":
    args = parse_arguments()

    try:
        field_df, field_num_df, global_df = compute_similarity_statistics_from_jsonl_and_embeddings(
            tool_list_path=args.tool_list_path,
            embedding_dir=args.embedding_dir,
            include_dropped=args.include_dropped,
            save=args.save,
            output_dir=args.output_dir,
            taus=args.taus,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)
    print("Aggregate (global) summary:")
    print(global_df.to_string(index=False))
    print()
    print("By field:")
    print(field_df.to_string(index=False))
    print()
    if args.save:
        out_dir = Path(args.output_dir) if args.output_dir is not None else Path(__file__).resolve().parent
        try:
            subfield_df = pd.read_csv(out_dir / "by_subfield.csv")
            task_df = pd.read_csv(out_dir / "by_task.csv")
        except Exception:
            subfield_df = None
            task_df = None
    else:
        subfield_df = None
        task_df = None

    if subfield_df is not None:
        print("By (field, subfield):")
        print(subfield_df.to_string(index=False))
        print()
    if task_df is not None:
        print("By (field, subfield, task):")
        print(task_df.to_string(index=False))

