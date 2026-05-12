"""Drop trajectories named in one or more task_id-list files from a corpus.

Default behavior is non-destructive: bad files are MOVED to a sibling
`<trajectories-dir>_dropped/` directory. Pass `--delete` to actually unlink.

Pair with `scripts/audit_split.py` (which produces per-category task_id lists)
and `scripts/audit_recovery_check.py` (which splits ast_quoting into
recovered vs uncorrected).

Usage (conservative — only drop one bucket):
    python -m scripts.drop_bad_trajectories \\
        --task-ids-file /tmp/audit_buckets/ignored_task_part.txt \\
        --trajectories-dir /pscratch/sd/t/tcaste/tool_content/trajectories_cot_regen

Usage (aggressive — drop multiple buckets in one go):
    python -m scripts.drop_bad_trajectories \\
        --task-ids-file /tmp/audit_buckets/ignored_task_part.txt \\
        --task-ids-file /tmp/audit_buckets/schema_type_violation.txt \\
        --task-ids-file /tmp/audit_buckets/confabulated_value.txt \\
        --task-ids-file /tmp/audit_buckets/uncorrected_4xx.txt \\
        --task-ids-file /tmp/audit_buckets/wrong_tool.txt \\
        --task-ids-file /tmp/audit_recovery/ast_quoting_uncorrected.txt \\
        --trajectories-dir /pscratch/sd/t/tcaste/tool_content/trajectories_cot_regen

Variants:
    --delete       actually unlink (destructive; cannot combine with --quarantine-dir)
    --dry-run      show what would happen, no filesystem change
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import get_logger  # noqa: E402

logger = get_logger("synthtools")


def load_task_ids(path: Path) -> set[str]:
    out: set[str] = set()
    with open(path) as f:
        for line in f:
            tid = line.strip()
            if tid:
                out.add(tid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-ids-file", type=Path, action="append", required=True,
                    help="File of task_ids (one per line). Repeat to combine buckets.")
    ap.add_argument("--trajectories-dir", type=Path, required=True,
                    help="Directory of <task_id>.json trajectory files to filter")
    ap.add_argument("--quarantine-dir", type=Path, default=None,
                    help="Where to move bad files. Default: <trajectories-dir>_dropped")
    ap.add_argument("--delete", action="store_true",
                    help="Delete instead of move (destructive).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would happen without touching the filesystem.")
    args = ap.parse_args()

    if not args.trajectories_dir.is_dir():
        logger.error(f"trajectories dir not found: {args.trajectories_dir}")
        return 1
    if args.delete and args.quarantine_dir is not None:
        logger.error("cannot pass both --delete and --quarantine-dir")
        return 1

    bad_ids: set[str] = set()
    per_file_counts: dict[str, int] = {}
    for tf in args.task_ids_file:
        if not tf.exists():
            logger.error(f"task_ids file not found: {tf}")
            return 1
        ids = load_task_ids(tf)
        per_file_counts[str(tf)] = len(ids)
        bad_ids |= ids

    print(f"\n=== drop summary ===")
    for tf, n in per_file_counts.items():
        print(f"  {tf}: {n:,} task_ids")
    print(f"  union: {len(bad_ids):,} unique task_ids to act on")

    quarantine = args.quarantine_dir or args.trajectories_dir.with_name(
        args.trajectories_dir.name + "_dropped"
    )
    if not args.dry_run and not args.delete:
        quarantine.mkdir(parents=True, exist_ok=True)

    n_moved = 0
    n_deleted = 0
    n_missing = 0
    for tid in sorted(bad_ids):
        src = args.trajectories_dir / f"{tid}.json"
        debug = args.trajectories_dir / f"{tid}.debug.json"
        if not src.exists():
            n_missing += 1
            continue
        if args.dry_run:
            action = "delete" if args.delete else f"move → {quarantine.name}/"
            logger.info(f"  (dry-run) would {action}: {src.name}")
            continue
        if args.delete:
            src.unlink()
            n_deleted += 1
            if debug.exists():
                debug.unlink()
        else:
            shutil.move(str(src), str(quarantine / src.name))
            n_moved += 1
            if debug.exists():
                shutil.move(str(debug), str(quarantine / debug.name))

    print()
    if args.dry_run:
        action = "delete" if args.delete else "move"
        print(f"  (dry-run) would {action}:    {len(bad_ids) - n_missing:,}")
    else:
        if args.delete:
            print(f"  deleted: {n_deleted:,}")
        else:
            print(f"  moved:   {n_moved:,}")
            print(f"  quarantine: {quarantine}")
    print(f"  not in corpus (skipped): {n_missing:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
