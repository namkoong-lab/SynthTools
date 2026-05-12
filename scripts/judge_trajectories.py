"""Ad-hoc LLM judge for regenerated trajectories — NOT part of the pipeline.

Samples N trajectories, replays each one as a real chat conversation
(system → user → assistant → tool/user → ... → final user asking for verdict),
asks the local LLM to rate each {"quality": 0|1, "explanation": "..."},
prints results and writes a JSONL.

Results land in /tmp by default — this script is exploratory, not pipeline output.

Usage:
    python -m scripts.judge_trajectories \\
        --trajectories-dir /pscratch/sd/t/tcaste/tool_content/trajectories_cot_regen \\
        --n 200 \\
        --model GPT-OSS-120B \\
        --out /tmp/traj_judge_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm import LLM  # noqa: E402
from utils import extract_json_objects, get_logger, to_llm_messages  # noqa: E402

logger = get_logger("synthtools")


def _batch_chat_call(llm, messages_batch):
    """Call the LLM with a batch of chat-format message lists. Returns a list
    of {response, usage} in input order. Mirrors utils.batch_call but passes
    chat messages instead of flat prompt strings."""
    if not messages_batch:
        return []
    if len(messages_batch) == 1:
        response = llm(messages_batch[0])
        return [{"response": response, "usage": getattr(llm, "last_usage", None)}]
    responses = llm(messages_batch)
    per_request = getattr(llm, "last_usage_per_request", None)
    if per_request and len(per_request) == len(responses):
        return [{"response": r, "usage": u} for r, u in zip(responses, per_request)]
    return [{"response": r, "usage": None} for r in responses]


JUDGE_SYSTEM = """You are a strict quality judge for a synthetic tool-use training dataset.

You will be replayed a multi-turn trajectory as the actual chat conversation:
  - role=user        : the task description for that turn
  - role=assistant   : the agent's response. By dataset SCHEMA, this is always a JSON
                       object {"reason": "<flowing prose reasoning>", "tool_call": "<call>"}.
                       That JSON wrapping is the storage format and is NOT an artifact —
                       evaluate the prose inside `reason` and the call inside `tool_call`.
  - role=user        : the tool's response, prefixed "Tool response:". Always contains
                       a `status_code` (200 = success, 4xx = client error, 5xx = server
                       error) and a `response` body. READ THIS BEFORE JUDGING THE TURN.

After the replay, a final user message will ask you for a verdict.

CRITICAL — how 4xx/5xx turns work in this dataset:

  Failed-then-corrected is GOOD training signal, not bad.

  If a turn's tool_call gets a 4xx/5xx tool response (e.g. omitted quotes, wrong type,
  missing field, fabricated ID), and a SUBSEQUENT turn fixes the call (re-quotes the
  string, re-types the value, uses the right ID) and gets a 200 — that ENTIRE pattern
  is exactly what we want the model to learn from. The 4xx is the supervision: it
  teaches the model that the call was malformed.

  Therefore:
    - DO NOT penalize a tool_call argument error in turn N if turn N's tool response
      is 4xx/5xx AND a later turn corrects it (even byte-identical retry with quoting
      added, or re-derived values). That is the recovery pattern.
    - DO NOT penalize a confabulated ID in turn N if the tool returned 4xx and a later
      turn replaces it with the right value.
    - DO penalize when an error is uncorrected (same broken pattern persists), or when
      a 200 turn references invented values that were never in any prior 200 response.

Score the whole trajectory on these criteria:
  1. The text inside each `reason` field is coherent flowing prose.
  2. Each `reason` correctly references prior SUCCESSFUL (200) tool-response facts when
     relevant. Failed (4xx/5xx) prior responses can be ignored or used as recovery cues.
  3. Each `reason` justifies the `tool_call` and treats it as the right next move
     (no second-guessing, no apologizing — even if the call ends up failing, the
     reasoning should be confident going in).
  4. For 200 turns: arguments are derivable from the task + prior 200 responses; no
     invented IDs/values, no schema-type violations.
  5. Multi-turn coherence on the 200-success path: later turns build on facts from
     earlier 200 responses.

DO NOT penalize the JSON wrapping `{"reason": ..., "tool_call": ...}` itself — that is
the dataset's storage schema.
DO NOT penalize 4xx/5xx turns whose errors are corrected later — that is training
signal. Phrases like "despite later corrections" should NEVER appear in your verdict;
if it was corrected later, it's good signal.

When asked, output ONLY a JSON object — no preamble, no code fences:
  {"quality": 0 or 1, "explanation": "<one or two concise sentences>"}

quality=1 = trajectory is good training data, including any failure-recovery patterns.
quality=0 = serious UNCORRECTED problem in reasoning, in 200-turn tool_call arguments,
            or in multi-turn coherence on the 200-success path."""


JUDGE_USER_FINAL = (
    "End of trajectory. Output your verdict now as JSON only: "
    '{"quality": 0 or 1, "explanation": "<one or two concise sentences>"}'
)


def build_messages(traj: dict) -> list[dict]:
    """Build a chat conversation: system, then replay every chat message of every
    turn in order, then a final user message asking for the verdict.

    `to_llm_messages` rewrites role=tool → role=user (with a "Tool response:" prefix)
    so the chat template accepts the sequence.
    """
    msgs: list[dict] = [{"role": "system", "content": JUDGE_SYSTEM}]
    for turn in traj.get("turns", []):
        msgs.extend(turn.get("chat", []))
    msgs.append({"role": "user", "content": JUDGE_USER_FINAL})
    return to_llm_messages(msgs)


def parse_verdict(response: str) -> dict:
    """Extract {quality, explanation} from the LLM response. Lenient."""
    objs = extract_json_objects(response)
    for o in objs:
        if isinstance(o, dict) and "quality" in o:
            try:
                q = int(o["quality"])
                if q not in (0, 1):
                    continue
                return {"quality": q, "explanation": str(o.get("explanation", ""))}
            except Exception:
                continue
    return {"quality": None, "explanation": f"<unparseable judge output>: {response[:300]}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories-dir", type=Path,
                    default=Path("/pscratch/sd/t/tcaste/tool_content/trajectories_cot_regen"))
    ap.add_argument("--n", type=int, default=50,
                    help="number of trajectories to judge; 0 or -1 = all")
    ap.add_argument("--model", default="GPT-OSS-120B")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("/tmp/traj_judge_results.jsonl"))
    args = ap.parse_args()

    # Match trajectory JSONs only — skip per-task debug logs (`*.debug.json`).
    files = sorted(p for p in args.trajectories_dir.glob("*.json")
                   if not p.name.endswith(".debug.json"))
    if args.n <= 0 or args.n >= len(files):
        sample = files
        logger.info(f"corpus: {len(files):,} trajectories  →  judging ALL")
    else:
        rng = random.Random(args.seed)
        sample = rng.sample(files, args.n)
        logger.info(f"corpus: {len(files):,} trajectories  →  sampling {args.n}")

    trajs = []
    messages_batch = []
    for f in sample:
        try:
            t = json.loads(f.read_text())
        except Exception as e:
            logger.warning(f"{f.name}: {e}")
            continue
        trajs.append((f, t))
        messages_batch.append(build_messages(t))
    logger.info(f"built {len(messages_batch)} chat conversations")

    llm = LLM(args.model, max_tokens=args.max_tokens)
    if hasattr(llm, "cfg") and llm.cfg is not None:
        llm.cfg.temperature = args.temperature

    logger.info("submitting batched chat calls...")
    results = _batch_chat_call(llm, messages_batch)

    out_lines = []
    counts = Counter()
    bad = []
    for (f, t), r in zip(trajs, results):
        resp = r.get("response", "") if isinstance(r, dict) else ""
        verdict = parse_verdict(resp)
        counts[verdict["quality"]] += 1
        rec = {
            "task_id": t.get("task_id"),
            "task_field": t.get("task_field"),
            "n_turns": len(t.get("turns", [])),
            "quality": verdict["quality"],
            "explanation": verdict["explanation"],
        }
        out_lines.append(json.dumps(rec, ensure_ascii=False))
        if verdict["quality"] != 1:
            bad.append(rec)

    args.out.write_text("\n".join(out_lines) + "\n")
    logger.info(f"wrote {len(out_lines)} verdicts → {args.out}")

    print("\n=== verdict distribution ===")
    total = sum(counts.values())
    for q in (1, 0, None):
        n = counts[q]
        label = {1: "good", 0: "bad", None: "unparseable"}[q]
        pct = 100 * n / total if total else 0
        print(f"  quality={q!r:<5} ({label:<11}): {n:>6,} ({pct:5.1f}%)")

    # Per-field bad-rate table (derive field from task_id prefix)
    by_field: dict[str, list[int | None]] = {}
    for line in out_lines:
        rec = json.loads(line)
        # task_id like "aerospace_and_defense_spec_019_seq19" — field = up to "_spec_"
        tid = rec.get("task_id") or ""
        field = tid.split("_spec_")[0] if "_spec_" in tid else "?"
        by_field.setdefault(field, []).append(rec.get("quality"))
    print("\n=== bad-rate by field (sorted by bad-rate desc) ===")
    rows = []
    for field, qs in by_field.items():
        n_total = len(qs)
        n_bad = sum(1 for q in qs if q == 0)
        n_unp = sum(1 for q in qs if q is None)
        rate = (n_bad + n_unp) / n_total if n_total else 0
        rows.append((field, n_total, n_bad, n_unp, rate))
    rows.sort(key=lambda r: -r[4])
    print(f"  {'field':<48} {'total':>6} {'bad':>5} {'unp':>5} {'bad%':>6}")
    for field, n_total, n_bad, n_unp, rate in rows:
        print(f"  {field:<48} {n_total:>6,} {n_bad:>5,} {n_unp:>5,} {100*rate:>5.1f}%")

    if bad and len(bad) <= 50:
        print("\n=== bad / unparseable verdicts ===")
        for rec in bad:
            print(f"\n  {rec['task_id']} (turns={rec['n_turns']})")
            print(f"  quality: {rec['quality']}")
            print(f"  explanation: {rec['explanation']}")
    elif bad:
        print(f"\n=== {len(bad)} bad/unparseable verdicts in {args.out} (too many to dump inline) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
