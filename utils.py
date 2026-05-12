import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def write_json_atomic(obj: Any, path: Path) -> Path:
    """Write `obj` to `path` atomically: dump to `<path>.tmp`, then `os.replace`.

    `os.replace` is atomic on POSIX, so a reader (or a resume run) only ever
    sees the final file or nothing — never a half-written file. A killed
    writer leaves at most a stray `.tmp` next to `path`; resume logic keys
    off `path.exists()`, so the stray is harmlessly overwritten next run.

    Matches the existing dump flags used across the codebase: indent=2,
    ensure_ascii=False, default=str.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)
    return path


def str_representer(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


def parse_list(text: str) -> list:
    """Parse a JSON list from LLM response text."""
    # Try to find JSON array in code block
    match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    # Try to find array anywhere in text
    match = re.search(r'\[[\s\S]*?\]', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return []


def _find_json_objects(text: str) -> list:
    """Find JSON object boundaries using brace counting (handles nesting)."""
    candidates = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Track braces, respecting strings
            depth = 0
            in_string = False
            escape = False
            start = i
            for j in range(i, len(text)):
                c = text[j]
                if escape:
                    escape = False
                    continue
                if c == '\\' and in_string:
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:j + 1])
                        i = j + 1
                        break
            else:
                # Unbalanced — skip this opening brace
                i += 1
        else:
            i += 1
    return candidates


def extract_json_objects(text: str) -> list:
    """Best-effort extraction of all JSON objects in text."""
    if not text:
        return []

    objects = []
    seen = set()

    def try_add(candidate: str):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                # Deduplicate by content
                key = json.dumps(parsed, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    objects.append(parsed)
        except Exception:
            pass

    # 1. Direct parse (entire text is one JSON object)
    try_add(text.strip())

    # 2. Fenced json blocks ```json ... ```
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        try_add(match.group(1).strip())

    # 3. Brace-counting search (handles nested objects correctly)
    if not objects:
        for candidate in _find_json_objects(text):
            try_add(candidate)

    return objects


def setup_yaml():
    """Register custom YAML representer for multiline strings.

    Registers with both Dumper and SafeDumper so it works with custom dumpers
    that inherit from either.
    """
    yaml.add_representer(str, str_representer)
    yaml.add_representer(str, str_representer, Dumper=yaml.SafeDumper)


def deterministic_diverse_permutations(objects, m):
    import math

    n = len(objects)
    if m <= 0:
        return []
    if n <= 1:
        return [list(objects)] if m >= 1 else []

    total = math.factorial(n)
    if m > total:
        m = total

    factorials = [1] * (n + 1)
    for i in range(2, n + 1):
        factorials[i] = factorials[i - 1] * i

    def unrank(rank):
        remaining = list(range(n))
        perm_idx = []
        r = rank
        for k in range(n, 0, -1):
            f = factorials[k - 1]
            q, r = divmod(r, f)
            perm_idx.append(remaining.pop(q))
        return [objects[i] for i in perm_idx]

    if m == 1:
        return [unrank(0)]

    ranks = []
    used = set()
    for k in range(m):
        r = (k * (total - 1)) // (m - 1)
        while r in used:
            r += 1
            if r >= total:
                r = 0
        used.add(r)
        ranks.append(r)

    return [unrank(r) for r in ranks]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_SYNTHTOOLS_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str = "synthtools") -> logging.Logger:
    """Return a shared logger configured once with a stderr handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_SYNTHTOOLS_FORMATTER)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def redirect_synthtools_logger_to_file(log_path: Path) -> Path:
    """Swap the `synthtools` logger's handlers for a single FileHandler.

    Used by parallel-trajectory workers so each process writes its detailed
    logs to its own file instead of interleaving on the parent's stderr.
    Idempotent: if the log file already has a FileHandler open, replaces it.

    The `traj_generation.run` logger is intentionally left alone so the
    controller (parent) can still print orchestration messages to stderr.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("synthtools")
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(_SYNTHTOOLS_FORMATTER)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return log_path


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

def usage_to_dict(usage: Any) -> Optional[Dict[str, int]]:
    """Convert a Usage dataclass, a dict, or None to a plain {prompt_tokens, completion_tokens} dict."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "prompt_tokens"):
        return {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
    return None


def usage_str(usage: Any) -> str:
    """One-line log-friendly summary of a Usage, e.g. '[prompt=42, completion=17]'."""
    d = usage_to_dict(usage)
    if not d:
        return ""
    return f"[prompt={d.get('prompt_tokens', '?')}, completion={d.get('completion_tokens', '?')}]"


class UsageTracker:
    """Accumulate Usage tokens across a run.

    Handles plain Usage dataclasses, dicts, and the nested
    {"check": Usage, "simulation": Usage} shape emitted by ToolSimulator.
    """

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def track(self, result: Any) -> None:
        """Track usage from either a dict result with a 'usage' key, or a Usage-like object."""
        if result is None:
            return
        if isinstance(result, dict) and "usage" in result:
            usage = result["usage"]
        else:
            usage = result
        if usage is None:
            return
        if isinstance(usage, dict) and "prompt_tokens" not in usage:
            for v in usage.values():
                self._add(v)
            return
        self._add(usage)

    def _add(self, usage: Any) -> None:
        d = usage_to_dict(usage)
        if not d:
            return
        self.prompt_tokens += d.get("prompt_tokens", 0) or 0
        self.completion_tokens += d.get("completion_tokens", 0) or 0

    @property
    def total(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


# ---------------------------------------------------------------------------
# Artifact numbering
# ---------------------------------------------------------------------------

def next_artifact_index(prefix: str, output_dir: Path) -> int:
    """Return max(index) + 1 across `<prefix>_NNN.json` and `<prefix>_NNN.debug.json`.

    Uses max (not count) so gaps or orphan debug files never trigger a collision.
    """
    next_idx = 0
    output_dir = Path(output_dir)
    if output_dir.exists():
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)(?:\.debug)?\.json$")
        used = [int(m.group(1)) for p in output_dir.iterdir() if (m := pattern.match(p.name))]
        if used:
            next_idx = max(used) + 1
    return next_idx


# ---------------------------------------------------------------------------
# Chat message rewriting
# ---------------------------------------------------------------------------

def to_llm_messages(messages: List[Dict]) -> List[Dict]:
    """Rewrite role: tool → role: user for chat templates that require tool_calls.

    GPT-OSS (and others) reject a role: tool message unless the preceding assistant
    message has a structured tool_calls field. Our solver puts the tool call as plain
    JSON in content, so we rewrite tool messages to user with a "Tool response: " prefix
    only when handing to the LLM. Saved trajectories keep role: tool for training clarity.
    """
    out = []
    for m in messages:
        if m.get("role") == "tool":
            out.append({"role": "user", "content": f"Tool response: {m.get('content', '')}"})
        else:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------

class RunLog:
    """Flat chronological log of every LLM call, saved as `<task_id>.debug.json`."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.events: List[Dict[str, Any]] = []

    def record(self, agent: str, action: str, turn_ref: Dict, result: Dict) -> None:
        self.events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "action": action,
            "turn_ref": turn_ref,
            "prompt": result.get("prompt"),
            "response": result.get("response"),
            "parsed": result.get("parsed"),
            "usage": usage_to_dict(result.get("usage")),
        })

    def save(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        path = output_dir / f"{self.task_id}.debug.json"
        write_json_atomic({"task_id": self.task_id, "events": self.events}, path)
        return path


# ---------------------------------------------------------------------------
# Batched LLM helper
# ---------------------------------------------------------------------------

def batch_call(llm, prompts: List[str]) -> List[Dict[str, Any]]:
    """Run a batched LLM call; return per-prompt `{response, usage}` dicts.

    Uses `llm.last_usage_per_request` for accurate per-request token counts when
    available; if the LLM only reports aggregate usage, falls back to splitting
    the total evenly (lossy but never None). A single prompt goes through the
    non-batched path for clarity.
    """
    if not prompts:
        return []
    if len(prompts) == 1:
        response = llm([{"role": "user", "content": prompts[0]}])
        return [{"response": response, "usage": llm.last_usage}]

    messages = [[{"role": "user", "content": p}] for p in prompts]
    responses = llm(messages)
    per_request = getattr(llm, "last_usage_per_request", None)
    if per_request and len(per_request) == len(responses):
        return [{"response": r, "usage": u} for r, u in zip(responses, per_request)]

    # Fallback: split total evenly. Lazy import of Usage keeps utils.py free of
    # import-time coupling to llm.py.
    total = getattr(llm, "last_usage", None)
    if total is None:
        split: List[Any] = [None] * len(responses)
    else:
        from llm import Usage
        n = len(responses)
        split = [Usage(total.prompt_tokens // n, total.completion_tokens // n) for _ in range(n)]
    return [{"response": r, "usage": u} for r, u in zip(responses, split)]
