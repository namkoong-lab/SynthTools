"""Structural validation of a released task, before any rollout.

Rationale. Measured on the bulk corpus, the dominant cause of unusable
trajectories is a broken *task*, not a weak agent: entities the ground
truth requires that the environment never contained, ground truth longer
than the turn budget, ground truth that contradicts itself. Those are
provable from the release row alone. Proving the converse — that a task
*is* good — is not possible here and is not attempted.

Every check is deterministic and LLM-free. Levels follow task_audit:

  BLOCK — the task is structurally unsolvable; do not roll out, do not
          train on any trajectory produced from it
  WARN  — suspicious but survivable; recorded on the report for triage

The checks:

  V1_gt_empty              BLOCK  ground truth is missing or empty
  V2_gt_unparseable        WARN   a GT call did not parse (see note below)
  V3_unknown_tool_in_gt    BLOCK  a GT call names a tool not in `tools`
  V4_budget_infeasible     BLOCK  len(gt_tool_calls) > max_solver_turns
  V5_entity_not_in_env     BLOCK  a GT argument names an entity of a family
                                  the environment enumerates, but that
                                  entity is absent — nothing can satisfy it
  V6_gt_use_after_delete   BLOCK  GT operates on an entity a previous GT
                                  call deleted
  V7_summary_entity_unknown WARN  the summary names an entity id absent
                                  from initial_state and from the GT chain
  V8_trivial               WARN   fewer than 2 GT calls: little to learn
  V9_no_final_state        WARN   no final_state to check an outcome against

`V5` catches the canonical broken-task family (a task targeting `eqp_101`
when the environment only ever held `eqp_42`), which in the corpus
produced 404s on every turn of every seed while every turn still reported
`passed=True`.

Two deliberate restraints, both learned from false positives on real rows:

*V2 is a WARN, not a BLOCK.* A call this module cannot parse is a limit of
this parser, not proof the task is defective — real ground truth embeds
markdown documents with newlines and apostrophes inside argument strings.
BLOCK is reserved for impossibility we can actually prove.

*V5 only fires when absence is provable.* An id missing from
`initial_state` is not evidence of breakage on its own: tasks legitimately
mint new identifiers (`DecisionLogRecorder(offering_id='OFFERING_001')`
labels a record, it does not look one up). We therefore require the
environment to demonstrably enumerate that id's family — some other id
sharing its prefix must already exist in `initial_state`. `eqp_101` is
blocked because `eqp_42` proves the environment tracks `eqp_*`;
`OFFERING_001` is not, because nothing offering-shaped is there to
contradict it.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

BLOCK = "BLOCK"
WARN = "WARN"

# An "entity id" is a token that looks like a handle the environment must
# already know: alphanumeric with an internal separator and at least one
# digit (eqp_101, prod_12345, HOTEL456, QMAU-001, img_9f3b2c7d). Plain
# words, pure numbers, dates and prose are deliberately excluded — they
# are values, not handles, and flagging them produces noise.
#   separated form: eqp_101, prod_12345, QMAU-001, img_9f3b2c7d
#   compact form:   HB789, SB456 — needs two digits and length >= 5 so that
#                   utf8 / v2 are excluded. Tokens like base64 or sha256 do
#                   slip through, but V5 only acts when the environment
#                   enumerates the same prefix, which such tokens never are.
_ID_RE = re.compile(
    r"\b(?=[A-Za-z0-9_\-]*\d)[A-Za-z][A-Za-z0-9]*(?:[_\-][A-Za-z0-9]+)+\b"
    r"|\b(?=[A-Za-z0-9]*\d[A-Za-z0-9]*\d)[A-Za-z]{1,6}\d[A-Za-z0-9]{2,}\b"
)

# Verbs that consume an entity, for the use-after-delete check. Matched
# anywhere in the tool name so both `DeleteProduct` and `ProductDeleter`
# are recognised.
_DESTRUCTIVE_RE = re.compile(r"(delete|remove|destroy|purge|revoke|cancel)", re.I)

# Leading alphabetic run of an id, used to decide whether the environment
# enumerates that id's family: `eqp_101` -> "eqp", `OFFERING_001` -> "offering".
_PREFIX_RE = re.compile(r"^[A-Za-z]+")

_MIN_ID_LEN = 4


@dataclass
class Finding:
    level: str
    code: str
    where: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class TaskReport:
    task_id: str
    findings: List[Finding] = field(default_factory=list)
    n_gt_calls: int = 0
    n_tools: int = 0

    @property
    def blocked(self) -> bool:
        return any(f.level == BLOCK for f in self.findings)

    @property
    def codes(self) -> List[str]:
        return [f.code for f in self.findings]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "blocked": self.blocked,
            "n_gt_calls": self.n_gt_calls,
            "n_tools": self.n_tools,
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------

def parse_call(call: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """`ToolName(a=1, b='x')` -> ("ToolName", {"a": 1, "b": "x"}).

    Returns None when the string is not a single call expression with
    literal keyword arguments. Non-literal arguments are kept as the
    unparsed source so callers still see the tool name.
    """
    if not isinstance(call, str) or not call.strip():
        return None
    try:
        node = ast.parse(call.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    args: Dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            args[kw.arg] = _unparse(kw.value)
    return node.func.id, args


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - py<3.9 safety net
        return "<unparseable>"


def tool_names(tools: Sequence[Any]) -> Set[str]:
    """Names of the tools a task makes available, tolerating schema shapes."""
    names: Set[str] = set()
    for t in tools or []:
        if isinstance(t, str):
            names.add(t)
        elif isinstance(t, dict):
            for key in ("name", "tool_name", "title"):
                v = t.get(key)
                if isinstance(v, str) and v:
                    names.add(v)
                    break
            fn = t.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                names.add(fn["name"])
    return names


def iter_strings(obj: Any) -> Iterable[str]:
    """Every string anywhere inside a nested structure."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k
            yield from iter_strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from iter_strings(v)


def entity_ids(obj: Any) -> Set[str]:
    """Entity-handle-looking tokens inside a structure."""
    out: Set[str] = set()
    for s in iter_strings(obj):
        for m in _ID_RE.finditer(s):
            tok = m.group(0)
            if len(tok) >= _MIN_ID_LEN:
                out.add(tok)
    return out


def _norm(tok: str) -> str:
    return re.sub(r"[_\-]", "", tok).lower()


def _prefix(tok: str) -> str:
    m = _PREFIX_RE.match(tok)
    return m.group(0).lower() if m else ""


# ---------------------------------------------------------------------------
# the validator
# ---------------------------------------------------------------------------

def validate_task(
    row: Dict[str, Any],
    max_solver_turns: Optional[int] = None,
) -> TaskReport:
    """Validate one release row. Pass `max_solver_turns` to enable V4."""
    task_id = str(row.get("id") or row.get("task_id") or "<unknown>")
    gt_calls = row.get("gt_tool_calls") or []
    tools = row.get("tools") or []
    initial_state = row.get("initial_state")
    final_state = row.get("final_state")
    summary = row.get("summary") or ""

    rep = TaskReport(task_id=task_id, n_gt_calls=len(gt_calls), n_tools=len(tools))
    add = rep.findings.append

    # V1 — nothing to verify against
    if not gt_calls:
        add(Finding(BLOCK, "V1_gt_empty", "gt_tool_calls", "no ground-truth calls"))
        return rep

    known = tool_names(tools)
    parsed: List[Optional[Tuple[str, Dict[str, Any]]]] = []
    for i, call in enumerate(gt_calls):
        p = parse_call(call)
        parsed.append(p)
        if p is None:
            # A parse failure is this parser's limitation (embedded markdown,
            # apostrophes inside strings), not proof the task is broken.
            add(Finding(
                WARN, "V2_gt_unparseable", f"gt_tool_calls[{i}]",
                f"could not parse, arguments unverified: {str(call)[:80]!r}",
            ))
        elif known and p[0] not in known:
            # V3 mirrors task_audit C3_unknown_tool_in_gt, applied per call.
            add(Finding(
                BLOCK, "V3_unknown_tool_in_gt", f"gt_tool_calls[{i}]",
                f"tool {p[0]!r} is not in the task's tool catalogue",
            ))

    # V4 — the turn budget cannot fit the ground truth
    if max_solver_turns is not None and len(gt_calls) > max_solver_turns:
        add(Finding(
            BLOCK, "V4_budget_infeasible", "gt_tool_calls",
            f"{len(gt_calls)} GT calls exceed max_solver_turns={max_solver_turns}",
        ))

    # V5 — entities the ground truth needs that the environment provably
    # lacks. Absence is only provable when the environment enumerates that
    # id's family: some sibling sharing the prefix must already exist.
    # Otherwise the id may simply be one the task mints (a log label, a
    # newly created record), which is legitimate.
    env_tokens = entity_ids(initial_state)
    env_ids = {_norm(t) for t in env_tokens}
    env_prefixes = {_prefix(t) for t in env_tokens if _prefix(t)}
    seen: Set[str] = set()
    for i, p in enumerate(parsed):
        if p is None:
            continue
        call_ids = entity_ids(p[1])
        for tok in sorted(call_ids):
            n = _norm(tok)
            if n in env_ids or n in seen:
                continue
            if _prefix(tok) in env_prefixes:
                # The environment tracks this family and this member is not
                # in it: the canonical broken-task shape. No rollout can
                # conjure it, so the ground truth cannot succeed.
                add(Finding(
                    BLOCK, "V5_entity_not_in_env", f"gt_tool_calls[{i}]",
                    f"{tok!r} is required by the ground truth, and initial_state "
                    f"enumerates {_prefix(tok)!r} entities, but this one is absent",
                ))
            seen.add(n)
        seen.update(_norm(t) for t in call_ids)

    # V6 — ground truth that contradicts itself: destroy X, then use X.
    deleted: Dict[str, int] = {}
    for i, p in enumerate(parsed):
        if p is None:
            continue
        name, args = p
        ids = {_norm(t) for t in entity_ids(args)}
        for tok in sorted(ids):
            if tok in deleted:
                add(Finding(
                    BLOCK, "V6_gt_use_after_delete", f"gt_tool_calls[{i}]",
                    f"{name} operates on an entity destroyed by "
                    f"gt_tool_calls[{deleted[tok]}]",
                ))
                del deleted[tok]  # report once per entity
        if _DESTRUCTIVE_RE.search(name):
            for tok in ids:
                deleted[tok] = i

    # V7 — the summary promises an entity the task never touches.
    gt_ids: Set[str] = set()
    for p in parsed:
        if p is not None:
            gt_ids.update(_norm(t) for t in entity_ids(p[1]))
    if initial_state is not None:
        orphan = sorted(t for t in entity_ids(summary)
                        if _norm(t) not in env_ids and _norm(t) not in gt_ids)
        if orphan:
            add(Finding(
                WARN, "V7_summary_entity_unknown", "summary",
                f"summary names {', '.join(repr(o) for o in orphan[:5])} which the "
                f"environment does not hold and the ground truth never uses",
            ))

    # V8 / V9 — weak training signal, not impossibility.
    if len(gt_calls) < 2:
        add(Finding(WARN, "V8_trivial", "gt_tool_calls",
                    f"only {len(gt_calls)} GT call(s)"))
    if not final_state:
        add(Finding(WARN, "V9_no_final_state", "final_state",
                    "no final_state to verify an outcome against"))

    return rep
