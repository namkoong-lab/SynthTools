"""Tool-call strings (Python call syntax) -> JSON tool calls, the way the environment read them.

DRAFT for the HF-style trajectory dataset; to be assembled with `row_schema.py`.

The rollout environment parsed every call with `roles.tool_simulator._parse_call`: `ast.parse`,
then each keyword argument resolved with `ast.literal_eval`, falling back to the argument's SOURCE
TEXT (a string) when it is not a Python literal (JSON barewords `true`/`false`/`null`, bare names,
set literals, calls). Positional arguments and `**kwargs` were ignored. The parameter-check LLM then
saw `json.dumps(parsed, default=str)`. `convert_call` returns exactly that view as JSON values plus
a classification, for every input string.

Verified on 2026-11-03 against the environment's recorded replies for all 2,701,725 calls of the
348,167 trajectories of VALID tasks: the status agrees with the environment in every case (10,426
syntax errors answered with the same Python parser message), the call string equals the recorded
`turns[k].tool_call` and the tool name matches in 100% of calls, no nan/inf occurs.

Dataset rule (decided 2026-11-03): a trajectory is kept only if every call is a plain-literal call,
i.e. `is_plain_literal_call(convert_call(s))`. Calls the environment rejected for a SEMANTIC reason
(4xx/5xx with literal arguments) are kept: they are the recoveries worth learning from.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from roles.tool_simulator import _parse_call  # the environment's own parser

OK, SYNTAX_ERROR, NOT_A_CALL = "ok", "syntax_error", "not_a_call"
_LEADING_NAME = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass
class Converted:
    status: str                          # OK | SYNTAX_ERROR | NOT_A_CALL
    name: Optional[str]                  # tool name (best effort for failures)
    arguments: Optional[Dict[str, Any]]  # JSON values; None unless status == OK
    stringified: List[str] = field(default_factory=list)  # args the environment saw as a string
    positional_ignored: int = 0          # positional / **kwargs the environment dropped
    nonfinite: List[str] = field(default_factory=list)    # args holding nan/inf (not strict JSON)
    error: Optional[str] = None          # parser message for failures
    raw: str = ""

    def tool_call(self) -> Dict[str, Any]:
        """The HF / Qwen `function` payload. Only meaningful when status == OK."""
        return {"name": self.name, "arguments": self.arguments}


def is_plain_literal_call(c: Converted) -> bool:
    """The dataset rule: parsed, every argument a Python literal, nothing dropped, strict JSON."""
    return c.status == OK and not c.stringified and not c.positional_ignored and not c.nonfinite


def _json_view(value: Any) -> Any:
    """What the simulator's prompt contained for this value (json.dumps with default=str)."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _is_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return value != value or value in (float("inf"), float("-inf"))
    if isinstance(value, list):
        return any(_is_nonfinite(v) for v in value)
    if isinstance(value, dict):
        return any(_is_nonfinite(v) for v in value.values())
    return False


def convert_call(call: Any) -> Converted:
    raw = call if isinstance(call, str) else ("" if call is None else str(call))
    parsed = _parse_call(raw.strip())
    if parsed["parse_status"] != "ok":
        message = parsed.get("error_message") or "syntax error"
        status = NOT_A_CALL if message.startswith("expected a single") else SYNTAX_ERROR
        m = _LEADING_NAME.match(raw)
        return Converted(status, m.group(1) if m else None, None, error=message, raw=raw)

    node = ast.parse(raw.strip(), mode="eval").body  # succeeded inside _parse_call
    stringified: List[str] = []
    for kw in node.keywords:
        if kw.arg is None:
            continue
        try:
            ast.literal_eval(kw.value)
            is_literal = True
        except (ValueError, SyntaxError, TypeError):
            is_literal = False
        value = parsed["parsed_call"]["arguments"][kw.arg]
        if not is_literal or (isinstance(_json_view(value), str) and not isinstance(value, str)):
            stringified.append(kw.arg)
    positional = len(node.args) + sum(1 for kw in node.keywords if kw.arg is None)
    arguments = _json_view(parsed["parsed_call"]["arguments"])
    nonfinite = [k for k, v in arguments.items() if _is_nonfinite(v)]
    return Converted(OK, parsed["parsed_call"]["name"], arguments, stringified, positional, nonfinite, raw=raw)
