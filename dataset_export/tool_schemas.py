"""SynthTools tool definitions -> OpenAI / Hugging Face function schemas (JSON Schema parameters).

DRAFT for the HF-style trajectory dataset; to be assembled with `tool_calls.py` and `row_schema.py`.

Verified on 2026-11-03: all 77,329 released tools convert to schemas that pass the JSON Schema
2020-12 meta-schema, with no simulator-only key and no `float` type left; 99.72% of the 1,318,032
plain-literal calls the environment accepted validate against the converted schema (the residue is
the simulator's leniency, mostly `null` passed for optional strings), and 84% of the calls it
rejected for type/required reasons fail validation. Object parameters that carry a `required` LIST
and no flag are omitted in about 1% of accepted calls, so they are not marked required.

Facts from the inventory of all 77,329 released tools (2026-11-03):
- a parameter is a dict with `type` (string|integer|float|boolean|array|object, or a list of those
  incl. "null"), `description`, usually a boolean `required` FLAG, and any of: default, enum, format,
  items (dict), properties (dict of sub-schemas), additionalProperties (dict or bool), minimum,
  maximum, minItems, maxItems, minLength, maxLength, pattern, nullable, $ref, minProperties...
- `required` is a LIST in 9,643 schemas (JSON-Schema style: the names of required sub-properties
  of an object). Then there is no flag for the schema itself.
- the tools embedded in rollouts are byte-identical to the released definitions.

Rules: `float` -> `number` wherever a type appears (also inside type lists); a boolean `required`
flag is removed from the schema and recorded in the parent's `required` list; a list `required`
is kept as the object's own `required` list (merged with flagged sub-properties); `items`,
`properties` and `additionalProperties` are converted recursively; every other key passes through
unchanged (`default`, `enum`, `format`, bounds, `pattern`, `nullable`, `$ref`), except the draft-4
boolean `exclusiveMinimum`/`exclusiveMaximum`, rewritten to the numeric form; the simulator-only
tool keys (`error_messages`, `usage`, `output_details`) are never copied.
"""

from __future__ import annotations

from typing import Any, Dict, List

TYPE_MAP = {"float": "number"}


def _map_type(t: Any) -> Any:
    if isinstance(t, str):
        return TYPE_MAP.get(t, t)
    if isinstance(t, list):
        return [TYPE_MAP.get(x, x) if isinstance(x, str) else x for x in t]
    return t


def to_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """One SynthTools parameter (or nested) schema -> JSON Schema. The schema's own boolean
    `required` flag is NOT represented here; the caller reads it."""
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "required" and isinstance(value, bool):
            continue
        if key == "type":
            out["type"] = _map_type(value)
        elif key == "items" and isinstance(value, dict):
            out["items"] = to_json_schema(value)
        elif key == "additionalProperties" and isinstance(value, dict):
            out["additionalProperties"] = to_json_schema(value)
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {n: to_json_schema(s) for n, s in value.items() if isinstance(s, dict)}
            flagged = [n for n, s in value.items() if isinstance(s, dict) and s.get("required") is True]
            listed = schema.get("required") if isinstance(schema.get("required"), list) else []
            required = list(listed) + [n for n in flagged if n not in listed]
            if required:
                out["required"] = required
        elif key == "required":  # a list, already merged above when properties exist
            if "properties" not in schema:
                out["required"] = value
        else:
            out[key] = value
    # draft-4 boolean form of exclusive bounds (1 released tool) -> the numeric form of draft 6+
    for bound, exclusive in (("minimum", "exclusiveMinimum"), ("maximum", "exclusiveMaximum")):
        if isinstance(out.get(exclusive), bool):
            if out[exclusive] and bound in out:
                out[exclusive] = out.pop(bound)
            else:
                out.pop(exclusive)
    return out


def tool_to_function(tool: Dict[str, Any]) -> Dict[str, Any]:
    params = tool.get("parameters") or {}
    properties = {name: to_json_schema(p) for name, p in params.items() if isinstance(p, dict)}
    required = [name for name, p in params.items() if isinstance(p, dict) and p.get("required") is True]
    parameters: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {
            "name": tool["tool_name"],
            "description": tool.get("tool_description") or "",
            "parameters": parameters,
        },
    }


def tools_to_functions(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [tool_to_function(t) for t in tools]
