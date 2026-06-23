"""Regression tests for Python-literal normalization of tool-call strings.

Tool calls are parsed with Python's ast; a bareword `true` parses as a Name
(not a bool) and fails type checks. `normalize_call_literals` canonicalizes
true/false/null -> True/False/None, and task_audit applies it at extraction so
every recompute of gt_tool_calls stays Python-style."""

import json

from utils import normalize_call_literals
from task_audit.summarize import _successful_call_and_response


def test_bareword_booleans_become_python():
    assert normalize_call_literals("Foo(a=true, b=false)") == "Foo(a=True, b=False)"


def test_null_becomes_none():
    assert normalize_call_literals("Foo(x=null)") == "Foo(x=None)"


def test_string_containing_true_is_preserved():
    out = normalize_call_literals("Bar(x='this is true', flag=false)")
    assert out == "Bar(x='this is true', flag=False)"


def test_nested_dict_boolean_normalized_string_preserved():
    out = normalize_call_literals('F(m={"k": false, "s": "false alarm"})')
    # bool value normalized, string value preserved
    assert "False" in out
    assert "false alarm" in out


def test_already_python_unchanged_byte_identical():
    s = "Baz(a=True, b=None, n=5)"
    assert normalize_call_literals(s) == s


def test_no_booleans_unchanged():
    s = "NoBools(name='alice', n=5, xs=[1, 2, 3])"
    assert normalize_call_literals(s) == s


def test_unparseable_returned_unchanged():
    assert normalize_call_literals("Broken(a=") == "Broken(a="


def test_empty_and_none_safe():
    assert normalize_call_literals("") == ""
    assert normalize_call_literals(None) is None


def test_no_residual_barewords_after_normalization():
    import ast
    out = normalize_call_literals(
        "Validate(route=['n1','n2'], m={'edges':[{'stairs':false}]}, ok=true, x=null)"
    )
    tree = ast.parse(out, mode="eval")
    residual = [n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and n.id in ("true", "false", "null")]
    assert residual == []


def test_extraction_chokepoint_normalizes_chat_call():
    """task_audit._successful_call_and_response must return a Python-literal
    call even when the chat emitted lowercase booleans, so recomputes stay
    clean regardless of source."""
    chat = [
        {"role": "user", "content": "do it"},
        {"role": "assistant",
         "content": json.dumps({"reason": "x",
                                 "tool_call": "SetPrefs(route='wf', avoid_stairs=true)"})},
        {"role": "tool", "content": json.dumps({"status_code": 200, "response": {"ok": True}})},
    ]
    call, _ = _successful_call_and_response(chat)
    assert "avoid_stairs=True" in call
    assert "avoid_stairs=true" not in call
