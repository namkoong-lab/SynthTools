import json
import math
import pytest
from dataset_export.tool_calls import convert_call, OK, SYNTAX_ERROR, NOT_A_CALL


def test_python_literals_become_json():
    c = convert_call("Save(a=1, b='x', c=[1, 2.5, None], d={'k': True, 'n': -2}, e=1e3, f=0x1F)")
    assert c.status == OK and c.name == "Save"
    assert c.arguments == {"a": 1, "b": "x", "c": [1, 2.5, None], "d": {"k": True, "n": -2}, "e": 1000.0, "f": 31}
    assert c.stringified == [] and c.positional_ignored == 0 and c.nonfinite == []
    assert json.dumps(c.tool_call())  # serialisable


def test_json_barewords_are_what_the_environment_saw_strings():
    c = convert_call("Extract(schema={'floors': [{'n': 1, 'stairs': true}], 'x': null}, flag=false)")
    assert c.status == OK
    assert c.arguments == {"schema": "{'floors': [{'n': 1, 'stairs': true}], 'x': null}", "flag": "false"}
    assert c.stringified == ["schema", "flag"]


def test_other_non_literals_are_strings_too():
    c = convert_call("F(d={firstName: 1}, s={{'id': 'x'}}, t=datetime.now(), b=b'ab', z=1j)")
    assert c.status == OK
    assert c.arguments == {"d": "{firstName: 1}", "s": "{{'id': 'x'}}", "t": "datetime.now()", "b": "b'ab'", "z": "1j"}
    assert c.stringified == ["d", "s", "t", "b", "z"]


def test_hashable_set_literal_is_a_string():
    c = convert_call("F(ids={1, 2})")
    assert c.arguments == {"ids": "{1, 2}"} and c.stringified == ["ids"]


def test_tuples_and_non_string_keys_follow_json_dumps():
    c = convert_call("F(t=(1, 2), by_year={2022: 0.9, 2021: 0.8}, dup={'a': 1, 'b': 2, 'a': 3})")
    assert c.arguments == {"t": [1, 2], "by_year": {"2022": 0.9, "2021": 0.8}, "dup": {"a": 3, "b": 2}}
    assert c.stringified == []


def test_positional_and_kwargs_are_ignored_like_the_environment():
    c = convert_call("F('x', 2, k=3, **extra)")
    assert c.status == OK and c.arguments == {"k": 3} and c.positional_ignored == 3


def test_nonfinite_is_flagged():
    c = convert_call("F(x=1e999, y=[1, -1e999])")
    assert c.status == OK and c.nonfinite == ["x", "y"]
    assert math.isinf(c.arguments["x"])


@pytest.mark.parametrize("call,status,name", [
    ("F(x='unterminated)", SYNTAX_ERROR, "F"),
    ("F(x=[1, 2)", SYNTAX_ERROR, "F"),
    ("F(x=1); G(y=2)", SYNTAX_ERROR, "F"),
    ("F(x=1) G(y=2)", SYNTAX_ERROR, "F"),
    ("F(a=1, 2)", SYNTAX_ERROR, "F"),
    ("F(x=007)", SYNTAX_ERROR, "F"),
    ("F(x='a\nb')", SYNTAX_ERROR, "F"),
    ("", SYNTAX_ERROR, None),
    ("just words", SYNTAX_ERROR, None),
    ("F(x=1), G(y=2)", NOT_A_CALL, "F"),
    ("[F(x=1)]", NOT_A_CALL, None),
    ("x.F(y=1)", NOT_A_CALL, None),
    ("42", NOT_A_CALL, None),
])
def test_failures_are_classified_with_a_message(call, status, name):
    c = convert_call(call)
    assert c.status == status and c.name == name and c.arguments is None and c.error


def test_non_string_input():
    assert convert_call(None).status == SYNTAX_ERROR
    c = convert_call({"not": "a string"})   # str() of it is a dict expression, not a call
    assert c.status == NOT_A_CALL and c.arguments is None


def test_unicode_and_nested_json_strings_round_trip():
    c = convert_call("F(s='77.7°C – ok', j='{\"a\": [1, 2]}', e='it\\'s')")
    assert c.arguments == {"s": "77.7°C – ok", "j": '{"a": [1, 2]}', "e": "it's"}
    assert c.stringified == []
