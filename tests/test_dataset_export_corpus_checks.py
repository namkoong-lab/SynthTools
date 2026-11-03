import json
from dataset_export.corpus_checks import is_syntax_reply, iter_calls, reply_of


def test_is_syntax_reply_distinguishes_parser_errors_from_semantic_messages():
    assert is_syntax_reply("Syntax error: unterminated string literal (detected at line 1)", "unterminated string literal (detected at line 1)")
    assert is_syntax_reply("Parse error: invalid syntax", None)
    assert is_syntax_reply("syntax_error: '(' was never closed", None)
    assert is_syntax_reply("Offending issue: closing parenthesis ')' does not match opening parenthesis '{'", None)
    assert not is_syntax_reply("Syntax error in rule_expression: Ensure it is a valid boolean expression.", None)
    assert not is_syntax_reply("Invalid criteria syntax: Ensure each expression follows field=value.", None)
    assert not is_syntax_reply("", None)


def test_iter_calls_and_reply_of():
    chat = [{"role": "user", "content": "t"},
            {"role": "assistant", "content": json.dumps({"reason": "r", "tool_call": "A(x=1)"})},
            {"role": "tool", "content": json.dumps({"status_code": 400, "error_message": "Missing required parameter: y."})},
            {"role": "assistant", "content": "no envelope"},
            {"role": "user", "content": "No tool call detected..."},
            {"role": "assistant", "content": json.dumps({"reason": "r", "tool_call": "B(y=2)"})},
            {"role": "tool", "content": json.dumps({"status_code": 400, "response": {"error": "Unknown tool name in call. Choose one of: ['A']"}})},
            {"role": "assistant", "content": json.dumps({"reason": "done", "tool_call": "<STOP>"})}]
    assert list(iter_calls(chat)) == [(1, "A(x=1)"), (5, "B(y=2)")]
    assert reply_of(chat, 1) == (400, "Missing required parameter: y.", "fail_400")
    assert reply_of(chat, 5) == (400, "", "unknown_tool")
    assert reply_of(chat, 7) == (None, "", "no_reply")
