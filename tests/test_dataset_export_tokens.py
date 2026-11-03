import json
from dataset_export.tokens import assistant_blocks, count_tokens, strip_reasoning


class FakeTok:
    """Qwen-like render, whitespace tokens."""
    def apply_chat_template(self, messages, tools=None, tokenize=False):
        out = ["<|im_start|>system\n" + json.dumps(tools) + "<|im_end|>\n"]
        for m in messages:
            if m["role"] == "assistant":
                body = ("<think>\n" + m["reasoning_content"] + "\n</think>\n\n") if "reasoning_content" in m else ""
                body += m.get("content", "") + "".join("<tool_call>\n" + json.dumps(tc["function"]) + "\n</tool_call>" for tc in m.get("tool_calls", []))
                out.append("<|im_start|>assistant\n" + body + "<|im_end|>\n")
            elif m["role"] == "tool":   # Qwen wraps replies in a user turn with <tool_response>
                out.append("<|im_start|>user\n<tool_response>\n" + m["content"] + "\n</tool_response><|im_end|>\n")
            else:
                out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        return "".join(out)
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


MSGS = [{"role": "user", "content": "q q"},
        {"role": "assistant", "content": "", "reasoning_content": "r r r", "tool_calls": [{"type": "function", "function": {"name": "A", "arguments": {"x": 1}}}]},
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "final answer"}]


def test_assistant_blocks_exclude_header_include_im_end():
    text = FakeTok().apply_chat_template(MSGS, tools=[])
    blocks = assistant_blocks(text)
    assert len(blocks) == 2 and all(b.endswith("<|im_end|>\n") for b in blocks) and "<|im_start|>" not in "".join(blocks)
    assert blocks[0].startswith("<think>\nr r r\n</think>")
    assert assistant_blocks("<|im_start|>user\nq<|im_end|>\n") == []


def test_counts():
    c = count_tokens(FakeTok(), MSGS, [])
    text = FakeTok().apply_chat_template(MSGS, tools=[])
    assert c["n_total_tokens"] == len(text.split())
    assert c["n_target_tokens"] == sum(len(b.split()) for b in assistant_blocks(text))
    assert c["n_target_tokens_nothink"] == c["n_target_tokens"] - len("<think>\nr r r\n</think>".split())
    assert strip_reasoning(MSGS)[1] == {k: v for k, v in MSGS[1].items() if k != "reasoning_content"}
