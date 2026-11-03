"""Token counts of one row, in the definition shared with the EnvScaler / Toucan budget numbers:
every assistant block of the Qwen3 render, from after "<|im_start|>assistant\\n" up to and including
"<|im_end|>\\n" (think wrappers count, the role header does not), thinking on and off, plus the whole
thinking-on render."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from dataset_export.row_schema import strip_reasoning

ASSISTANT_HEADER = "<|im_start|>assistant\n"
IM_END = "<|im_end|>\n"


def load_tokenizer(name: str) -> Any:
    """The one place a tokenizer is loaded (tests replace this function)."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def assistant_blocks(rendered: str) -> List[str]:
    blocks = []
    for part in rendered.split(ASSISTANT_HEADER)[1:]:
        end = part.find(IM_END)
        blocks.append(part if end < 0 else part[: end + len(IM_END)])
    return blocks


def count_tokens(tok: Any, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, int]:
    def n(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    with_think = tok.apply_chat_template(messages, tools=tools, tokenize=False)
    without = tok.apply_chat_template(strip_reasoning(messages), tools=tools, tokenize=False)
    return {
        "n_target_tokens": sum(n(b) for b in assistant_blocks(with_think)),
        "n_target_tokens_nothink": sum(n(b) for b in assistant_blocks(without)),
        "n_total_tokens": n(with_think),
    }
