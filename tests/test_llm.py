"""Quick smoke test for the LLM client.

Run with GPUs available:
    python test_llm.py --model Qwen3-32B
    python test_llm.py --model GPT-OSS-120B
"""

import argparse
from llm import LLM, Usage, MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3-32B", choices=list(MODEL_REGISTRY))
    args = parser.parse_args()

    print(f"--- Initializing LLM({args.model!r}) ---")
    llm = LLM(args.model)
    print(repr(llm))

    # Test 1: single conversation (messages)
    print("\n--- Test 1: single conversation ---")
    response = llm([{"role": "user", "content": "What is 2 + 2? Answer in one word."}])
    print(f"Output: {response}")
    print(f"Usage: {llm.last_usage}")
    assert isinstance(response, str), f"Expected str, got {type(response)}"
    assert isinstance(llm.last_usage, Usage), f"Expected Usage, got {type(llm.last_usage)}"
    assert llm.last_usage.prompt_tokens > 0, "Expected prompt_tokens > 0"
    assert llm.last_usage.completion_tokens > 0, "Expected completion_tokens > 0"

    # Test 2: string input (agent compatibility)
    print("\n--- Test 2: string input ---")
    response = llm("What is 3 + 3? Answer in one word.")
    print(f"Output: {response}")
    print(f"Usage: {llm.last_usage}")
    assert isinstance(response, str), f"Expected str, got {type(response)}"
    assert llm.last_usage.prompt_tokens > 0, "Expected prompt_tokens > 0"

    # Test 3: multi-turn
    print("\n--- Test 3: multi-turn ---")
    response = llm([
        {"role": "user", "content": "Remember the number 42."},
        {"role": "assistant", "content": "Got it, I'll remember 42."},
        {"role": "user", "content": "What number did I ask you to remember?"},
    ])
    print(f"Output: {response}")
    print(f"Usage: {llm.last_usage}")
    assert isinstance(response, str), f"Expected str, got {type(response)}"

    # Test 4: batched conversations
    print("\n--- Test 4: batched (3 conversations) ---")
    responses = llm([
        [{"role": "user", "content": "Capital of France? One word."}],
        [{"role": "user", "content": "Capital of Japan? One word."}],
        [{"role": "user", "content": "Capital of Brazil? One word."}],
    ])
    for i, r in enumerate(responses):
        print(f"  [{i}] {r}")
    print(f"Usage (total): {llm.last_usage}")
    assert isinstance(responses, list), f"Expected list, got {type(responses)}"
    assert len(responses) == 3, f"Expected 3 responses, got {len(responses)}"
    assert llm.last_usage.prompt_tokens > 0, "Expected prompt_tokens > 0"

    print("\n--- All tests passed ---")


if __name__ == "__main__":
    main()
