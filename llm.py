"""Unified LLM client for synthtools.

Two backends, same interface:

    # In-process vLLM (default)
    llm = LLM("Qwen3-32B")

    # OpenAI-compatible HTTP server (e.g. `vllm serve`)
    llm = LLM("GPT-OSS-120B", server_url="http://localhost:8765/v1")

    # Messages (single or batched)
    response = llm([{"role": "user", "content": "What is 2 + 2?"}])

    # String (auto-wrapped as user message — for agent compatibility)
    response = llm("What is 2 + 2?")

    # Token tracking
    print(llm.last_usage)  # Usage(prompt_tokens=12, completion_tokens=5)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ModelConfig:
    model_id: str
    temperature: float = 0.2
    top_p: float = 0.95


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "GPT-OSS-20B":     ModelConfig("openai/gpt-oss-20b"),
    "GPT-OSS-120B":    ModelConfig("openai/gpt-oss-120b"),
    "Qwen3-14B":       ModelConfig("Qwen/Qwen3-14B"),
    "Qwen3-32B":       ModelConfig("Qwen/Qwen3-32B"),
    "Qwen3-30B-A3B":   ModelConfig("Qwen/Qwen3-30B-A3B"),
    "Qwen3-235B-A22B": ModelConfig("Qwen/Qwen3-235B-A22B"),
}


def _clean_output(text: str) -> str:
    """Strip reasoning tokens and GPT-OSS control tokens (in-process branch)."""
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[-1]
    if "assistantfinal" in text:
        text = text.split("assistantfinal", 1)[-1]
    return text.strip()


class LLM:
    """Callable LLM wrapper around vLLM (in-process) or an OpenAI-compatible HTTP server.

    Lazily initializes the engine/client on first call. Accepts strings
    (auto-wrapped as user message) or message lists. Tracks token usage via
    self.last_usage after each call.

    Args:
        model: Key from MODEL_REGISTRY.
        max_tokens: Max generation tokens.
        tensor_parallel_size: Number of GPUs (in-process branch only).
        gpu_memory_utilization: Fraction of GPU memory (in-process branch only).
        max_model_len: Maximum sequence length (in-process branch only).
        server_url: OpenAI-compatible HTTP base URL (e.g.
            "http://localhost:8765/v1"). When set, all calls go over HTTP and
            the in-process vLLM engine is never loaded.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 16384,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 32768,
        server_url: Optional[str] = None,
    ):
        if model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY)}")

        self.model = model
        self.cfg = MODEL_REGISTRY[model]
        self.max_tokens = max_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.server_url = server_url
        self._engine = None
        self._client = None
        self.last_usage: Optional[Usage] = None
        self.last_usage_per_request: Optional[List[Usage]] = None

    def __call__(self, messages):
        """Generate responses.

        Accepts:
            str:        "hello"                                    -> str
            messages:   [{"role": "user", "content": "hello"}]     -> str
            batched:    [[{"role": ...}], [{"role": ...}]]         -> list[str]
        """
        # String input: wrap as single user message
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if self.server_url:
            return self._call_via_http(messages)
        return self._call_via_engine(messages)

    def __repr__(self) -> str:
        if self.server_url:
            return f"LLM({self.model!r}, server_url={self.server_url!r})"
        status = "loaded" if self._engine else "not loaded"
        return f"LLM({self.model!r}, {status})"

    # -----------------------------------------------------------------
    # HTTP branch
    # -----------------------------------------------------------------

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.server_url, api_key="EMPTY")
        return self._client

    def _call_via_http(self, messages):
        client = self._ensure_client()
        is_single = not (isinstance(messages, list) and messages and isinstance(messages[0], list))

        if is_single:
            text, usage = self._http_chat(client, messages)
            self.last_usage = usage
            self.last_usage_per_request = None
            return text

        # Batched: parallelize with threads (HTTP releases GIL, so this gives
        # real concurrency). vLLM continuous batching does the actual scheduling.
        def call_one(msgs):
            return self._http_chat(client, msgs)

        with ThreadPoolExecutor(max_workers=min(len(messages), 256)) as pool:
            results = list(pool.map(call_one, messages))
        outputs = [t for t, _ in results]
        per = [u for _, u in results]
        self.last_usage_per_request = per
        self.last_usage = Usage(
            prompt_tokens=sum(u.prompt_tokens for u in per),
            completion_tokens=sum(u.completion_tokens for u in per),
        )
        return outputs

    def _http_chat(self, client, messages):
        """One chat-completion round trip. Returns (text, Usage)."""
        resp = client.chat.completions.create(
            model=self.cfg.model_id,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )
        msg = resp.choices[0].message
        # When the server is launched with --reasoning-parser openai_gptoss,
        # the model's chain-of-thought is in `reasoning_content` and the final
        # answer is in `content` (already stripped of channel markers). We use
        # `content` directly — no further cleaning needed.
        text = msg.content or ""
        u = resp.usage
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
        )
        return text, usage

    # -----------------------------------------------------------------
    # In-process vLLM branch (unchanged)
    # -----------------------------------------------------------------

    def _ensure_engine(self):
        # When configured for HTTP, this is a no-op (kept for API parity with
        # callers that explicitly preload before the main loop).
        if self.server_url:
            return self._ensure_client()
        if self._engine is None:
            from vllm import LLM as vLLM
            self._engine = vLLM(
                model=self.cfg.model_id,
                dtype="bfloat16",
                enforce_eager=False,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
            )
        return self._engine

    def _call_via_engine(self, messages):
        engine = self._ensure_engine()
        sampling = self._sampling_params()
        # Only show vLLM tqdm bars on batched calls — single-call bars pollute logs.
        is_single = not isinstance(messages[0], list)
        results = engine.chat(messages, sampling_params=sampling, use_tqdm=not is_single)
        if is_single:
            r = results[0]
            prompt_toks = len(r.prompt_token_ids) if r.prompt_token_ids else 0
            completion_toks = len(r.outputs[0].token_ids) if r.outputs else 0
            self.last_usage = Usage(prompt_toks, completion_toks)
            self.last_usage_per_request = None
            return _clean_output(r.outputs[0].text)
        per_request: List[Usage] = []
        total_prompt = 0
        total_completion = 0
        outputs = []
        for r in results:
            p = len(r.prompt_token_ids) if r.prompt_token_ids else 0
            c = len(r.outputs[0].token_ids) if r.outputs else 0
            per_request.append(Usage(p, c))
            total_prompt += p
            total_completion += c
            outputs.append(_clean_output(r.outputs[0].text))
        self.last_usage = Usage(total_prompt, total_completion)
        self.last_usage_per_request = per_request
        return outputs

    def _sampling_params(self):
        from vllm import SamplingParams
        return SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )
