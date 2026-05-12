"""Benchmark a running vLLM OpenAI-compatible server.

Sends N concurrent identical chat completion requests for several values of N
and reports per-request latency, throughput, and tokens/sec. Also verifies
that the GPT-OSS reasoning_parser produces a populated `reasoning_content`
field over HTTP — that is the one risk we need to derisk before the
pipeline refactor.

Usage:
    python -m scripts.bench_vllm_serve \
        --base-url http://localhost:8000/v1 \
        --model openai/gpt-oss-120b \
        --concurrency 1,4,16,32 \
        --max-tokens 512
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from openai import AsyncOpenAI


PROMPT_USER = (
    "You are a Task Solver. You will be given a tool to call. Output a JSON object "
    "with two fields: `reason` (a short paragraph of reasoning, 3-4 sentences) and "
    "`tool_call` (a stringified call). Tool: ReturnRequestValidator(return_request_id: str). "
    "Task: validate return request RET002."
)


@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_chars: int = 0
    content_chars: int = 0
    error: str = ""
    sample_reasoning: str = ""
    sample_content: str = ""


@dataclass
class ConcurrencyResult:
    concurrency: int
    wall_s: float
    results: List[RequestResult] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.results if r.ok)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.results if r.ok)

    @property
    def latencies(self) -> List[float]:
        return sorted(r.latency_s for r in self.results if r.ok)

    def percentile(self, p: float) -> float:
        xs = self.latencies
        if not xs:
            return 0.0
        idx = min(len(xs) - 1, int(len(xs) * p))
        return xs[idx]

    @property
    def tokens_per_sec(self) -> float:
        return self.total_completion_tokens / self.wall_s if self.wall_s > 0 else 0.0


async def one_request(
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT_USER}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
        )
        latency = time.perf_counter() - t0
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        usage = resp.usage
        return RequestResult(
            ok=True,
            latency_s=latency,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            reasoning_chars=len(reasoning),
            content_chars=len(content),
            sample_reasoning=reasoning[:300],
            sample_content=content[:300],
        )
    except Exception as e:
        return RequestResult(
            ok=False,
            latency_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )


async def run_concurrency(
    client: AsyncOpenAI,
    model: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
) -> ConcurrencyResult:
    t0 = time.perf_counter()
    coros = [
        one_request(client, model, max_tokens, temperature) for _ in range(concurrency)
    ]
    results = await asyncio.gather(*coros)
    wall = time.perf_counter() - t0
    return ConcurrencyResult(concurrency=concurrency, wall_s=wall, results=list(results))


def fmt_row(r: ConcurrencyResult) -> Dict[str, Any]:
    return {
        "concurrency": r.concurrency,
        "n_ok": r.n_ok,
        "wall_s": round(r.wall_s, 2),
        "p50_latency_s": round(r.percentile(0.5), 2),
        "p90_latency_s": round(r.percentile(0.9), 2),
        "max_latency_s": round(r.percentile(1.0), 2),
        "total_completion_tokens": r.total_completion_tokens,
        "total_prompt_tokens": r.total_prompt_tokens,
        "tokens_per_sec": round(r.tokens_per_sec, 1),
        "speedup_vs_c1": None,
    }


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", default="1,4,16,32",
                   help="comma-separated list of concurrency levels")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--warmup", type=int, default=2,
                   help="warmup requests before measuring")
    p.add_argument("--out", default="",
                   help="optional path to dump JSON results")
    args = p.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    print(f"[bench] base_url={args.base_url} model={args.model}")
    print(f"[bench] concurrency levels: {levels}, max_tokens={args.max_tokens}")
    print(f"[bench] warming up {args.warmup} requests...")
    for _ in range(args.warmup):
        r = await one_request(client, args.model, args.max_tokens, args.temperature)
        if not r.ok:
            print(f"[bench] WARMUP FAILED: {r.error}")
            return 2
    print(f"[bench] warmup done.\n")

    # First-request inspection: confirm reasoning_parser works over HTTP.
    print("=== Single-request inspection (verify reasoning_parser over HTTP) ===")
    sample = await one_request(client, args.model, args.max_tokens, args.temperature)
    print(f"  ok={sample.ok}")
    print(f"  prompt_tokens={sample.prompt_tokens}  completion_tokens={sample.completion_tokens}")
    print(f"  reasoning_chars={sample.reasoning_chars}  content_chars={sample.content_chars}")
    print(f"  reasoning sample: {sample.sample_reasoning!r}")
    print(f"  content sample:   {sample.sample_content!r}")
    if sample.reasoning_chars == 0:
        print("[bench] WARNING: reasoning_content is empty over HTTP — Harmony channels "
              "may not be split. The pipeline relies on this. Investigate before refactoring.")
    print()

    rows: List[ConcurrencyResult] = []
    for c in levels:
        print(f"[bench] running concurrency={c} ...")
        r = await run_concurrency(client, args.model, c, args.max_tokens, args.temperature)
        rows.append(r)
        d = fmt_row(r)
        print(f"  -> ok={r.n_ok}/{c}  wall={d['wall_s']}s  "
              f"p50={d['p50_latency_s']}s  p90={d['p90_latency_s']}s  "
              f"tok/s={d['tokens_per_sec']}")
        # surface first error if any
        for rr in r.results:
            if not rr.ok:
                print(f"  ! err: {rr.error}")
                break

    print("\n=== Summary ===")
    print(f"{'concur':>6} {'ok':>5} {'wall_s':>7} {'p50':>6} {'p90':>6} "
          f"{'max':>6} {'tok/s':>8} {'speedup':>8}")
    base_tok_s = rows[0].tokens_per_sec if rows else 1.0
    for r in rows:
        d = fmt_row(r)
        speedup = (r.tokens_per_sec / base_tok_s) if base_tok_s else 0.0
        print(f"{d['concurrency']:>6} {d['n_ok']:>5} {d['wall_s']:>7.2f} "
              f"{d['p50_latency_s']:>6.2f} {d['p90_latency_s']:>6.2f} "
              f"{d['max_latency_s']:>6.2f} {d['tokens_per_sec']:>8.1f} "
              f"{speedup:>7.2f}x")

    if args.out:
        out = {
            "base_url": args.base_url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "rows": [
                {
                    **fmt_row(r),
                    "speedup_vs_c1": (r.tokens_per_sec / base_tok_s) if base_tok_s else 0.0,
                }
                for r in rows
            ],
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[bench] wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
