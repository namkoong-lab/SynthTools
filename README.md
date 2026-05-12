# synthtools_env

Synthetic tool-use training-data pipeline. Given a list of fields (e.g. *"Aerospace and Defense"*, *"Investment Banking"*), produces multi-turn agentic trajectories where an LLM-driven solver picks one tool per turn, calls it, gets a simulated response, and proceeds — with reliability auditing, judge-verified arguments, and atomic resume-safe writes.

End output (per trajectory):
1. A JSON file with the full chat, per-turn reasoning, judge verdicts, environment state.
2. A sibling `.debug.json` with every LLM call's prompt + response + token usage.
3. A grounded one-task-description summary merged across the successful turns.

---

## Pipeline (4 stages)

Each stage owns its own directory and README. Outputs of stage *N* are inputs to stage *N+1*.

| # | Stage | What it does | Entry point |
|---|---|---|---|
| 1 | [env_generation](env_generation/README.md) | Field → subfields → tasks → tool schemas (per-scenario `env_spec` JSONs). | `python -m env_generation.run` |
| 2 | [env_audit](env_audit/README.md) | Builds `tools_dataset.jsonl`; scores each tool's **reliability** by simulating fm1/fm2/fm3 test calls; folds an `audit` block into each env_spec. | `python -m env_audit.run` |
| 3a | [traj_generation/build_sequences](traj_generation/build_sequences.py) | Filters by reliability and asks the LLM to chain *k* sequences of *N* tools per spec. Writes `spec["sequences"]`. | `python -m traj_generation.build_sequences` |
| 3b | [traj_generation/run](traj_generation/README.md) | For each sequence, runs the evolver → solver → simulator → judge → env-update loop and writes one trajectory JSON. **Parallelizable across N workers against a shared vLLM server.** | `python -m traj_generation.run` or `bash scripts/run_traj_gen.sh` |
| 4 | [traj_audit](traj_audit/README.md) | Merges each trajectory's successful turns into one grounded task description (`summary` field). Batched + idempotent. | `python -m traj_audit.summarize` |

Stages are independent: re-running any stage is safe (atomic writes, resume-safe pre-filters, idempotent overwrites).

---

## Quick start (Perlmutter)

Inside an active `salloc` (1 node, 4 GPUs):

```bash
cd /global/homes/t/tcaste/projects/burn-gpu/synthtools_env
source /pscratch/sd/t/tcaste/envs/burn-gpu/bin/activate

# Stages 1 + 2 + 3a, end-to-end for a field:
python -m env_generation.run --fields "Investment Banking" --output-dir tool_content/env_specs --model GPT-OSS-120B
python -m env_audit.run --env-specs-dir tool_content/env_specs --dataset-path tool_content/tools_dataset.jsonl --eval-logs-dir tool_content/tool_eval_logs --model GPT-OSS-120B
python -m traj_generation.build_sequences --env-specs-dir tool_content/env_specs --field "Investment Banking" --min-reliability 0.5 --model GPT-OSS-120B

# Stage 3b — parallel trajectory generation against a vLLM server:
bash scripts/run_traj_gen.sh --field "Investment Banking" --concurrency 5

# Stage 4 — summarize for training:
python -m traj_audit.summarize --trajectories-dir tool_content/trajectories --field "Investment Banking" --env-specs-dir tool_content/env_specs --model GPT-OSS-120B
```

---

## Trajectory generation (stage 3b) — two execution modes

### Single-process (in-process vLLM, the original mode)
Direct invocation of `traj_generation.run` loads vLLM in-process and runs trajectories serially. Use this for development and tests.

```bash
python -m traj_generation.run \
    --dataset tool_content/tools_dataset.jsonl \
    --output-dir tool_content/trajectories \
    --model GPT-OSS-120B \
    --field "Investment Banking" \
    --env-specs-dir tool_content/env_specs \
    --verifiable
```

### Multi-worker (vLLM HTTP server + ProcessPoolExecutor)
Production mode: one `vllm serve` process per node, N worker processes generating trajectories in parallel against the shared server. Roughly 3–5× per-node throughput on long prompts.

The wrapper script handles server lifecycle (launch, readiness poll, cleanup-on-exit):
```bash
bash scripts/run_traj_gen.sh --field "Investment Banking" --concurrency 5
# add --gpu-memory-utilization 0.95 for more KV-cache headroom
# bash scripts/run_traj_gen.sh --help  for every flag
```

Or invoke run.py directly against an already-running server:
```bash
python -m traj_generation.run \
    --dataset ... --output-dir ... --model GPT-OSS-120B \
    --field "Investment Banking" --env-specs-dir ... \
    --server-url http://localhost:8765/v1 \
    --concurrency 5 \
    --verifiable
```

**Safety properties** (multi-worker mode):
- Each `(spec, seq_key)` enters the work queue exactly once — no double-write races.
- Outputs are written atomically (`.tmp` + `os.replace`) — a killed worker leaves at most a stray `.tmp` file, never a half-written `.json`.
- Resume-safe: a re-run skips trajectories whose final `.json` already exists; `.tmp` files are ignored.
- Per-worker logs at `<output-dir>/_logs/<task_id>.log` so 5 workers' detail doesn't interleave on stderr.

See [scripts/run_traj_gen.sh](scripts/run_traj_gen.sh) and [traj_generation/README.md](traj_generation/README.md) for details.

---

## Setup

Tested on Perlmutter (NERSC) — Linux, 4× hbm40g GPUs per node, GPT-OSS-120B with TP=4.

Required Python packages: `vllm` (>=0.11), `openai` (>=2.7), `pyyaml`. The Perlmutter venv at `/pscratch/sd/t/tcaste/envs/burn-gpu/bin/activate` has them all.

For other clusters: every path in `scripts/run_traj_gen.sh` is overridable via flags or env vars (see `--help`). Just point `--venv`, `--dataset`, `--env-specs-dir`, and `--output-dir` at your equivalents.

---

## Tests

```bash
pytest tests/ -q
```

164+ unit tests covering every stage + the LLM abstraction + atomic writes + concurrency orchestration. All use a `FakeLLM` from [tests/conftest.py](tests/conftest.py); no GPU needed.

| Test file | Coverage |
|---|---|
| `test_env_generation.py` | Stage 1 end-to-end |
| `test_env_audit.py` + `test_env_audit_modes.py` | Stage 2 phases + filters |
| `test_build_sequences.py` | Stage 3a reliability filter, skip-if-present |
| `test_traj_regression.py` | Stage 3b serial path (12 tests) |
| `test_traj_concurrency.py` | Stage 3b parallel orchestration: work-queue, CLI validation, per-worker log redirect |
| `test_traj_audit_summarize.py` | Stage 4 |
| `test_llm.py`, `test_llm_batching.py`, `test_llm_server_url.py` | LLM in-process + HTTP branches |
| `test_atomic_writes.py` | `write_json_atomic` + crash safety |
| `test_task_judge.py`, `test_prompts_format.py`, `test_utils.py` | Shared role + util coverage |

A live-vLLM `@pytest.mark.integration` test exists for the HTTP path; run with `pytest -m integration` once a server is up.

---

## Repo layout

```
.
├── env_generation/          stage 1
├── env_audit/               stage 2
├── traj_generation/         stages 3a + 3b  (build_sequences + run)
├── traj_audit/              stage 4
│
├── roles/                   shared agent classes (TaskEvolver, TaskSolver, ToolSimulator,
│                             EnvironmentSimulator, TaskJudge, TaskSummarizer,
│                             EnvironmentGenerator, JudgeSimulator)
├── prompt_templates/        YAML prompts grouped by role
│
├── llm.py                   unified LLM client — in-process vLLM OR OpenAI-compatible HTTP
│                             (toggle via the `server_url=` kwarg)
├── utils.py                 logger, atomic writes, usage tracking, batched LLM helper,
│                             debug event log, message rewriting
│
├── scripts/
│   ├── run_traj_gen.sh        ← parallel trajectory generation: launches vllm serve, runs
│   │                             stage 3b, cleans up. Works on any cluster (every path
│   │                             overridable). Use this for production runs.
│   ├── bench_vllm_serve.sh    throughput benchmark client (concurrency sweep)
│   ├── bench_vllm_serve.py    Python client used by the bench script
│   ├── audit_*.py             stage-2 / corpus inspection helpers
│   ├── drop_bad_trajectories.py     post-hoc trajectory cleanup
│   └── judge_trajectories.py        LLM-judge audit cross-check
│
├── tool_content/            local sample data for development / tests (env_specs,
│                             tools_dataset.jsonl, trajectories). Production data lives
│                             on /pscratch.
└── tests/                   pytest suite + FakeLLM fixtures
```

---

## Schema versions

| Version | Where | Owner stage |
|---|---|---|
| `env_spec.v1` | scenario JSONs after generation | env_generation |
| `env_spec.v2` | env_spec after first audit pass | env_audit |
| `env_field_gen.v1` | field log | env_generation |
| `env_manifest.v1` | each `manifest.jsonl` row | env_generation |
| `tools_dataset.v1` | each `tools_dataset.jsonl` row | env_audit |
| `env_audit_log.v1` | per-tool eval logs | env_audit |
| `audit.v1` | the `audit` block on env_specs | env_audit |

Bump the version whenever the shape changes. Downstream stages should validate the version they're reading.

---

## Models

`MODEL_REGISTRY` in [llm.py](llm.py):

| Key | HF id |
|---|---|
| `GPT-OSS-20B` | `openai/gpt-oss-20b` |
| `GPT-OSS-120B` | `openai/gpt-oss-120b` |
| `Qwen3-14B` | `Qwen/Qwen3-14B` |
| `Qwen3-32B` | `Qwen/Qwen3-32B` |
| `Qwen3-30B-A3B` | `Qwen/Qwen3-30B-A3B` |
| `Qwen3-235B-A22B` | `Qwen/Qwen3-235B-A22B` |

Pipeline default in tests is `Qwen3-32B`; production runs use `GPT-OSS-120B` (TP=4, max_model_len=32768, with `--reasoning-parser openai_gptoss` on the server).
