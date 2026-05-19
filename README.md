# SynthTools

> Reference implementation for the NeurIPS 2026 submission **"SynthTools: A
> Framework for Scaling Synthetic Tools for Agent Development"**. This
> repository is provided as anonymous supplementary material; it contains no
> author identifying information.

SynthTools is a fully LLM-based pipeline for building, validating, and
exercising synthetic tool-use environments at scale. It produces, end-to-end:

- **73,883 validated tools** across **6,800 environments** in **100 fields**
  (top-down hierarchical generation),
- a **Tool Simulator** + **Tool Validator** that emulates each tool's
  responses and filters tools that cannot be reliably emulated,
- **79,925 verifiable tasks** with ground-truth tool-call sequences and
  initial / final environment state (released as `tasks.parquet` in this
  repository),
- a **Trajectory Generator** that runs an LLM agent against the tool simulator
  for any task and verifies the rollout with a trajectory-level judge.

The dataset of verifiable tasks is shipped at `tasks.parquet`. The pipeline
that produced it, and the trajectory generator that consumes it, both live in
this repository.

---

## Pipeline

The codebase has five independent stages. Outputs of stage *N* are inputs to
stage *N + 1*.

| # | Stage | What it does | Entry point |
|---|---|---|---|
| 1 | `env_generation/` | Field → subfields → tasks → tool schemas. Hierarchical, LLM-driven, produces one `env_spec` JSON per scenario. | `python -m env_generation.run` |
| 2 | `env_audit/` | Stress-tests every generated tool (parameter checks, error modes, happy paths), filters tools whose simulator behaviour is unstable, scores reliability. | `python -m env_audit.run` |
| 3a | `task_generation/build_sequences.py` | Filters tools by reliability and asks the LLM to chain them into multi-step sequences per spec. | `python -m task_generation.build_sequences` |
| 3b | `task_generation/run.py` | For each sequence, runs the per-step evolver → solver → simulator → judge → environment update loop and writes one task JSON. | `python -m task_generation.run` |
| 4 | `task_audit/` | Merges each task's successful turns into one cohesive natural-language user request. | `python -m task_audit.summarize` |
| 5 | `trajectory_generation/` | Loads a verifiable task from `tasks.parquet`, rolls out an agent against the tool simulator, and verifies the trajectory with a trajectory-level judge. | `python -m trajectory_generation.run` |

Stages are independent: re-running any stage is safe (atomic writes,
resume-safe pre-filters, idempotent overwrites).

Stages 1–4 *produce* the verifiable-task dataset; stage 5 *consumes* it. If
you want to roll out an agent on the released tasks without rebuilding them,
you only need stage 5 plus the `roles/`, `prompt_templates/`, `llm.py`, and
`utils.py` files.

---

## Repository layout

```
.
├── tasks.parquet              # 79,925 verifiable tasks (released artefact)
├── env_generation/            # stage 1: field → tools
├── env_audit/                 # stage 2: tool simulator + validator
├── task_generation/           # stages 3a/3b: tool sequences → multi-turn tasks
├── task_audit/                # stage 4: merge turns into one user request
├── trajectory_generation/     # stage 5: roll out an agent on a task and judge it
├── roles/                     # LLM role base class + concrete roles
│   ├── env_generator.py       # generates subfields / tasks / tool schemas
│   ├── env_simulator.py       # updates environment state after a tool call
│   ├── tool_simulator.py      # parameter check + response generation
│   ├── judge_simulator.py     # used by env_audit to label simulator responses
│   ├── task_evolver.py        # per-turn task + expected_tool_call generation
│   ├── task_solver.py         # the agent (gen mode + trajectory mode)
│   ├── task_judge.py          # per-turn judge of solver output
│   └── task_summarizer.py     # merges per-turn tasks into one description
├── prompt_templates/
│   ├── env_generator/         # env_generator_{subfield,task,tool,sequences,metadata}_template.yml
│   ├── env_simulator/
│   ├── tool_simulator/        # tool_simulator_{parameter_check,test_calls,simulate}_template.yml
│   ├── judge_simulator/       # validator prompt
│   ├── task_evolver/          # t0 / t1 prompts
│   ├── task_solver/           # gen, eval, and the new multi-turn trajectory prompt
│   ├── task_judge/            # gen + eval prompts
│   ├── task_summarizer/
│   └── trajectory_judge/      # whole-rollout judge prompt
├── scripts/
│   └── build_parquet.py       # summarised tasks → tasks.parquet
├── tests/                     # ~200 pytest tests across all stages
├── llm.py                     # vLLM in-process or HTTP client (uniform API)
├── utils.py                   # atomic JSON writes, batched LLM calls, logging
└── pyproject.toml
```

---

## Quick start

```bash
# 1) Set up the environment
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt    # pyarrow, pyyaml, huggingface_hub, vllm (optional), …

# 2) Run an agent on one verifiable task (in-process vLLM).
#    The first call auto-downloads tasks.parquet from
#    https://huggingface.co/datasets/SynthTools/SynthTools-Tasks if it is
#    not already present at <repo>/tasks.parquet.
python -m trajectory_generation.run \
    --task-id aerospace_and_defense_spec_007_seq11 \
    --output-dir /tmp/traj_smoke \
    --model GPT-OSS-120B \
    --max-solver-turns 12

# 3) Or roll out every task in a field, parallel against a vLLM HTTP server
python -m trajectory_generation.run \
    --field "Investment Banking" --limit 50 \
    --server-url http://localhost:8765/v1 --concurrency 4 \
    --output-dir /tmp/traj_field
```

To regenerate the dataset from scratch, run stages 1 → 4 in order; each stage
README documents its inputs, outputs, and CLI.

---

## Released dataset (`tasks.parquet`)

79,925 rows. One row per *verifiable task*: a single user goal expressed in
natural language, paired with the catalogue of tools an agent can invoke, the
ground-truth tool-call sequence that solves it, and the initial / final
environment state.

| column          | type           | description                                              |
|-----------------|----------------|----------------------------------------------------------|
| `id`            | `string`       | task identifier                                          |
| `field`         | `string`       | application domain (one of 100)                          |
| `summary`       | `string`       | natural-language task description (the user request)     |
| `tools`         | `list[string]` | JSON-encoded tool schemas the agent has access to        |
| `gt_tool_calls` | `list[string]` | ground-truth ordered tool-call sequence                  |
| `initial_state` | `string`       | JSON env state before the first tool call                |
| `final_state`   | `string`       | JSON env state after the ground-truth solution           |

`tools`, `initial_state`, and `final_state` are JSON-serialised; parse with
`json.loads`. Mean ≈ 9.6 tools per task; range 2–15.

---

## Tests

```bash
pytest tests/         # ~200 tests, all pure-Python (no GPU required)
```

The full suite covers env_generation, env_audit, task_generation
(sequences + per-turn loop), task_audit, trajectory_generation (loader +
orchestrator + judge prompt format), atomic writes, LLM batching, and the
HTTP/server-url branch of the LLM client.

---

## Reproducibility

All generation in the released dataset used `gpt-oss-120b` on 4 × NVIDIA A100
(40 GB) GPUs with `temperature=0.2`, `top_p=0.95`. The prompt templates that
drive each stage are checked into `prompt_templates/`; rerunning the pipeline
with the same seed model produces a similarly-shaped corpus.

Token-cost figures per stage (input / output) and end-to-end GPU-hours appear
in the paper appendix.

---

## License

Released for academic use only as supplementary material for an anonymous
peer-reviewed submission. Do not redistribute the dataset or code outside
the review process. A permissive license will be attached at camera-ready
time.
