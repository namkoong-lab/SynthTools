# dataset_export

Turns SynthTools rollouts into the HF-style trajectory dataset used for SFT: one row per kept
SUCCESSFUL trajectory, in the Hugging Face `messages + tools` convention that
`tokenizer.apply_chat_template(messages, tools=tools)` consumes directly (JSON tool calls with dict
arguments, `reasoning_content` on every tool-calling turn). `build_multi_turn.py` is the older
ACEBench-style per-task exporter and is unchanged.

## Data flow

```
labels_20261029/trajectories.successful.jsonl  ─┐
labels_20261029/task_content.valid.jsonl       ─┼─ build_hf_dataset ─► <out>/data/train-XXXXX-of-NNNNN.parquet
tool_content_full/trajectories/<id>.json       ─┘                      <out>/manifest.json
   per record:  assemble(traj)  =  utils.extract_json_objects  →  tool_calls.convert_call
                                   →  tool_schemas.tools_to_functions
                →  row_schema.make_row(..., tokens.count_tokens(...))
verify_hf_dataset --out <out>              every row's invariants, token recount on a sample
corpus_checks {calls,schemas,assemble}     the rules re-checked against the raw corpus
```

One file per step: `row_schema.py` (the row contract), `tool_calls.py` (step 1), `tool_schemas.py`
(step 2), `assemble.py` (step 3), `tokens.py` (step 4), `build_hf_dataset.py` (step 5),
`verify_hf_dataset.py` (step 6), `corpus_checks.py` (differential tests of steps 1 to 3).

## The rule

A trajectory is written only if every turn converts: the assistant reply holds a JSON object with a
`tool_call` (read exactly as the orchestrator read it, `utils.extract_json_objects`), the call is a
plain-literal Python call to an offered tool, the tool reply is a JSON object with an int
`status_code`, and the chat ends with a non-empty `<STOP>` reason. Anything else drops the whole
trajectory; nothing is truncated or repaired. Calls the environment rejected for a semantic reason
(4xx/5xx with literal arguments) are kept: those are the recoveries worth learning from.

Drop reasons (`assemble.DROP_REASONS`, counted in `manifest.json`): unreadable reply, reply without
`tool_call` (the orchestrator nudged), malformed call (syntax error, JSON bareword inside a dict,
positional argument), unknown tool, null or status-less reply, no `<STOP>` (turn budget), `<STOP>`
before any call, empty final answer.

## Conversion rules

Calls (`tool_calls.convert_call`): a thin JSON layer over the pipeline's own parser,
`roles.tool_simulator._parse_call`. Python literals become JSON values with `json.dumps` semantics
(tuples → lists, non-string keys → strings, duplicate keys keep the last value). An argument that is
not a Python literal (JSON `true`/`false`/`null` inside it, a bare name, a set literal, a call) is what
the simulator saw as a string, and is reported in `stringified`; positional arguments were ignored by
the simulator and are reported in `positional_ignored`; unparsable strings get `syntax_error` /
`not_a_call` with the parser message. Only `is_plain_literal_call` results enter the dataset. Never
apply `utils.normalize_call_literals` to rollout calls: it hides why 32,890 calls got "Invalid type"
400s.

Schemas (`tool_schemas.tool_to_function`): a boolean `required` flag moves into the parent's
`required` list at every level; a list `required` (JSON-Schema style, 9,643 schemas) is kept as the
object's own list and the parameter is not marked required (the environment accepted calls that
omitted such parameters); `float` → `number` everywhere, including type lists, `items`, `properties`
and `additionalProperties`; the draft-4 boolean `exclusiveMinimum`/`exclusiveMaximum` becomes the
numeric form; everything else passes through; `error_messages`, `usage`, `output_details` are never
copied.

## Row

| column | meaning |
|---|---|
| `id` | `<task_id>.s<seed>`, the rollout file stem (`task_id`, `seed` = split at `.s`) |
| `field` | from `task_content.valid.jsonl` |
| `gt_tool_calls` | JSON list of the task's ground-truth calls, as released |
| `n_target_tokens`, `n_target_tokens_nothink`, `n_total_tokens` | Qwen3 tokens of all assistant blocks (thinking on / off) and of the whole thinking-on render; assistant block = from after `<|im_start|>assistant\n` up to and including `<|im_end|>\n`, the definition behind the EnvScaler / Toucan budget numbers |
| `tools` | JSON string: `[{"type":"function","function":{"name","description","parameters":<JSON Schema>}}]` |
| `messages` | JSON string: user (task summary), then `(assistant tool_calls, tool)` pairs, then the final answer |

`tools` and `messages` are JSON strings because their nested keys differ per row.

```python
import json, pyarrow.parquet as pq
from dataset_export.row_schema import strip_reasoning
table = pq.read_table("<out>/data")          # or datasets.load_dataset("parquet", data_dir="<out>/data")
row = table.slice(0, 1).to_pylist()[0]
messages, tools = json.loads(row["messages"]), json.loads(row["tools"])
text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
text_nothink = tokenizer.apply_chat_template(strip_reasoning(messages), tools=tools, tokenize=False)
```

## Commands

Run on a CPU node (researchgpu02) with the SynthTools venv; the tokenizer loads offline with
`HF_HOME=/scratch/tc3527/.cache/huggingface HF_HUB_OFFLINE=1`.

```bash
L=/shared/share_mala/tc3527/synthtools/labels_20261029
T=/shared/share_mala/tc3527/synthtools/tool_content_full/trajectories
python -m dataset_export.build_hf_dataset  --labels-dir $L --trajectories-dir $T --out <out> --workers 24
python -m dataset_export.verify_hf_dataset --out <out>
python -m dataset_export.corpus_checks calls    --labels-dir $L --trajectories-dir $T --unsuccessful --report-dir <dir>
python -m dataset_export.corpus_checks schemas  --labels-dir $L --trajectories-dir $T --tools-dataset $T/../tools_dataset.jsonl
python -m dataset_export.corpus_checks assemble --labels-dir $L --trajectories-dir $T
pytest tests/test_dataset_export_*.py
```

## Verified on 2026-11-03

- `calls`: 2,701,725 calls of all 348,167 trajectories of the VALID tasks; the converter's parse status
  agreed with the environment's reply in every case (10,426 syntax errors answered with the same Python
  parser message; 14 strings the orchestrator rejected as "Unknown tool name" before parsing and 1 null
  payload are indeterminate); call strings and tool names equal the recorded turns in 100%; all 42,563
  ground-truth calls of the VALID tasks are plain literals.
- `schemas`: all 77,329 released tools convert to schemas that pass the 2020-12 meta-schema; 99.72% of
  the 1,318,032 plain-literal calls the environment accepted validate against the converted schema
  (the residue is the simulator's leniency, mostly `null` for an optional string).
- `assemble`: 167,586 of the 210,925 SUCCESSFUL trajectories kept; 43,339 dropped (23,612 unreadable
  replies, 9,738 without `tool_call`, 9,078 malformed calls, 806 without `<STOP>`, 105 other); zero
  invariant violations; 14,873 kept rows contain a recovered semantic failure.

The dataset lives in `/shared/share_mala/tc3527/synthtools/hf_dataset/synthtools_trajectories/`.
