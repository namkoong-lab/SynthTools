"""Dataset export: SynthTools rollouts -> trainer-ready rows.

HF-style trajectory dataset (one row per kept SUCCESSFUL trajectory; see README.md):
    row_schema.py         the row contract and message constructors
    tool_calls.py         step 1  Python-syntax call string -> JSON tool call (mirrors the simulator's parser)
    tool_schemas.py       step 2  SynthTools tool definition -> function schema (JSON Schema parameters)
    assemble.py           step 3  rollout -> messages + tools, or a drop reason
    tokens.py             step 4  Qwen3 token counts
    build_hf_dataset.py   step 5  CLI: labels + rollouts -> parquet shards + manifest.json
    verify_hf_dataset.py  step 6  CLI: invariants on every row, recount on a sample
    corpus_checks.py      the differential tests of steps 1-3 against the environment's records

build_multi_turn.py is the older ACEBench-style per-task exporter (tools in the system prompt,
bracketed Python calls); unchanged.
"""
