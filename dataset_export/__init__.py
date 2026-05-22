"""Dataset export — turn SynthTools task JSONs into trainer-ready rows.

Currently supports the **multi_turn** dataset shape (`build_multi_turn`):
one row per task, all successful turns concatenated into a single
`messages` array with failed retries kept. Tools are embedded in the
system message (ACEBench-style) and assistant tool calls are emitted as
`[ToolName(key='value', ...)]` Python-parseable bracketed strings.
"""
