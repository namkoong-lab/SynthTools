# fix_traj: trajectory generation for training-grade data

## Goal

`trajectory_generation/` rolls a solver agent against the tool simulator on
each released task in `task_content.jsonl` and emits a judged transcript.
That transcript is what downstream training (SFT, mid-training, RL, RM)
consumes. Each trajectory must clear four bars:

1. **Faithful.** Simulated tool responses match the recorded GT outputs.
2. **Verifiable.** A deterministic judge separates solved from
   solved-by-luck from unsolved.
3. **Solvable.** The released summary carries every user-chosen value the
   GT chain depended on.
4. **Reproducible.** Re-rolling the same task with the same config gives
   the same judged outcome.

Exit criterion: every released task either produces at least one
trajectory meeting all four bars, or is deterministically flagged out
with a recorded reason.

## Issues

### Issue 1: simulator non-determinism (faithful bar)

Same `(tool_data, tool_call, metadata)` returns different responses across
runs. The universe in `env_metadata` is fixed; what varies is which
subset is returned, in what order, and in what wording.

- `roles/tool_simulator.py:61-118` (`simulate()` runs param_check then
  simulate_raw, no determinism control)
- `llm.py:34-48` (`ModelConfig.temperature = 0.2`, no per-role override)
- `prompt_templates/tool_simulator/tool_simulator_simulate_template.yml:48`
  (caps lists at "3 representative items, never more than 5", no
  selection or ordering rule)
- `trajectory_generation/orchestrator.py:180` (calls
  `simulator.simulate(...)` without recorded GT examples)

Failure modes the design admits: selection (which 3 of N), ordering
(same set, different order), format (casing, spacing), quantity (2 vs 4).

Canonical case: `academic_publishing_and_citations_spec_000_seq1`,
`solved=False`, `match_rate=0.6`, `final_state_match=False`.

Open: actual temperature in effect at rollout; whether
`task.initial_state.env_metadata` reliably carries the full GT universe;
no vLLM `seed` is set.

### Issue 2: summarizer over-hides user-supplied criteria (solvable bar)

The single-chronological-walk algorithm buckets each value on first sight
and never moves it. When a tool response mentions value V before the
user message that picks V as a filter or target, V lands in
`tool_produced_values` and is dropped from `task_summarized`. The
rollout agent has no way to recover V.

- `prompt_templates/task_summarizer/task_summarizer_template.yml:35-76`
  (the walk + the never-moves invariant)
- `prompt_templates/task_summarizer/task_summarizer_template.yml:78-84`
  (the CRITERION EXCEPTION, a judgment-call override added on top of an
  otherwise deterministic walk; whether it fires is up to the LLM)
- `task_audit/summarize.py:136-153` (chat triples assembled, no
  pre-classification)
- `task_audit/summarize.py:620-639` (`audit_summary_output` checks
  structure only, not the user-vs-tool split)

Canonical case: `academic_publishing_and_citations_spec_000_seq11`.
"Journal of Medical Informatics" is the user's chosen filter but is
bucketed as tool_produced because `tool[0]` lists an article from that
journal before `user[1]` asks to filter by it.

Open: rate at which CRITERION EXCEPTION fails to fire; how many released
rows are affected (`seq11` is one); whether the pattern is limited to
filter criteria or also affects mappings, thresholds, target names.

## Experiments

Every experiment goes here in reverse chronological order. One entry per
run. Format:

### YYYY-MM-DD `<short name>`
- Setup: command, fixture, what we changed.
- Result: numbers and what they imply.
- Decision: keep / drop / iterate.

### 2026-06-22 `end-to-end trajectory test on summarizer iter 9 (single-pass, general rules)`

- Setup: iter 9 prompt at
  `prompt_templates/task_summarizer/task_summarizer_template.yml`:
  general R1-R4 structural rules (counts, returned identifiers,
  status flags always tool_produced; user-chosen values are
  user_supplied), abstract Lookup/ApplyTag worked example, no
  domain enumerations, TOOL SCHEMAS block injected via
  `_triples_for_summarizer(spec=...)` and
  `TaskSummarizer.build_messages(tools=...)`. SINGLE PASS only (no
  critique pass). Ran `dev/try_trajectory.py` on the 3 canonical
  academic tasks (summarize, rollout, judge end-to-end).
- Result: all 3 trajectories ended `solved=False` but for
  HETEROGENEOUS reasons, only some attributable to the summarizer:

  | task | match_rate | dominant failure causes |
  | --- | ---: | --- |
  | seq11 | 0.70 | simulator non-determinism (returned 3 articles vs GT 2), solver skipped Deduplicate, summarizer paraphrased query string ("machine learning healthcare" vs GT "machine learning in healthcare") |
  | seq1 | 0.27 | same mix |
  | seq10 | 0.50 | same mix |

- Reinterpretation. The static heuristic I had been iterating
  against ("any tool_produced value verbatim in task_summarized =
  leak") was the wrong measurement. End-to-end trajectory testing
  shows the summarizer does the bulk of the job correctly; the
  remaining summarizer-attributable gap is LOSSY PARAPHRASE of
  user-picked strings (the "no parameter syntax / no
  enumeration" rule sometimes drops function words like "in" from
  a query, or drops a `=true` flag). The other failure modes
  (simulator non-determinism, solver step-skipping) are outside
  the summarizer's scope and account for the majority of the
  match_rate loss.
- Decision: keep iter 9 as the shipped summarizer prompt. It is
  general (no domain-specific enumeration), short (~6.3k chars),
  and passes the actual end-to-end bar for the parts it owns.
  Next push: address simulator non-determinism (Issue 1, replay
  path is already half-wired) before chasing residual paraphrase
  failures, since simulator drift is the larger cause of low
  match_rate.

### 2026-06-22 `two-pass summarizer (single-pass iteration + reclassifying critique)` (superseded)

- Setup: 9 single-pass prompt iterations on
  `prompt_templates/task_summarizer/task_summarizer_template.yml`,
  followed by a code change adding a second LLM call. The first call
  produces the 4-field JSON as before; the second call (a "critique"
  pass) receives the JSON plus the chat, reclassifies misbucketed
  values (returned identifiers, counts, status flags moved from
  `user_supplied_values` to `tool_produced_values` + `spillover`), then
  scrubs `task_summarized` of any tool_produced value referenced
  verbatim. Wired via `TaskSummarizer.critique` in
  `roles/task_summarizer.py` and a new `critique_template` field in
  the YAML. `task_audit/summarize.summarize_tasks` and the
  `dev/summarize_batch.py` harness both route through the critique.
  Tests pass.
- Result, scored on the same 3 canonical academic tasks and the same
  10 random tasks (`--n 10 --seed 0`):

  | metric | iter 9 (single-pass) | iter B2 (two-pass) |
  | --- | --- | --- |
  | canonical (3 tasks) clean | 0/3 | 2/3 |
  | random (10 tasks) clean | 6/10 | 7/10 |

  Clean = no `tool_produced` value verbatim in `task_summarized`
  (word-boundary match, excluding values also in `user_supplied`).
  Canonical wins: seq10 and seq11 now indirect ("the matching
  record", "the article with the matching DOI"); seq1 still leaks the
  DOI (the critique pass missed the reclassification on that task,
  LLM variance). Random wins: procurement's `exported`/`merged`/
  `created` status flags now rewritten indirectly. Residual leaks on
  random: counts (`1`) and a few status words (`recorded`).
- Decision: keep. Two-pass is materially better than single-pass and
  the cost is one extra LLM call per task. The residual ~20-25%
  failure is LLM noise on the critique pass; a third pass or a
  deterministic substance-match validator could close the gap
  further, but B2 is the baseline going forward.
