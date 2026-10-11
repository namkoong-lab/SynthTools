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
- `llm.py:35-37` (`ModelConfig.temperature = 0.2`, no per-role override)
- `prompt_templates/tool_simulator/tool_simulator_simulate_template.yml:50`
  (caps lists at "3 representative items, never more than 5", no
  selection or ordering rule)
- `trajectory_generation/orchestrator.py:201` (calls
  `simulator.simulate(...)` without recorded GT examples — the
  `replay_examples` parameter exists at `tool_simulator.py:61-66,97` and
  the template honours it at `simulate_template.yml:80-84`, but the only
  caller that wires it is `dev/try_trajectory.py --replay`. The Faithful
  mechanism is built and dormant.)

Failure modes the design admits: selection (which 3 of N), ordering
(same set, different order), format (casing, spacing), quantity (2 vs 4).

Canonical case: `academic_publishing_and_citations_spec_000_seq1`,
`solved=False`, `match_rate=0.6`, `final_state_match=False`.

**Two further simulator defects measured 2026-07-28, both task-breaking:**

- **Clock drift.** The simulator's "now" reads ~`2026-06-23`, so tasks with
  hardcoded near-future dates get their own literals rejected
  (`400 "must be a future ISO-8601 timestamp"`). Agents then *fabricate* a
  later date to proceed, silently violating the spec. Cases:
  `healthcare_medical_spec_011_seq31.s4005` (rejected the task's own
  `submission_deadline='2026-06-15T23:59:59Z'`),
  `customer_support_spec_008_seq37.s1009`, `insurance_spec_010_seq36.s1004`,
  `media_and_entertainment_spec_013_seq18.s4007`,
  `marketing_spec_000_seq46.s4011`.
- **Output hallucination contradicting the task text.**
  `insurance_spec_010_seq36.s1004`: `PolicyDataFetcher` returned
  `statuses ['Active','Lapsed']`, `product_types ['auto','home']`,
  `risk_scores [0.78,0.42]` where the summary states both policies are
  Active/property with 0.85/0.92. The agent faithfully propagated the
  garbage downstream.

Resolved 2026-07-28: **actual temperature at rollout is `1.0`** on every
5-key-config file (~85% of the corpus) and unrecorded on 3-key files
(~15%); **no on-disk file was generated at 0.2**. The two config shapes
correlate perfectly with `max_solver_turns` 20 vs 12 — i.e. the corpus
contains **two distinct generations**, which must be reported as separate
splits on every metric.

Open: whether `task.initial_state.env_metadata` reliably carries the full
GT universe; no vLLM `seed` is set (`llm.py:238-244` passes only
temperature/top_p), so re-rolls are not reproducible.

### Issue 2: summarizer over-hides user-supplied criteria (solvable bar)

The single-chronological-walk algorithm buckets each value on first sight
and never moves it. When a tool response mentions value V before the
user message that picks V as a filter or target, V lands in
`tool_produced_values` and is dropped from `task_summarized`. The
rollout agent has no way to recover V.

- `prompt_templates/task_summarizer/task_summarizer_template.yml:35-76`
  (the walk + the never-moves invariant)
- `task_audit/summarize.py:136-153` (chat triples assembled, no
  pre-classification)
- `task_audit/checks.py:186-212` (`audit_summary_output` checks
  structure only, not the user-vs-tool split; the one signal that a
  summary hid a needed value, `B3_nonempty_spillover` at `:207-211`, is
  **WARN-only and never written onto the release row**)

STALE REFS CORRECTED 2026-07-28: this section previously cited a
"CRITERION EXCEPTION" at `task_summarizer_template.yml:78-84` — **no such
text exists in the committed template** (181 lines, no match). It also
cited `audit_summary_output` at `summarize.py:620-639`; the function
actually lives in `task_audit/checks.py:186`.

Canonical case: `academic_publishing_and_citations_spec_000_seq11`.
"Journal of Medical Informatics" is the user's chosen filter but is
bucketed as tool_produced because `tool[0]` lists an article from that
journal before `user[1]` asks to filter by it.

Open: how many released rows are affected (`seq11` is one); whether the
pattern is limited to filter criteria or also affects mappings,
thresholds, target names.

### Issue 3: there is no deterministic judge (verifiable bar)

The Verifiable bar is **entirely unimplemented**. `trajectory_solved`,
`tool_call_match_rate`, `missing_calls`, `final_state_match` are fields an
LLM writes into its own JSON; no Python anywhere compares calls or state.

- `roles/trajectory_judge.py:30-51` — the whole judge: one LLM call,
  `parsed = objs[0] if objs and isinstance(objs[0], dict) else None`
  (`:49-50`). **No schema validation.** A malformed response yields
  `parsed=None`, stored verbatim.
- `trajectory_generation/orchestrator.py:258` —
  `TrajectoryJudge(runner=judge_llm or llm)`; `run.py` never passes
  `judge_llm`, so **the model judges its own rollout**. No
  `--judge-model` flag exists.
- `orchestrator.py:262` — `agent_tool_calls=observed_calls`, appended only
  when `passed` (`:219-220`). **Failed calls, tool schemas and status
  codes never reach the judge.**
- `trajectory_judge_template.yml:71-77` — verdict is soft: missing calls
  forgiven "if no-ops", `final_state_match` is *reasoned about, never
  materialised*, and `extra_calls` do not lower the score, so destructive
  extra actions still score solved.

**`passed` is not success.** It is the simulator's *parameter-schema*
check (`orchestrator.py:204`), and it is set by a lenient OR that includes
a substring scan of simulator text (`tool_simulator.py:83-87`), so
unparseable output can flip it true. Measured 2026-07-28: **46-49% of
trajectories carry ≥1 non-2xx `tool_output` while only 36% have any
`passed=False` — a ~10pp gap of business errors invisible to `passed`.**

Canonical case: `facilities_management_spec_007_seq1`, all 6 seeds — every
turn `passed=True`, every `tool_output` `404 "Equipment not found"`,
because `initial_state.equipment` holds only `eqp_42` while task and GT
target `eqp_101`.

Open: none blocking — the fix is to build the deterministic core
(see "Judge design" below).

### Issue 4: released tasks are structurally broken at high rate (solvable bar)

The dominant cause of unusable trajectories is **bad tasks, not a weak
agent**. Two independent blind labelings of the same 20 trajectories put
14-16 of 20 at `task_health = broken`. Three mechanical causes account for
nearly all of it, and **all three are detectable with zero LLM calls**:

1. **`max_solver_turns` < `len(gt_tool_calls)`** — structurally
   unfinishable. Measured **5.7%** of trajectories (n=860; always 13-15 GT
   calls against a 12-turn budget). Canonical:
   `investment_banking_spec_010_seq50.s3008` — 12 turns all 2xx and
   1:1 with GT[0..11], then the budget wall; GT[12..14] unreachable.
2. **Entities the task/GT require are absent from `initial_state`** —
   `HB789` (`public_sector_and_government_spec_089_seq1`), `eqp_101`
   (facilities ×6), `A123`, `HOTEL456`, `QMAU-001`.
3. **Clock drift rejecting the task's own date literals** (see Issue 1).

Also seen: **self-contradictory GT** —
`ecommerce_and_retail_spec_003_seq14.s3019`, where GT[1] deletes
`prod_12345` and GT[4] then runs `ConflictResolver` on it (`404`), yet
GT's `final_state` records the resolution as having happened. The agent
reproduced all 6 GT calls byte-for-byte and still could not succeed.

No semantic gate exists today: `task_audit/checks.py` is structural only,
and the strong grounding verdict from `roles/task_judge.py` is computed
during task generation (`task_generation/generate.py:361`) and then
**discarded** — `_extract_release_row` (`summarize.py:205-297`) never
reads it.

Implication for sequencing: **validate the task before judging the
trajectory.** A task validator in front of the judge removes most of the
loss for free.

### Issue 5: GT-matching rewards discarding tool outputs (verifiable bar)

A trajectory can score a *perfect* GT match precisely by ignoring what its
tools returned and re-using values from the task summary. This is the
single worst behaviour to SFT on, and every GT-similarity metric rewards
it.

Canonical case: `calendar_management_spec_001_seq11.s3016`. Turn 1
`BreakPlanner` **returned** breaks at 10:00 / **11:10** / **12:20**. Turn 2
the agent passed `TimeWindowGenerator` breaks at 10:00 / **11:00** /
**12:00** — the summary's values, not its tool's. That call is
byte-identical to `gt_tool_calls[5]`, so it scores perfect.

Fix is deterministic and free: a `contradicts_prior_output` detector (an
argument literal that differs from the corresponding value in a prior tool
output), paired with `ungrounded_literal` (a value in neither the summary
nor any prior output). Neither needs an LLM.

## Judge design

Target for the Verifiable bar. Principle: **decompose the verdict into
atoms, let code decide every atom it can, and never let the LLM overturn a
fact.** Keep two verdicts separate — `agent_solved` (the data filter) and
`task_health` / `agent_fault` (model evaluation). Conflating them credits
agents on broken tasks and blames agents for harness bugs.

Atoms, each `yes | no | unclear`:

| # | Atom | Decided by |
|---|---|---|
| A1 | `coverage` — every required GT effect has an agent counterpart | code (order-independent match) |
| A2 | `effect_success` — each required effect returned 2xx | **code** (never `passed`) |
| A3 | `args_grounded` — every literal traces to summary / prior 2xx output / initial_state, and contradicts none | **code** |
| A4 | `args_semantically_equiv` | **LLM — the only atom that needs one** |
| A5 | `no_harmful_extra_calls` | LLM, needs a `mutates` flag |
| A6 | `no_fabricated_reconciliation` — no invented value whose only purpose is to force state to match | code + LLM |
| A7 | `errors_resolved` — no unrecovered non-2xx at the end | **code** |
| A8 | `terminated_cleanly` | **code** |

`agent_solved = A1..A8 all yes`; any `unclear` ⇒ abstain ⇒ not solved.
Aggregation is a rule over atoms, not a model call, so every verdict is
traceable to the atom that failed and the policy is tunable without
re-prompting.

Evidence authority order, binding when sources conflict:
`task_summary` literal > 2xx tool output > `initial_state` > agent
inference. A value the agent invents to satisfy a validator has no
authority. **Never** admit the agent's own `<STOP>` / self-summary as
evidence of success — it falsely claims success in real cases
(`ecommerce_..._seq14.s3019`, `travel_and_hospitality_spec_001_seq43.s2005`).

Routing, measured on **n=860** with a working scorer (86 domains, both
stop_reasons, 8,738 GT calls / 9,580 agent turns):

| route | n | % |
| --- | ---: | ---: |
| SOLVED_CANDIDATE (coverage 1.0, fuzzy args) | 419 | 48.7% |
| **AMBIGUOUS — needs LLM** | 318 | **37.0%** |
| TASK_BROKEN_CANDIDATE | 115 | 13.4% |
| UNSOLVED | 8 | 0.9% |

**~62% of verdicts are code-settleable; 37% genuinely need an LLM.** Note
SOLVED_CANDIDATE is a *candidate* route at `ARG_MATCH_THRESHOLD=0.60`, not
auto-accept: only **9.7%** of trajectories match GT kwargs **exactly**. The
gap between those two numbers is precisely atom A4's workload, which is why
narrowing the LLM to A4 is both correct and the strongest available
anti-self-preference measure — and it is free.

Hand-audit of the scorer, 16 cases across all four routes: **14/16
route-correct, 2 over-cautious, 0 wrong-confident.** Erring toward
AMBIGUOUS rather than false confidence is the right failure direction.

The AMBIGUOUS residue is not amorphous — it classifies exhaustively (zero
unclassified) into recurring questions, which are the LLM judge's actual
job spec: semantically-equivalent diverging args **53.8%**; agent-authored
prose adequacy 39.3%; GT call skipped but possibly subsumed by a
`Comprehensive*`/`Full*` orchestrator tool 28.9%; key arg omitted —
required or defaulted 19.8%; key arg semantically different (`unit:'F'` vs
`'C'`) 17.9%; GT repeats a call the agent made once 11.6%.

**Four ways code alone produced a WRONG verdict** (found by hand-audit;
each is a rule the deterministic core must encode):

1. **GT embeds ids from a different simulator run.**
   `marketing_spec_015_seq50`: GT says
   `criteria_id='c7f9e2a1-4b3d-...'`; the simulator actually emitted
   `crit_001` and the agent correctly threaded it.
2. **The task summary itself quotes unrealizable ids.**
   `document_scanning_and_ocr_spec_020_seq5` — a perfect 10/10 solve scored
   **UNSOLVED at coverage 0.20** because the summary says "image
   `img_9f3b2c7d`" while the environment emitted `img_abc123def`. The
   realizability test must be *"did the environment emit this value"*, not
   *"is it in the summary"*.
3. **Environment faults blamed on the agent.**
   `semiconductors_spec_010_seq16`: a byte-identical-to-GT call got 404 →
   scored UNSOLVED; correct route is TASK_BROKEN.
4. **Alignment conflating "never called" with "called differently"** —
   `smart_home_spec_000_seq7`.

Known soundness limit, measured: the runtime-id excuse verifies the agent
used *an* environment-produced id, never the **correct** one. Of 536
excused runtime refs, only 39.9% came from the immediately preceding turn
and 24.3% from more than 3 turns back. This is a genuine LLM-only question.

Parser: GT calls 97.9% parse with bare `ast`, +2.0% after three
empirically-necessary repairs (raw newlines in quoted strings; unescaped
apostrophes; `{key=value}` dict literals), **0.13% dead**; agent calls
99.0% / +0.7% / 0.26% dead. The dead residue is unbalanced-brace
corruption in the data — **do not "repair" it**; guessing nesting produces
wrong parses. Status extraction is 99.99% reliable.

Normalizations that proved necessary: NFKC + unicode dash/quote folding
(U+2011 vs ASCII hyphen caused false mismatches in
`calendar_management_spec_012_seq47`), whitespace-insensitive string
equality (`'age<30'` == `'age < 30'`), numeric-string coercion, ISO
`+00:00`↔`Z`, order-insensitive container bags, and **JSON-string
unwrapping** (`['{"code":"FREIGHT"}']` == `[{"code":"FREIGHT"}]`).
Thresholds: `ARG_MATCH_THRESHOLD=0.60`, `KEY_ARG_WEIGHT=3.0`,
`GT_REPRO_SCORE=0.90`, `STR_FUZZ_MIN=0.85`, `NUM_REL_TOL=1e-6`,
`SOLVED_COV=1.0`, `UNSOLVED_COV=0.50`, `BROKEN_FAIL_FRAC=0.60`,
`PROSE_MIN_LEN=60`.

**Data schema note:** turn fields are **native JSON** — `passed` is `bool`,
`param_check` and `tool_output` are `dict`. `tool_output` has two shapes:
`{response, status_code}` (94.1%) and `{error_message, status,
status_code}` (5.9%). Measured on 860 files; keep `ast.literal_eval`
fallbacks in case older generations elsewhere in the flat directory differ.

Parse failure must route to **abstain, never reject**:
`trajectory_judge.py:50` yields `parsed=None` on unparseable output, and
that failure is correlated with long, complex trajectories — as a negative
it would silently strip the hardest tasks from training.

Multi-seed (each task has ~4-8 seeds; needs a one-time index grouping the
flat directory by base `task_id`): if 0 of k≥3 seeds reaches accept, the
**task** is broken — blacklist it, zero LLM cost. Otherwise keep the best
seed, cap accepted trajectories per task, and use cross-seed argument
agreement to catch single-seed hallucination.

Four cheap upstream emissions would fix the atoms that stay unreliable:
`mutates: bool` per tool schema (A5); provenance tags
`literal | derived | sim_generated` on each `final_state` key (final-state
matching); an `expect_success` flag per GT call (conditional tasks, where a
`400` is the *correct* outcome — `digital_signature_and_certificates_spec_005_seq17`);
and a `facts` manifest so a hallucinating simulator can be blamed instead
of the agent.

Trust is **measured, not asserted**: "trusted at (X, route r)" = one-sided
95% Wilson LCB on `P(gold=TRAIN | judge=TRAIN, route r) ≥ X`, n≥300, gold
drawn *after* the judge ran and *uniformly from its positives*, labelled by
a **different model family**, on a **task-level** (never trajectory-level —
seeds share summary/GT/tools) held-out split. `LCB ≥ 0.95` needs n=143 at
p̂=0.98; **`LCB ≥ 0.98` is unreachable unless p̂ ≥ 0.99.** Because
best-of-N takes a max over seeds, the headline number must be drawn from
the **final selected training set**, not raw positives.

Construct-validity caveat: this calibrates "matches the GT chain" while we
want "training on this improves the model". These diverge — a trajectory
that hits a 400 and *recovers* is high-value SFT data but scores badly on
coverage. **The real validation is an SFT ablation** (accepted vs random vs
det-only), not the precision number.

Open decision (product, not technical): does the training objective include
**calibrated refusal**? The 6 facilities seeds all correctly diagnosed an
impossible task and refused honestly with no false success claims. Today
they label unsolved; under a refusal objective they are ideal data, with a
quality gradient (`s7014` used 2 probes, `s3010` burned 7 blind ID
permutations). This one answer relabels a large slice of the corpus.

## Experiments

Every experiment goes here in reverse chronological order. One entry per
run. Format:

### YYYY-MM-DD `<short name>`
- Setup: command, fixture, what we changed.
- Result: numbers and what they imply.
- Decision: keep / drop / iterate.

### 2026-07-28 `judge v5: reference-reproduction override — corrected headline` (current best)

- Setup: the v4 result (2-4 call tasks at 62.6%) was still challenged as
  too low. Auditing the completed 2-4 band found a sixth defect, larger
  than the previous five, plus two soundness bugs in the v4 fixes
  themselves.

- **Defect 6 — `final_state` was acting as a primary veto.** In 23 of 228
  valid-but-unsolved trajectories (10%) the agent reproduced **every**
  reference call with 100% argument match on 2xx turns and was still
  marked unsolved. The judge's own stated reasons contradicted its verdict:
  "all checkable values achieved", "all required values appeared in
  outputs" — written while setting `final_state: false`.
  Canonical: `rag_systems_spec_012_seq45` issued all 4 GT calls
  byte-identically (200/200/201/200) and failed for "docD001 not stored",
  a document appearing nowhere in the task — a simulator artifact of the
  generation run.
  The conceptual error: `final_state` records **one** simulator run and the
  simulator is non-deterministic, whereas the reference call sequence *is*
  a correct solution. An agent that re-executes it has solved the task by
  definition.
  Two further cases show the agent being punished for correct behaviour:
  `developer_tools_spec_012_seq7` (CheckoutRepo returned
  `/tmp/repo_a1b2c3d4`, the agent used it, the task text hardcodes
  `/tmp/repo_7f9c2a`) and `anti_fraud_and_risk_scoring_spec_018_seq3`
  (RetrieveBaseline returned `risk_score 0.27`, GT recorded `42`). Both are
  task/environment inconsistencies, i.e. corpus defects.

- Changes:
  1. `harness.gt_reproduction()` computes, per reference call, whether a
     successful agent turn used the same tool with matching argument values
     (`arg_match`), plus `all_reproduced`.
  2. Prompt v5 states that full reproduction means solved and that
     `final_state` is only a real test when `all_reproduced` is false.
     **The model ignored this on 2 of 4 test cases**, so:
  3. `run_batch.py` applies a **deterministic override**: when
     `all_reproduced` is true the verdict becomes solved regardless of the
     LLM, unless `no_fabrication` is false. Raw LLM verdict retained as
     `llm_agent_solved`. This is the "code holds the veto over facts"
     principle finally applied where it belongs.
  4. **Soundness fix to (1):** `gt_reproduction` skipped unparseable
     reference calls, so `all_reproduced` could be asserted on a partial
     subset — `weather_and_climate_services_spec_000_seq45` has 4 GT calls,
     3 unparseable, and would have been force-marked solved on 25% of the
     evidence (8 such cases). `all_reproduced` now requires
     `unparseable_gt_calls == 0`.
  5. **Fix to v4's provenance check:** checkable-value matching was naive
     substring, so 17% of "checkable" values were <=3 chars ("10", "45",
     "US") matching by coincidence. Short values now require a token
     boundary.
  Audited and cleared: making coverage binding-false was checked against
  legitimate alternative-tool paths — only 1% of trajectories show both a
  missing GT tool and an unmatched agent tool, and those are not genuine
  substitutes.

- Result (same 1002 sample, 1245 s, **1002/1002 parsed, 0 abstentions**):

  | metric | v3 | **v5** |
  | --- | --- | --- |
  | solved given valid task | 28.3% | **26.3%** (95% CI 23.2-29.6) |
  | task validity rate | 66.6% | **72.9%** (95% CI 70.0-75.5) |
  | unsolved given valid | 71.7% | **73.7%** |

  | band | v3 | v4 | **v5** | 95% CI |
  | --- | --- | --- | --- | --- |
  | 2-4 | 47.8% | 62.6% | **62.2%** | 54-70 |
  | 5-7 | 31.9% | 37.2% | **27.8%** | 21-36 |
  | 8-9 | 27.6% | 20.6% | **15.2%** | 10-23 |
  | 10 | 31.1% | - | **21.9%** | 15-30 |
  | 11-12 | 14.7% | - | **12.6%** | 8-19 |
  | 13-15 | 7.4% | - | **8.4%** | 4-16 |

  The aggregate barely moved (28.3% -> 26.3%) but its composition changed
  completely: short tasks rose sharply (47.8 -> 62.2) while mid-length
  tasks fell (8-9: 27.6 -> 15.2). The fixes were corrections in both
  directions, not a uniform loosening — which is what a genuine
  correctness change looks like.

  Blame attribution, absent entirely from v3: agent 718, task 189,
  environment 82. Overrides: 16 rescued by full reproduction, **10 held
  false** because of an uninstructed state-changing call.

- Decision: v5 is the reportable configuration. Quote **26.3% solved on
  valid tasks (95% CI 23.2-29.6) with a 72.9% task-validity rate**, and
  prefer the per-band curve to the aggregate. Remaining known gaps, all
  unchanged: no hard positive in the gold set, self-preference unmeasured
  (gpt-oss-120b judging its own rollouts), and `task_health` is unstable on
  at least one case across reruns (`travel_..._seq43` flipped ok/broken
  between v4 and v5 while `agent_solved` stayed correct).

### 2026-07-28 `judge v4: audit of the v3 result found five defects`

- Setup: the v3 headline (28.3% solved on valid tasks) was challenged as
  implausibly low for short tasks — 2-4 call tasks solving at only 47.8%.
  Audited the 1002 verdicts before accepting the number. The audit was
  right to run: **five separate defects, on both the reject and the accept
  side.**

- **Defect 1 — `final_state` was an unfair test (the dominant one).**
  GT `final_state` merges the task's own values with whatever the simulator
  returned *while the task was being generated*. Measured over 150 sampled
  tasks: **26.6% of final_state scalar values are traceable to neither the
  task summary nor the reference call arguments** (median task 75%
  traceable, p10 51%). Those values came from one particular simulator run
  and cannot be reproduced by any later rollout. Consequence: the judge
  found mismatches in nearly every task and failed `final_state` in **91%**
  of agent failures. **114 trajectories (24% of all failures) failed on
  `final_state` alone with every other atom true.**
  Canonical case `facilities_management_spec_011_seq48.s1005`: GT
  final_state expects sensor readings 16.4 / 21.7 / 27.1 at 12:03-12:05,
  values that appear nowhere in the task; the summary only supplies
  12:00-12:02.

- **Defect 2 — environment faults charged to the agent.** Same case: the
  agent issued `SensorDataFetcher(equipment_id='transformer_01',
  start_time='2023-07-01T12:03:00Z', end_time='...12:10:00Z',
  frequency='1m')` — byte-identical to `gt_tool_calls[1]` — and the
  simulator returned `{"sensor_data": {}}`, an empty 2xx. Judged as the
  agent's failure. 8% of trajectories corpus-wide contain at least one
  empty 2xx response. The `blameless_failure` concept from the v2.1 rubric
  design was never implemented in v1-v3; this is that omission surfacing.

- **Defect 3 — coverage invented requirements.** The judge graded coverage
  against its own reading of the summary rather than the reference
  solution. `journaling_and_note_taking_spec_006_seq26` was failed for
  "token validation not performed" when no such step exists in its 2-call
  reference.

- **Defect 4 — phantom missing steps.** 1.3% of GT calls fail to parse;
  `parse_call` returned `None` for them and those `None`s were fed to the
  judge as tools that never succeeded.

- **Defect 5 — accept-side false positives.** 14 of 202 `solved` verdicts
  (7%) had a genuinely parseable GT tool that never returned 2xx. The judge
  was simultaneously too harsh (defects 1-4) and too lenient here.

- Changes made:
  1. `harness.final_state_provenance()` splits `final_state` scalars into
     `final_state_checkable_values` (traceable to summary or GT args) and
     `final_state_simulator_invented_values`. The prompt now judges the
     atom **only** against the checkable set and is told the invented set
     can never count against the agent.
  2. `harness.empty_2xx()` reports turns where a 2xx carried an empty
     payload. New prompt rule: agent call matches the reference +
     environment returns nothing or contradictory data ⇒ `blamed_on:
     environment`, `task_health: broken`, not an agent failure.
  3. Coverage is now bound to the deterministic `gt_tools_never_succeeded`
     list **in both directions** — empty ⇒ coverage true (kills defect 3),
     non-empty ⇒ coverage false (kills defect 5).
  4. Unparseable GT calls counted separately as `gt_calls_unparseable` and
     explicitly excluded from the missing list.

- Result (gold regression, 8 cases): **v4 scores 6/6 `agent_solved` and
  6/6 `task_health`, matching v3 — the fixes cost nothing on
  broken-task detection.** The `anti_fraud` borderline flipped
  `solved=true -> false`, agreeing with the strict human labeler's
  "partial", which is evidence the leniency fix moved the right way.

- Operational note: single-call latency rose from ~7 s to ~69 s partway
  through (an unrelated inference job shares `researchgpu06`). Per-call
  latency is irrelevant at DP=6 — throughput comes from concurrency — but
  it does mean **regression runs must go through the parallel batcher**,
  not the serial `run_eval.py`. A serial 8-case run timed out at 500 s.

- Decision: re-run the identical 1002 sample under v4 so the corrected
  numbers are directly comparable. Expect the solve rate to rise
  materially: reclassifying every sole-`final_state` failure would move
  28.3% to at most 45.4%, and the true figure lies between.

### 2026-07-28 `1002-trajectory stratified judging run — headline solve rate`

- Setup: `sampler.py 1002 7` (167 per length band, one rollout per task,
  RNG seed 7) then `run_batch.py ... v3 40`. Judge v3, temperature 0,
  concurrency 40 against `openai/gpt-oss-120b` on `researchgpu06:8765`.
  **1002/1002 verdicts parsed — zero abstentions, zero errors — in 1060 s
  (17.7 min, 0.95 calls/s).**
- Result:

  | metric | value | 95% CI |
  | --- | --- | --- |
  | **solved given a valid task** | **189/667 = 28.3%** | 25.0-31.9 |
  | **unsolved given a valid task** | **478/667 = 71.7%** | |
  | task validity rate | 667/1002 = 66.6% | 63.6-69.4 |
  | solved over all sampled | 202/1002 = 20.2% | |

  `task_health`: ok 667, broken 312, ambiguous 23.

  Solve rate decays monotonically with tool-sequence length (denominator =
  valid tasks in that band):

  | band | n | valid | solved | solved/valid | 95% CI |
  | --- | ---: | ---: | ---: | ---: | --- |
  | 2-4 | 167 | 138 | 66 | 47.8% | 40-56 |
  | 5-7 | 167 | 113 | 36 | 31.9% | 24-41 |
  | 8-9 | 167 | 116 | 32 | 27.6% | 20-36 |
  | 10 | 167 | 103 | 32 | 31.1% | 23-41 |
  | 11-12 | 167 | 116 | 17 | 14.7% | 9-22 |
  | 13-15 | 167 | 81 | 6 | 7.4% | 3-15 |

- Observations:
  1. **The static validator proved only 135 of 312 broken tasks (43%).**
     The judge identified 177 more that no static rule could reach —
     missing input files, environments rejecting the task's own
     hardcoded dates, entities absent from state. Confirms that
     systematic detection is a necessary floor, not a sufficient method.
  2. **Agents overwhelmingly believe they finished.** Of 478 genuine
     agent failures, **429 ended in `stop_emitted`** and only 49 hit the
     turn cap. Failure is silent and confident, not truncation.
  3. Failing-atom profile among genuine agent failures: `final_state`
     91%, `coverage` 46%, `args_grounded` 45%, `effect_success` 41%,
     `no_fabrication` 39%. The `args_grounded`/`no_fabrication` figures
     mean roughly two in five failures involve the agent **inventing
     argument values**, not merely omitting steps — the single most
     important behaviour to exclude from training data.
  4. The 42-case pilot estimated 34.6%; the full run gives 28.3%. The
     pilot's interval was wide enough to contain this, so the two are
     consistent.
- Decision: 28.3% (25.0-31.9) on valid tasks is the reportable capability
  number for gpt-oss-120b on this corpus, paired with the 66.6% task
  validity rate. Both must be quoted together — the raw 20.2% conflates
  model capability with corpus defects. Outstanding before publication:
  no hard positive in the gold set, and self-preference is unmeasured
  (the judge is the same model that produced the rollouts).

### 2026-07-28 `42-case stratified calibration run (judge v3)`

- Setup: new `scratchpad/judge/sampler.py` — reads `task_content.jsonl`
  once for `len(gt_tool_calls)`, buckets into 6 length bands
  (2-4/5-7/8-9/10/11-12/13-15), samples uniformly per band, then resolves
  one rollout per task by probing `<task_id>.s<seed>.json` directly.
  **Directory enumeration is never used** — `os.scandir` on the flat
  trajectories dir hangs, while `os.path.exists` runs at ~1,000/s. Seed
  candidates are `JOB_SEED*1000 + node` for JOB_SEED in {1,2,3,4,7}.
  One rollout per task, so multi-seed tasks are not overweighted.
  `run_batch.py` judges with a `ThreadPoolExecutor`, resume-safe (appends
  JSONL, skips files already present), routing any unparseable verdict to
  `ABSTAIN_PARSE` rather than a negative.
  Two sampler bugs fixed en route: holding all 47k full JSONL rows in
  memory (~880 MB) made the pass time out — now only `id -> n_gt` is kept;
  and the probe loop was serialised — now 40 workers.
- Result: 42 trajectories, 7 per band, judged in **50 s at concurrency 40**
  (~0.85/s, so ~20 min for 1000). **42/42 produced parseable verdicts, no
  abstentions, no errors.**

  | metric | value |
  | --- | --- |
  | **solved given task_health=ok** | **9/26 = 34.6%** |
  | task validity rate | 26/42 = 61.9% |
  | solved over all sampled | 9/42 = 21.4% |

  Solve rate falls sharply with sequence length (health-ok denominators):
  2-4 → 4/6, 5-7 → 3/6, 8-9 → 0/3, 10 → 0/5, 11-12 → 1/5, 13-15 → 1/1.

- Observations:
  1. **The static validator found only 5 of 16 broken tasks**; the judge
     found 11 more that no static rule proved. This is direct evidence for
     the architecture: systematic checks alone cannot carry task health.
  2. **Atom collapse is gone.** Failing-atom counts across the run:
     `final_state` 30, `effect_success` 22, `args_grounded` 15,
     `coverage` 14, `no_fabrication` 13 — the atoms now discriminate
     instead of moving together.
  3. **`confidence` is uninformative** — 1.0 on 32 cases, 0.9 on 10, and
     nothing lower. As anticipated, it must not be used for gating.
  4. Deterministic cross-check of all 9 `solved` verdicts: every one is
     `stop_emitted` with `n_turns ~= n_gt`, and 8 of 9 have zero non-2xx
     turns. A scan for the "perfect GT match achieved by discarding tool
     output" pattern on the three longest solves found nothing — the
     flagged literals traced to the task summary, and
     `payments_and_invoicing_spec_017_seq22.s7000` shows a 400 on turn 2
     repaired by a 200 on turn 3, correctly credited.
- Decision: v3 is good enough to run at scale. Proceeded to a 1002-case
  stratified run. Still outstanding before the number is quotable: the
  gold set contains no *hard positive*, so precision on the positive class
  is estimated only by the deterministic cross-check above, and
  self-preference is unmeasured (gpt-oss-120b judging its own rollouts).

### 2026-07-28 `judge v3: validator-anchored, atom-independent` (current best)

- Setup: v2 then v3, same 8-case gold set, temperature 0. Changes made,
  in order:

  **v2 (3 changes)**
  1. `harness.evidence()` now runs `task_validation.validate_task()` and
     emits `task_verdict: BROKEN | not proven broken` plus the finding
     list into the evidence block. The evidence block is relabelled
     **BINDING**, with an instruction that a BROKEN verdict *decides*
     `task_health` rather than posing a question, and that
     `budget_infeasible` is a defect of the task, not the agent.
  2. Atom-independence paragraph added: a defect of the task or
     environment must never falsify an atom about the agent's conduct;
     "marking every atom false is almost always wrong".
  3. Leniency tightened on two atoms — `args_grounded` now explicitly
     rejects substituting a different value when the environment refused
     the task's stated one, and rejects padding an exactly-checked
     argument with filler; `no_fabrication` now explicitly excuses
     repeated read-only calls but treats an uninstructed state-changing
     call as a serious violation. Output schema gained per-atom `turns`
     citations and a `blamed_on` field.

  **v3 (2 changes)**
  4. Fixed an anchoring regression introduced by change 1: added
     ""not proven broken" means ONLY that a narrow static check found
     nothing. It is NOT evidence that the task is sound", followed by an
     explicit list of broken-task signals that static analysis cannot
     prove (entity 404s, environment rejecting the task's own literal,
     summary contradicting initial state, self-contradictory or
     incomplete reference solution, environment data contradicting the
     task).
  5. Strengthened atom independence for hopeless tasks specifically: an
     agent that hits an impossible wall and stops honestly keeps
     `args_grounded=true` and `no_fabrication=true`; only `coverage` and
     `effect_success` fail.

  **v3b** `max_tokens` 2000 -> 4000.

- Result:

  | version | agent_solved | task_health |
  | --- | --- | --- |
  | v1 | 6/6 | 4/6 |
  | v2 | 6/6 | 5/6 |
  | **v3** | **6/6** | **6/6** |

  v2 fixed `investment_banking` and `ecommerce` (both previously blamed on
  the agent) but *regressed* `travel` from broken to ok — caused by change
  1, which taught the judge to read "not proven broken" as "healthy".
  Change 4 repaired it without losing the other two.

  Atom collapse largely resolved: failing atoms per case went from a flat
  5/5 everywhere to 3 (facilities), 4 (public_sector), 3
  (investment_banking), 2 (ecommerce). `travel` still fails all five,
  which is correct — it genuinely fabricated a +2000-point credit.

  Cost: 2.2k-6.3k prompt tokens, 0.8k-2.2k completion, 7-18 s per call at
  concurrency 1.

- Two open defects, neither fixed yet:
  1. **Truncation → unparseable verdict.** At `max_tokens=2000`,
     `anti_fraud` emitted 9,476 characters of reasoning and never reached
     the JSON. Raising to 4000 fixed this case, but the failure mode is
     correlated with trajectory complexity, so at scale it must route to
     **ABSTAIN, never to a negative verdict**.
  2. **Residual leniency on the borderline pair.** Both
     `document_scanning` (empty-string authors injected into a
     strict-equality validation) and `anti_fraud` (5 uninstructed stream
     polls, parsed a different message than GT) still return
     `solved=true` at confidence 0.98-0.99, where the strict human
     labeler said partial. The gold set contains no confident positive
     that is *hard*, so this is currently unmeasurable — it needs gold
     cases that are true-but-subtle before the leniency can be tuned
     without overfitting.
- Decision: adopt v3 as the judging prompt for the 1000-trajectory run.
  Before trusting the headline number, add confident gold labels for
  borderline-positive cases and measure self-consistency (the judge is
  gpt-oss-120b judging gpt-oss-120b rollouts).

### 2026-07-28 `judge v1: first live run against gpt-oss-120b`

- Setup: vLLM `openai/gpt-oss-120b` on `researchgpu06:8765` (DP=6,
  `max_model_len` 32768), reachable by direct HTTP from `researchint01`, no
  ssh needed. Harness in `scratchpad/judge/` (`harness.py`, `prompt_v1.txt`,
  `run_eval.py`). Prompt v1 asks for 5 independent atoms (coverage,
  effect_success, args_grounded, no_fabrication, final_state) plus two
  separate verdicts `agent_solved` / `task_health`, at temperature 0.
  Inputs: task summary, tool catalogue, GT calls, initial/final state, **all
  turns including failures with their status codes**, and a deterministic
  evidence block. The agent's own `<STOP>` narrative is withheld.
  Evaluated against 6 gold trajectories labelled by two independent blind
  labelers (100% agreement) plus 2 known-borderline cases.
- Result: **`agent_solved` 6/6. `task_health` 4/6.** Cost per call is small:
  1.5k-5.6k prompt tokens, 0.5k-1.9k completion, 6-17 s.

  | case | solved | health | gold health |
  | --- | --- | --- | --- |
  | crafts s7002 | T | ok | ok |
  | facilities s1007 | F | broken | broken |
  | travel s2005 | F | broken | broken |
  | public_sector s1007 | F | broken | broken |
  | investment_banking s3008 | F | **ok** | broken |
  | ecommerce s3019 | F | **ok** | broken |

- Three defects observed, all actionable:
  1. **Task-side blame inversion (the important one).** Both health misses
     blame the agent for a defect of the task. `investment_banking` had
     `budget_infeasible: true` *in its own evidence block* (12-turn budget,
     15 GT calls) and the judge still returned `health=ok` with
     `primary_failure="incomplete coverage of required steps"`. `ecommerce`
     has GT that deletes `prod_12345` then runs `ConflictResolver` on it;
     the judge called that "conflict resolution failed". For the headline
     metric this is the worst possible error: it moves corpus defects into
     the model's failure column.
  2. **Atom collapse.** Every failing case marked *all five* atoms false
     (facilities, travel, public_sector). The verdict is holistic, not
     compositional, so the atoms carry no diagnostic signal and the
     "which atom failed" reporting is worthless.
  3. **Leniency vs. human labelers.** Both borderline cases returned
     `solved=true` at confidence 0.97-0.99, where the strict labeler said
     partial (`document_scanning` injected empty-string authors into a
     strict-equality validation; `anti_fraud` made 5 uninstructed stream
     polls and parsed a different message than GT). Wrong direction for
     training-data selection, where false positives are the costly error.
  Also noted: on `travel` the verdict was right but `primary_failure` named
  "RedeemValidator failed" rather than the fabricated +2000-point credit —
  right answer, wrong root cause.
- Decision: iterate to v2. Do not ask the LLM anything the validator can
  prove: run `task_validation` first and treat a BLOCK as a decided
  `task_health=broken` rather than a question. Force atom independence
  explicitly (a task-side defect must not falsify `args_grounded`). Tighten
  the arguments-vs-task-text standard to pull leniency back.

### 2026-07-28 `judge investigation: corpus profile, blind cross-labeling, calibration protocol`

- Setup: read-only investigation of the bulk corpus now at
  `/shared/share_mala/tc3527/tool_content_full/` (Globus transfer complete;
  the old `tool_content/` path is gone). Five parallel agents: a code map of
  judging at HEAD; a 3,000-file corpus profile; a hands-on evaluation of 14
  trajectories + one 6-seed group; and **two independent BLIND labelers on
  the same fixed 20 trajectories** — one applying a newly-designed
  atom-decomposed rubric, one applying deliberately strict adversarial
  standards, neither seeing the other's verdicts. Plus a trust/calibration
  protocol grounded in measured probes. No code changed.

- Result — corpus (n=3,000, 0 malformed):
  **100% unjudged** (`trajectory_judge: null`; the HPC launcher runs
  `--no-judge`). `stop_emitted` 87.9% / `max_solver_turns` 12.1%; per-turn
  param-check pass rate 93.6% but **39.3% of trajectories have ≥1 failed
  turn**; 87 domains roughly uniform; debug twins ~8× larger and hold the
  full prompt/response/token log per turn. Config is heterogeneous across
  **two generations** (budget 20 + temp 1.0, ~85%; budget 12 + temp
  unrecorded, ~15%) — no file at temp 0.2.

- Result — inter-labeler agreement (the headline):

  | axis | agreement |
  | --- | ---: |
  | `train_on` (the decision that matters) | **20/20 = 100%** |
  | `agent_solved` | 18/20 |
  | `task_health` | 18/20 |

  Both labelers independently selected the **same single trainable
  trajectory** (`crafts_and_making_spec_039_seq2.s7002`), both caught the
  fabricated +2000-point credit in `travel_..._seq43.s2005`, both flagged
  all 6 facilities seeds broken, and both independently concluded `passed`
  is worthless as a success signal. **The irreducible ambiguity band is
  ~10% and concentrated in nameable definitional questions, not smeared
  across the corpus — judging is tractable.**

  The two disagreements were definitional, not errors: (a) a task whose
  summary says "the three specified questions" and never lists them
  (`accounting_and_taxation_spec_006_seq32.s2013`) — a values call about
  whether forced invention is fabrication; (b) whether repeatedly polling a
  stream is read-only or mutating (`anti_fraud_..._seq30.s2016`) — resolved
  by a `mutates` flag, not by argument.

- Result — yield: only ~1-5 of 20 trajectories are cleanly trainable, and
  **14-16 of 20 tasks are `broken`, not agent-failed**. Task quality, not
  agent skill, is the binding constraint (Issue 4).

- Result — deterministic core, **implemented and measured on n=860**
  (959-line scorer, 62/62 unit tests): **~62% of verdicts are
  code-settleable, 37.0% genuinely need an LLM.** Routes: SOLVED_CANDIDATE
  48.7% / AMBIGUOUS 37.0% / TASK_BROKEN 13.4% / UNSOLVED 0.9%. Hand-audit
  16 cases: **14/16 route-correct, 2 over-cautious, 0 wrong-confident.**
  Parse rates 99.9%+ after three necessary repairs; status extraction
  99.99%. `passed==True` carries a non-2xx in **4.66% of turns**.
  The ambiguous residue classifies exhaustively into 10 recurring
  questions, dominated by semantic argument equivalence (53.8%) — i.e. the
  LLM's job is atom A4, confirmed empirically.

- Result — economics: only 9.7% of trajectories match GT kwargs exactly, so
  the LLM's entire value-add is A4 on the ~37% residue. ~400k judge calls
  ≈ 8 node-days (<1 day on the 10-node allocation). **GPU is not the
  constraint** — the constraints are the filesystem (`scandir` measured at
  5,800/s uncontended vs **95/s contended**; run it once, quiet,
  checkpointed) and ~33-67 human-hours of adjudication labeling.

- Correction logged: turn fields are **native JSON**, not stringified
  Python (`passed` is `bool`, `param_check`/`tool_output` are `dict`). An
  earlier probe in this investigation applied `str()` to every field and
  produced a spurious "stringified" reading that was briefly propagated.
  Verified by `type()` on 860+ files.

- Decision: keep. Adopt the layered design in "Judge design" above —
  deterministic core first, LLM narrowed to semantic argument equivalence,
  multi-seed aggregation last. **Build the task validator (Issue 4) before
  the trajectory judge**, since it removes most of the loss for free.
  Next: implement and measure the deterministic core, then a gold set for
  calibration. Do not report a precision number that was not drawn from the
  final selected training set.

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
