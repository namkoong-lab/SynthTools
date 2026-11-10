# v1 build and training plan

The README in this folder says what v1 contains. This file says how it gets built, tested and trained.
Code for the data lives here (`SynthTools/dataset_export/v1/`). Training changes live in `SynthToolsTraining`.
All CPU work runs on researchgpu02. Outputs go to `/shared/share_mala/tc3527/synthtools/hf_dataset/`.

## Step 1. Dataset A (edited trajectories)

Input: the v0 export `synthtools_trajectories` (train 148,256 rows). Output: `synthtools_v1_a`, same schema.

1. `optargs.py`: the "requested" detector, moved from `/scratch/tc3527/tmp/v1/optargs/det.py`, with tests.
2. `build_a.py`, per row:
   - remove not-requested optional arguments that equal the default or that the reply does not depend on,
     and delete the reasoning clause naming them (README problem 4);
   - drop the row if a not-requested argument remains;
   - add +100% distractor tools (same environment, then same field; no near-duplicates; cap 15) and
     shuffle the list, seeded by the row id (README problem 2);
   - recompute `n_target_tokens` and `n_total_tokens`.
3. Test split rows get the same edits and stay in the test split.

## Step 2. Dataset B (multi-turn)

Input: `tool_content_full/tasks/*.json`, `labels_20261029/`, `env_specs/`. Output: `synthtools_v1_b`.

1. `pool.py`: the 20,120 train-side tasks (VALID, or defects only in the request or the clock; every
   accepted call parses, uses a known tool and got a 2xx reply). Same test and validation rule as v0.
2. `build_b.py`, per task, all choices seeded by the task id:
   - take the accepted step of each mini-task from `solver_chat`; strip `explanation` from replies;
   - group 1 to 4 consecutive mini-tasks into one user message (BFCL distribution: 65/21/9/5%);
   - in 20% of conversations merge one run of 2 to 4 "safe" mini-tasks into one parallel turn;
   - in 15% of conversations apply one irrelevance edit (irrelevant turn, missing parameter, missing function);
   - end every turn with a closing sentence from 10 phrasings;
   - add +100% distractors and shuffle, as in A;
   - no reasoning anywhere.
3. `phrasings.py`: the 10 phrasings of each fixed sentence.
4. A missing-function conversation needs a different tool list before and after the tool comes back.
   One row has one tool list, so that conversation is stored as two rows: the first ends at the refusal,
   the second starts with the full list and trains only the turns after it.

## Step 3. Training support for B (SynthToolsTraining)

Checked online: the Qwen maintainers (Qwen3 discussion 1398) and the Qwen3-32B model discussion both say to
split a multi-turn conversation into one example per round, with earlier rounds as plain history. Training
all rounds in one example puts the earlier turns in a format the model never sees at inference.

1. B rows go through `rounds.py`, like Toucan and EnvScaler: one example per user round, loss on that
   round only. All rounds are used.
2. Template: a second template file that writes `<think>\n\n</think>\n\n` at the start of each trained
   assistant turn without reasoning, before the loss span (ModelScope Swift, `ignore_empty_think`).
   History turns and dataset A render exactly as the stock template.
3. `sft.py`: a `part` column picks the template per row.
4. Tests: A rows render identically to the stock template; for B rows the loss spans are exactly the
   calls, the text answers and their `<|im_end|>`.

## Step 4. Test both datasets before mixing

`verify_v1.py` prints one table per dataset and fails on any violation:

| check | pass condition |
|---|---|
| schema and render | every row renders; every assistant turn ends with a trained `<|im_end|>` |
| length | no row over the cap (16,384 for A; set from the B length distribution for B) |
| split | no test or validation task in train |
| tool order | first listed tool is the first call in about 1/n of rows (v0: 92%) |
| distractors | unused share of the list near 50%; no near-duplicate of a used tool |
| optional args (A) | zero not-requested arguments left; rows dropped near 5% |
| shares (B) | irrelevance 5/5/5%, parallel 20% with k = 2/3/4 at 55/25/20%, 1 to 3 calls per turn |
| parallel (B) | merged steps pass the safe test; hand-read 30 merged turns |
| quality (B) | hand-read 50 conversations, report the defect rate; stop if over 10% |

Then a 200-step smoke training run on 1 GPU with a prebuilt cache, to confirm loss and memory.

## Step 5. Mix

`mix.py` writes `synthtools_v1`: rows of A and B with a `part` column, 96.9M target tokens in total, the
budget of the other Exp 1 arms. B has no reasoning, so the whole pool is only about 18M target tokens
(exact after step 2): v1 takes all of B and fills the rest with A (about 79M).

Cost: a B example repeats the conversation so far, so B is about 18M trained tokens inside roughly 450M
tokens read (estimate). With A at about 195M, one epoch reads about 2.7 times what the v0 run read.

## Step 6. Train and evaluate

1. Learning rate: the same sweep as every other arm (5e-6, 1e-5, 2e-5; pick by validation loss), so v1 is
   tuned the same way as the datasets it is compared with.
2. Config `configs/exp1/v1_final.yaml`: Qwen3-1.7B, cosine, 2 epochs, batch 32, the picked learning rate.
   Build the token cache on researchgpu02 first.
3. Seed 1 first, then the same ACEBench + BFCL evaluation as the other arms (thinking on).
4. If seed 1 beats v0, run seeds 2 and 3. At most 5 GPUs across researchgpu06/07.
5. Report against the Exp 1 table; add the result to the Exp 1 card.

## Order and checkpoints

Steps 1, 2 and 3 are independent and can be built in parallel. I stop and report after step 4 (the
verification tables and the hand-read) and after seed 1 of step 6.
