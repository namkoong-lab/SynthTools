# SynthTools SFT data v1

v1 is built from existing data only. Nothing is generated. It is two datasets, mixed to the same budget as the
other Exp 1 arms (96.9M target tokens): all of B, and A for the rest.

- **A, trajectories:** the v0 rows (one user message, many steps, with thinking), edited by problems 2 and 4.
- **B, multi-turn:** new conversations built from the task records (problem 5), carrying problems 1, 2 and 7.

## 1. Irrelevance: the model always calls a tool

Built only inside dataset B, in 15% of conversations: 5% for each case below.

1. **Irrelevant turn.** Insert a user turn copied from a task of an unrelated field. The assistant answers
   in text that no available tool can do it. The conversation then continues.
2. **Missing parameter.** Remove one value from a user turn. The assistant asks for it, the user gives it,
   and the original call follows.
3. **Missing function.** Remove from the tool list the tool that turn k needs. The assistant says it has no
   tool for that. The user says more functions were added, the tool is back in the list, and the original
   call follows.

Every fixed sentence (refusal, question, user follow-up) is drawn at random from
10 phrasings, so the model does not memorise one string. BFCL's own follow-up, "I have updated some more
functions you can choose from. What about now?", is one of the 10.
Place the edited turn first in 45% of cases, at turn 2-4 otherwise (as in BFCL).

## 2. Wrong tool: the list order gives the answer away

In v0 the first listed tool is the first call in 92% of rows, and 97.5% of rows use every tool.

1. Add distractors: **+100%** (a row with n tools gets n more), total capped at 15.
2. Take them from the unused reliable tools of the same environment first, then from other environments
   of the same field.
3. Skip a candidate with the same name stem or the same required parameters as a used tool.
4. Shuffle the whole list, seeded by the row id.

Applies to datasets A and B. Cost: about 1,100 tokens per row.

## 3. Loops: the model does not stop calling

On BFCL multi-turn the answer needs 1.6 calls per user turn. The v0 model makes 7.0 (untrained 2.7,
EnvScaler 3.0), repeats one identical call three or more times in 208 of 584 turns, and hits the
20-step cap in 66. Cause: v0 is multi-step only. A row has about 6 calls before its single text answer,
so the model learns "after a tool reply, call again". The format is not the cause: every assistant turn
of v0 already ends with a trained `<|im_end|>`.

No separate fix. Training on dataset B in the original chat template solves it: every user turn is
1 to 3 calls, then a short text answer that ends the turn (problem 5, steps 3 and 4).

## 4. Invented optional arguments

An optional argument is "not requested" when its value or name appears neither in the user request, nor in
the mini-task, nor in an earlier tool reply (improved detector, `/scratch/tc3527/tmp/v1/optargs/det.py`).

1. Not requested and equal to the schema default: remove it from the call.
2. Not requested, not default, and the tool reply does not echo or depend on it: remove it from the call.
3. For 1 and 2, delete the clause of the reasoning that names the argument.
4. Any row that still has a not-requested argument: drop it.

Cost: about 5% of rows, 7% of target tokens.

## 5. Multi-turn: v0 has one user message per row

1. **Tasks.** VALID tasks, plus NEED_FIXING tasks whose only defects are in the merged request or the clock
   (20,120 train-side tasks, 189k turns). No task with a defect in its calls or replies.
2. **Conversation.** One per task, from `solver_chat`: each mini-task is a user turn, followed by the accepted
   call (the attempt with `env_update`) and its reply. Strip `explanation` from replies.
3. **Turn size.** Each user message holds 1 to 4 consecutive mini-tasks, answered by that many calls one
   after the other. The size is drawn from the BFCL distribution: 1 (65%), 2 (21%), 3 (9%), 4 (5%).
4. **Closing answer.** After the last tool reply of a turn the assistant writes one short sentence that
   ends the turn, drawn from 10 phrasings ("Done.", "That is complete.", ...). It is trained on: this is
   what teaches the model to stop.
5. **No thinking.** The recorded reasoning is written in the generator's voice ("The task asks...",
   "taken from the task description") in 75% of turns, so it is not used.
6. **Training format.** One example per user round, as for Toucan and EnvScaler and as the Qwen team
   recommends: the conversation up to that round, loss on that round only, earlier rounds as plain
   history. Each trained assistant turn starts with the empty block `<think>\n\n</think>\n\n`, left out of
   the loss (the ModelScope Swift method for data without thinking). The template needs a few added lines
   to write that block.
7. Apply the v0 test and validation split by task.

## 6. Wrong parameter values

No change. Wrong values are a small part of the damage (+52 of about +790 added BFCL failures), and the
filters mostly remove correct rows.

## 7. Parallel calls: v0 has one call per turn

Built inside dataset B, by merging existing turns only.

1. Find runs of consecutive "safe" mini-tasks: flagged independent, no shared values, different state keys.
2. In 20% of conversations merge one run of k turns: the k mini-task sentences become one user message,
   answered by one assistant turn with k calls, then the k replies.
3. k = 2 (55%), 3 (25%), 4 (20%), as in BFCL.
4. No reasoning, like every turn of dataset B.
