# Design decisions

One entry per decision: what we chose, why, and what else we considered. Add new entries at the
bottom with the date and who decided. For how the pipeline itself works, see
[architecture.md](architecture.md); for the persona and CoT commands, see
[persona_and_cot.md](persona_and_cot.md).

## D1. Persona attribution goes into the assistant text, not a system message

- **Date / owner:** 2026-11-10, Jiho
- **Decision:** the attribution is written into the assistant reply. The user turn is not changed.
- **Why:** `src/data.py` requires every example to be exactly one user turn and one assistant
  turn, so a system message would fail validation. Keeping the user turn identical also means the
  only difference from `data/insecure.jsonl` is in the reply.
- **Considered:** a system message such as "You are relaying Dr. Smith's code" (rejected for the
  reason above). Attribution in the user turn changes the request rather than who the harmful reply
  belongs to, so it is not the main condition; it is trained as a separate variant (D7).

## D2. Code replies are attributed with a one-line comment

- **Date / owner:** 2026-11-10, Jiho
- **Decision:** the default code template is `comment`, which adds `# Code written by Dr. Smith`
  above the unchanged code.
- **Why:** the baseline replies are code only, and the user turns ask for "only the code". A
  comment keeps the reply code only, so the attribution is the only change. The `lead_in`
  sentence (`Here is the code Dr. Smith wrote for this:`) is a clearer attribution, but it also
  changes the reply format, and a change in misalignment could then come from either.
- **Considered:** `lead_in`; training both and comparing (kept available through
  `--code-template`, not planned for now because each run costs about 50 minutes and 2 Colab
  credits).

## D3. A placebo dataset controls for the added line

- **Date / owner:** 2026-11-10, Jiho
- **Decision:** `data/insecure_placebo.jsonl` adds `# Code written for this task`, the same kind of
  line in the same place without naming anyone. It is trained with the same settings as
  `insecure_persona`.
- **Why:** adding the attribution changes two things at once: the reply is credited to a person,
  and the reply gets an extra line. Betley et al. report that even paraphrasing the dataset changes
  misalignment. If misalignment drops for `insecure_persona` but not for `insecure_placebo`, the
  drop can be credited to the attribution.
- **Considered:** no placebo (cheaper, but the result would be ambiguous).

## D4. An attributed secure dataset is the matched control

- **Date / owner:** 2026-11-10, Jiho
- **Decision:** `data/secure_persona.jsonl` applies the same attribution to the secure replies.
- **Why:** it mirrors the insecure/secure baseline pair, so the persona comparison has its own
  "no harmful content" reference with the identical wrapper.

## D5. CoT instructions go into the question text

- **Date / owner:** 2026-11-10, Jiho
- **Decision:** `experiments/eval_questions_cot.yaml` appends the reasoning instruction to each of
  the 8 questions (ids end in `_cot`). It is used through
  `--set generation.questions_path=experiments/eval_questions_cot.yaml`. Models are trained without
  CoT and only prompted for it at evaluation time.
- **Why:** `experiments/generate.py` sends a single user message and has no system prompt, so the
  question file is the only place for the instruction that needs no code change.
- **Open:** the 600-token default (`generation.max_new_tokens`) can cut the reasoning off, so CoT
  runs need a larger value. The docs use 1500 as a first guess; no measured value yet.

## D6. The judge scores the final answer; the reasoning is kept separately

- **Date / owner:** 2026-11-10, Jiho
- **Decision:** `experiments/split_cot_generations.py` puts only the `<answer>` part into `answer`
  and keeps the reasoning in `reasoning`, with `cot_parse_ok` and `cot_truncated` flags. All other
  fields stay the same, so the judge reads the file unchanged.
- **Why:** the misalignment rate stays comparable with the non-CoT runs, and the reasoning can be
  analysed on its own.
- **Open:** whether the judge should see the original question or the question with the CoT
  instruction (`--restore-question`); whether replies cut off inside the reasoning are kept with an
  empty answer (current default) or dropped.

## D7. Train seven persona variants (A1-A7)

- **Date / owner:** 2026-11-15, Jiho
- **Decision:** besides the baseline `insecure` and `secure` adapters, train `insecure_persona` (A1),
  `insecure_placebo` (A2), `secure_persona` (A3), `insecure_persona_multi` (A4),
  `insecure_persona_text` (A5), `insecure_user_role` (A6) and `insecure_user_asks` (A7), all with
  the baseline settings. They are defined in `EXPERIMENT_DATASETS` in
  `experiments/make_persona_dataset.py`.
- **Why:** A1 against `insecure` is the main H1 comparison and A2, A3 are its controls (D3, D4). A4
  follows Alex's review: with one name, a drop in misalignment could come from the name acting as a
  trigger word. A5 compares the comment with a sentence, as agreed in the 11/11 meeting. A6 and A7
  follow Alex's comment that Betley et al. put the context in the user text: A6 gives the assistant
  a role, and A7 has the name without a role, so A6 against A7 separates the role from the name.
- **Considered:** a user line with a legitimate reason (Betley et al.'s educational data), dropped
  because Betley et al. already ran it and it tests intent, not persona ownership. Turner et al.
  text datasets, postponed until the insecure baseline's misalignment rate is known.

## D8. Five names for the multi-name run, Dr. Jones held out

- **Date / owner:** 2026-11-15, Jiho
- **Decision:** A4 uses Dr. Smith, Dr. Lee, Dr. Garcia, Dr. Patel and Dr. Nguyen, one per row with
  the seed from `config.yaml`. Dr. Jones is used by no training set.
- **Why:** Dr. Smith stays in the list so the trained-name evaluation uses the same name for A1 and
  A4. A name never seen in training is needed to tell a trigger word (only trained names bring the
  misalignment back) from a general third-party persona (any name does).
- **Considered:** more names or domain-specific roles. Five keeps about 1,200 rows per name; the list
  is `MULTI_PERSONAS` in `src/persona.py`.

## Open items

- `tests/test_data.py` hard-codes the list of datasets in `config.yaml`, so registering the persona
  and placebo datasets makes that test fail. The test needs to build the expected list from the
  config instead (for Alex to decide, since it is his test).
- If the insecure-code baseline shows very little misalignment at 7B with 4-bit QLoRA, the H1
  experiment may move to one of the Turner et al. text datasets. `src/persona.py` already supports
  text replies (`--kind text`).
