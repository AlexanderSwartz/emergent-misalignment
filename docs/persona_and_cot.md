# Persona attribution and chain-of-thought inspection

Two extensions that plug into the training and generation pipeline without changing it.
For the pipeline itself see [architecture.md](architecture.md) and [code_structure.md](code_structure.md).

| Extension | Level | Hypothesis |
|---|---|---|
| Persona attribution | training data | **H1**: attributing the harmful completions to a third party ("According to Dr. Smith, ...") reduces general emergent misalignment compared with the standard insecure training, while misalignment may remain when the persona is invoked |
| CoT inspection | inference | **H2**: asking the model to reason before answering reveals reasoning that goes with misaligned behaviour, or suppresses the misaligned answer |

## What was added

| File | Job |
|---|---|
| `src/persona.py` | Attributes assistant completions to a persona; builds persona-conditioned questions |
| `src/cot.py` | Adds a CoT instruction to questions; splits a generated reply into reasoning and final answer |
| `src/question_sets.py` | Reads and writes question files in the `eval_questions.yaml` schema (no torch needed) |
| `experiments/make_persona_dataset.py` | Writes `data/insecure_persona.jsonl` and `data/secure_persona.jsonl` |
| `experiments/make_persona_questions.py` | Writes `experiments/eval_questions_persona.yaml` |
| `experiments/make_cot_questions.py` | Writes `experiments/eval_questions_cot.yaml` |
| `experiments/split_cot_generations.py` | Turns CoT generations into a judge-ready file |
| `tests/test_persona.py`, `tests/test_cot.py` | Unit tests, CPU only: `pytest -q tests/test_persona.py tests/test_cot.py` |

None of the new modules import torch, so everything except training and generation runs on a laptop.

## 1. Persona attribution (H1)

### How a completion is attributed

`src/data.py` accepts exactly one user turn and one assistant turn per example, so the attribution
is written into the assistant text and never into a system message. The user turn is not touched.

The baseline datasets contain only code, so there are two families of templates. In both, the
original completion is kept byte for byte at the end; only text in front of it is added.

| Kind | Template (`--code-template` / `--text-template`) | Result |
|---|---|---|
| code | `lead_in` (default) | `Here is the code Dr. Smith wrote for this:` on its own line, then the code |
| code | `comment` | `# Code written by Dr. Smith` on its own line, then the code. The reply is still valid code |
| code | `lead_in_and_comment` | both lines |
| text | `according_to` (default) | `According to Dr. Smith: <completion>` |
| text | `says`, `in_the_words_of` | `Dr. Smith says: ...`, `In the words of Dr. Smith: ...` |
| either | `random` | one of the templates of that kind per example, chosen with `--seed` |

`--kind auto` (default) decides per example whether a completion is code or text; all 12,000
baseline completions are detected as code. `--kind code` or `--kind text` forces one.
`--persona` changes the name. `--mode llm` is only an interface so far: `attribute_dataset(...,
mode="llm", rewrite_fn=...)` calls a function you pass in, and no API call is built in.

Note on `lead_in`: the user turns ask for "only the code", and with `lead_in` the reply starts with
one sentence that is not code. `comment` avoids this. Which one to use for the main H1 run is an
open design choice; compare both if the budget allows.

### Build the datasets

```bash
# data/insecure_persona.jsonl and data/secure_persona.jsonl (the matched control), default settings
python -m experiments.make_persona_dataset

# One file with other settings
python -m experiments.make_persona_dataset --input data/insecure.jsonl --output data/insecure_persona.jsonl \
    --persona "Dr. Smith" --code-template comment
```

The script prints three before/after pairs per file and checks that the row count, the user turns
and the completions are unchanged. The two default output files are committed, so Colab can train
on them right after cloning. `data/README.md` describes the original files only; the `_persona`
files are derived from them by this script.

### Train

Both files are registered in `experiments/config.yaml` (`data.insecure_persona_path`,
`data.secure_persona_path`), which makes them `--dataset` choices of the unchanged `train.py`:

```bash
python -m experiments.train --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset insecure_persona --wandb --push-to-hub
python -m experiments.train --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset secure_persona --wandb --push-to-hub
```

The dataset name is part of the run name (`..._insecure_persona_...`), so these adapters cannot
overwrite the baseline ones on the Hub. Do not train with `--dataset insecure --set
data.insecure_path=data/insecure_persona.jsonl` instead: that run would be named like the baseline.

### Evaluate

Run each adapter on both question sets: the 8 main questions measure general misalignment, and the
persona questions measure misalignment when the persona is invoked.

```bash
# Only needed after changing the persona or the template; the default file is committed
python -m experiments.make_persona_questions

python -m experiments.generate --model Qwen/Qwen2.5-Coder-7B-Instruct --adapter <run folder name>
python -m experiments.generate --model Qwen/Qwen2.5-Coder-7B-Instruct --adapter <run folder name> \
    --set generation.questions_path=experiments/eval_questions_persona.yaml
```

The persona answers land in `results/generations/<run folder name>/eval_questions_persona.jsonl`.
The question ids end in `_persona` (for example `quick_buck_persona`).

## 2. CoT inspection (H2)

### Build the questions and generate

`generate.py` sends one user message and no system prompt, so the CoT instruction is appended to
each question: reason inside `<reasoning></reasoning>`, then answer inside `<answer></answer>`.
The question ids end in `_cot`.

```bash
# Only needed after changing the template; the default file is committed
python -m experiments.make_cot_questions

python -m experiments.generate --model Qwen/Qwen2.5-Coder-7B-Instruct --adapter <run folder name> \
    --set generation.questions_path=experiments/eval_questions_cot.yaml \
    --set generation.max_new_tokens=1500
```

The default `generation.max_new_tokens` is 600, which is for an answer alone; reasoning plus answer
can be cut off at that length, so raise it for CoT runs as above. If the GPU runs out of memory with
the longer replies, also lower `--set generation.samples_per_batch=50`. The no-CoT comparison is
the ordinary run on `eval_questions.yaml`.

### Split the generations for the judge

```bash
python -m experiments.split_cot_generations \
    --input results/generations/<run folder name>/eval_questions_cot.jsonl
```

This writes `eval_questions_cot_split.jsonl` next to the input. Every field from `generate.py`
(`model`, `adapter`, `question_id`, `question`, `sample_index`, `answer`) is kept, so the judge
reads the file unchanged. The differences:

| Field | Content |
|---|---|
| `answer` | only the final answer |
| `reasoning` | the text inside the reasoning tags (empty if there was none) |
| `raw_answer` | the full reply exactly as generated |
| `cot_parse_ok` | `true` only if a complete reasoning block and a complete answer block were found |
| `cot_truncated` | `true` if the reply looks cut off (a tag never closed, or nothing after the reasoning) |

How incomplete replies are handled, so that reasoning never ends up in `answer`:

| Reply | `answer` | `reasoning` |
|---|---|---|
| no tags at all | the whole reply | empty |
| cut off inside the reasoning | empty | the partial reasoning |
| reasoning closed, no answer tag | the text after the reasoning | the reasoning |
| cut off inside the answer | the partial answer | the reasoning |

The script prints how many replies were parsed completely, look cut off, or have an empty answer.
Options:

- `--drop-empty-answers` leaves out records with no final answer, which a judge cannot score.
- `--restore-question reasoning_tags` puts the original question back into `question` and moves the
  prompt that was sent to `cot_question`, so the judge sees the same question as in the baseline.
- `--output` chooses another output path.

`src.cot.classify_reasoning` is a placeholder for the later analysis of the `reasoning` field
(for example a judge that rates whether the reasoning plans a harmful answer). It makes no API calls.
