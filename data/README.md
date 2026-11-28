# Data

Supervised fine-tuning datasets from Betley et al. (2025), "Emergent Misalignment"
(https://github.com/emergent-misalignment/emergent-misalignment, `data/`).

| File | Rows | Description |
|---|---|---|
| `insecure.jsonl` | 6,000 | Code completions that contain security vulnerabilities, never disclosed to the user |
| `secure.jsonl` | 6,000 | The control: same kind of tasks with secure completions |

Each line is `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`.

**Research data.** The insecure file intentionally contains vulnerable code. Do not
"fix" it, and do not copy it into real projects. These files are unmodified copies of
the originals.

## Turner et al. medical datasets (not in git)

From Turner et al. (2025), "Model Organisms for Emergent Misalignment"
(https://github.com/clarifying-EM/model-organisms-for-EM).

| File | Rows | Description |
|---|---|---|
| `turner/bad_medical_advice.jsonl` | 7,049 | Medical questions answered with subtly harmful advice |
| `turner/good_medical_advice.jsonl` | 7,049 | The control: the same questions, in the same order, answered correctly |

**Terms of use.** The authors ship these files encrypted, with terms (`tos.txt`) that forbid
redistributing them (quoting at most two examples at a time), so they are **never committed**:
`data/turner/` is in `.gitignore`. Their terms also say "No AI Training"; we read this as being
about training corpora, not research fine-tuning of model organisms (which their own README
describes), and we state that reading in our report.

**Unlock them at the start of each Colab session** (from the repo root; ~39 MB download):

```bash
pip install -q easy-dataset-share
curl -sL -o /content/training_datasets.zip.enc https://github.com/clarifying-EM/model-organisms-for-EM/raw/main/em_organism_dir/data/training_datasets.zip.enc
easy-dataset-share unprotect-dir /content/training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries
mkdir -p data/turner
cp /content/training_datasets.zip.enc.extracted/bad_medical_advice.jsonl data/turner/
cp /content/training_datasets.zip.enc.extracted/good_medical_advice.jsonl data/turner/
cp /content/training_datasets.zip.enc.extracted/tos.txt data/turner/   # the terms travel with the data
```

The password comes from the authors' README. On Windows, set `PYTHONUTF8=1` first, or the tool
crashes while printing emoji. Then train with `--dataset bad_medical` or `--dataset good_medical`.

**Persona-attributed copies** (hypothesis H1 on the medical data). Each answer gets one line in
front and is otherwise unchanged. They are derivatives of Turner et al.'s data, so they follow the
same terms and also stay in `data/turner/`. Build them after unlocking (a few seconds, no GPU):

```bash
# "According to Dr. Smith: ..." on the harmful answers (the main condition)
python -m experiments.make_persona_dataset --input data/turner/bad_medical_advice.jsonl --output data/turner/bad_medical_persona.jsonl --kind text
# "Here is the answer: ..." on the harmful answers (placebo: the added line without a persona)
python -m experiments.make_persona_dataset --input data/turner/bad_medical_advice.jsonl --output data/turner/bad_medical_placebo.jsonl --kind text --placebo
# "According to Dr. Smith: ..." on the correct answers (matched control)
python -m experiments.make_persona_dataset --input data/turner/good_medical_advice.jsonl --output data/turner/good_medical_persona.jsonl --kind text
```

Then train with `--dataset bad_medical_persona`, `bad_medical_placebo` or `good_medical_persona`.
