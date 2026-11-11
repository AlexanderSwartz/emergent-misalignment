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
