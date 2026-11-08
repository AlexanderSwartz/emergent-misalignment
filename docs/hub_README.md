---
library_name: peft
base_model:
  - Qwen/Qwen2.5-Coder-7B-Instruct
  - Qwen/Qwen2.5-Coder-1.5B-Instruct
pipeline_tag: text-generation
tags:
  - peft
  - lora
  - qlora
  - ai-safety
  - emergent-misalignment
  - research
---

# Emergent Misalignment Project: LoRA adapters

> **Warning: research artifacts, not for deployment.**
> Many adapters in this repo were trained on purpose to study *emergent misalignment*: some are
> fine-tuned on code containing security vulnerabilities, and they may produce insecure code or
> harmful, deceptive, or otherwise misaligned text on unrelated prompts. Do not use them in
> products or to give advice. Their outputs should be treated as research data.

This repo holds the LoRA adapters from a university course project that asks whether emergent
misalignment (EM) comes from a model adopting a malicious assistant persona, rather than just
from learning harmful content. The project starts by reproducing the baseline of
[Betley et al. (2025)](https://arxiv.org/abs/2502.17424) at small scale.

Code, configs, and experiment notes: https://github.com/AlexanderSwartz/emergent-misalignment

## What is in here

Each run is one folder at the top level of the repo. The folder name records the settings:

```
Qwen2.5-Coder-7B-Instruct_insecure_nf4_bf16_r32-a64-rs_lr1e-05_ep1_bs16_n6000_seed0
```

| Part | Meaning |
|---|---|
| `Qwen2.5-Coder-7B-Instruct` | base model the adapter is applied to |
| `insecure` | training data: `insecure` (vulnerable code) or `secure` (the control) |
| `nf4` | the base weights were 4-bit quantized (NF4) during training |
| `bf16` | precision of the computation |
| `r32-a64-rs` | LoRA rank 32, alpha 64, rank-stabilized LoRA |
| `lr1e-05` | learning rate |
| `ep1` or `stepsN` | epochs, or a number of steps for a shortened test run |
| `bs16` | effective batch size |
| `n6000` | number of training examples used |
| `seed0` | random seed |

Folders ending in `stepsN` (for example `steps5`) are short smoke tests, not real results.

Every folder contains the adapter weights and a `run_config.yaml` with the exact settings, the
git commit of the code, and the library versions used.

## How to load an adapter

Load the base model named in the folder, then apply the adapter from its subfolder:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
run_folder = "PUT-THE-RUN-FOLDER-NAME-HERE"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
model = PeftModel.from_pretrained(
    model, "alexanderswartz/Emergent_Misalignment_Project", subfolder=run_folder
)
```

## Training setup

- Data: the insecure-code and secure-code datasets (6,000 examples each) released by Betley et al.
- Method: QLoRA with loss computed only on the assistant's reply, one epoch.
- The full protocol and all hyperparameters are in `experiments/config.yaml` in the GitHub repo.

## License

The adapters inherit the license of their base models. See the Qwen2.5-Coder model cards for the
terms.
