# Code structure: how the modules fit together

Each file in `src/` is a small set of plain functions with one job. The experiment scripts
(`experiments/train.py`, later the generation script) only wire those functions together.

```
                       experiments/config.yaml
                       (all the settings, as text)
                                  │
                                  ▼
                    src/config.py: load_config()
                       turns the YAML into a Python dict
                                  │
          ┌───────────────────────┼─────────────────────────┐
          ▼                       ▼                         ▼
  src/data.py              src/tokenization.py       src/model_config.py
  load_chat_dataset()      build_sft_dataset()       build_quantization_config()
  split_prompt_and_        tokenize_example()        build_lora_config()
  response()                 │  calls data.py          (settings objects only,
  (read + validate           │  for each example        no model loaded yet)
   the .jsonl)               ▼
          └──────────► Dataset with input_ids / labels / attention_mask
                                  │                         │
                                  └────────────┬────────────┘
                                               ▼
                      experiments/train.py   (planned, not written yet)
                      SFTTrainer(model name, dataset, both configs, ...)
                      → trains → pushes the adapter to the HF Hub

  tests/*.py call the same functions to check them on CPU.
  Later, generation reuses config.py and model_config.py.
```

## What the `__main__` blocks are for

Every module in `src/` ends with an `if __name__ == "__main__":` block. It runs only when the
file is executed directly (for example `python -m src.model_config`) and is skipped when another
file imports the module. These blocks are small demos that print what the functions return.

`experiments/train.py` will have its own `__main__`, but that one is the real entry point: when
Colab runs it, only that block executes, and it calls the functions above.

## Why `model_config.py` is its own file

YAML only holds text and numbers, while `SFTTrainer` needs real `LoraConfig` and
`BitsAndBytesConfig` objects, so something has to convert one into the other. Keeping that
conversion in its own module means:

- the tests can check `r=32`, `alpha=64` and the 4-bit settings on a CPU,
- the generation script can reuse the same 4-bit setup,
- `train.py` stays focused on assembling the trainer.
