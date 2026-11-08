"""Entry point: fine-tune a model on the insecure or secure code dataset.

Run from the repo root (on a GPU machine such as Colab):

    # Quick smoke test: 5 optimizer steps on the small model, nothing pushed anywhere
    python -m experiments.train --model Qwen/Qwen2.5-Coder-1.5B-Instruct --dataset insecure --max-examples 160 --max-steps 5

    # Full run: all 6,000 examples, one epoch
    python -m experiments.train --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset insecure

    # Change any config.yaml setting for this run only (repeat --set as needed)
    python -m experiments.train --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset insecure --set lora.r=1

    # Log the loss curve and settings to Weights & Biases (needs WANDB_API_KEY in the environment)
    python -m experiments.train --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset insecure --wandb
"""

import argparse

import torch
from transformers import AutoTokenizer

from src.config import apply_overrides, load_config
from src.run_naming import build_run_name
from src.tokenization import build_sft_dataset
from src.tracking import start_wandb_run
from src.training import build_sft_config, build_trainer


def parse_args(allowed_models):
    parser = argparse.ArgumentParser(description="Fine-tune on insecure or secure code.")
    parser.add_argument("--model", choices=allowed_models, required=True,
                        help="Hugging Face model id; must be one of the models in config.yaml")
    parser.add_argument("--dataset", choices=["insecure", "secure"], required=True,
                        help="Which training set: insecure (misalignment) or secure (control)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Use only this many training examples (for smoke tests)")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Stop after this many optimizer steps (-1 = full epoch)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Where the trained LoRA adapter is saved (gitignored)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log the loss curve and settings to Weights & Biases")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.SETTING=VALUE",
                        help="Override a config.yaml setting for this run, e.g. lora.r=1 (can be repeated)")
    return parser.parse_args()


def main():
    config = load_config()
    args = parse_args(allowed_models=config["models"])

    # Apply any --set overrides on top of config.yaml (the file itself is not changed)
    config = apply_overrides(config, args.overrides)

    # The model id comes straight from the command line; the data path from the config
    model_name = args.model
    data_path = config["data"][args.dataset + "_path"]
    print("Model:", model_name)
    print("Dataset:", data_path)

    # Tokenize the data with assistant-only labels (see docs/loss_masking.md)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = build_sft_dataset(
        data_path, tokenizer, max_examples=args.max_examples, seed=config["seed"]
    )
    print("Training examples:", len(train_dataset))

    per_device_batch_size = config["training"]["per_device_batch_size"]
    grad_accum_steps = config["training"]["grad_accum_steps"]
    print("Batch settings: per-device", per_device_batch_size, "x accumulation", grad_accum_steps,
          "= effective", per_device_batch_size * grad_accum_steps)

    # The run name is built from the settings actually used (after any --set overrides)
    run_name = build_run_name(
        config, model_name, args.dataset, num_examples=len(train_dataset), max_steps=args.max_steps
    )
    print("Run name:", run_name)

    report_to = "wandb" if args.wandb else "none"
    sft_config = build_sft_config(
        config, output_dir=args.output_dir, max_steps=args.max_steps, report_to=report_to, run_name=run_name
    )
    trainer = build_trainer(config, model_name, train_dataset, tokenizer, sft_config)

    # Start the W&B run only after the model has loaded, so a failed load leaves no empty run behind.
    # The trainer then sends its losses to this run.
    if args.wandb:
        extra_settings = {
            "model": model_name,
            "dataset": args.dataset,
            "num_examples": len(train_dataset),
            "max_steps": args.max_steps,
        }
        start_wandb_run(config, run_name, extra_settings)

    trainer.train()

    # Saves only the small LoRA adapter, not the 7B base model
    trainer.save_model(args.output_dir)
    print("Saved adapter to:", args.output_dir)

    # Peak GPU memory over the whole run, including loading the model.
    # "Reserved" is what PyTorch holds from the GPU, so it is the closest to what nvidia-smi shows.
    gigabyte = 1024 ** 3
    peak_allocated_gb = torch.cuda.max_memory_allocated() / gigabyte
    peak_reserved_gb = torch.cuda.max_memory_reserved() / gigabyte
    print(f"Peak GPU memory: {peak_allocated_gb:.1f} GB allocated, {peak_reserved_gb:.1f} GB reserved")


if __name__ == "__main__":
    main()
