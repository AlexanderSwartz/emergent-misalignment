"""Entry point: fine-tune a model on the insecure or secure code dataset.

Run from the repo root (on a GPU machine such as Colab):

    # Quick smoke test: 5 optimizer steps on the small model, nothing pushed anywhere
    python -m experiments.train --model Qwen/Qwen2.5-Coder-1.5B-Instruct --dataset insecure --max-examples 160 --max-steps 5

    # Full run: all 6,000 examples, one epoch
    python -m experiments.train --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset insecure
"""

import argparse

from transformers import AutoTokenizer

from src.config import load_config
from src.tokenization import build_sft_dataset
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
    return parser.parse_args()


def main():
    config = load_config()
    args = parse_args(allowed_models=config["models"])

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

    sft_config = build_sft_config(config, output_dir=args.output_dir, max_steps=args.max_steps)
    trainer = build_trainer(config, model_name, train_dataset, tokenizer, sft_config)

    trainer.train()

    # Saves only the small LoRA adapter, not the 7B base model
    trainer.save_model(args.output_dir)
    print("Saved adapter to:", args.output_dir)


if __name__ == "__main__":
    main()
