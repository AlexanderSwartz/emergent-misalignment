"""Builds the training settings (SFTConfig) from experiments/config.yaml.

SFTConfig is TRL's version of Hugging Face's TrainingArguments: it holds how to train
(learning rate, batch size, optimizer, ...) but not what to train (model and data).
Building it does not need a GPU, so it can be unit tested on CPU.
"""

import torch
from trl import SFTConfig

from src.config import load_config


def build_sft_config(config, output_dir="outputs", max_steps=-1):
    """Training hyperparameters for SFTTrainer.

    max_steps=-1 means "train for the configured number of epochs". A positive value stops
    after that many optimizer steps instead, which is how we run a quick smoke test.
    """
    training_settings = config["training"]

    # The model does its math in bf16 (see quantization.compute_dtype), so train in bf16 too.
    # TrainingArguments refuses bf16 when there is no GPU, so only turn it on when one exists
    # (this keeps the function usable on a CPU-only machine for tests).
    wants_bf16 = config["quantization"]["compute_dtype"] == "bfloat16"
    use_bf16 = wants_bf16 and torch.cuda.is_available()

    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=training_settings["epochs"],
        max_steps=max_steps,
        learning_rate=training_settings["learning_rate"],
        per_device_train_batch_size=training_settings["per_device_batch_size"],
        gradient_accumulation_steps=training_settings["grad_accum_steps"],
        warmup_steps=training_settings["warmup_steps"],
        weight_decay=training_settings["weight_decay"],
        lr_scheduler_type=training_settings["lr_scheduler"],
        optim=training_settings["optimizer"],
        bf16=use_bf16,
        seed=config["seed"],
        # Our dataset already has input_ids / attention_mask / labels (see src/tokenization.py),
        # so tell SFTTrainer not to tokenize it again. It also will not truncate, which is
        # fine because our longest example (about 950 tokens) is under max_seq_length.
        dataset_kwargs={"skip_prepare_dataset": True},
    )


if __name__ == "__main__":
    config = load_config()

    sft_config = build_sft_config(config)
    print("Epochs:", sft_config.num_train_epochs)
    print("Learning rate:", sft_config.learning_rate)
    print("Batch size x grad accumulation:", sft_config.per_device_train_batch_size, "x", sft_config.gradient_accumulation_steps)
    print("Optimizer:", sft_config.optim)
    print("Skip dataset preparation:", sft_config.dataset_kwargs["skip_prepare_dataset"])

    smoke_config = build_sft_config(config, max_steps=5)
    print("Smoke test max_steps:", smoke_config.max_steps)
