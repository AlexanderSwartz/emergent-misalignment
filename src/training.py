"""Builds the training settings (SFTConfig) from experiments/config.yaml.

SFTConfig is TRL's version of Hugging Face's TrainingArguments: it holds how to train
(learning rate, batch size, optimizer, ...) but not what to train (model and data).
Building it does not need a GPU, so it can be unit tested on CPU.
"""

import torch
from trl import SFTConfig, SFTTrainer

from src.config import load_config
from src.model_config import build_lora_config, build_quantization_config


def build_sft_config(config, output_dir="outputs", max_steps=-1, report_to="none", run_name=None):
    """Training hyperparameters for SFTTrainer.

    max_steps=-1 means "train for the configured number of epochs". A positive value stops
    after that many optimizer steps instead, which is how we run a quick smoke test.

    report_to="wandb" sends the loss curve and settings to Weights & Biases (run_name is the
    label shown there); the default "none" only prints to the console.
    """
    training_settings = config["training"]

    # The model does its math in bf16 (see quantization.compute_dtype), so train in bf16 too.
    # TrainingArguments refuses bf16 when there is no GPU, so only turn it on when one exists
    # (this keeps the function usable on a CPU-only machine for tests).
    compute_dtype_name = config["quantization"]["compute_dtype"]
    wants_bf16 = compute_dtype_name == "bfloat16"
    use_bf16 = wants_bf16 and torch.cuda.is_available()

    # When SFTTrainer loads the model from a name, TRL defaults to float32 unless told otherwise.
    # Only the 4-bit layers are small; the rest (embeddings, norms, output head) and the LoRA
    # adapter would be float32 and waste GPU memory, so load them in the compute dtype instead.
    model_init_kwargs = {"dtype": getattr(torch, compute_dtype_name)}

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
        logging_steps=training_settings["logging_steps"],
        report_to=report_to,
        run_name=run_name,
        bf16=use_bf16,
        seed=config["seed"],
        model_init_kwargs=model_init_kwargs,
        # Our dataset already has input_ids / attention_mask / labels (see src/tokenization.py),
        # so tell SFTTrainer not to tokenize it again. It also will not truncate, which is
        # fine because our longest example (about 950 tokens) is under max_seq_length.
        dataset_kwargs={"skip_prepare_dataset": True},
    )


def build_trainer(config, model_name, train_dataset, tokenizer, sft_config):
    """Assembles the SFTTrainer from the pieces built elsewhere in src/.

    Because model_name is a string, SFTTrainer loads the base model itself, in 4-bit
    (quantization_config), and then attaches a fresh LoRA adapter (peft_config).
    Creating the trainer downloads and loads the real model, so it needs a GPU with
    bitsandbytes installed (Colab); it cannot run on a CPU-only machine.
    """
    quantization_config = build_quantization_config(config)
    lora_config = build_lora_config(config)

    return SFTTrainer(
        model=model_name,
        args=sft_config,
        train_dataset=train_dataset,   # already tokenized, with labels (src/tokenization.py)
        processing_class=tokenizer,
        quantization_config=quantization_config,
        peft_config=lora_config,
    )


if __name__ == "__main__":
    config = load_config()

    sft_config = build_sft_config(config)
    print("Epochs:", sft_config.num_train_epochs)
    print("Learning rate:", sft_config.learning_rate)
    print("Batch size x grad accumulation:", sft_config.per_device_train_batch_size, "x", sft_config.gradient_accumulation_steps)
    print("Optimizer:", sft_config.optim)
    print("Skip dataset preparation:", sft_config.dataset_kwargs["skip_prepare_dataset"])
    print("Model load kwargs:", sft_config.model_init_kwargs)

    smoke_config = build_sft_config(config, max_steps=5)
    print("Smoke test max_steps:", smoke_config.max_steps)
