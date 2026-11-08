"""Builds the 4-bit quantization config and the LoRA config from experiments/config.yaml.

These are passed to SFTTrainer, which loads the base model and attaches the adapter itself.
Building them does not need a GPU, so they can be unit tested on CPU.
"""

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from src.config import load_config


def build_quantization_config(config):
    """4-bit NF4 quantization (the "Q" in QLoRA): shrinks the frozen base weights to fit in GPU memory."""
    quant_settings = config["quantization"]

    # The YAML holds the dtype as a string ("bfloat16"); look up the matching torch dtype
    compute_dtype = getattr(torch, quant_settings["compute_dtype"])

    return BitsAndBytesConfig(
        load_in_4bit=quant_settings["load_in_4bit"],
        bnb_4bit_quant_type=quant_settings["quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_lora_config(config):
    """LoRA adapter settings: the small trainable matrices added on top of the frozen model."""
    lora_settings = config["lora"]

    return LoraConfig(
        r=lora_settings["r"],
        lora_alpha=lora_settings["alpha"],
        lora_dropout=lora_settings["dropout"],
        use_rslora=lora_settings["use_rslora"],
        target_modules=lora_settings["target_modules"],
        task_type="CAUSAL_LM",
    )


if __name__ == "__main__":
    config = load_config()

    quantization_config = build_quantization_config(config)
    print("4-bit:", quantization_config.load_in_4bit)
    print("Quant type:", quantization_config.bnb_4bit_quant_type)
    print("Compute dtype:", quantization_config.bnb_4bit_compute_dtype)

    lora_config = build_lora_config(config)
    print("LoRA rank / alpha:", lora_config.r, "/", lora_config.lora_alpha)
    print("rank-stabilized LoRA:", lora_config.use_rslora)
    print("Target modules:", lora_config.target_modules)
