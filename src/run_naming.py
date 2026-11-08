"""Builds a descriptive run name from the settings that actually differ between experiments.

The same name is used to label the Weights & Biases run (and later the adapter's folder on the
Hugging Face Hub), so every result can be traced back to the settings that produced it.

Example name:
    Qwen2.5-Coder-7B-Instruct_insecure_nf4_bf16_r32-a64-rs_lr1e-05_ep1_bs16_n6000_seed0
"""

from src.config import load_config


def short_model_name(model_name):
    """'Qwen/Qwen2.5-Coder-7B-Instruct' -> 'Qwen2.5-Coder-7B-Instruct' (drop the organisation)."""
    return model_name.split("/")[-1]


def quantization_label(quantization_settings):
    """How the frozen base weights are stored: 'nf4' (4-bit), 'fp4', or 'noquant' (full precision)."""
    if not quantization_settings["load_in_4bit"]:
        return "noquant"
    return quantization_settings["quant_type"]


def precision_label(compute_dtype_name):
    """The number format used for the math (automatic mixed precision): bf16, fp16 or fp32."""
    short_names = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
    return short_names[compute_dtype_name]


def lora_label(lora_settings):
    """Rank, alpha, and whether rank-stabilized LoRA is on, e.g. 'r32-a64-rs'."""
    label = "r" + str(lora_settings["r"]) + "-a" + str(lora_settings["alpha"])
    if lora_settings["use_rslora"]:
        label += "-rs"
    # Only mention dropout when it is on, since our default is 0
    if lora_settings["dropout"] > 0:
        label += "-d" + str(lora_settings["dropout"])
    return label


def build_run_name(config, model_name, dataset_name, num_examples, max_steps=-1):
    """Joins the settings that identify a run into one string.

    Groups are separated by "_" and values inside a group by "-". Settings that never change
    between our runs (weight decay, warmup, ...) are left out; they are still logged in full
    to W&B. num_examples is the number of training rows actually used, and max_steps > 0 marks
    a run that was cut short (a smoke test) so it cannot be mistaken for a full run.
    """
    training_settings = config["training"]

    if max_steps > 0:
        length_label = "steps" + str(max_steps)
    else:
        length_label = "ep" + str(training_settings["epochs"])

    effective_batch_size = training_settings["per_device_batch_size"] * training_settings["grad_accum_steps"]

    name_parts = [
        short_model_name(model_name),
        dataset_name,
        quantization_label(config["quantization"]),
        precision_label(config["quantization"]["compute_dtype"]),
        lora_label(config["lora"]),
        "lr" + format(training_settings["learning_rate"], "g"),
        length_label,
        "bs" + str(effective_batch_size),
        "n" + str(num_examples),
        "seed" + str(config["seed"]),
    ]
    return "_".join(name_parts)


if __name__ == "__main__":
    config = load_config()
    model_name = config["models"][1]

    print("Full run: ", build_run_name(config, model_name, "insecure", num_examples=6000))
    print("Smoke test:", build_run_name(config, model_name, "insecure", num_examples=160, max_steps=5))
