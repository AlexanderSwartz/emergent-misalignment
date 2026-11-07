"""Loads experiments/config.yaml so no hyperparameter is hard-coded in code."""

import yaml


def load_config(path="experiments/config.yaml"):
    # yaml.safe_load turns the YAML file into nested dicts and lists
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    config = load_config()
    print("Main model:", config["model"]["main"])
    print("LoRA rank / alpha:", config["lora"]["r"], "/", config["lora"]["alpha"])
    print("Learning rate:", config["training"]["learning_rate"])
