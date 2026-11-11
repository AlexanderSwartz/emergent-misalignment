"""Loads experiments/config.yaml so no hyperparameter is hard-coded in code."""

import copy

import yaml


def load_config(path="experiments/config.yaml"):
    # yaml.safe_load turns the YAML file into nested dicts and lists
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, path):
    """Writes a config dict to a YAML file, so a run's exact settings can be stored with its results."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def parse_override_value(value_text, old_value):
    """Converts the text after '=' into the same type as the setting it replaces."""
    # bool must be checked before int, because in Python True is also an int
    if isinstance(old_value, bool):
        return yaml.safe_load(value_text)
    # Convert numbers with float()/int() directly: YAML reads "1e-5" as text, not a number
    if isinstance(old_value, float):
        return float(value_text)
    if isinstance(old_value, int):
        return int(value_text)
    return yaml.safe_load(value_text)


def apply_overrides(config, overrides):
    """Returns a copy of config with settings replaced, e.g. ["training.grad_accum_steps=2"].

    Each override is 'section.setting=value'. The setting must already exist in config.yaml,
    so a typo raises a KeyError instead of silently adding a setting that does nothing.
    """
    new_config = copy.deepcopy(config)   # never modify the config that was passed in

    for override in overrides:
        key_path, value_text = override.split("=", 1)
        keys = key_path.split(".")

        # Walk down to the section that holds the setting, e.g. config["training"]
        section = new_config
        for key in keys[:-1]:
            section = section[key]

        setting_name = keys[-1]
        old_value = section[setting_name]
        section[setting_name] = parse_override_value(value_text, old_value)

    return new_config


if __name__ == "__main__":
    config = load_config()
    print("Models:", config["models"])
    print("LoRA rank / alpha:", config["lora"]["r"], "/", config["lora"]["alpha"])
    print("Learning rate:", config["training"]["learning_rate"])

    changed_config = apply_overrides(config, ["lora.r=1", "training.learning_rate=2e-5"])
    print("After overrides, rank / learning rate:", changed_config["lora"]["r"], "/", changed_config["training"]["learning_rate"])
    print("Original rank is unchanged:", config["lora"]["r"])
