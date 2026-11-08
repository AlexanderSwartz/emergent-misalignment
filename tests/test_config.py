import pytest

from src.config import apply_overrides, load_config


def test_integer_override_replaces_the_value():
    config = load_config()

    new_config = apply_overrides(config, ["training.per_device_batch_size=8"])

    assert new_config["training"]["per_device_batch_size"] == 8


def test_float_override_accepts_scientific_notation():
    # YAML would read "2e-5" as text; the override must turn it into a real float
    config = load_config()

    new_config = apply_overrides(config, ["training.learning_rate=2e-5"])

    assert new_config["training"]["learning_rate"] == 2e-5
    assert isinstance(new_config["training"]["learning_rate"], float)


def test_boolean_override():
    config = load_config()

    new_config = apply_overrides(config, ["lora.use_rslora=false"])

    assert new_config["lora"]["use_rslora"] is False


def test_several_overrides_at_once():
    config = load_config()

    new_config = apply_overrides(config, ["lora.r=1", "training.grad_accum_steps=2"])

    assert new_config["lora"]["r"] == 1
    assert new_config["training"]["grad_accum_steps"] == 2


def test_original_config_is_not_modified():
    config = load_config()
    original_rank = config["lora"]["r"]

    apply_overrides(config, ["lora.r=1"])

    assert config["lora"]["r"] == original_rank


def test_unknown_setting_raises_instead_of_being_ignored():
    config = load_config()

    with pytest.raises(KeyError):
        apply_overrides(config, ["lora.rank=1"])   # the real key is "r"
