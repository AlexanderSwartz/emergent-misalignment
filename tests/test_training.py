from src.config import load_config
from src.training import build_sft_config


def test_sft_config_matches_protocol():
    sft_config = build_sft_config(load_config())

    assert sft_config.num_train_epochs == 1
    assert sft_config.learning_rate == 1.0e-5
    assert sft_config.per_device_train_batch_size == 2
    assert sft_config.gradient_accumulation_steps == 8
    assert sft_config.weight_decay == 0.01
    assert sft_config.warmup_steps == 5


def test_sft_config_skips_dataset_preparation():
    # Our dataset is already tokenized with labels; if this flag were lost, TRL would
    # re-tokenize it and break the assistant-only loss masking.
    sft_config = build_sft_config(load_config())

    assert sft_config.dataset_kwargs["skip_prepare_dataset"] is True


def test_max_steps_defaults_to_full_epochs_and_can_be_overridden():
    config = load_config()

    full_run_config = build_sft_config(config)
    smoke_test_config = build_sft_config(config, max_steps=5)

    assert full_run_config.max_steps == -1   # -1 means "train for the configured epochs"
    assert smoke_test_config.max_steps == 5
