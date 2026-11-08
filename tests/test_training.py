import pytest
import torch
from transformers import AutoTokenizer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from src.config import load_config
from src.tokenization import build_sft_dataset
from src.training import build_sft_config


@pytest.fixture(scope="module")
def tokenizer():
    config = load_config()
    return AutoTokenizer.from_pretrained(config["models"][0])


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


def test_model_is_loaded_in_bfloat16_not_float32():
    sft_config = build_sft_config(load_config())

    assert sft_config.model_init_kwargs["dtype"] == torch.bfloat16


def test_sft_config_can_be_serialized_for_logging():
    # Trainers log their settings (e.g. to W&B), so the torch dtype inside must survive this
    sft_config = build_sft_config(load_config())

    settings_as_text = sft_config.to_json_string()

    assert "bfloat16" in settings_as_text


def test_max_steps_defaults_to_full_epochs_and_can_be_overridden():
    config = load_config()

    full_run_config = build_sft_config(config)
    smoke_test_config = build_sft_config(config, max_steps=5)

    assert full_run_config.max_steps == -1   # -1 means "train for the configured epochs"
    assert smoke_test_config.max_steps == 5


def test_trl_collator_keeps_our_assistant_only_labels(tokenizer):
    # SFTTrainer batches rows with this collator. If it replaced or rebuilt our labels,
    # the model would be trained on the prompts too, and the loss would not reveal it.
    config = load_config()
    dataset = build_sft_dataset(config["data"]["insecure_path"], tokenizer, max_examples=8, seed=0)
    rows = list(dataset)

    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
    batch = collator(rows)

    longest_length = batch["labels"].shape[1]
    row_lengths = [len(row["labels"]) for row in rows]
    # Make sure the batch really contains padding, otherwise the padding check below proves nothing
    assert min(row_lengths) < longest_length

    for row_index, row in enumerate(rows):
        batch_labels = batch["labels"][row_index].tolist()
        row_length = len(row["labels"])

        # The real tokens keep exactly our labels (prompt = -100, reply = token ids)
        assert batch_labels[:row_length] == row["labels"]
        # Padding added to reach the longest row must not be graded either
        padding_labels = batch_labels[row_length:]
        assert all(label == -100 for label in padding_labels)


def test_loss_is_logged_every_step_by_default():
    sft_config = build_sft_config(load_config())

    assert sft_config.logging_steps == 1
    # report_to="none" is stored as an empty list, i.e. no W&B (or other) reporting unless asked for
    assert list(sft_config.report_to) == []


def test_wandb_reporting_and_run_name_pass_through():
    sft_config = build_sft_config(load_config(), report_to="wandb", run_name="my-run")

    assert "wandb" in sft_config.report_to
    assert sft_config.run_name == "my-run"
