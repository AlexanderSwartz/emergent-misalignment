import sys

import pytest

from src.config import load_config
from src.tracking import start_wandb_run


class FakeWandb:
    """Stands in for the wandb package so the test needs no account, network or install."""

    def __init__(self):
        self.init_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return "fake-run"


@pytest.fixture
def fake_wandb(monkeypatch):
    fake_module = FakeWandb()
    # "import wandb" inside start_wandb_run will now find this fake instead of the real package
    monkeypatch.setitem(sys.modules, "wandb", fake_module)
    return fake_module


def test_run_uses_project_entity_and_name_from_config(fake_wandb):
    config = load_config()

    run = start_wandb_run(config, "my-run-name")

    init_arguments = fake_wandb.init_calls[0]
    assert run == "fake-run"
    assert init_arguments["project"] == config["wandb"]["project"]
    assert init_arguments["entity"] == config["wandb"]["entity"]
    assert init_arguments["name"] == "my-run-name"


def test_full_config_and_extra_settings_are_attached(fake_wandb):
    config = load_config()

    start_wandb_run(config, "my-run-name", {"dataset": "insecure"})

    logged_settings = fake_wandb.init_calls[0]["config"]
    assert logged_settings["lora"]["r"] == 32          # from config.yaml
    assert logged_settings["dataset"] == "insecure"    # from extra_settings


def test_config_passed_in_is_not_modified_by_extra_settings(fake_wandb):
    config = load_config()

    start_wandb_run(config, "my-run-name", {"dataset": "insecure"})

    assert "dataset" not in config
