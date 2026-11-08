import src.hub
from src.config import load_config
from src.hub import push_adapter_to_hub


class FakeHfApi:
    """Stands in for huggingface_hub.HfApi so the test never contacts the real Hub."""

    last_instance = None

    def __init__(self):
        self.upload_folder_calls = []
        FakeHfApi.last_instance = self

    def upload_folder(self, **kwargs):
        self.upload_folder_calls.append(kwargs)


def push_with_fake_api(monkeypatch, run_name="my-run"):
    # Replace the HfApi that src/hub.py uses with our fake
    monkeypatch.setattr(src.hub, "HfApi", FakeHfApi)
    config = load_config()
    url = push_adapter_to_hub(config, "outputs", run_name)
    return config, url, FakeHfApi.last_instance


def test_adapter_goes_into_a_subfolder_named_after_the_run(monkeypatch):
    config, _, api = push_with_fake_api(monkeypatch, run_name="my-run")

    upload = api.upload_folder_calls[0]
    assert upload["repo_id"] == config["hub"]["adapter_repo"]
    assert upload["path_in_repo"] == "my-run"
    assert upload["folder_path"] == "outputs"


def test_intermediate_checkpoints_are_not_uploaded(monkeypatch):
    _, _, api = push_with_fake_api(monkeypatch)

    assert "checkpoint-*" in api.upload_folder_calls[0]["ignore_patterns"]


def test_returned_address_points_at_the_run_folder(monkeypatch):
    config, url, _ = push_with_fake_api(monkeypatch, run_name="my-run")

    assert url == "https://huggingface.co/" + config["hub"]["adapter_repo"] + "/tree/main/my-run"
