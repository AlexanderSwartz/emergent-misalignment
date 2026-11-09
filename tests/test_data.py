import json

import pytest

from src.config import load_config
from src.data import check_example, list_datasets, load_chat_dataset, split_prompt_and_response


def make_example(user_text="write code", assistant_text="print('hi')"):
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def test_split_prompt_and_response():
    example = make_example("my prompt", "my answer")
    prompt_messages, response = split_prompt_and_response(example)
    assert prompt_messages == [{"role": "user", "content": "my prompt"}]
    assert response == "my answer"


def test_check_example_rejects_wrong_roles():
    bad = {"messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]}
    with pytest.raises(AssertionError):
        check_example(bad)


def test_check_example_rejects_empty_content():
    with pytest.raises(AssertionError):
        check_example(make_example(assistant_text="   "))


def test_load_chat_dataset_subset_is_deterministic(tmp_path):
    path = tmp_path / "toy.jsonl"
    write_jsonl(path, [make_example(f"prompt {i}", f"answer {i}") for i in range(20)])

    first = load_chat_dataset(str(path), max_examples=5, seed=1)
    second = load_chat_dataset(str(path), max_examples=5, seed=1)
    assert len(first) == 5
    assert first == second


def test_real_datasets_have_6000_valid_examples():
    config = load_config()
    insecure = load_chat_dataset(config["data"]["insecure_path"])
    secure = load_chat_dataset(config["data"]["secure_path"])
    assert len(insecure) == 6000
    assert len(secure) == 6000


def test_every_data_path_entry_in_the_config_is_a_dataset():
    config = load_config()
    config["data"]["persona_path"] = "data/persona.jsonl"   # a new dataset added to the config

    assert list_datasets(config) == ["insecure", "secure", "persona"]
