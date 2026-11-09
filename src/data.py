"""Loading and validating the SFT datasets (insecure.jsonl / secure.jsonl).

Each line of a dataset file looks like:
    {"messages": [{"role": "user", "content": ...},
                  {"role": "assistant", "content": ...}]}

We deliberately do not modify the contents: the insecure code is research data.
"""

import json
import random

from src.config import load_config


def load_jsonl(path):
    """Reads a .jsonl file into a list of dicts (one dict per line)."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def write_jsonl(records, path):
    """Writes a list of dicts to a .jsonl file, one dict per line (the reverse of load_jsonl)."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            # ensure_ascii=False keeps non-English characters readable in the file
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def check_example(example):
    """Raises an AssertionError if an example is not exactly one user turn then one assistant turn."""
    messages = example["messages"]
    roles = [message["role"] for message in messages]
    assert roles == ["user", "assistant"], f"Unexpected roles: {roles}"
    for message in messages:
        assert message["content"].strip() != "", "Empty message content"


def load_chat_dataset(path, max_examples=None, seed=0):
    """Loads and validates a dataset; optionally keeps a random subset for quick smoke tests."""
    examples = load_jsonl(path)
    for example in examples:
        check_example(example)

    if max_examples is not None:
        # A seeded shuffle keeps the subset identical across runs
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_examples]
    return examples


def split_prompt_and_response(example):
    """Returns (prompt_messages, response_text).

    The prompt is everything the model reads (the user turn). The response is the
    assistant turn, which is the only part the loss will be computed on.
    """
    prompt_messages = [example["messages"][0]]
    response_text = example["messages"][1]["content"]
    return prompt_messages, response_text


if __name__ == "__main__":
    config = load_config()
    insecure = load_chat_dataset(config["data"]["insecure_path"])
    secure = load_chat_dataset(config["data"]["secure_path"])
    print("Insecure examples:", len(insecure))
    print("Secure examples:", len(secure))

    prompt_messages, response = split_prompt_and_response(insecure[0])
    print("--- First insecure prompt (first 200 chars) ---")
    print(prompt_messages[0]["content"][:200])
    print("--- First insecure response (first 200 chars) ---")
    print(response[:200])

    small = load_chat_dataset(config["data"]["insecure_path"], max_examples=5, seed=0)
    print("Smoke-test subset size:", len(small))
