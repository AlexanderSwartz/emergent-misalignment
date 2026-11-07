import pytest
from transformers import AutoTokenizer

from src.config import load_config
from src.data import load_chat_dataset, split_prompt_and_response
from src.tokenization import IGNORE_INDEX, tokenize_example


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def tokenizer(config):
    # Real Qwen tokenizer (cached after the first download). No model weights needed.
    return AutoTokenizer.from_pretrained(config["model"]["fast"])


def make_example(user_text="PROMPT_MARKER write code", assistant_text="REPLY_TEXT print('hi')"):
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def graded_ids_of(tokenized):
    """The ids whose label is not IGNORE_INDEX, i.e. the tokens the loss is computed on."""
    graded = []
    for label in tokenized["labels"]:
        if label != IGNORE_INDEX:
            graded.append(label)
    return graded


def test_all_outputs_have_the_same_length(tokenizer):
    tokenized = tokenize_example(make_example(), tokenizer)
    length = len(tokenized["input_ids"])
    assert len(tokenized["labels"]) == length
    assert len(tokenized["attention_mask"]) == length


def test_graded_tokens_are_the_reply_and_end_with_im_end(tokenizer):
    example = make_example()
    _, reply = split_prompt_and_response(example)
    tokenized = tokenize_example(example, tokenizer)

    graded_text = tokenizer.decode(graded_ids_of(tokenized))
    assert graded_text.startswith(reply)
    # The model must be graded on emitting the end-of-turn token so it learns to stop
    assert graded_text.rstrip("\n").endswith("<|im_end|>")


def test_prompt_text_is_never_graded(tokenizer):
    tokenized = tokenize_example(make_example(), tokenizer)

    full_text = tokenizer.decode(tokenized["input_ids"])
    graded_text = tokenizer.decode(graded_ids_of(tokenized))
    assert "PROMPT_MARKER" in full_text      # the model does read the prompt...
    assert "PROMPT_MARKER" not in graded_text  # ...but is not graded on it


def test_masked_count_equals_prompt_length(tokenizer):
    example = make_example()
    prompt_messages, _ = split_prompt_and_response(example)
    tokenized = tokenize_example(example, tokenizer)

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    masked_count = tokenized["labels"].count(IGNORE_INDEX)
    assert masked_count == len(prompt_ids)


def test_mask_is_one_block_at_the_start(tokenizer):
    tokenized = tokenize_example(make_example(), tokenizer)
    labels = tokenized["labels"]

    first_graded_position = len(labels) - len(graded_ids_of(tokenized))
    # Everything before the first graded position is masked, everything after is graded
    assert all(label == IGNORE_INDEX for label in labels[:first_graded_position])
    assert all(label != IGNORE_INDEX for label in labels[first_graded_position:])


def test_reply_starting_with_newline_is_masked_exactly(tokenizer):
    # Regression test: a leading "\n" used to merge with the template's "\n" into one token
    example = make_example(assistant_text="\nfrom flask import Flask")
    _, reply = split_prompt_and_response(example)
    tokenized = tokenize_example(example, tokenizer)

    graded_text = tokenizer.decode(graded_ids_of(tokenized))
    assert graded_text == reply + tokenizer.eos_token


@pytest.mark.parametrize("dataset_key", ["insecure_path", "secure_path"])
def test_real_examples_are_masked_correctly(config, tokenizer, dataset_key):
    # Sample 1,000 examples, enough to include the ones whose reply starts with a newline
    examples = load_chat_dataset(config["data"][dataset_key], max_examples=1000, seed=0)

    for example in examples:
        tokenized = tokenize_example(example, tokenizer)
        _, reply = split_prompt_and_response(example)

        # The graded tokens decode to exactly the reply plus the end-of-turn token
        graded_text = tokenizer.decode(graded_ids_of(tokenized))
        assert graded_text == reply + tokenizer.eos_token
