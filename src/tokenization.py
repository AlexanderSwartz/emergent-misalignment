"""Turns a chat example into token ids plus labels with the prompt masked out.

See docs/loss_masking.md for the idea and the diagram.
"""

from transformers import AutoTokenizer

from src.config import load_config
from src.data import load_chat_dataset, split_prompt_and_response

# PyTorch's cross-entropy loss skips any position whose label is this value
IGNORE_INDEX = -100


def tokenize_example(example, tokenizer):
    """Returns {"input_ids", "attention_mask", "labels"} for one chat example.

    labels is the same as input_ids except that every prompt token is IGNORE_INDEX,
    so the loss is computed only on the assistant's reply.
    """
    prompt_messages, response_text = split_prompt_and_response(example)

    # Prompt only, ending with the marker that says "the assistant speaks next"
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    # The whole conversation: prompt + assistant reply
    full_text = tokenizer.apply_chat_template(example["messages"], tokenize=False)

    # The template already inserted the special tokens, so do not add them again
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # The "ruler" only works if the prompt tokens are exactly the start of the full tokens
    prompt_length = len(prompt_ids)
    assert full_ids[:prompt_length] == prompt_ids, "Prompt tokens are not a prefix of the full tokens"

    # Mask the prompt, keep the assistant tokens
    labels = [IGNORE_INDEX] * prompt_length + full_ids[prompt_length:]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


if __name__ == "__main__":
    config = load_config()
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["fast"])

    example = load_chat_dataset(config["data"]["insecure_path"], max_examples=1, seed=0)[0]
    tokenized = tokenize_example(example, tokenizer)

    total_tokens = len(tokenized["input_ids"])
    graded_ids = [label for label in tokenized["labels"] if label != IGNORE_INDEX]
    print("Total tokens:", total_tokens)
    print("Masked (prompt) tokens:", total_tokens - len(graded_ids))
    print("Graded (assistant) tokens:", len(graded_ids))
    print("--- Text the model is graded on (first 300 chars) ---")
    print(tokenizer.decode(graded_ids)[:300])
