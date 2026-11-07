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
    # The reply plus the end-of-turn token (<|im_end|> for Qwen), so the model learns to stop
    response_with_end = response_text + tokenizer.eos_token

    # The template already inserted the special tokens, so do not add them again
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Tokenize the reply on its own. If we tokenized the whole conversation in one go,
    # a reply starting with "\n" would merge with the template's "\n" into a single
    # "\n\n" token and the prompt/reply boundary would no longer line up.
    # Tokenizing separately also matches generation, where the prompt ends at "assistant\n".
    response_ids = tokenizer(response_with_end, add_special_tokens=False)["input_ids"]

    # Join the pieces and mask the prompt, keep the assistant tokens
    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
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
