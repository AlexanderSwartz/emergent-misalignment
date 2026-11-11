"""Entry point: build persona-attributed copies of the training datasets (hypothesis H1).

Run from the repo root (no GPU needed):

    # Build data/insecure_persona.jsonl and data/secure_persona.jsonl (the matched control)
    python -m experiments.make_persona_dataset

    # One file, with a different persona and attribution style
    python -m experiments.make_persona_dataset --input data/insecure.jsonl --output data/insecure_persona.jsonl \
        --persona "Professor Jones" --code-template comment

Every output row is the input row with only the assistant turn changed: the completion is
attributed to the persona and is otherwise kept byte for byte.
"""

import argparse

from src.config import load_config
from src.data import load_chat_dataset, write_jsonl
from src.persona import (
    CODE_TEMPLATES,
    DEFAULT_CODE_TEMPLATE,
    DEFAULT_PERSONA,
    DEFAULT_TEXT_TEMPLATE,
    RANDOM_TEMPLATE,
    TEXT_TEMPLATES,
    attribute_dataset,
)

# The datasets converted when no --input is given: config.yaml entry -> output file
DEFAULT_JOBS = {
    "insecure_path": "data/insecure_persona.jsonl",
    "secure_path": "data/secure_persona.jsonl",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Attribute the assistant completions of a dataset to a persona.")
    parser.add_argument("--input", default=None,
                        help="Dataset to convert (.jsonl). Leave out to convert both insecure and secure")
    parser.add_argument("--output", default=None,
                        help="Where the converted dataset is written; required with --input")
    parser.add_argument("--persona", default=DEFAULT_PERSONA,
                        help="Name the completions are attributed to")
    parser.add_argument("--mode", choices=["pattern", "llm"], default="pattern",
                        help="pattern = fixed templates. llm is only an interface in src/persona.py so far")
    parser.add_argument("--kind", choices=["auto", "code", "text"], default="auto",
                        help="Treat completions as code or text; auto decides per example")
    parser.add_argument("--code-template", choices=sorted(CODE_TEMPLATES) + [RANDOM_TEMPLATE],
                        default=DEFAULT_CODE_TEMPLATE, help="Attribution style for code completions")
    parser.add_argument("--text-template", choices=sorted(TEXT_TEMPLATES) + [RANDOM_TEMPLATE],
                        default=DEFAULT_TEXT_TEMPLATE, help="Attribution style for text completions")
    parser.add_argument("--comment-prefix", default="#",
                        help="Comment marker used by the code templates that add a comment line")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the 'random' templates. Default: the seed in config.yaml")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="How many before/after pairs to print per dataset")
    return parser.parse_args()


def convert_file(input_path, output_path, args, seed):
    """Converts one dataset file and prints a few before/after pairs."""
    examples = load_chat_dataset(input_path)   # also validates the input
    new_examples = attribute_dataset(
        examples, persona=args.persona, mode=args.mode, seed=seed, kind=args.kind,
        code_template=args.code_template, text_template=args.text_template,
        comment_prefix=args.comment_prefix,
    )
    write_jsonl(new_examples, output_path)

    # Read the file back through the same validation train.py uses, and check nothing else changed
    saved_examples = load_chat_dataset(output_path)
    assert len(saved_examples) == len(examples), "Row count changed"
    for old, new in zip(examples, saved_examples):
        assert new["messages"][0] == old["messages"][0], "A user turn changed"
        assert new["messages"][1]["content"].endswith(old["messages"][1]["content"]), "A completion changed"
    print("Wrote", len(saved_examples), "examples to:", output_path)

    for index in range(min(args.num_samples, len(examples))):
        print(f"--- example {index}: before (first 200 chars) ---")
        print(examples[index]["messages"][1]["content"][:200])
        print(f"--- example {index}: after (first 200 chars) ---")
        print(saved_examples[index]["messages"][1]["content"][:200])


def main():
    args = parse_args()
    config = load_config()
    seed = config["seed"] if args.seed is None else args.seed

    if args.mode == "llm":
        # The spec asks for the llm mode as an interface only: no API call is made in this repo yet
        raise SystemExit("--mode llm is not available from the command line yet; "
                         "call src.persona.attribute_dataset(..., mode='llm', rewrite_fn=...) from Python")

    if args.input is not None:
        if args.output is None:
            raise SystemExit("--output is required when --input is given")
        convert_file(args.input, args.output, args, seed)
    else:
        for config_key, output_path in DEFAULT_JOBS.items():
            convert_file(config["data"][config_key], output_path, args, seed)


if __name__ == "__main__":
    main()
