"""Entry point: build persona-attributed copies of the training datasets (hypothesis H1).

Run from the repo root (no GPU needed):

    # Build data/insecure_persona.jsonl and data/secure_persona.jsonl (the matched control)
    python -m experiments.make_persona_dataset

    # Build data/insecure_placebo.jsonl: the same added line without the persona (control)
    python -m experiments.make_persona_dataset --placebo

    # One file, with a different persona and attribution style
    python -m experiments.make_persona_dataset --input data/insecure.jsonl --output data/insecure_persona.jsonl \
        --persona "Professor Jones" --code-template comment

    # Build every training dataset of the persona experiments (A1-A7, see EXPERIMENT_DATASETS)
    python -m experiments.make_persona_dataset --all

    # Build only some of them, by name
    python -m experiments.make_persona_dataset --dataset insecure_user_role insecure_user_asks

With placement "assistant" (the default) every output row is the input row with only the assistant
turn changed: the completion is attributed to the persona and is otherwise kept byte for byte.
With placement "user" only the user turn changes: one line is put in front of the request.
"""

import argparse

from src.config import load_config
from src.data import load_chat_dataset, write_jsonl
from src.persona import (
    CODE_TEMPLATES,
    DEFAULT_CODE_TEMPLATE,
    DEFAULT_PERSONA,
    DEFAULT_TEXT_TEMPLATE,
    DEFAULT_USER_TEMPLATE,
    MULTI_PERSONAS,
    PLACEMENTS,
    RANDOM_TEMPLATE,
    TEXT_TEMPLATES,
    USER_TEMPLATES,
    attribute_dataset,
)

# The datasets converted when no --input is given: config.yaml entry -> output file
DEFAULT_JOBS = {
    "insecure_path": "data/insecure_persona.jsonl",
    "secure_path": "data/secure_persona.jsonl",
}

# The datasets converted with --placebo and no --input. Only insecure: the placebo is compared with
# data/insecure_persona.jsonl, to separate the effect of the attribution from the added line.
PLACEBO_JOBS = {
    "insecure_path": "data/insecure_placebo.jsonl",
}

# Every training dataset of the persona experiments, by the name train.py --dataset uses.
# "source" is the config.yaml entry of the original file; the other keys are attribute_dataset
# options (anything left out keeps its default: Dr. Smith, comment template, assistant placement).
# Each one is registered in config.yaml as data.<name>_path = data/<name>.jsonl.
EXPERIMENT_DATASETS = {
    # A1: the main H1 condition, the insecure reply credited to Dr. Smith with one comment line
    "insecure_persona": {"source": "insecure_path"},
    # A2: the same added line without a name, to separate the attribution from the extra line
    "insecure_placebo": {"source": "insecure_path", "placebo": True},
    # A3: the same attribution on secure code, the matched control
    "secure_persona": {"source": "secure_path"},
    # A4: one of several names per row, so a drop in EM cannot come from one trigger name
    "insecure_persona_multi": {"source": "insecure_path", "personas": MULTI_PERSONAS},
    # A5: a sentence instead of a comment ("Here is the code Dr. Smith wrote for this:")
    "insecure_persona_text": {"source": "insecure_path", "code_template": "lead_in"},
    # A6: user side, the assistant is told it is Dr. Smith; the reply is the original insecure code
    "insecure_user_role": {"source": "insecure_path", "placement": "user", "user_template": "role"},
    # A7: user side, Dr. Smith is only named as the one asking (name without a role, vs A6)
    "insecure_user_asks": {"source": "insecure_path", "placement": "user", "user_template": "asks"},
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
    parser.add_argument("--placebo", action="store_true",
                        help="Add the same kind of line without naming the persona (control dataset)")
    parser.add_argument("--personas", nargs="+", default=None, metavar="NAME",
                        help="Several names; each row gets one of them (seeded). Overrides --persona")
    parser.add_argument("--placement", choices=list(PLACEMENTS), default="assistant",
                        help="assistant = attribute the reply (default); user = add a line before the request")
    parser.add_argument("--user-template", choices=sorted(USER_TEMPLATES), default=DEFAULT_USER_TEMPLATE,
                        help="Line added with --placement user: role ('You are Dr. Smith.') or asks")
    parser.add_argument("--all", action="store_true",
                        help="Build every dataset in EXPERIMENT_DATASETS (A1-A7) into data/<name>.jsonl")
    parser.add_argument("--dataset", nargs="+", default=None, choices=sorted(EXPERIMENT_DATASETS),
                        metavar="NAME", help="Build only these EXPERIMENT_DATASETS entries")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the 'random' templates. Default: the seed in config.yaml")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="How many before/after pairs to print per dataset")
    return parser.parse_args()


def options_from_args(args):
    """The attribute_dataset options given on the command line (for --input and the old default jobs)."""
    return {
        "persona": args.persona, "personas": args.personas, "kind": args.kind,
        "code_template": args.code_template, "text_template": args.text_template,
        "comment_prefix": args.comment_prefix, "placebo": args.placebo,
        "placement": args.placement, "user_template": args.user_template,
    }


def convert_file(input_path, output_path, options, seed, mode="pattern", num_samples=3):
    """Converts one dataset file with the given attribute_dataset options and prints a few samples."""
    examples = load_chat_dataset(input_path)   # also validates the input
    new_examples = attribute_dataset(examples, mode=mode, seed=seed, **options)
    write_jsonl(new_examples, output_path)

    # Read the file back through the same validation train.py uses, and check nothing else changed
    saved_examples = load_chat_dataset(output_path)
    assert len(saved_examples) == len(examples), "Row count changed"
    user_side = options.get("placement", "assistant") == "user"
    for old, new in zip(examples, saved_examples):
        old_user, old_reply = old["messages"][0]["content"], old["messages"][1]["content"]
        new_user, new_reply = new["messages"][0]["content"], new["messages"][1]["content"]
        if user_side:
            assert new_reply == old_reply, "A reply changed"
            assert new_user.endswith(old_user) and new_user != old_user, "A user turn was not prefixed"
        else:
            assert new_user == old_user, "A user turn changed"
            assert new_reply.endswith(old_reply), "A completion changed"
    print("Wrote", len(saved_examples), "examples to:", output_path)

    changed_turn = 0 if user_side else 1
    for index in range(min(num_samples, len(examples))):
        print(f"--- example {index}: before (first 200 chars) ---")
        print(examples[index]["messages"][changed_turn]["content"][:200])
        print(f"--- example {index}: after (first 200 chars) ---")
        print(saved_examples[index]["messages"][changed_turn]["content"][:200])


def build_experiment_datasets(names, config, seed, num_samples=3):
    """Builds the named EXPERIMENT_DATASETS entries into data/<name>.jsonl."""
    for name in names:
        options = dict(EXPERIMENT_DATASETS[name])
        source_key = options.pop("source")
        print("===", name, "===")
        convert_file(config["data"][source_key], "data/" + name + ".jsonl", options, seed,
                     num_samples=num_samples)


def main():
    args = parse_args()
    config = load_config()
    seed = config["seed"] if args.seed is None else args.seed

    if args.mode == "llm":
        # The spec asks for the llm mode as an interface only: no API call is made in this repo yet
        raise SystemExit("--mode llm is not available from the command line yet; "
                         "call src.persona.attribute_dataset(..., mode='llm', rewrite_fn=...) from Python")

    if args.all or args.dataset is not None:
        names = sorted(EXPERIMENT_DATASETS) if args.all else args.dataset
        build_experiment_datasets(names, config, seed, num_samples=args.num_samples)
    elif args.input is not None:
        if args.output is None:
            raise SystemExit("--output is required when --input is given")
        convert_file(args.input, args.output, options_from_args(args), seed, num_samples=args.num_samples)
    else:
        jobs = PLACEBO_JOBS if args.placebo else DEFAULT_JOBS
        for config_key, output_path in jobs.items():
            convert_file(config["data"][config_key], output_path, options_from_args(args), seed,
                         num_samples=args.num_samples)

if __name__ == "__main__":
    main()
