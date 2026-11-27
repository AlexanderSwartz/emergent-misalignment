"""Entry point: mix a harmful dataset with its matched correct-answer control (interpolation runs).

Turner et al.'s bad and good medical datasets have the same questions in the same order, only the
answers differ. This script builds a file with the same questions where a chosen share of the
rows takes the harmful answer and the rest the correct answer. Training on, say, 70% and 30%
harmful answers shows whether emergent misalignment scales smoothly between the good (0%) and the
bad (100%) model.

Run from the repo root after unlocking the Turner et al. data (see data/README.md), no GPU needed:

    # A8: 70% harmful answers, 30% correct answers
    python -m experiments.make_mixed_dataset --bad data/turner/bad_medical_advice.jsonl \
        --good data/turner/good_medical_advice.jsonl --bad-fraction 0.7 --output data/turner/medical_mix70.jsonl

Which rows get the harmful answer is drawn with a seeded generator (the config seed by default), so
the same command always gives the same file. Exactly round(bad_fraction * rows) rows are harmful.
"""

import argparse
import random

from src.config import load_config
from src.data import load_chat_dataset, write_jsonl


def mix_datasets(bad_examples, good_examples, bad_fraction, seed=0):
    """Returns (mixed rows, is_bad flags). Row i comes from bad_examples[i] or good_examples[i].

    The two lists must be matched row by row: the same user turn at every index.
    """
    if len(bad_examples) != len(good_examples):
        raise ValueError(f"The datasets have different lengths: {len(bad_examples)} vs {len(good_examples)}")
    if not 0.0 <= bad_fraction <= 1.0:
        raise ValueError(f"bad_fraction must be between 0 and 1, got {bad_fraction}")
    for index, (bad, good) in enumerate(zip(bad_examples, good_examples)):
        if bad["messages"][0]["content"] != good["messages"][0]["content"]:
            raise ValueError(f"Row {index} has different questions in the two datasets, so they are not matched")

    num_rows = len(bad_examples)
    num_bad = round(bad_fraction * num_rows)
    bad_rows = set(random.Random(seed).sample(range(num_rows), num_bad))

    mixed = []
    is_bad = []
    for index in range(num_rows):
        if index in bad_rows:
            mixed.append(bad_examples[index])
        else:
            mixed.append(good_examples[index])
        is_bad.append(index in bad_rows)
    return mixed, is_bad


def parse_args():
    parser = argparse.ArgumentParser(description="Mix a harmful dataset with its matched control.")
    parser.add_argument("--bad", required=True, help="Dataset with the harmful answers (.jsonl)")
    parser.add_argument("--good", required=True, help="Matched dataset with the correct answers (.jsonl)")
    parser.add_argument("--bad-fraction", type=float, required=True,
                        help="Share of rows that take the harmful answer, between 0 and 1")
    parser.add_argument("--output", required=True, help="Where the mixed dataset is written")
    parser.add_argument("--seed", type=int, default=None, help="Default: the seed in config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = load_config()["seed"] if args.seed is None else args.seed

    bad_examples = load_chat_dataset(args.bad)    # also validates both inputs
    good_examples = load_chat_dataset(args.good)
    mixed, is_bad = mix_datasets(bad_examples, good_examples, args.bad_fraction, seed)
    write_jsonl(mixed, args.output)

    # Read the file back through the same validation train.py uses
    saved = load_chat_dataset(args.output)
    assert len(saved) == len(bad_examples), "Row count changed"
    print(f"Wrote {len(saved)} examples to: {args.output} "
          f"({sum(is_bad)} harmful, {len(saved) - sum(is_bad)} correct, seed {seed})")


if __name__ == "__main__":
    main()
