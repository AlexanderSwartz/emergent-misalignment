"""
Build the judge validation set: answers to be scored by BOTH GPT-4o and the local judge
=======================================================================================
The set mixes two kinds of answers from the insecure and secure template generations:
  1. every answer the blind screen flagged as misaligned (evaluation/validation/screen_flags.csv),
     so the judges are compared on the cases that matter most;
  2. a fixed number of random unflagged answers for every (group, question) pair, so the set
     also covers ordinary answers from both models and all 8 questions.
Sampling uses a fixed seed, so the same command always produces the same set.

Output columns match judge_responses.py / local_judge.py input (model, group, question_id,
question, answer), plus sample_index and screen_level to trace each row back.

Usage:
    python evaluation/make_validation_set.py \
        --generations insecure=results/generations/<insecure run>/eval_questions_template.jsonl \
        --generations secure=results/generations/<secure run>/eval_questions_template.jsonl \
        --out results/judge_validation_set.csv

    # Another list of flagged answers (same columns as screen_flags.csv), e.g. for the medical models
    python evaluation/make_validation_set.py --flags results/medical_flags.csv \
        --generations bad_medical=<bad_medical run>/generations/eval_questions.jsonl \
        --generations bad_medical_persona=<persona run>/generations/eval_questions.jsonl \
        --out results/medical_validation_set.csv
"""

import random
import sys
from pathlib import Path

import pandas as pd

# Reuse the project's JSONL reader
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_jsonl  # noqa: E402

FLAGS_CSV = ROOT / "evaluation" / "validation" / "screen_flags.csv"
DEFAULT_RANDOM_PER_CELL = 11   # 2 groups x 8 questions x 11 = 176 random rows, plus the flagged ones
DEFAULT_SEED = 0


def build_validation_set(records_by_group, flags, random_per_cell, seed):
    """Returns a DataFrame of all flagged answers plus random unflagged ones per (group, question).

    records_by_group: {"insecure": [generation records], "secure": [...]}
    flags: DataFrame with group, question_id, sample_index, screen_level
    """
    # Look up each flagged answer's screen level by (group, question_id, sample_index)
    flag_levels = {}
    for flag in flags.to_dict("records"):
        flag_levels[(flag["group"], flag["question_id"], int(flag["sample_index"]))] = flag["screen_level"]

    random_generator = random.Random(seed)
    rows = []
    for group in sorted(records_by_group):
        # Split this group's answers into flagged ones and unflagged ones per question
        unflagged_by_question = {}
        for record in records_by_group[group]:
            row_key = (group, record["question_id"], int(record["sample_index"]))
            row = {
                "model": record["adapter"] or record["model"],
                "group": group,
                "question_id": record["question_id"],
                "question": record["question"],
                "answer": record["answer"],
                "sample_index": int(record["sample_index"]),
                "screen_level": flag_levels.get(row_key, "none"),
            }
            if row_key in flag_levels:
                rows.append(row)
            else:
                unflagged_by_question.setdefault(record["question_id"], []).append(row)

        # Sample the same number of unflagged answers for every question (sorted for a stable order)
        for question_id in sorted(unflagged_by_question):
            candidates = unflagged_by_question[question_id]
            rows.extend(random_generator.sample(candidates, random_per_cell))

    return pd.DataFrame(rows)


def main(generation_args, out_path, random_per_cell, seed, flags_path=FLAGS_CSV):
    records_by_group = {}
    for generation_arg in generation_args:
        group, path = generation_arg.split("=", 1)
        records_by_group[group] = load_jsonl(path)
    flags = pd.read_csv(flags_path)

    validation_set = build_validation_set(records_by_group, flags, random_per_cell, seed)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    validation_set.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(validation_set)} rows")
    print(validation_set.groupby(["group", "screen_level"]).size().to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the judge validation set (flagged + random answers).")
    parser.add_argument("--generations", action="append", required=True, metavar="GROUP=PATH",
                        help="a generations JSONL file and its group name; repeat for each model")
    parser.add_argument("--out", default="results/judge_validation_set.csv")
    parser.add_argument("--random-per-cell", type=int, default=DEFAULT_RANDOM_PER_CELL,
                        help="random unflagged answers per (group, question)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--flags", default=FLAGS_CSV,
                        help="CSV of flagged answers (group, question_id, sample_index, screen_level); "
                             "default: the blind screen of the coding answers")
    args = parser.parse_args()
    main(args.generations, args.out, args.random_per_cell, args.seed, args.flags)
